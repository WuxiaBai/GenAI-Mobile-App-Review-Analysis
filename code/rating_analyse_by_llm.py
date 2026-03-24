import csv
import json
import os
import re
import shutil
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import httpx
import hdbscan
import numpy as np
import pandas as pd
from openai import BadRequestError, OpenAI
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import StandardScaler


LOW_RATING_PROMPT = (
    "You are a professional app review analyst. Analyze the following user review "
    "about a GenAI-powered mobile app. Please identify and summarize the "
    "main issues or concerns that might have led to a low rating. Provide a clear "
    "and objective analysis."
)

HIGH_RATING_PROMPT = (
    "You are a professional app review analyst. Analyze the following user review "
    "about a GenAI-powered mobile app. Please identify and summarize the "
    "main strengths or features that might have led to a high rating. Provide a "
    "clear and objective analysis."
)


def split_text_by_max_length(text: str, max_length: int = 131072) -> List[str]:
    segments: List[str] = []
    remaining = str(text)
    while len(remaining) > max_length:
        split_pos = remaining.rfind(" ", 0, max_length)
        if split_pos == -1:
            split_pos = max_length
        segments.append(remaining[:split_pos])
        remaining = remaining[split_pos:].strip()
    if remaining:
        segments.append(remaining)
    return segments


def create_openai_client(base_url: str, api_key: str, disable_ssl_verify: bool = True) -> OpenAI:
    http_client = httpx.Client(verify=not disable_ssl_verify)
    return OpenAI(base_url=base_url, api_key=api_key, http_client=http_client)


def resolve_openai_base_url(base_url: str) -> str:
    resolved = (base_url or "").strip()
    if not resolved:
        resolved = os.getenv("OPENAI_BASE_URL", "").strip()
    return resolved


def analyze_single_review(
    client: OpenAI,
    review_text: str,
    prompt: str,
    model_name: str,
) -> str:
    segments = split_text_by_max_length(review_text)
    reasons: List[str] = []
    for segment in segments:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": segment},
            ],
            temperature=0,
            stream=False,
        )
        reasons.append(response.choices[0].message.content.strip())
    return " ".join(item for item in reasons if item)


def write_failure_record(target_path: Path, comment: str, reason: str) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    error_df = pd.DataFrame({"Comment": [comment], "Reason": [reason]})
    error_df.to_csv(target_path, index=False)


def analyze_reviews_with_prompt(
    all_reviews: Sequence[str],
    app_names: Sequence[str],
    results_folder: Path,
    failed_folder: Path,
    prompt: str,
    model_name: str = "gpt-5.1",
    base_url: str = "",
    api_key: str = "",
) -> None:
    if not api_key:
        raise RuntimeError("Missing API key. Set OPENAI_API_KEY.")
    base_url = resolve_openai_base_url(base_url)

    results_folder.mkdir(parents=True, exist_ok=True)
    failed_folder.mkdir(parents=True, exist_ok=True)
    client = create_openai_client(base_url=base_url, api_key=api_key)

    for app_name, review in zip(app_names, all_reviews):
        result_file = results_folder / f"{app_name}.csv"
        failed_file = failed_folder / f"{app_name}.csv"
        original_review = str(review)

        if result_file.exists():
            continue

        try:
            reason = analyze_single_review(
                client=client,
                review_text=original_review,
                prompt=prompt,
                model_name=model_name,
            )
            pd.DataFrame({"Comment": [original_review], "Reason": [reason]}).to_csv(result_file, index=False)
        except BadRequestError as exc:
            error_msg = str(exc)
            if result_file.exists():
                if failed_file.exists():
                    failed_file.unlink()
                shutil.move(str(result_file), str(failed_file))
            else:
                if "content management policy" in error_msg.lower() or "content filtered" in error_msg.lower():
                    write_failure_record(
                        target_path=failed_file,
                        comment=original_review,
                        reason="[processing_failed: content_filtered]",
                    )
                else:
                    write_failure_record(
                        target_path=failed_file,
                        comment=original_review,
                        reason=f"[processing_failed: {error_msg[:160]}]",
                    )
        except Exception as exc:
            if result_file.exists():
                if failed_file.exists():
                    failed_file.unlink()
                shutil.move(str(result_file), str(failed_file))
            else:
                write_failure_record(
                    target_path=failed_file,
                    comment=original_review,
                    reason=f"[processing_failed: {type(exc).__name__}: {str(exc)[:160]}]",
                )


def analyze_low_rating_reviews(
    all_reviews: Sequence[str],
    app_names: Sequence[str],
    results_folder: Path,
    failed_folder: Path,
    model_name: str = "gpt-5.1",
    base_url: str = "",
    api_key: str = "",
) -> None:
    analyze_reviews_with_prompt(
        all_reviews=all_reviews,
        app_names=app_names,
        results_folder=results_folder,
        failed_folder=failed_folder,
        prompt=LOW_RATING_PROMPT,
        model_name=model_name,
        base_url=base_url,
        api_key=api_key,
    )


def analyze_high_rating_reviews(
    all_reviews: Sequence[str],
    app_names: Sequence[str],
    results_folder: Path,
    failed_folder: Path,
    model_name: str = "gpt-5.1",
    base_url: str = "",
    api_key: str = "",
) -> None:
    analyze_reviews_with_prompt(
        all_reviews=all_reviews,
        app_names=app_names,
        results_folder=results_folder,
        failed_folder=failed_folder,
        prompt=HIGH_RATING_PROMPT,
        model_name=model_name,
        base_url=base_url,
        api_key=api_key,
    )


def load_reviews_from_csv_folder(folder_path: Path, text_column: str = "body") -> Tuple[List[str], List[str]]:
    app_names: List[str] = []
    reviews: List[str] = []
    for csv_file in sorted(folder_path.glob("*.csv")):
        try:
            df = pd.read_csv(csv_file)
        except Exception:
            continue
        if text_column not in df.columns:
            continue
        for text in df[text_column].dropna().astype(str):
            app_names.append(csv_file.stem)
            reviews.append(text)
    return app_names, reviews


def extract_reasons_from_csv(folder_path: Path, output_json_file: Path, unique_file: Path) -> set[str]:
    """Extract reason phrases from CSV files and save JSON/CSV outputs."""
    results: Dict[str, List[str]] = {}
    unique_reasons: List[str] = []
    strings_to_remove = [
        "Possible Reason",
        "Possible Reasons",
        "Issue",
        "Reason",
        "Example",
        "Impact",
        "Review",
        "Issue:",
        "Possible Cause:",
        "Reason:",
        "Example:",
        "Analysis:",
        "Problem:",
        "Review:",
        "User Complaints:",
        "Impact:",
        "Reviews Mentioned",
        "User Feedback",
        "Analysis",
        "Summary",
    ]

    if not folder_path.is_dir():
        output_json_file.parent.mkdir(parents=True, exist_ok=True)
        unique_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_json_file, "w", encoding="utf-8") as json_file:
            json.dump({}, json_file, ensure_ascii=False, indent=4)
        pd.DataFrame(columns=["Unique reasons"]).to_csv(unique_file, index=False)
        return set()

    for csv_path in sorted(folder_path.glob("*.csv")):
        try:
            df = pd.read_csv(csv_path)
        except Exception:
            continue

        if "Reason" not in df.columns:
            continue

        reasons = df["Reason"].dropna().tolist()
        extracted_reasons: List[str] = []

        for reason in reasons:
            matches = re.findall(r"\. \*\*(.*?)\*\*?", str(reason))
            extracted_reasons.extend([match.replace(":", "") for match in matches])

        extracted_reasons = [s for s in extracted_reasons if s not in strings_to_remove]
        unique_reasons.extend(extracted_reasons)
        results[csv_path.stem] = extracted_reasons

    unique_strings = set(unique_reasons)
    output_json_file.parent.mkdir(parents=True, exist_ok=True)
    unique_file.parent.mkdir(parents=True, exist_ok=True)
    df_unique = pd.DataFrame(unique_strings, columns=["Unique reasons"])
    df_unique.to_csv(unique_file, index=False)

    with open(output_json_file, "w", encoding="utf-8") as json_file:
        json.dump(results, json_file, ensure_ascii=False, indent=4)

    return unique_strings


def extract_three_level_lists(taxonomy: Dict[str, Any]) -> Tuple[List[str], List[str], List[str]]:
    level_1_list: List[str] = []
    level_2_list: List[str] = []
    level_3_list: List[str] = []
    for level_1, level_2_dict in taxonomy.items():
        if level_1 not in level_1_list:
            level_1_list.append(level_1)
        if not isinstance(level_2_dict, dict):
            continue
        for level_2, level_3_dict in level_2_dict.items():
            if level_2 not in level_2_list:
                level_2_list.append(level_2)
            if not isinstance(level_3_dict, dict):
                continue
            for level_3 in level_3_dict.keys():
                if level_3 not in level_3_list:
                    level_3_list.append(level_3)
    return level_1_list, level_2_list, level_3_list


def cluster_with_hdbscan(
    csv_file: Path,
    output_file: Path,
    topics_csv_file: Path,
    min_cluster_size: int,
    min_samples: int | None = None,
    cluster_selection_method: str = "leaf",
    cluster_selection_epsilon: float = 0.0,
) -> pd.DataFrame:
    text_column = "Unique reasons"
    df = pd.read_csv(csv_file)
    comments = df[text_column].dropna().tolist()
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(comments)
    embeddings = StandardScaler().fit_transform(embeddings)
    if min_samples is None:
        min_samples = min_cluster_size
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_method=cluster_selection_method,
        cluster_selection_epsilon=cluster_selection_epsilon,
        prediction_data=True,
    )
    labels = clusterer.fit_predict(embeddings)

    initial_noise_count = int((labels == -1).sum())
    initial_noise_ratio = initial_noise_count / len(labels) * 100

    noise_mask = labels == -1
    if noise_mask.sum() > 0:
        membership_vectors = hdbscan.all_points_membership_vectors(clusterer)
        noise_indices = np.where(noise_mask)[0]
        for i in noise_indices:
            cluster_probs = membership_vectors[i]
            best_cluster = int(np.argmax(cluster_probs))
            labels[i] = best_cluster

    df["Cluster"] = labels

    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)

    cluster_topics: Dict[int, str] = {}
    for cluster_label in unique_labels:
        cluster_mask = labels == cluster_label
        cluster_embeddings = embeddings[cluster_mask]
        cluster_indices = np.where(cluster_mask)[0]

        if len(cluster_embeddings) > 0:
            cluster_center = np.mean(cluster_embeddings, axis=0)
            distances = np.linalg.norm(cluster_embeddings - cluster_center, axis=1)
            closest_idx = cluster_indices[int(np.argmin(distances))]
            cluster_topics[int(cluster_label)] = comments[closest_idx]

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as file:
        file.write("HDBSCAN clustering results\n")
        file.write("=" * 60 + "\n")
        file.write(f"Total points: {len(comments)}\n")
        file.write(f"Clusters: {n_clusters}\n")
        file.write(f"Noise points: {initial_noise_count} ({initial_noise_ratio:.2f}%)\n")
        file.write(f"min_cluster_size: {min_cluster_size}\n")
        file.write(f"min_samples: {min_samples}\n")
        file.write(f"cluster_selection_method: {cluster_selection_method}\n")
        file.write(f"cluster_selection_epsilon: {cluster_selection_epsilon}\n")
        file.write("=" * 60 + "\n\n")

        for cluster_label in sorted(df["Cluster"].unique()):
            file.write(f"Cluster {cluster_label}:\n")
            cl = int(cluster_label)
            if cl in cluster_topics:
                file.write(f"  Topic: {cluster_topics[cl]}\n")
            cluster_comments = df[df["Cluster"] == cluster_label][text_column].tolist()
            file.write(f"  Count: {len(cluster_comments)}\n")
            for comment in cluster_comments:
                file.write(f"  - {comment}\n")
            file.write("\n")

    topics_df = pd.DataFrame(
        {text_column: [cluster_topics[label] for label in sorted(cluster_topics.keys())]}
    )
    topics_csv_file.parent.mkdir(parents=True, exist_ok=True)
    topics_df.to_csv(topics_csv_file, index=False, encoding="utf-8")

    return df.sort_values(by="Cluster")


def classify_reviews_with_taxonomy(
    input_folder: Path,
    taxonomy_json: Path,
    output_folder: Path,
    batch_size: int = 100,
    model_name: str = "gpt-5.1",
    base_url: str = "",
    api_key: str = "",
) -> None:
    if not api_key:
        raise RuntimeError("Missing API key. Set OPENAI_API_KEY.")
    base_url = resolve_openai_base_url(base_url)

    output_folder.mkdir(parents=True, exist_ok=True)
    with open(taxonomy_json, "r", encoding="utf-8") as f:
        taxonomy = json.load(f)
    _, _, level_3_list = extract_three_level_lists(taxonomy)

    taxonomy_prompt = f"Here is the taxonomy:\n\n{json.dumps(taxonomy, ensure_ascii=False, indent=4)}\n\n"
    client = create_openai_client(base_url=base_url, api_key=api_key)

    for csv_path in sorted(input_folder.glob("*.csv")):
        output_path = output_folder / f"{csv_path.stem}.txt"
        if output_path.exists():
            continue

        comments: List[str] = []
        with open(csv_path, "r", encoding="utf-8") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                body = row.get("body", "")
                if body and len(str(body).strip()) > 3:
                    comments.append(str(body).strip())

        comments = list(dict.fromkeys(comments))

        for i in range(0, len(comments), batch_size):
            batch_comments = comments[i : i + batch_size]
            comments_prompt = "\n".join(f"- {c}" for c in batch_comments if c.strip())
            if not comments_prompt:
                continue

            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert in categorizing text and evaluating information quality.",
                    },
                    {
                        "role": "user",
                        "content": (
                            "There is a batch of user review data and a pre-constructed taxonomy. "
                            "Classify the reviews according to the taxonomy; each review may belong "
                            "to multiple categories.\n\n"
                            "The classification results must be selected from the leaf-level categories "
                            "of the taxonomy. Only provide classifications clearly supported by the "
                            "comment text. If the comment does not belong to any category, return None.\n\n"
                            f"Here are the reviews to classify:\n{comments_prompt}\n\n"
                            f"{taxonomy_prompt}"
                            "Here is the range of taxonomy leaf-level categories:\n"
                            f"{level_3_list}\n\n"
                            "For each review, return the result in the following format without index "
                            "numbers or Markdown formatting:\n"
                            "comment: [review text]\n"
                            'Classification: ["Category A", "Category B", ...] or None\n'
                        ),
                    },
                ],
                stream=False,
            )
            content = response.choices[0].message.content
            with open(output_path, "a", encoding="utf-8") as txt_file:
                txt_file.write(content)


def run_post_analysis_pipeline(
    market: str,
    rating_band: str,
    data_root: Path,
    api_key: str,
    base_url: str,
) -> None:
    """Extract reasons, run two HDBSCAN stages, then taxonomy tagging for one market and rating band."""
    if rating_band not in ("low", "high"):
        raise ValueError("rating_band must be 'low' or 'high'")

    # 1. extract reasons for low/high rating reviews
    reasons_folder = data_root / f"{market}_reasons_for_{rating_band}_rating_analyse_by_llm"
    taxonomy_name = "low_rated_taxonomy.json" if rating_band == "low" else "high_rated_taxonomy.json"
    taxonomy_path = data_root / taxonomy_name

    out_base = data_root / "clustering_results" / f"{rating_band}_rating"
    out_base.mkdir(parents=True, exist_ok=True)

    reasons_json = out_base / f"{market}_{rating_band}_rating_reasons.json"
    unique_csv = out_base / f"{market}_unique_reasons.csv"
    extract_reasons_from_csv(reasons_folder, reasons_json, unique_csv)

    # 2. run two HDBSCAN stages
    first_txt = out_base / f"{market}_hdbscan_first_clustering_results.txt"
    first_topics = out_base / f"{market}_hdbscan_first_clustering_results_topics.csv"
    cluster_with_hdbscan(
        unique_csv,
        first_txt,
        first_topics,
        min_cluster_size=5,
        min_samples=3,
        cluster_selection_method="leaf",
        cluster_selection_epsilon=0.0,
    )

    second_txt = out_base / f"{market}_hdbscan_second_clustering_results.txt"
    second_topics = out_base / f"{market}_hdbscan_second_clustering_results_topics.csv"
    cluster_with_hdbscan(
        first_topics,
        second_txt,
        second_topics,
        min_cluster_size=3,
        min_samples=1,
        cluster_selection_method="leaf",
        cluster_selection_epsilon=0.0,
    )

    # 3. classify reviews with taxonomy
    reviews_input = data_root / f"{market}_reviews_{rating_band}_rating_selected_genai_related"
    tagging_out = out_base / f"{market}_reviews_tagging_by_llm"
    classify_reviews_with_taxonomy(
        input_folder=reviews_input,
        taxonomy_json=taxonomy_path,
        output_folder=tagging_out,
        model_name="gpt-5.1",
        base_url=base_url,
        api_key=api_key,
    )


def main() -> None:
    data_root = "/data"
    failed_folder = data_root / "failed_apps/failed_tagging_apps"
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    base_url = os.getenv("OPENAI_BASE_URL", "").strip()

    markets: Tuple[str, ...] = ("google", "ios")

    for market in markets:
        low_input = data_root / f"{market}_reviews_low_rating_selected_genai_related"
        high_input = data_root / f"{market}_reviews_high_rating_selected_genai_related"
        low_out = data_root / f"{market}_reasons_for_low_rating_analyse_by_llm"
        high_out = data_root / f"{market}_reasons_for_high_rating_analyse_by_llm"

        # 1. tagging low rating reviews with LLM
        low_app_names, low_reviews = load_reviews_from_csv_folder(low_input, text_column="body")
        analyze_low_rating_reviews(
            all_reviews=low_reviews,
            app_names=low_app_names,
            results_folder=low_out,
            failed_folder=failed_folder,
            model_name="gpt-5.1",
            base_url=base_url,
            api_key=api_key,
        )

        # 2. tagging high rating reviews with LLM
        high_app_names, high_reviews = load_reviews_from_csv_folder(high_input, text_column="body")
        analyze_high_rating_reviews(
            all_reviews=high_reviews,
            app_names=high_app_names,
            results_folder=high_out,
            failed_folder=failed_folder,
            model_name="gpt-5.1",
            base_url=base_url,
            api_key=api_key,
        )

    for market in markets:
        for rating_band in ("low", "high"):
            run_post_analysis_pipeline(
                market=market,
                rating_band=rating_band,
                data_root=data_root,
                api_key=api_key,
                base_url=base_url,
            )


if __name__ == "__main__":
    main()
