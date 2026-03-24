import csv
import json
import os
import re
import shutil
import threading
from pathlib import Path
from queue import Queue
from threading import Lock
from typing import Dict, List, Optional, Set, Tuple
import openai


def load_keywords_from_csv(csv_path: Path) -> Set[str]:
    keywords: Set[str] = set()
    with csv_path.open("r", encoding="utf-8", newline="") as csv_file:
        reader = csv.reader(csv_file)
        for row in reader:
            keywords.update(item.strip() for item in row if item.strip())
    return keywords


def keyword_matches_text(keyword: str, text: str) -> bool:
    pattern = r"\b" + re.escape(keyword).replace(r"\*", r".*") + r"\b"
    return bool(re.search(pattern, text, re.IGNORECASE))


def run_keyword_filter(
    input_folder: Path,
    output_folder: Path,
    keywords: Set[str],
    stats_dir: Path,
) -> None:
    output_folder.mkdir(parents=True, exist_ok=True)
    stats_dir.mkdir(parents=True, exist_ok=True)

    keyword_stats: Dict[str, int] = {keyword: 0 for keyword in keywords}
    file_keyword_stats: Dict[str, List[Optional[str]]] = {}

    for folder, _, filenames in os.walk(input_folder):
        for filename in filenames:
            if not filename.endswith(".txt"):
                continue

            file_path = Path(folder) / filename
            output_name = filename[:-4]
            output_path = output_folder / f"{output_name}.txt"

            if output_path.exists():
                continue

            text = file_path.read_text(encoding="utf-8").lower()
            matched_keywords: List[str] = []

            for keyword in keywords:
                if keyword_matches_text(keyword.lower(), text):
                    matched_keywords.append(keyword)
                    keyword_stats[keyword] += 1

            if matched_keywords:
                shutil.copyfile(file_path, output_path)

            file_keyword_stats[output_name] = [str(Path(folder).resolve()), *matched_keywords]

    file_stats_path = stats_dir / "file_keyword_stats.csv"
    with file_stats_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "source_folder", "matched_keywords"])
        for filename, values in file_keyword_stats.items():
            source_folder = values[0] if values else ""
            matched = ";".join(values[1:]) if len(values) > 1 else ""
            writer.writerow([filename, source_folder, matched])

    keyword_stats_path = stats_dir / "keyword_stats.csv"
    with keyword_stats_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["keyword", "occurrences"])
        for keyword, count in keyword_stats.items():
            writer.writerow([keyword, count])


def read_processed_filenames(csv_file: Path) -> Set[str]:
    filenames: Set[str] = set()
    if not csv_file.exists():
        return filenames

    try:
        with csv_file.open("r", encoding="utf-8", newline="") as csvfile:
            reader = csv.reader(csvfile)
            next(reader, None)
            for row in reader:
                if not row:
                    continue
                name = row[0].strip()
                filenames.add(name[:-4] if name.endswith(".txt") else name)
    except Exception:
        return filenames

    return filenames


def extract_json_values(text_content: str) -> Optional[str]:
    try:
        data = json.loads(text_content)
    except json.JSONDecodeError:
        return None
    return " ".join(str(value) for value in data.values())


def evaluate_description_with_llm(
    extracted_content: str,
    model_name: str,
    base_url: str,
    api_key: str,
) -> str:
    SYSTEM_PROMPT = (
        "Given the following Android app description, determine whether it clearly indicates that the app is an GenAI-powered application with generative AI (AIGC) capabilities (such as text generation, image generation, or audio/video generation). Answer strictly in the format: Yes/No, with a brief explanation based only on the description."
    )
    client = openai.OpenAI(api_key=api_key, base_url=base_url)
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": extracted_content},
        ],
        temperature=0,
        stream=False,
    )
    return response.choices[0].message.content.strip().lower()


def process_single_description(
    file_info: Tuple[Path, str],
    processed_filenames: Set[str],
    positive_writer: csv.writer,
    negative_writer: csv.writer,
    file_locks: Dict[str, Lock],
    stats: Dict[str, int],
    model_name: str,
    base_url: str,
    api_key: str,
) -> None:
    file_path, file_name_without_ext = file_info

    if file_name_without_ext in processed_filenames:
        stats["skipped"] += 1
        return

    try:
        content = file_path.read_text(encoding="utf-8")
        extracted_content = extract_json_values(content)
        if extracted_content is None:
            stats["error"] += 1
            return

        llm_result = evaluate_description_with_llm(
            extracted_content=extracted_content,
            model_name=model_name,
            base_url=base_url,
            api_key=api_key,
        )

        if llm_result.startswith("yes"):
            with file_locks["positive_csv"]:
                positive_writer.writerow([file_name_without_ext, extracted_content, llm_result])
            stats["positive"] += 1
        elif llm_result.startswith("no"):
            with file_locks["negative_csv"]:
                negative_writer.writerow([file_name_without_ext, extracted_content, llm_result])
            stats["negative"] += 1
        else:
            stats["error"] += 1
            return

        stats["processed"] += 1
    except Exception:
        stats["error"] += 1


def run_llm_filter(
    input_folder: Path,
    positive_output_csv: Path,
    negative_output_csv: Path,
    model_name: str,
    base_url: str,
    api_key: str,
    num_threads: int = 8,
) -> Dict[str, int]:
    positive_output_csv.parent.mkdir(parents=True, exist_ok=True)
    negative_output_csv.parent.mkdir(parents=True, exist_ok=True)
    positive_output_csv.touch(exist_ok=True)
    negative_output_csv.touch(exist_ok=True)

    processed_filenames = read_processed_filenames(positive_output_csv)
    processed_filenames.update(read_processed_filenames(negative_output_csv))

    positive_outfile = positive_output_csv.open("a", encoding="utf-8", newline="")
    negative_outfile = negative_output_csv.open("a", encoding="utf-8", newline="")
    positive_writer = csv.writer(positive_outfile)
    negative_writer = csv.writer(negative_outfile)

    if positive_output_csv.stat().st_size == 0:
        positive_writer.writerow(["filename", "description", "model_check_result"])
    if negative_output_csv.stat().st_size == 0:
        negative_writer.writerow(["filename", "description", "model_check_result"])

    file_locks = {"positive_csv": Lock(), "negative_csv": Lock()}
    stats = {"processed": 0, "positive": 0, "negative": 0, "skipped": 0, "error": 0}

    files_to_process: List[Tuple[Path, str]] = []
    for root, _, files in os.walk(input_folder):
        for file_name in files:
            if not file_name.endswith(".txt"):
                continue
            file_path = Path(root) / file_name
            if file_name.endswith(".txt.txt"):
                stem = file_name[:-8]
            else:
                stem = file_name[:-4]
            if stem not in processed_filenames:
                files_to_process.append((file_path, stem))

    def worker(file_queue: Queue) -> None:
        while True:
            file_info = file_queue.get()
            if file_info is None:
                file_queue.task_done()
                break
            try:
                process_single_description(
                    file_info=file_info,
                    processed_filenames=processed_filenames,
                    positive_writer=positive_writer,
                    negative_writer=negative_writer,
                    file_locks=file_locks,
                    stats=stats,
                    model_name=model_name,
                    base_url=base_url,
                    api_key=api_key,
                )
            finally:
                file_queue.task_done()

    file_queue: Queue = Queue()
    for info in files_to_process:
        file_queue.put(info)

    threads = []
    for _ in range(num_threads):
        t = threading.Thread(target=worker, args=(file_queue,), daemon=True)
        t.start()
        threads.append(t)

    file_queue.join()

    for _ in range(num_threads):
        file_queue.put(None)
    for t in threads:
        t.join()

    positive_outfile.close()
    negative_outfile.close()
    return stats


def main() -> None:
    # 1. filter the app metadata by keywords
    keywords_csv = "app_metadata_filter_keywords.csv"
    keyword_input_folder = "data/app_metadata/description"
    keyword_output_folder = "data/app_metadata/description_filtered_by_keywords"
    keyword_stats_dir = "data/app_metadata/keyword_stats"

    keywords = load_keywords_from_csv(keywords_csv)
    run_keyword_filter(
        input_folder=keyword_input_folder,
        output_folder=keyword_output_folder,
        keywords=keywords,
        stats_dir=keyword_stats_dir,
    )

    # 2. identify the candidate app by LLM
    llm_input_folder = keyword_output_folder
    positive_output_csv = "data/app_metadata/llm_positive_output.csv"
    negative_output_csv = "data/app_metadata/llm_negative_output.csv"

    stats = run_llm_filter(
        input_folder=llm_input_folder,
        positive_output_csv=positive_output_csv,
        negative_output_csv=negative_output_csv,
        model_name="gpt-5.1",
        base_url= "Your base URL",
        api_key= "Your API key",
        num_threads=8,
    )
    print(
        f"Processed={stats['processed']} Positive={stats['positive']} "
        f"Negative={stats['negative']} Skipped={stats['skipped']} Error={stats['error']}"
    )


if __name__ == "__main__":
    main()
