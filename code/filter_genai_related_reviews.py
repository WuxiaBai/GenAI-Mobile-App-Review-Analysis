from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Pattern
import ast
import pandas as pd
import re

try:
    from langdetect import LangDetectException, detect
except ImportError:
    LangDetectException = Exception
    detect = None


def load_keywords(keyword_csv: Path) -> List[str]:
    keywords_df = pd.read_csv(keyword_csv)
    keywords = keywords_df.iloc[:, 0].dropna().astype(str).str.strip().tolist()
    return [keyword for keyword in keywords if keyword]


def build_keyword_pattern(keyword: str) -> Pattern[str]:
    if keyword.endswith("*"):
        prefix = keyword[:-1].strip()
        return re.compile(rf"(?<!\w){re.escape(prefix)}\w*(?!\w)", re.IGNORECASE)
    return re.compile(rf"(?<!\w){re.escape(keyword)}(?!\w)", re.IGNORECASE)


def build_keyword_patterns(keywords: List[str]) -> Dict[str, Pattern[str]]:
    return {keyword: build_keyword_pattern(keyword) for keyword in keywords}


def contains_any_keyword(text: str, keyword_patterns: Dict[str, Pattern[str]]) -> bool:
    return any(pattern.search(text) for pattern in keyword_patterns.values())


def normalize_text_value(value: object) -> Optional[str]:
    if not isinstance(value, str):
        return None

    text = value.strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return None

    return text


def extract_translation_body(value: object) -> Optional[str]:
    text = normalize_text_value(value)
    if text is None:
        return None

    try:
        parsed_value = ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return text

    if isinstance(parsed_value, dict):
        return normalize_text_value(parsed_value.get("body"))

    return text


def looks_like_english(text: str) -> bool:
    letters = [character for character in text if character.isalpha()]
    if not letters:
        return False

    ascii_letters = sum(1 for character in letters if character.isascii())
    return ascii_letters / len(letters) >= 0.85


def is_english_review(text: str, detected_language: object = None) -> bool:
    language = normalize_text_value(detected_language)
    if language is not None:
        return language.lower().startswith("en")

    if detect is not None:
        try:
            return detect(text) == "en"
        except LangDetectException:
            pass

    return looks_like_english(text)


def select_text_for_keyword_matching(row: pd.Series) -> Optional[str]:
    body_text = normalize_text_value(row.get("body"))
    if body_text is None:
        return None

    if is_english_review(body_text, row.get("detected_language")):
        return body_text

    return extract_translation_body(row.get("translation"))


def collect_keyword_occurrences(
    text: str,
    keyword_patterns: Dict[str, Pattern[str]],
    stats: Dict[str, int],
) -> None:
    for keyword, pattern in keyword_patterns.items():
        stats[keyword] += len(pattern.findall(text))


def filter_genai_related_reviews(
    input_folder: Path,
    keywords: List[str],
    output_folder: Path,
    stats_output_file: Path,
) -> int:
    output_folder.mkdir(parents=True, exist_ok=True)
    stats_output_file.parent.mkdir(parents=True, exist_ok=True)

    keyword_stats: Dict[str, int] = defaultdict(int)
    keyword_patterns = build_keyword_patterns(keywords)

    for file_path in sorted(input_folder.glob("*.csv")):
        try:
            source_df = pd.read_csv(file_path)
        except Exception:
            continue

        if "body" not in source_df.columns:
            continue

        match_texts = source_df.apply(select_text_for_keyword_matching, axis=1)
        matched_mask = match_texts.apply(
            lambda value: isinstance(value, str) and contains_any_keyword(value, keyword_patterns)
        )
        matched_rows = source_df[matched_mask]
        for match_text in match_texts[matched_mask].dropna().astype(str):
            collect_keyword_occurrences(match_text, keyword_patterns, keyword_stats)

        if matched_rows.empty:
            continue

        destination_path = output_folder / file_path.name
        if destination_path.exists():
            existing_df = pd.read_csv(destination_path)
            merged_df = pd.concat([existing_df, matched_rows], ignore_index=True, sort=False)
            merged_df.to_csv(destination_path, index=False)
        else:
            matched_rows.to_csv(destination_path, index=False)

    stats_df = pd.DataFrame(
        [{"Keyword": keyword, "Occurrences": keyword_stats.get(keyword, 0)} for keyword in keywords]
    ).sort_values("Keyword")
    stats_df.to_csv(stats_output_file, index=False)

    return int(stats_df["Occurrences"].sum())


def filter_ratings(
    input_folder: Path,
    low_rating_output_folder: Path,
    high_rating_output_folder: Path,
) -> None:
    low_rating_output_folder.mkdir(parents=True, exist_ok=True)
    high_rating_output_folder.mkdir(parents=True, exist_ok=True)

    for file_path in sorted(input_folder.glob("*.csv")):
        if file_path.stat().st_size <= 0:
            continue

        try:
            df = pd.read_csv(file_path)
        except Exception:
            continue

        if "rating" not in df.columns:
            continue

        df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
        low_rating_df = df[df["rating"] <= 3]
        high_rating_df = df[df["rating"] > 3]

        if not low_rating_df.empty:
            low_rating_df.to_csv(low_rating_output_folder / file_path.name, index=False)
        if not high_rating_df.empty:
            high_rating_df.to_csv(high_rating_output_folder / file_path.name, index=False)


def main() -> None:
    # 1. split the reviews into two parts: low rating and high rating
    market = "google"  # "ios" or "google"
    reviews_input_folder = Path(f"data/{market}_reviews")
    reviews_low_rating_output_folder = Path(f"data/{market}_reviews_low_rating_selected")
    reviews_high_rating_output_folder = Path(f"data/{market}_reviews_high_rating_selected")
    filter_ratings(reviews_input_folder, reviews_low_rating_output_folder, reviews_high_rating_output_folder)

    # 2. filter the reviews by keywords
    keyword_csv = Path("reviews_genai_keywords.csv")
    keywords = load_keywords(keyword_csv)
    reviews_low_rating_genai_related_output_folder = Path(
        f"data/{market}_reviews_low_rating_selected_genai_related"
    )
    reviews_high_rating_genai_related_output_folder = Path(
        f"data/{market}_reviews_high_rating_selected_genai_related"
    )
    filter_genai_related_reviews(
        input_folder=reviews_low_rating_output_folder,
        keywords=keywords,
        output_folder=reviews_low_rating_genai_related_output_folder,
        stats_output_file=reviews_low_rating_genai_related_output_folder / "keyword_stats.csv",
    )
    filter_genai_related_reviews(
        input_folder=reviews_high_rating_output_folder,
        keywords=keywords,
        output_folder=reviews_high_rating_genai_related_output_folder,
        stats_output_file=reviews_high_rating_genai_related_output_folder / "keyword_stats.csv",
    )


if __name__ == "__main__":
    main()
