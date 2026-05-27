# The-Promise-and-Pitfalls-of-GenAI-Powered-Mobile-Apps

Lightweight open-source pipeline for:

- Identifying candidate GenAI apps from app metadata,
- Filtering GenAI-related user reviews,
- Tagging review with LLMs,
- Clustering reasons with HDBSCAN,
- Classifying reviews with taxonomy.

## Repository Layout

```text
open_source/
├── README.md
├── code/
│   ├── app_metadata_genai_keywords.csv
│   ├── identify_candidate_app.py
│   ├── reviews_genai_keywords.csv
│   ├── filter_genai_related_reviews.py
│   └── rating_analyse_by_llm.py
└── data/
    ├── app_metadata/
    ├── clustering_results/
    ├── manual_verification_samples/
    ├── reviews/
    └── taxonomy/
```

## What Each Script Does

- `code/identify_candidate_app.py`
  - Loads metadata-description text files and applies the app-metadata keyword pre-filter.
  - Uses `code/app_metadata_genai_keywords.csv` as the keyword source.
  - Writes keyword-level and file-level match statistics.
  - Sends keyword-matched descriptions to an LLM for final GenAI-app validation.
  - Outputs:
    - `data/app_metadata/description_filtered_by_keywords/`
    - `data/app_metadata/keyword_stats/`
    - `data/app_metadata/llm_positive_output.csv`
    - `data/app_metadata/llm_negative_output.csv`

- `code/filter_genai_related_reviews.py`
  - Splits app reviews into low-rating (`rating <= 3`) and high-rating (`rating > 3`) groups.
  - Filters each group for GenAI-related reviews using `reviews_genai_keywords.csv`.
  - Treats a trailing `*` in review keywords as a wildcard, for example `chat*` can match `chat`, `chatbot`, or `chatting`.
  - For non-English `body` text, uses the row's `translation` field when available.
  - Writes per-app filtered CSVs and `keyword_stats.csv` into rating-specific output folders.

- `code/rating_analyse_by_llm.py`
  - Extracts low-rating issues and high-rating strengths from filtered reviews with an LLM.
  - Aggregates extracted reasons into JSON/CSV files.
  - Runs first-stage and second-stage HDBSCAN clustering over unique reasons.
  - Classifies reviews against `data/taxonomy/low_rated_taxonomy.json` or `data/taxonomy/high_rated_taxonomy.json`.
  - Runs the full post-analysis loop for both `google` and `ios`.

## Data Notes

- `data/taxonomy/low_rated_taxonomy.json`
- `data/taxonomy/high_rated_taxonomy.json`
- `data/clustering_results/{low_rating,high_rating}/...`
- `data/manual_verification_samples/...`

## Requirements

Python 3.10+ recommended.

Install dependencies:

```bash
pip install pandas numpy openai httpx langdetect sentence-transformers hdbscan scikit-learn
```

## Environment Variables

Set before running:

```bash
export OPENAI_API_KEY="your_api_key"
export OPENAI_BASE_URL="OPENAI_BASE_URL"
```

## Quick Start

Run in this order:

```bash
python code/identify_candidate_app.py
python code/filter_genai_related_reviews.py
python code/rating_analyse_by_llm.py
```

## Reproducibility Tips

- Keep taxonomy files and input folder names consistent with script defaults.
- Validate on a small sample first (especially `body`, `rating`, and `Reason` columns).
- LLM inference + clustering can be time-consuming; keep intermediate outputs.
