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
├── code/
│   ├── identify_candidate_app.py
│   ├── filter_genai_related_reviews.py
│   └── rating_analyse_by_llm.py
└── data/
    ├── app_metadata/
    ├── clustering_results/
    ├── manual_verification_samples/
    └── taxonomy/
```

## What Each Script Does

- `code/identify_candidate_app.py`
  - Keyword pre-filter on app descriptions
  - LLM-based validation for candidate GenAI apps
  - Outputs:
    - `data/app_metadata/llm_positive_output.csv`
    - `data/app_metadata/llm_negative_output.csv`

- `code/filter_genai_related_reviews.py`
  - Split reviews into low/high rating groups
  - Filter GenAI-related reviews by keywords

- `code/rating_analyse_by_llm.py`
  - LLM reason extraction for low/high reviews
  - Two-stage HDBSCAN clustering
  - Taxonomy-based review tagging
  - Runs full loops for both `google` and `ios`

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
