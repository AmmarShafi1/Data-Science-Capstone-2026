# Data Science Capstone 2026

## Project Overview

This project analyzes AI exposure across occupations using O*NET data and Bureau of Labor Statistics (BLS) employment data. It computes AI similarity scores for Standard Occupational Classification (SOC) codes using sentence embeddings and contrastive learning.

## Key Components

### 1. SOC AI Similarity Calculator (`soc_ai_similarity.py`)
Computes AI exposure scores for occupations using:
- **Technology Skills** from O*NET database
- **Contrastive scoring**: `ai_exposure = ai_sim - max(automation_sim, 0)`
- Sentence embeddings via `sentence-transformers/all-MiniLM-L6-v2`

### 2. BLS CES Earnings Data (`bls_ces_earnings.py`)
Fetches monthly average hourly earnings (2022-2025) from the BLS API for:
- Total Private sector
- Key NAICS supersectors (Information, Professional & Business Services, etc.)

## Data Sources

- **O*NET Database**: [onetcenter.org](https://www.onetcenter.org/database.html)
- **BLS API**: [bls.gov](https://www.bls.gov/developers/)

## Setup

```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
.\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

```bash
# Run AI similarity analysis
python soc_ai_similarity.py

# Fetch BLS earnings data
python bls_ces_earnings.py
```

## Output Files

- `soc_ai_similarity.csv` - AI exposure scores by SOC code
- `ces_hourly_earnings_2022_2025.csv` - BLS earnings data
- `ai_intensity_by_naics2.csv` - AI intensity by 2-digit NAICS
- `ai_intensity_by_ces_bucket.csv` - AI intensity by CES bucket

## Requirements

- Python 3.10+
- pandas
- numpy
- sentence-transformers
- scikit-learn
- requests
