# Lifetime Value (LTV) Prediction for Digital Neobanks

A practical, production-style implementation of a **1-Year Lifetime Value (LTV)** prediction system for a digital neobank.

Early scoring window (KYC + 30 days) · LightGBM + Tweedie · Fully reproducible on a public Kaggle dataset.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![LightGBM](https://img.shields.io/badge/LightGBM-Tweedie-green)](https://lightgbm.readthedocs.io/)

---

## Why this repository?

Most public LTV examples are either:
- Toy regressions on synthetic data, or
- Black-box production systems that cannot be shared

This repo bridges the gap. It implements a clean, realistic pipeline inspired by how leading digital banks approach early-lifecycle user value, generalized and free of any proprietary references. It is trained and evaluated on the real public [FinTech Customer Life Time Value (LTV) Dataset](https://www.kaggle.com/datasets/harunrai/fintech-customer-life-time-value-ltv-dataset).

---

## Key Design Choices

| Aspect                  | Choice                                      |
|-------------------------|---------------------------------------------|
| Scoring window          | KYC + **30 days**                           |
| Prediction target       | **1-Year** Lifetime Value                   |
| Model                   | LightGBM with Tweedie (variance power 1.5)  |
| Feature style           | Demographics + transaction + engagement + support + engineered ratios |
| Validation              | 70/15/15 split + ranking & relative error metrics |

---

## Results on the Kaggle Dataset (7,000 users)

| Metric                  | Validation | Test   |
|-------------------------|------------|--------|
| MAE                     | 3,975      | 4,090  |
| RMSE                    | 5,945      | 5,936  |
| R²                      | 0.9998     | 0.9998 |
| Spearman ρ              | 0.9999     | 0.9999 |
| MdAPE                   | 0.84 %     | 0.85 % |
| Precision@top 20 %      | 99.5 %     | 99.0 % |

> **Note:** The public Kaggle dataset has very strong (almost deterministic) relationships between `Total_Spent` and LTV. In a real neobank production setting you should expect Spearman in the 0.65–0.75 range and MdAPE of 7–15 %. The architecture and feature philosophy remain the same.

### Visualizations

| Predicted vs Actual | Feature Importance | Residuals |
|---------------------|--------------------|-----------|
| ![pred](images/pred_vs_actual.png) | ![imp](images/feature_importance.png) | ![res](images/residuals.png) |

---

## Repository Structure

```
ltv-neobank/
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
├── docs/
│   └── TECHNICAL_DOCUMENTATION.md   ← Full technical reference
├── src/
│   └── train_ltv_model.py           ← Reproducible training script
├── images/                          ← Result plots
├── data/                            ← Place the Kaggle CSV here
├── notebooks/
│   └── ltv_prediction.ipynb         ← Full exploratory + training notebook
```

---

## Quick Start

### 1. Clone and install

```bash
git clone https://github.com/<your-username>/ltv-neobank.git
cd ltv-neobank
pip install -r requirements.txt
```

### 2. Get the data

Download the dataset from Kaggle and place it in the `data/` folder:

```bash
# Option A: manual download
# https://www.kaggle.com/datasets/harunrai/fintech-customer-life-time-value-ltv-dataset
# Rename / move the CSV to: data/digital_wallet_ltv_dataset.csv

# Option B: with kaggle CLI (after kaggle.json setup)
kaggle datasets download -d harunrai/fintech-customer-life-time-value-ltv-dataset -p data/ --unzip
```

### 3. Train

**Option A — Script (recommended for reproducibility)**

```bash
python src/train_ltv_model.py
```

**Option B — Notebook**

```bash
jupyter notebook notebooks/ltv_prediction.ipynb
```

Both produce the same metrics and plots.

---

## Documentation

Full technical documentation is available in [`docs/TECHNICAL_DOCUMENTATION.md`](docs/TECHNICAL_DOCUMENTATION.md).

It covers:
- Design principles
- Feature engineering
- Model architecture
- Monitoring considerations
- Limitations and next steps

---

## License

MIT License — free to use, modify and redistribute. See [LICENSE](LICENSE).

---

## Disclaimer

This is a demonstration project. The metrics shown are on a public dataset with strong linear relationships. Real production LTV models face more noise, delayed costs, behavioral drift and regulatory constraints. Always validate thoroughly before using any model for business decisions.
