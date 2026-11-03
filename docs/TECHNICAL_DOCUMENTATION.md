# Lifetime Value (LTV) Prediction for a Digital Neobank

## Technical Documentation (Real Kaggle Dataset)

**Version:** 1.0  
**Date:** November 2026  
**Dataset:** [FinTech Customer Life Time Value (LTV) Dataset](https://www.kaggle.com/datasets/harunrai/fintech-customer-life-time-value-ltv-dataset) by harunrai  
**Audience:** Data Scientists, ML Engineers, Marketing Analysts, Product Managers  

---

### Purpose of the Document

This document is the technical reference for a **1-Year Lifetime Value (LTV) Prediction** system designed for a digital neobank (comparable to modern European or global digital banks). It describes the data approach, feature engineering, model architecture, and validation results obtained by applying the methodology to the public Kaggle FinTech Customer LTV dataset.

The approach is inspired by production user-value systems used by leading neobanks, fully generalized and free of any proprietary or company-specific references.

---

### Key Design Principles

| Dimension                  | Approach                                                                 |
|---------------------------|--------------------------------------------------------------------------|
| Scoring window            | Features collected through early lifecycle (KYC + 30 days)              |
| Prediction target         | 1-Year Lifetime Value / Contribution margin                             |
| Model structure           | Single global LightGBM model                                             |
| Objective                 | Tweedie (variance power 1.5) for right-skewed positive targets          |
| Feature engineering       | Demographics, transaction behavior, engagement, support, derived ratios |
| Score behavior            | Static snapshot at first scoring                                         |

---

### Feature Engineering (applied)

| Category                      | Features used / engineered                                      |
|-------------------------------|-----------------------------------------------------------------|
| Demographics                  | Age, Location, Income_Level                                     |
| Transaction behavior          | Total_Transactions, Avg/Max/Min_Transaction_Value, Total_Spent  |
| Engagement                    | Active_Days, Loyalty_Points_Earned, Referral_Count, App_Usage_Frequency |
| Support                       | Support_Tickets_Raised, Issue_Resolution_Time, Customer_Satisfaction_Score |
| Engineered                    | spend_to_txn_ratio, avg_spend_per_active_day, recency_score, support_intensity, engagement_score, high_value_txn_flag, income_num, app_usage_num |

Categorical features (Location, Income_Level, App_Usage_Frequency, Preferred_Payment_Method) are handled natively by LightGBM.

---

### Model Architecture

- **Algorithm:** LightGBM  
- **Objective:** Tweedie (variance power = 1.5)  
- **Training:** 70% train / 15% validation / 15% test  
- **Early stopping:** on validation MAE (max 500 rounds)  
- **Final profit / LTV** predicted directly (single model; revenue/cost split is optional in production)

---

### Results on Real Kaggle Dataset (7,000 users)

**LTV distribution:** mean ≈ 511,920 · median ≈ 387,818 · max ≈ 1,956,988

#### Validation & Test Metrics

| Metric                  | Validation | Test (OOT) |
|-------------------------|------------|------------|
| MAE                     | 3,975      | 4,090      |
| RMSE                    | 5,945      | 5,936      |
| R²                      | 0.9998     | 0.9998     |
| Spearman ρ              | 0.9999     | 0.9999     |
| MdAPE                   | 0.84 %     | 0.85 %     |
| Precision@top 20 %      | 99.5 %     | 99.0 %     |

**Interpretation:**  
The Kaggle dataset contains very strong, nearly deterministic relationships between Total_Spent and LTV (plus a few supporting features). This produces exceptionally high ranking quality and extremely low relative error. In a real neobank production setting (with more noise, delayed costs, behavioral drift, and multi-market variation) one typically observes Spearman in the 0.65–0.75 range and MdAPE of 7–15 %. The architecture and feature philosophy remain the same.

#### Top Features by Gain

| Rank | Feature                      | Notes                          |
|------|------------------------------|--------------------------------|
| 1    | Total_Spent                  | Dominant driver                |
| 2    | avg_spend_per_active_day     | Engineered intensity          |
| 3    | Total_Transactions           | Volume signal                  |
| 4    | spend_to_txn_ratio           | Engineered                     |
| 5    | Active_Days                  | Engagement                     |
| 6–8  | Avg/Max/Min_Transaction_Value| Transaction profile            |
| 9    | Loyalty_Points_Earned        | Engagement                     |
| 10   | Customer_Satisfaction_Score  | Support quality                |

---

### Data Monitoring & Operational Considerations

- Monitor Population Stability Index (PSI) on core features.  
- Track score distribution (mean, median, p90) by segment (Location, Income_Level).  
- Retrain when new cohorts mature or when PSI > 0.25 on important features.  
- Human review remains in the loop for marketing budget allocation.  
- Demographic features are proxies, not explicit targeting criteria.

---

### Limitations

- Static early-lifecycle score at KYC + 30 days (does not update after the observation window).  
- Synthetic / idealized relationships in the public dataset → real-world metrics will be lower.  
- No monthly cash-flow shape (predicts total value, not timing).  
- Treasury / pure interest income often excluded or modeled separately in production.

---

### Next Steps

- Shadow-mode validation against an existing production model (if applicable).  
- Optional dedicated treasury / savings model layered on top.  
- Fairness monitoring across Location, Income_Level, Age.  
- Continuous retraining cadence.

---

**Code & artifacts:** Accompanying training script, sample predictions CSV, and plots are available.  
**Dataset source:** https://www.kaggle.com/datasets/harunrai/fintech-customer-life-time-value-ltv-dataset
