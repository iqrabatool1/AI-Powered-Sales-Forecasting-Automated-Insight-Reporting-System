# AI-Powered Sales Forecasting & Automated Insight Reporting System

An end-to-end system that predicts daily store sales using machine learning, explains its predictions, detects unusual sales patterns, and automatically generates plain-English business reports using generative AI — all wrapped in an interactive dashboard and a scheduled automation pipeline.

## Overview

This project combines three layers into a single working system:

1. **Predictive ML** — XGBoost regression model forecasting daily sales per store
2. **Explainable AI** — SHAP analysis showing which factors drive each prediction
3. **Generative AI Automation** — an LLM (via Groq) that turns model output into readable business insight reports, triggered automatically on a schedule

## Dataset

[Rossmann Store Sales](https://www.kaggle.com/competitions/rossmann-store-sales) — daily sales data for 1,115 stores, including promotions, holidays, and competition information.

## Key Results

- XGBoost reduced prediction error (MAE) by **~60%** compared to a naive rolling-average baseline (663 vs 1,681)
- SHAP analysis identified `Promo`, `Sales_Lag_14`, and `Sales_RollingMean_30` as the strongest predictors of daily sales
- Automated anomaly detection flagged ~2.2% of predictions as statistically unusual, surfacing specific stores worth business investigation

## Architecture
