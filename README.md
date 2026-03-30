# Predictive Denial Prevention AI Platform

## Overview
This platform acts as an anomaly and audit risk early-warning system. It flags high-risk healthcare claims prior to submission and uses LLMs to explain the raw statistical metrics (SHAP values) into human-readable reasons and corrective actions.

## GitHub Structure
- `data/`: Contains downloaded Kaggle Data (Inpatient, Outpatient, Beneficiary, Train mappings). Handled automatically by KaggleHub in notebooks.
- `notebooks/`: Contains the Jupyter notebooks for execution inside Databricks.
  - `EDA_Day1.ipynb`: Data Ingestion, Basic PySpark EDA, Join logic verification.
- `architecture.png`: Visual representation of the data and ML flow.

## Architecture Stack
- **Data Engineering:** Azure Databricks (PySpark, Delta Lake)
- **Machine Learning:** Spark MLlib / Scikit-Learn
- **Explanations:** Databricks Foundation Model APIs (Model Serving)
- **Tracking:** MLflow
- **Inference:** HAPI FHIR Server (Synthea)
