# Capstone-Abacus Table and Column Inventory

This inventory is extracted from the notebooks in `notebooks/` and reflects the tables/columns used in engineering, modeling, and LLM stages.

## 1) capstone.bronze.claims_engineered

Main claim-level engineered table (Day 2). Built from inpatient + outpatient + beneficiary + labels joins.

### Core columns used in notebooks/dashboard

- Provider
- BeneID
- ClaimID
- PotentialFraud
- claim_type
- InscClaimAmtReimbursed
- DeductibleAmtPaid
- ClaimStartDt
- ClaimEndDt
- AdmissionDt
- DischargeDt
- DOB
- DOD
- State
- NoOfMonths_PartACov
- NoOfMonths_PartBCov
- IPAnnualReimbursementAmt
- OPAnnualReimbursementAmt
- AttendingPhysician
- OperatingPhysician
- OtherPhysician
- ClmDiagnosisCode_1 ... ClmDiagnosisCode_9
- ClmProcedureCode_1 ... ClmProcedureCode_6

### Chronic condition fields

- ChronicCond_Alzheimer
- ChronicCond_Heartfailure
- ChronicCond_KidneyDisease
- ChronicCond_Cancer
- ChronicCond_ObstrPulmonary
- ChronicCond_Depression
- ChronicCond_Diabetes
- ChronicCond_IschemicHeart
- ChronicCond_Osteoporasis
- ChronicCond_rheumatoidarthritis
- ChronicCond_stroke
- HasRenalDisease

### Engineered columns (Day 2 feature engineering)

- DiagnosisCodeCount
- ProcedureCodeCount
- PhysicianCount
- HasOperatingPhysician
- HasOtherPhysician
- ClaimDuration
- LengthOfStay
- Age
- IsDeceased
- ChronicConditionCount
- HasPrimaryDiagnosis
- ICD_Chapter
- MedicareCoverageMonths
- IPtoOPReimbursementRatio
- class_weight

## 2) capstone.bronze.claims_engineered_test

Holdout test split produced in Day 2 test pipeline. Same schema pattern as `claims_engineered`.

## 3) capstone.bronze.train_features

Provider-level aggregated training table (Day 3).

### Keys/label

- Provider
- PotentialFraud

### Provider feature columns

- ClaimCount
- UniqueBeneficiaryCount
- UniqueClaimCount
- ClaimsPerBeneficiary
- AvgClaimAmt
- StdClaimAmt
- MaxClaimAmt
- TotalClaimAmt
- AvgDeductible
- AvgClaimDuration
- AvgLOS
- MaxLOS
- AvgDiagnosisCodeCount
- AvgProcedureCodeCount
- MaxDiagnosisCodeCount
- MaxProcedureCodeCount
- AvgPhysicianCount
- InpatientRate
- DeceasedPatientRate
- RenalDiseaseRate
- OperatingPhysicianRate
- OtherPhysicianRate
- PrimaryDiagnosisRate
- AlzheimerRate
- HeartfailureRate
- KidneyDiseaseRate
- CancerRate
- ObstrPulmonaryRate
- DepressionRate
- DiabetesRate
- IschemicHeartRate
- OsteoporasiaRate
- RheumatoidRate
- StrokeRate
- AvgChronicCondCount
- AvgPatientAge
- AvgMedicareCoverage
- AvgIPtoOPRatio

## 4) capstone.bronze.test_features

Provider-level aggregated holdout features. Same feature columns as `train_features`.

## 5) capstone.bronze.provider_lookup

Lookup table used by Day 4 LLM/risk inference.

### Columns

- Provider
- All 38 provider feature columns listed above (same names/order as FEATURE_COLS)

## 6) capstone.bronze.llm_predictions

Operational sink table written in Day 4 (`log_and_sink`) for each LLM-assisted review.

### Columns

- run_id
- patient_id
- provider_id
- risk_score
- claim_amount
- claim_duration
- top_shap_feature_1
- top_shap_value_1
- denial_reasons
- corrective_actions
- risk_summary
- prediction_ts
- prompt_version
- mock_mode

## Notebook sources used

- notebooks/day2_data_engineering.ipynb
- notebooks/day2_data_engineering_test.ipynb
- notebooks/day3_ml_modeling.ipynb
- notebooks/day3_ml_modeling_test file.ipynb
- notebooks/day4_llm_layer.ipynb
- notebooks/denial_prevention_agent_model.py
