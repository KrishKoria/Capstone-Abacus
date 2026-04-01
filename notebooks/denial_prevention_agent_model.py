import json
import os
import time
import uuid
from datetime import date, datetime
from pathlib import Path
from typing import Optional

import mlflow
import numpy as np
import pandas as pd
import requests
import shap
from mlflow.models import set_model
from mlflow.pyfunc import ChatAgent
from mlflow.types.agent import ChatAgentMessage, ChatAgentResponse, ChatAgentChunk

mlflow.set_registry_uri("databricks-uc")

FHIR_BASE_URL = os.getenv("FHIR_BASE_URL", "https://hapi.fhir.org/baseR4")
LLM_ENDPOINT = os.getenv("LLM_ENDPOINT", "databricks-meta-llama-3-3-70b-instruct")
PROMPT_TEMPLATE_VERSION = os.getenv("PROMPT_TEMPLATE_VERSION", "v1.1")
TOP_N_SHAP = int(os.getenv("TOP_N_SHAP", "3"))
MOCK_MODE = os.getenv("MOCK_MODE", "False").lower() == "true"

UC_MODEL_NAME = os.getenv("UC_MODEL_NAME", "capstone.default.denial-risk-model")
UC_MODEL_VERSION = os.getenv("UC_MODEL_VERSION", "").strip()
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "databricks")
MLFLOW_REGISTRY_URI = os.getenv("MLFLOW_REGISTRY_URI", "databricks-uc")
os.environ.setdefault("MLFLOW_USE_DATABRICKS_SDK_MODEL_ARTIFACTS_REPO_FOR_UC", "True")

PROVIDER_FEATURE_COLS = [
    "ClaimCount", "UniqueBeneficiaryCount", "UniqueClaimCount", "ClaimsPerBeneficiary",
    "AvgClaimAmt", "StdClaimAmt", "MaxClaimAmt", "TotalClaimAmt", "AvgDeductible",
    "AvgClaimDuration", "AvgLOS", "MaxLOS",
    "AvgDiagnosisCodeCount", "AvgProcedureCodeCount", "MaxDiagnosisCodeCount",
    "MaxProcedureCodeCount", "AvgPhysicianCount",
    "InpatientRate", "DeceasedPatientRate", "RenalDiseaseRate", "OperatingPhysicianRate",
    "OtherPhysicianRate", "PrimaryDiagnosisRate",
    "AlzheimerRate", "HeartfailureRate", "KidneyDiseaseRate", "CancerRate",
    "ObstrPulmonaryRate", "DepressionRate", "DiabetesRate", "IschemicHeartRate",
    "OsteoporasiaRate", "RheumatoidRate", "StrokeRate", "AvgChronicCondCount",
    "AvgPatientAge", "AvgMedicareCoverage", "AvgIPtoOPRatio",
]

ICD_TO_CHRONIC_FLAG = {
    "G30": "ChronicCond_Alzheimer",
    "F00": "ChronicCond_Alzheimer",
    "F01": "ChronicCond_Alzheimer",
    "F02": "ChronicCond_Alzheimer",
    "F03": "ChronicCond_Alzheimer",
    "I50": "ChronicCond_Heartfailure",
    "N18": "ChronicCond_KidneyDisease",
    "N19": "ChronicCond_KidneyDisease",
    "C18": "ChronicCond_Cancer",
    "C34": "ChronicCond_Cancer",
    "C50": "ChronicCond_Cancer",
    "J44": "ChronicCond_ObstrPulmonary",
    "F32": "ChronicCond_Depression",
    "F33": "ChronicCond_Depression",
    "E10": "ChronicCond_Diabetes",
    "E11": "ChronicCond_Diabetes",
    "E13": "ChronicCond_Diabetes",
    "I25": "ChronicCond_IschemicHeart",
    "M80": "ChronicCond_Osteoporasis",
    "M81": "ChronicCond_Osteoporasis",
    "M05": "ChronicCond_rheumatoidarthritis",
    "M06": "ChronicCond_rheumatoidarthritis",
    "M15": "ChronicCond_rheumatoidarthritis",
    "I63": "ChronicCond_stroke",
    "I64": "ChronicCond_stroke",
    "G45": "ChronicCond_stroke",
}

ALL_CHRONIC_FLAGS = [
    "ChronicCond_Alzheimer",
    "ChronicCond_Heartfailure",
    "ChronicCond_KidneyDisease",
    "ChronicCond_Cancer",
    "ChronicCond_ObstrPulmonary",
    "ChronicCond_Depression",
    "ChronicCond_Diabetes",
    "ChronicCond_IschemicHeart",
    "ChronicCond_Osteoporasis",
    "ChronicCond_rheumatoidarthritis",
    "ChronicCond_stroke",
]

FEATURE_LABELS = {
    "ClaimCount": "Total claims submitted by provider",
    "UniqueBeneficiaryCount": "Number of unique beneficiaries served",
    "UniqueClaimCount": "Number of unique claims submitted",
    "ClaimsPerBeneficiary": "Average claims per beneficiary (reuse rate)",
    "AvgClaimAmt": "Provider average claim reimbursement amount ($)",
    "StdClaimAmt": "Standard deviation of provider claim amounts ($)",
    "MaxClaimAmt": "Provider maximum single claim amount ($)",
    "TotalClaimAmt": "Total reimbursement billed by provider ($)",
    "AvgDeductible": "Provider average deductible amount ($)",
    "AvgClaimDuration": "Average claim service duration (days)",
    "AvgLOS": "Average inpatient length of stay (days)",
    "MaxLOS": "Maximum inpatient length of stay (days)",
    "AvgDiagnosisCodeCount": "Average number of diagnosis codes per claim",
    "AvgProcedureCodeCount": "Average number of procedure codes per claim",
    "MaxDiagnosisCodeCount": "Maximum diagnosis codes on a single claim",
    "MaxProcedureCodeCount": "Maximum procedure codes on a single claim",
    "AvgPhysicianCount": "Average physician count per claim",
    "InpatientRate": "Proportion of inpatient claims",
    "DeceasedPatientRate": "Proportion of claims for deceased patients",
    "RenalDiseaseRate": "Proportion of patients with renal disease",
    "OperatingPhysicianRate": "Proportion of claims with an operating physician",
    "OtherPhysicianRate": "Proportion of claims with other physicians listed",
    "PrimaryDiagnosisRate": "Proportion of claims with a primary diagnosis code",
    "AlzheimerRate": "Proportion of Alzheimer / dementia patients",
    "HeartfailureRate": "Proportion of heart failure patients",
    "KidneyDiseaseRate": "Proportion of kidney disease patients",
    "CancerRate": "Proportion of cancer patients",
    "ObstrPulmonaryRate": "Proportion of COPD patients",
    "DepressionRate": "Proportion of depression patients",
    "DiabetesRate": "Proportion of diabetes patients",
    "IschemicHeartRate": "Proportion of ischemic heart disease patients",
    "OsteoporasiaRate": "Proportion of osteoporosis patients",
    "RheumatoidRate": "Proportion of rheumatoid arthritis patients",
    "StrokeRate": "Proportion of stroke patients",
    "AvgChronicCondCount": "Average chronic condition count per patient",
    "AvgPatientAge": "Average patient age across provider claims",
    "AvgMedicareCoverage": "Average Medicare coverage months per patient",
    "AvgIPtoOPRatio": "Average inpatient-to-outpatient reimbursement ratio",
}

SYSTEM_PROMPT = """You are a healthcare claims auditor AI assistant.
Your role is to explain why a medical provider has been flagged as high anomaly/audit risk,
based ONLY on the structured data provided to you.

STRICT RULES — violations make the output unusable:
1. Do NOT mention any diagnosis, condition, medication, or clinical detail that is not explicitly present in the input data.
2. Do NOT speculate about patient health beyond the chronic condition flags provided.
3. Frame all findings as \"anomalous billing patterns that historically trigger payer audits and systemic claim denials\" — NOT as proof of fraud.
4. Output ONLY a single valid JSON object. No preamble, no explanation, no markdown fences.
5. All three keys (denial_reasons, corrective_actions, risk_summary) are required.
"""

USER_PROMPT_TEMPLATE = """Analyze the following provider billing profile and generate a structured JSON explanation.

=== CURRENT PATIENT / CLAIM CONTEXT ===
Patient Age: {patient_age} years
Active Chronic Conditions (this patient): {chronic_conditions}
Current Claim Amount: ${claim_amount:,.2f}
Current Claim Duration: {claim_duration} days
Current Claim Type: {claim_type}
Primary Diagnosis Chapter (ICD-10): {icd_chapter}
Claim Amount Z-Score vs. Provider Average: {claim_amt_zscore:+.2f}

=== PROVIDER BILLING PROFILE (Day 3 model features) ===
Total Claims Submitted: {claim_count:,}
Unique Beneficiaries Served: {unique_beneficiary_count:,}
Average Claim Amount: ${avg_claim_amt:,.2f}
Max Single Claim Amount: ${max_claim_amt:,.2f}
Total Reimbursement Billed: ${total_claim_amt:,.2f}
Average Length of Stay (inpatient): {avg_los:.1f} days
Average Diagnosis Codes per Claim: {avg_diagnosis_code_count:.1f}
Inpatient Claim Rate: {inpatient_rate:.1%}
Deceased Patient Rate: {deceased_patient_rate:.1%}
Average Patient Age: {avg_patient_age:.1f} years
Average Chronic Conditions per Patient: {avg_chronic_cond_count:.2f}

=== MODEL OUTPUT ===
Anomaly Risk Score: {risk_score:.2f} (scale 0.0–1.0; threshold for high risk: 0.5)
Risk Interpretation: {risk_interpretation}

Top {top_n} Contributing Factors (SHAP feature importance values):
{shap_summary}

=== REQUIRED OUTPUT FORMAT ===
Return exactly this JSON schema — no extra keys, no missing keys:
{{
  "denial_reasons": [
    "Specific reason 1 grounded in the provider billing data above",
    "Specific reason 2 grounded in the provider billing data above",
    "Specific reason 3 grounded in the provider billing data above"
  ],
  "corrective_actions": [
    "Actionable step 1 the billing team can take before submission",
    "Actionable step 2",
    "Actionable step 3"
  ],
  "risk_summary": "A 2–3 sentence plain-English summary for a billing reviewer explaining the overall anomaly risk and what to do next."
}}"""


def _configure_mlflow():
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    except Exception:
        pass
    try:
        mlflow.set_registry_uri(MLFLOW_REGISTRY_URI)
    except Exception:
        pass


def _get_mlflow_client() -> mlflow.tracking.MlflowClient:
    _configure_mlflow()
    host = os.getenv("DATABRICKS_HOST", "").rstrip("/")
    token = os.getenv("DATABRICKS_TOKEN", "")
    if host and token:
        os.environ.setdefault("MLFLOW_TRACKING_TOKEN", token)
    return mlflow.tracking.MlflowClient(
        tracking_uri=MLFLOW_TRACKING_URI,
        registry_uri=MLFLOW_REGISTRY_URI,
    )


def _get_databricks_host() -> str:
    host = os.getenv("DATABRICKS_HOST")
    if host:
        return host.rstrip("/")
    try:
        from databricks.sdk import WorkspaceClient
        resolved = WorkspaceClient().config.host
        if resolved:
            return resolved.rstrip("/")
    except Exception:
        pass
    raise RuntimeError("Could not resolve DATABRICKS_HOST. Set it as an environment variable for serving.")


def _get_openai_client():
    from openai import OpenAI
    # In serving containers, these are auto-injected
    host = os.getenv("DATABRICKS_HOST", "").rstrip("/")
    token = os.getenv("DATABRICKS_TOKEN", "")
    if host and token:
        return OpenAI(
            base_url=f"{host}/serving-endpoints",
            api_key=token,
        )
    # Fallback for notebook testing
    from databricks.sdk import WorkspaceClient
    return WorkspaceClient().serving_endpoints.get_open_ai_client()


def _fhir_get(
    resource_type: str,
    params: dict | None = None,
    resource_id: str | None = None,
    max_retries: int = 3,
    base_sleep_seconds: float = 1.5,
) -> dict:
    if resource_id:
        url = f"{FHIR_BASE_URL}/{resource_type}/{resource_id}"
    else:
        url = f"{FHIR_BASE_URL}/{resource_type}"

    headers = {"Accept": "application/fhir+json"}

    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.get(url, params=params, timeout=20, headers=headers)
            if resp.status_code in {429, 500, 502, 503, 504}:
                raise requests.HTTPError(
                    f"Transient FHIR error {resp.status_code}: {resp.text[:300]}",
                    response=resp,
                )
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            last_err = e
            if attempt == max_retries:
                break
            time.sleep(base_sleep_seconds * (2 ** (attempt - 1)))

    raise RuntimeError(f"FHIR GET failed after {max_retries} attempts: {last_err}")


def _compute_age(birth_date_str: str, reference_date: Optional[date] = None) -> int:
    ref = reference_date or date.today()
    bd = datetime.strptime(birth_date_str, "%Y-%m-%d").date()
    years = ref.year - bd.year
    if (ref.month, ref.day) < (bd.month, bd.day):
        years -= 1
    return max(years, 0)


def _map_chronic_conditions(conditions: list[dict]) -> dict:
    flags = {flag: 0 for flag in ALL_CHRONIC_FLAGS}
    for cond in conditions:
        for coding in cond.get("code", {}).get("coding", []):
            code_prefix = (coding.get("code") or "")[:3].upper()
            matched_flag = ICD_TO_CHRONIC_FLAG.get(code_prefix)
            if matched_flag in flags:
                flags[matched_flag] = 1
    return flags


def _prepare_inference_df(feature_vector: dict, feature_cols: list[str]) -> pd.DataFrame:
    row = {k: v for k, v in feature_vector.items() if not k.startswith("_")}
    for col in feature_cols:
        if row.get(col) is None:
            row[col] = 0.0
    df = pd.DataFrame([row])
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0.0
    return df[feature_cols].fillna(0.0).astype("float64")


def _extract_positive_class_shap(model, X: pd.DataFrame) -> dict[str, float]:
    is_tree_model = (
        hasattr(model, "estimators_")
        or hasattr(model, "tree_")
        or hasattr(model, "get_booster")
        or model.__class__.__module__.startswith("xgboost")
    )

    if is_tree_model:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        if isinstance(shap_values, list):
            sv = shap_values[1][0] if len(shap_values) > 1 else shap_values[0][0]
        else:
            arr = np.array(shap_values)
            if arr.ndim == 3:
                sv = arr[0, :, 1] if arr.shape[-1] > 1 else arr[0, :, 0]
            elif arr.ndim == 2:
                sv = arr[0]
            elif arr.ndim == 1:
                sv = arr
            else:
                raise ValueError(f"Unexpected SHAP output shape for tree model: {arr.shape}")
        return {col: float(val) for col, val in zip(X.columns, sv)}

    if hasattr(model, "coef_"):
        background = np.zeros((1, X.shape[1]), dtype=float)
        explainer = shap.LinearExplainer(model, background)
        shap_values = explainer.shap_values(X)
        if isinstance(shap_values, list):
            sv = shap_values[1][0] if len(shap_values) > 1 else shap_values[0][0]
        else:
            arr = np.array(shap_values)
            if arr.ndim == 2:
                sv = arr[0]
            elif arr.ndim == 3:
                sv = arr[0, :, 1] if arr.shape[-1] > 1 else arr[0, :, 0]
            elif arr.ndim == 1:
                sv = arr
            else:
                raise ValueError(f"Unexpected SHAP output shape for linear model: {arr.shape}")
        return {col: float(val) for col, val in zip(X.columns, sv)}

    raise TypeError(f"Unsupported model type for SHAP explanation: {type(model)}")


def format_shap_for_prompt(shap_values: dict, top_n: int = TOP_N_SHAP) -> str:
    top = sorted(shap_values.items(), key=lambda x: abs(x[1]), reverse=True)[:top_n]
    lines = []
    for feature, value in top:
        label = FEATURE_LABELS.get(feature, feature)
        direction = "increased" if value > 0 else "decreased"
        magnitude = "significantly" if abs(value) > 0.2 else "moderately"
        sign = "+" if value > 0 else ""
        lines.append(
            f"  - {label}: {sign}{value:.3f}\n"
            f"    (This factor {magnitude} {direction} the anomaly risk score)"
        )
    return "\n".join(lines)


def build_prompt(feature_vector: dict, prediction: dict) -> tuple[str, str]:
    chronic_conditions = [
        flag.replace("ChronicCond_", "").replace("_", " ")
        for flag, val in feature_vector.get("_chronic_flags", {}).items()
        if val == 1
    ] or ["None documented"]

    claim_type_label = "Inpatient" if feature_vector.get("_claim_type") == 1 else "Outpatient"
    shap_summary = format_shap_for_prompt(prediction["shap_values"])
    risk_score = prediction["risk_score"]
    risk_interpretation = (
        "HIGH RISK — recommend pre-submission review"
        if risk_score >= 0.5 else "LOW RISK"
    )

    user_prompt = USER_PROMPT_TEMPLATE.format(
        patient_age=feature_vector.get("_patient_age", "Unknown"),
        chronic_conditions=", ".join(chronic_conditions),
        claim_amount=feature_vector.get("_claim_amount", 0.0),
        claim_duration=feature_vector.get("_claim_duration", 0),
        claim_type=claim_type_label,
        icd_chapter=feature_vector.get("_icd_chapter", "Unknown"),
        claim_amt_zscore=feature_vector.get("_claim_amt_zscore", 0.0),
        claim_count=int(feature_vector.get("ClaimCount", 0.0)),
        unique_beneficiary_count=int(feature_vector.get("UniqueBeneficiaryCount", 0.0)),
        avg_claim_amt=float(feature_vector.get("AvgClaimAmt", 0.0)),
        max_claim_amt=float(feature_vector.get("MaxClaimAmt", 0.0)),
        total_claim_amt=float(feature_vector.get("TotalClaimAmt", 0.0)),
        avg_los=float(feature_vector.get("AvgLOS", 0.0)),
        avg_diagnosis_code_count=float(feature_vector.get("AvgDiagnosisCodeCount", 0.0)),
        inpatient_rate=float(feature_vector.get("InpatientRate", 0.0)),
        deceased_patient_rate=float(feature_vector.get("DeceasedPatientRate", 0.0)),
        avg_patient_age=float(feature_vector.get("AvgPatientAge", 0.0)),
        avg_chronic_cond_count=float(feature_vector.get("AvgChronicCondCount", 0.0)),
        risk_score=risk_score,
        risk_interpretation=risk_interpretation,
        top_n=TOP_N_SHAP,
        shap_summary=shap_summary,
    )

    return SYSTEM_PROMPT, user_prompt


class DenialPreventionAgent(ChatAgent):
    def load_context(self, context):
        _configure_mlflow()

        lookup = pd.read_csv(context.artifacts["provider_lookup_csv"])
        if "Provider" not in lookup.columns:
            raise ValueError("provider_lookup_csv must contain a 'Provider' column.")
        self.provider_lookup = lookup.set_index("Provider", drop=False)

        self.feature_cols = PROVIDER_FEATURE_COLS.copy()
        self.risk_model = None
        self.risk_model_source = "uninitialized"

        local_feature_cols_path = context.artifacts.get("feature_columns_json")
        if local_feature_cols_path and Path(local_feature_cols_path).exists():
            with open(local_feature_cols_path, "r") as f:
                cols = json.load(f)
            if isinstance(cols, list) and cols:
                self.feature_cols = cols

        local_model_path = context.artifacts.get("risk_model")
        if local_model_path and Path(local_model_path).exists():
            self.risk_model = mlflow.sklearn.load_model(local_model_path)
            self.risk_model_source = f"artifact:{local_model_path}"

    def _get_provider_stats(self, provider_id: str) -> dict:
        if provider_id in self.provider_lookup.index:
            row = self.provider_lookup.loc[provider_id]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            return {col: float(row.get(col, 0.0) or 0.0) for col in PROVIDER_FEATURE_COLS}
        return {col: 0.0 for col in PROVIDER_FEATURE_COLS}

    def _resolve_registry_model_version(self):
        client = _get_mlflow_client()

        if UC_MODEL_VERSION:
            try:
                mv = client.get_model_version(name=UC_MODEL_NAME, version=UC_MODEL_VERSION)
                return mv
            except Exception as e:
                raise RuntimeError(
                    f"Pinned UC_MODEL_VERSION={UC_MODEL_VERSION} could not be loaded for {UC_MODEL_NAME}: {e}"
                ) from e

        try:
            mv = client.get_model_version_by_alias(UC_MODEL_NAME, "Champion")
            if mv is not None:
                return mv
        except Exception:
            pass

        try:
            versions = client.search_model_versions(f"name='{UC_MODEL_NAME}'")
        except Exception as e:
            raise RuntimeError(
                "MLflow registry lookup failed. This usually means the serving endpoint is not using "
                f"the UC registry URI, or the endpoint identity cannot access model {UC_MODEL_NAME}. "
                f"tracking_uri={MLFLOW_TRACKING_URI}, registry_uri={MLFLOW_REGISTRY_URI}. Root error: {e}"
            ) from e

        if not versions:
            raise RuntimeError(
                f"No registered model versions found for {UC_MODEL_NAME}. "
                "Either the name is wrong, the model was never registered in Unity Catalog, "
                "or the serving identity cannot see it."
            )
        return max(versions, key=lambda mv: int(mv.version))

    def _load_registry_model_if_needed(self):
        if self.risk_model is not None:
            return self.risk_model, self.feature_cols, self.risk_model_source

        mv = self._resolve_registry_model_version()
        model_uri = f"models:/{UC_MODEL_NAME}/{mv.version}"
        model = mlflow.sklearn.load_model(model_uri)

        feature_cols = self.feature_cols
        try:
            local_path = mlflow.artifacts.download_artifacts(
                artifact_uri=f"runs:/{mv.run_id}/feature_columns.json"
            )
            with open(local_path, "r") as f:
                cols = json.load(f)
            if isinstance(cols, list) and cols:
                feature_cols = cols
        except Exception:
            pass

        return model, feature_cols, f"registry:{model_uri}"

    def get_patient_from_fhir(self, patient_id: str) -> dict:
        patient = _fhir_get("Patient", resource_id=patient_id)
        birth_date = patient.get("birthDate", "1950-01-01")
        gender_raw = (patient.get("gender") or "unknown").lower()
        gender = 1 if gender_raw == "male" else 0

        condition_bundle = _fhir_get("Condition", params={"patient": patient_id, "_count": 200})
        conditions = [e["resource"] for e in condition_bundle.get("entry", [])]
        chronic_flags = _map_chronic_conditions(conditions)
        chronic_condition_count = int(sum(chronic_flags.values()))

        claim_bundle = _fhir_get("Claim", params={"patient": patient_id, "_sort": "-created", "_count": 1})
        entries = claim_bundle.get("entry", [])
        if not entries:
            raise ValueError(f"No Claim resources found for patient_id={patient_id}")

        claim = entries[0]["resource"]
        claim_amount = float(claim.get("total", {}).get("value", 0.0) or 0.0)

        claim_created = (claim.get("created") or date.today().isoformat())[:10]
        claim_start = datetime.strptime(claim_created, "%Y-%m-%d").date()
        claim_duration = 1

        diagnoses = [
            d.get("diagnosisCodeableConcept", {}).get("coding", [{}])[0].get("code", "UNKNOWN")
            for d in claim.get("diagnosis", [])
        ]
        primary_dx = diagnoses[0] if diagnoses else "UNKNOWN"
        has_primary_diagnosis = int(primary_dx != "UNKNOWN")
        icd_chapter = primary_dx[:3] if len(primary_dx) >= 3 else "UNK"

        procedures = [
            p.get("procedureCodeableConcept", {}).get("coding", [{}])[0].get("code")
            for p in claim.get("procedure", [])
        ]
        has_operating_physician = int(len([p for p in procedures if p]) > 0)

        claim_type_coding = claim.get("type", {}).get("coding", [{}])[0].get("code", "")
        claim_type = 1 if "institutional" in claim_type_coding.lower() else 0

        provider_ref = claim.get("provider", {}).get("reference", "UNKNOWN_PROVIDER")
        provider_id = provider_ref.split("/")[-1]
        provider_stats = self._get_provider_stats(provider_id)

        avg_amt = float(provider_stats.get("AvgClaimAmt", 0.0) or 0.0)
        std_amt = float(provider_stats.get("StdClaimAmt", 0.0) or 0.0)
        z_score = ((claim_amount - avg_amt) / std_amt) if std_amt > 0 else 0.0
        patient_age = _compute_age(birth_date, claim_start)

        feature_vector = {col: float(provider_stats.get(col, 0.0) or 0.0) for col in PROVIDER_FEATURE_COLS}
        feature_vector.update({
            "_patient_id": patient_id,
            "_provider_id": provider_id,
            "_patient_age": patient_age,
            "_gender": gender,
            "_chronic_flags": chronic_flags,
            "_chronic_count": chronic_condition_count,
            "_primary_diagnosis": primary_dx,
            "_icd_chapter": icd_chapter,
            "_has_primary_dx": has_primary_diagnosis,
            "_has_op_physician": has_operating_physician,
            "_claim_type": claim_type,
            "_claim_amount": float(claim_amount),
            "_claim_duration": int(claim_duration),
            "_claim_start_date": claim_start.isoformat(),
            "_claim_amt_zscore": float(round(z_score, 4)),
        })
        return feature_vector

    def predict_fraud_risk(self, feature_vector: dict) -> dict:
        if MOCK_MODE:
            return {
                "risk_score": 0.84,
                "shap_values": {
                    "AvgClaimAmt": 0.31,
                    "ClaimCount": 0.22,
                    "AvgDiagnosisCodeCount": -0.14,
                    "AvgChronicCondCount": 0.11,
                    "AvgLOS": 0.09,
                },
                "ml_run_id": "mock_run_000",
                "model_source": "mock",
            }

        model, feature_cols, model_source = self._load_registry_model_if_needed()
        X = _prepare_inference_df(feature_vector, feature_cols)

        if not hasattr(model, "predict_proba"):
            raise AttributeError("Loaded model does not expose predict_proba().")

        risk_score = float(model.predict_proba(X)[0][1])
        shap_dict = _extract_positive_class_shap(model, X)

        return {
            "risk_score": risk_score,
            "shap_values": shap_dict,
            "ml_run_id": getattr(model, "run_id", "unknown"),
            "model_source": model_source,
        }

    def call_llm(self, feature_vector: dict, prediction: dict) -> dict:
        system_prompt, user_prompt = build_prompt(feature_vector, prediction)
        client = _get_openai_client()

        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "denial_explanation",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "denial_reasons": {"type": "array", "items": {"type": "string"}},
                        "corrective_actions": {"type": "array", "items": {"type": "string"}},
                        "risk_summary": {"type": "string"},
                    },
                    "required": ["denial_reasons", "corrective_actions", "risk_summary"],
                    "additionalProperties": False,
                },
            },
        }

        t0 = time.time()
        response = client.chat.completions.create(
            model=LLM_ENDPOINT,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
            max_tokens=800,
            response_format=response_format,
        )
        latency_ms = int((time.time() - t0) * 1000)

        raw_content = response.choices[0].message.content
        if not isinstance(raw_content, str):
            raise ValueError(f"Unexpected LLM content type: {type(raw_content)}")

        parsed = json.loads(raw_content)
        required_keys = {"denial_reasons", "corrective_actions", "risk_summary"}
        missing = required_keys - set(parsed.keys())
        if missing:
            raise ValueError(f"LLM response missing required keys: {missing}")

        parsed["_latency_ms"] = latency_ms
        parsed["_input_tokens"] = getattr(response.usage, "prompt_tokens", 0)
        parsed["_output_tokens"] = getattr(response.usage, "completion_tokens", 0)
        parsed["_model_used"] = LLM_ENDPOINT
        return parsed

    def _maybe_log_interaction(
        self, patient_id: str, feature_vector: dict, prediction: dict, llm_response: dict
    ) -> str:
        try:
            with mlflow.start_run(run_name=f"llm_explanation_{patient_id[:8]}") as run:
                top_shap = dict(sorted(
                    prediction["shap_values"].items(), key=lambda x: abs(x[1]), reverse=True
                )[:TOP_N_SHAP])

                mlflow.log_params({
                    "prompt_template_version": PROMPT_TEMPLATE_VERSION,
                    "llm_endpoint": LLM_ENDPOINT,
                    "top_n_shap": TOP_N_SHAP,
                    "mock_mode": str(MOCK_MODE),
                    "model_source": prediction.get("model_source", "unknown"),
                })
                mlflow.log_metrics({
                    "risk_score": prediction["risk_score"],
                    "latency_ms": llm_response["_latency_ms"],
                    "input_tokens": llm_response["_input_tokens"],
                    "output_tokens": llm_response["_output_tokens"],
                    "total_tokens": llm_response["_input_tokens"] + llm_response["_output_tokens"],
                })
                mlflow.log_dict(top_shap, "shap_values_used.json")
                mlflow.log_dict({
                    "denial_reasons": llm_response["denial_reasons"],
                    "corrective_actions": llm_response["corrective_actions"],
                    "risk_summary": llm_response["risk_summary"],
                }, "llm_response_output.json")
                mlflow.set_tags({
                    "patient_id": feature_vector.get("_patient_id", patient_id),
                    "provider_id": feature_vector.get("_provider_id", "UNKNOWN"),
                    "ml_run_id": prediction.get("ml_run_id", "unknown"),
                })
                return run.info.run_id
        except Exception:
            return "not-logged"

    def _extract_patient_id(self, text: str) -> str:
        import re
        uuid_pattern = r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"
        match = re.search(uuid_pattern, text)
        if match:
            return match.group(0)
        cleaned = text.strip().split()[-1] if text.strip() else ""
        if cleaned.isdigit() and len(cleaned) > 4:
            return cleaned
        return ""

    def _format_chat_response(
        self, patient_id: str, prediction: dict, llm_response: dict, run_id: str
    ) -> str:
        risk = prediction["risk_score"]
        risk_icon = "🔴" if risk >= 0.7 else "🟡" if risk >= 0.5 else "🟢"
        level = "HIGH" if risk >= 0.7 else "MEDIUM" if risk >= 0.5 else "LOW"

        reasons = "\n".join(f"  {i+1}. {r}" for i, r in enumerate(llm_response["denial_reasons"]))
        actions = "\n".join(f"  {i+1}. {a}" for i, a in enumerate(llm_response["corrective_actions"]))

        footer = (
            f"*Prompt: `{PROMPT_TEMPLATE_VERSION}` · LLM endpoint: `{LLM_ENDPOINT}` · Model source: `{prediction.get('model_source', 'unknown')}`*"
            if run_id == "not-logged"
            else f"*MLflow run ID: `{run_id}` · Prompt: `{PROMPT_TEMPLATE_VERSION}` · LLM endpoint: `{LLM_ENDPOINT}` · Model source: `{prediction.get('model_source', 'unknown')}`*"
        )

        return f"""Claim Anomaly Risk Report
**Patient ID:** `{patient_id}`
**Anomaly Risk Score:** {risk:.2f} / 1.00  {risk_icon} **{level} RISK**

---

### Summary
{llm_response["risk_summary"]}

---

### Why This Claim Was Flagged
{reasons}

---

### Recommended Actions Before Submission
{actions}

---
{footer}
*⚠️ Note: Risk score is based on anomalous billing pattern detection, not a direct denial prediction.*"""

    def predict(
        self,
        messages: list[ChatAgentMessage],
        context=None,
        custom_inputs=None,
    ) -> ChatAgentResponse:
        last_user_msg = next((m.content for m in reversed(messages) if m.role == "user"), "")
        patient_id = self._extract_patient_id(last_user_msg)

        if not patient_id:
            return ChatAgentResponse(messages=[
                ChatAgentMessage(
                    role="assistant",
                    content=(
                        "Please provide a valid FHIR patient ID to analyze.\n\n"
                        "Example: `Analyze patient a7b3c9d2-1234-5678-abcd-ef0123456789`"
                    ),
                    id=str(uuid.uuid4()),
                )
            ])

        try:
            feature_vector = self.get_patient_from_fhir(patient_id)
            prediction = self.predict_fraud_risk(feature_vector)
            llm_response = self.call_llm(feature_vector, prediction)
            run_id = self._maybe_log_interaction(patient_id, feature_vector, prediction, llm_response)
            reply = self._format_chat_response(patient_id, prediction, llm_response, run_id)
        except Exception as e:
            reply = (
                f"⚠️ Error processing patient `{patient_id}`:\n```\n{str(e)}\n```\n\n"
                "This is not necessarily a FHIR patient-ID problem. It can also be caused by a missing / inaccessible ML model, "
                "wrong Unity Catalog model name, wrong MLflow registry URI, or serving-endpoint permissions."
            )

        return ChatAgentResponse(messages=[
            ChatAgentMessage(
                role="assistant",
                content=reply,
                id=str(uuid.uuid4()),
            )
        ])
    
    def predict_stream(
        self,
        messages: list[ChatAgentMessage],
        context=None,
        custom_inputs=None,
    ):
        response = self.predict(messages, context, custom_inputs)
        for msg in response.messages:
            yield ChatAgentChunk(
                delta=ChatAgentMessage(
                    role=msg.role,
                    content=msg.content,
                    id=msg.id,
                ),
            )


set_model(DenialPreventionAgent())
