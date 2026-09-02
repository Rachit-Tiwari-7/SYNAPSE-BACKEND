"""
SynapseOS — ml/diagnostics.py
Quantitative Disease Risk & Clinical Prediction Scoring.
Ported from AI-Healthcare-System diagnostics algorithms.
Calculates Framingham-like CVD risk, ADA Diabetes Score, KDIGO Renal Stage, and FIB-4 Liver Index.
"""

import math
from typing import Dict, Any
from pydantic import BaseModel, Field


class DiagnosticRiskRequest(BaseModel):
    age: int = Field(default=45, ge=18, le=100)
    gender: str = Field(default="male")  # "male" or "female"
    systolic_bp: float = Field(default=130.0)
    total_cholesterol: float = Field(default=200.0)
    hdl_cholesterol: float = Field(default=45.0)
    fasting_glucose: float = Field(default=105.0)
    hba1c: float = Field(default=5.8)
    creatinine: float = Field(default=1.1)
    ast: float = Field(default=28.0)
    alt: float = Field(default=32.0)
    platelet_count: float = Field(default=250.0)
    is_smoker: bool = Field(default=False)
    is_diabetic: bool = Field(default=False)


def calculate_clinical_risks(req: DiagnosticRiskRequest) -> Dict[str, Any]:
    """Calculates comprehensive quantitative risk assessments across 4 disease clusters."""
    
    # 1. 10-Year Cardiovascular Disease (CVD) Risk (Framingham Risk Proxy)
    cvd_points = 0
    if req.age > 40: cvd_points += int((req.age - 40) / 5) * 2
    if req.systolic_bp > 120: cvd_points += int((req.systolic_bp - 120) / 10) * 1.5
    if req.total_cholesterol > 200: cvd_points += int((req.total_cholesterol - 200) / 20)
    if req.hdl_cholesterol < 40: cvd_points += 2
    if req.is_smoker: cvd_points += 4
    if req.is_diabetic: cvd_points += 3
    
    cvd_percent = min(75.0, round(1.0 / (1.0 + math.exp(-(cvd_points - 10) * 0.25)) * 100, 1))
    cvd_tier = "Low" if cvd_percent < 10 else ("Moderate" if cvd_percent < 20 else "High / Elevated")

    # 2. Type 2 Diabetes Risk (ADA Risk Calculator Proxy)
    dm_score = 0
    if req.age >= 45: dm_score += 2
    if req.fasting_glucose >= 100: dm_score += 3
    if req.fasting_glucose >= 126: dm_score += 5
    if req.hba1c >= 5.7: dm_score += 3
    if req.hba1c >= 6.5: dm_score += 5
    if req.systolic_bp >= 130: dm_score += 2
    
    dm_tier = "Normal Glycemic" if dm_score <= 2 else ("Prediabetes Warning" if dm_score <= 6 else "Likely Diabetes / Dysglycemia")

    # 3. CKD / Renal Impairment Stage (CKD-EPI eGFR calculation)
    # Simple eGFR approximation
    k = 0.9 if req.gender == "male" else 0.7
    alpha = -0.302 if req.gender == "male" else -0.241
    scr_k = req.creatinine / k
    egfr = round(142.0 * (min(scr_k, 1.0) ** alpha) * (max(scr_k, 1.0) ** -1.200) * (0.9938 ** req.age), 1)
    
    if egfr >= 90: ckd_stage = "Stage 1 (Normal Kidney Function)"
    elif egfr >= 60: ckd_stage = "Stage 2 (Mild Reduction)"
    elif egfr >= 30: ckd_stage = "Stage 3 (Moderate Reduction - Monitor Creatinine)"
    else: ckd_stage = "Stage 4-5 (Severe Renal Impairment)"

    # 4. FIB-4 Liver Fibrosis Index (Age * AST) / (Platelet Count * sqrt(ALT))
    try:
        fib4 = round((req.age * req.ast) / (req.platelet_count * math.sqrt(max(1.0, req.alt))), 2)
    except Exception:
        fib4 = 1.1
    
    if fib4 < 1.3: fib_status = "Low Probability of Hepatic Fibrosis"
    elif fib4 <= 2.67: fib_status = "Indeterminate / Moderate Hepatic Fatty Risk"
    else: fib_status = "High Risk for Advanced Hepatic Fibrosis"

    return {
        "cardiovascular_risk": {
            "ten_year_probability_percent": cvd_percent,
            "category": cvd_tier,
            "framingham_points": round(cvd_points, 1)
        },
        "diabetes_risk": {
            "status": dm_tier,
            "fasting_glucose_mg_dl": req.fasting_glucose,
            "hba1c_percent": req.hba1c
        },
        "renal_health": {
            "estimated_gfr": egfr,
            "kdigo_stage": ckd_stage,
            "serum_creatinine_mg_dl": req.creatinine
        },
        "hepatic_index": {
            "fib4_score": fib4,
            "interpretation": fib_status
        }
    }
