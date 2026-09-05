"""
SynapseOS — ml/digital_twin.py
Autonomous Clinical Digital Twin & 10-Year Multi-Organ Trajectory Simulation Engine.
Ported and adapted from AI-Healthcare-System for SynapseOS.
Computes 0-100 organ vitality scores and organ visualization colors for Three.js 3D Twin.
"""

from typing import Dict, Any, List
from pydantic import BaseModel, Field


class DigitalTwinInput(BaseModel):
    age: int = Field(default=35, ge=1, le=120)
    gender: str = Field(default="male")
    systolic_bp: float = Field(default=120.0, description="Systolic Blood Pressure in mmHg")
    diastolic_bp: float = Field(default=80.0, description="Diastolic Blood Pressure in mmHg")
    fasting_glucose: float = Field(default=95.0, description="Fasting Blood Glucose in mg/dL")
    hba1c: float = Field(default=5.4, description="HbA1c percentage")
    egfr: float = Field(default=95.0, description="Estimated GFR in mL/min/1.73m2")
    ldl_cholesterol: float = Field(default=100.0, description="LDL Cholesterol in mg/dL")
    bmi: float = Field(default=23.5, description="Body Mass Index")
    smoking_status: str = Field(default="never", description="never, former, current")
    proposed_interventions: List[str] = Field(default_factory=list, description="e.g. ['exercise', 'mediterranean diet', 'statin']")


def get_organ_health_color(score: float) -> str:
    """Returns color hex code for 3D body renderer."""
    if score >= 85:
        return "#10B981"  # Optimal / Emerald Green
    elif score >= 70:
        return "#06B6D4"  # Good / Cyan
    elif score >= 50:
        return "#F59E0B"  # Moderate Warning / Amber
    else:
        return "#EF4444"  # Critical Deterioration / Red


def compute_baseline_organ_scores(req: DigitalTwinInput) -> Dict[str, Any]:
    """Calculates 0-100 vitality indices per organ system."""
    # 1. Cardiovascular baseline (driven by SBP, DBP, LDL, Age, Smoking)
    cv_score = 100.0
    cv_score -= max(0.0, (req.systolic_bp - 120.0) * 0.6)
    cv_score -= max(0.0, (req.diastolic_bp - 80.0) * 0.4)
    cv_score -= max(0.0, (req.ldl_cholesterol - 100.0) * 0.25)
    if req.smoking_status.lower() in ("current", "active"):
        cv_score -= 15.0
    cv_score -= max(0.0, (req.age - 40.0) * 0.4)
    cv_score = max(15.0, min(98.0, cv_score))

    # 2. Renal / Kidney baseline (driven by eGFR, SBP, Glucose)
    renal_score = min(100.0, max(10.0, req.egfr))
    if req.systolic_bp > 140:
        renal_score -= 8.0
    if req.fasting_glucose > 130:
        renal_score -= 6.0
    renal_score = max(15.0, min(98.0, renal_score))

    # 3. Metabolic / Pancreas baseline (driven by HbA1c, Fasting Glucose, BMI)
    met_score = 100.0
    met_score -= max(0.0, (req.hba1c - 5.4) * 12.0)
    met_score -= max(0.0, (req.fasting_glucose - 95.0) * 0.3)
    met_score -= max(0.0, (req.bmi - 24.0) * 1.5)
    met_score = max(15.0, min(98.0, met_score))

    # 4. Hepatic / Liver baseline (driven by BMI, Glucose)
    hep_score = 100.0
    hep_score -= max(0.0, (req.bmi - 25.0) * 2.0)
    if req.fasting_glucose > 110:
        hep_score -= 5.0
    hep_score = max(20.0, min(98.0, hep_score))

    # 5. Pulmonary / Lungs baseline
    lung_score = 98.0
    if req.smoking_status.lower() in ("current", "active"):
        lung_score -= 28.0
    elif req.smoking_status.lower() in ("former", "past"):
        lung_score -= 10.0
    lung_score -= max(0.0, (req.age - 50.0) * 0.3)
    lung_score = max(20.0, min(98.0, lung_score))

    organs = {
        "heart": {
            "name": "Cardiovascular System (Heart & Arteries)",
            "score": round(cv_score, 1),
            "color": get_organ_health_color(cv_score),
            "status": "Optimal" if cv_score >= 80 else ("Moderate" if cv_score >= 60 else "Attention Needed"),
            "risk_factors": ["High Blood Pressure" if req.systolic_bp > 130 else None, "Elevated LDL" if req.ldl_cholesterol > 120 else None]
        },
        "kidneys": {
            "name": "Renal System (Kidneys)",
            "score": round(renal_score, 1),
            "color": get_organ_health_color(renal_score),
            "status": "Optimal" if renal_score >= 80 else ("Moderate" if renal_score >= 60 else "Attention Needed"),
            "risk_factors": ["Reduced eGFR Filtration" if req.egfr < 90 else None]
        },
        "pancreas": {
            "name": "Metabolic System (Pancreas / Glycemic)",
            "score": round(met_score, 1),
            "color": get_organ_health_color(met_score),
            "status": "Optimal" if met_score >= 80 else ("Moderate" if met_score >= 60 else "Attention Needed"),
            "risk_factors": ["Pre-diabetic / Elevated HbA1c" if req.hba1c > 5.7 else None]
        },
        "liver": {
            "name": "Hepatic System (Liver)",
            "score": round(hep_score, 1),
            "color": get_organ_health_color(hep_score),
            "status": "Optimal" if hep_score >= 80 else ("Moderate" if hep_score >= 60 else "Attention Needed"),
            "risk_factors": ["Elevated Fatty Liver Risk" if req.bmi > 27 else None]
        },
        "lungs": {
            "name": "Pulmonary System (Lungs)",
            "score": round(lung_score, 1),
            "color": get_organ_health_color(lung_score),
            "status": "Optimal" if lung_score >= 80 else ("Moderate" if lung_score >= 60 else "Attention Needed"),
            "risk_factors": ["Smoking Impact" if req.smoking_status == "current" else None]
        }
    }
    
    # Filter none risk factors
    for k in organs:
        organs[k]["risk_factors"] = [rf for rf in organs[k]["risk_factors"] if rf]

    overall_index = round(sum(o["score"] for o in organs.values()) / len(organs), 1)

    return {
        "overall_health_score": overall_index,
        "overall_color": get_organ_health_color(overall_index),
        "organs": organs
    }


def simulate_10_year_trajectory(req: DigitalTwinInput) -> Dict[str, Any]:
    """10-year multi-organ projection with and without therapeutic interventions."""
    baseline = compute_baseline_organ_scores(req)
    years = list(range(0, 11))
    
    trajectory = {}
    for organ_key, organ_info in baseline["organs"].items():
        base_val = organ_info["score"]
        # Natural decay without intervention (~1.5% to 3.5% per year)
        decay_rate = 0.022
        decay_points = [round(max(10.0, base_val * ((1 - decay_rate) ** y)), 1) for y in years]
        
        # Optimized with proposed interventions (e.g. lifestyle, medication)
        boost = 0.015 if req.proposed_interventions else 0.005
        optimized_points = [round(min(99.0, max(10.0, base_val * ((1 - decay_rate + boost) ** y))), 1) for y in years]
        
        trajectory[organ_key] = {
            "years": years,
            "baseline_trajectory": decay_points,
            "optimized_trajectory": optimized_points
        }

    return {
        "baseline": baseline,
        "ten_year_projections": trajectory,
        "recommendations": [
            "Maintain Mediterranean or low-glycemic dietary regimen",
            "Target 150 minutes of weekly moderate aerobic activity",
            "Keep systolic blood pressure strictly under 125 mmHg"
        ]
    }
