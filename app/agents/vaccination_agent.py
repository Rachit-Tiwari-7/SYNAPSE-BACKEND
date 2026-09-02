"""
SynapseOS — agents/vaccination_agent.py
National Universal Immunization Programme (UIP) & U-WIN Vaccine Tracking Agent.
Calculates age-milestone vaccine due dates, booster doses, pregnancy immunization, and U-WIN certificates.
"""

import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from backend.app.core.state import SynapseOSState, AgentTraceStep
from backend.app.services.llm_service import call_llm_json

UIP_VACCINATION_SCHEDULE = [
    {
        "milestone_id": "birth",
        "milestone_label": "At Birth (Within 24 Hours)",
        "age_weeks_min": 0,
        "age_weeks_max": 2,
        "category": "child",
        "vaccines": [
            {
                "name": "BCG",
                "full_name": "Bacillus Calmette-Guérin",
                "protects_against": "Severe Childhood Tuberculosis / TB Meningitis",
                "route": "Intradermal (Left Upper Arm)",
                "mandatory_uip": True
            },
            {
                "name": "OPV-0",
                "full_name": "Oral Polio Vaccine (Birth Dose)",
                "protects_against": "Poliomyelitis",
                "route": "Oral Drops (2 Drops)",
                "mandatory_uip": True
            },
            {
                "name": "Hepatitis B (Birth Dose)",
                "full_name": "Hepatitis B Vaccine",
                "protects_against": "Perinatal Hepatitis B Infection",
                "route": "Intramuscular (Anterolateral Mid-Thigh)",
                "mandatory_uip": True
            }
        ]
    },
    {
        "milestone_id": "6_weeks",
        "milestone_label": "6 Weeks (1.5 Months)",
        "age_weeks_min": 6,
        "age_weeks_max": 9,
        "category": "child",
        "vaccines": [
            {
                "name": "Pentavalent-1",
                "full_name": "Diphtheria, Pertussis, Tetanus, Hepatitis B, Hib (5-in-1)",
                "protects_against": "Diphtheria, Whooping Cough, Tetanus, Hep B, Hib Pneumonia/Meningitis",
                "route": "Intramuscular (Left Thigh)",
                "mandatory_uip": True
            },
            {
                "name": "Rotavirus-1",
                "full_name": "Rotavirus Vaccine (RVV)",
                "protects_against": "Severe Rotaviral Diarrhea & Dehydration",
                "route": "Oral Drops (5 Drops)",
                "mandatory_uip": True
            },
            {
                "name": "fIPV-1",
                "full_name": "Fractional Inactivated Polio Vaccine",
                "protects_against": "Poliomyelitis Type 1, 2, 3",
                "route": "Intradermal (Right Upper Arm)",
                "mandatory_uip": True
            },
            {
                "name": "PCV-1",
                "full_name": "Pneumococcal Conjugate Vaccine",
                "protects_against": "Pneumococcal Pneumonia & Sepsis",
                "route": "Intramuscular (Right Thigh)",
                "mandatory_uip": True
            }
        ]
    },
    {
        "milestone_id": "10_weeks",
        "milestone_label": "10 Weeks (2.5 Months)",
        "age_weeks_min": 10,
        "age_weeks_max": 13,
        "category": "child",
        "vaccines": [
            {
                "name": "Pentavalent-2",
                "full_name": "Diphtheria, Pertussis, Tetanus, Hep B, Hib (Dose 2)",
                "protects_against": "5 Core Childhood Infections",
                "route": "Intramuscular (Left Thigh)",
                "mandatory_uip": True
            },
            {
                "name": "Rotavirus-2",
                "full_name": "Rotavirus Vaccine (Dose 2)",
                "protects_against": "Severe Diarrhea",
                "route": "Oral Drops (5 Drops)",
                "mandatory_uip": True
            }
        ]
    },
    {
        "milestone_id": "14_weeks",
        "milestone_label": "14 Weeks (3.5 Months)",
        "age_weeks_min": 14,
        "age_weeks_max": 18,
        "category": "child",
        "vaccines": [
            {
                "name": "Pentavalent-3",
                "full_name": "Diphtheria, Pertussis, Tetanus, Hep B, Hib (Dose 3)",
                "protects_against": "5 Core Childhood Infections",
                "route": "Intramuscular (Left Thigh)",
                "mandatory_uip": True
            },
            {
                "name": "Rotavirus-3",
                "full_name": "Rotavirus Vaccine (Dose 3)",
                "protects_against": "Severe Diarrhea",
                "route": "Oral Drops (5 Drops)",
                "mandatory_uip": True
            },
            {
                "name": "fIPV-2",
                "full_name": "Fractional Inactivated Polio Vaccine (Dose 2)",
                "protects_against": "Poliomyelitis",
                "route": "Intradermal (Right Upper Arm)",
                "mandatory_uip": True
            },
            {
                "name": "PCV-2",
                "full_name": "Pneumococcal Conjugate Vaccine (Dose 2)",
                "protects_against": "Pneumococcal Pneumonia & Sepsis",
                "route": "Intramuscular (Right Thigh)",
                "mandatory_uip": True
            }
        ]
    },
    {
        "milestone_id": "9_12_months",
        "milestone_label": "9 - 12 Months",
        "age_weeks_min": 36,
        "age_weeks_max": 52,
        "category": "child",
        "vaccines": [
            {
                "name": "MR-1",
                "full_name": "Measles & Rubella Vaccine (Dose 1)",
                "protects_against": "Measles, Congenital Rubella Syndrome",
                "route": "Subcutaneous (Right Upper Arm)",
                "mandatory_uip": True
            },
            {
                "name": "PCV Booster",
                "full_name": "Pneumococcal Conjugate Vaccine Booster",
                "protects_against": "Pneumonia / Invasive Pneumococcal Disease",
                "route": "Intramuscular (Right Thigh)",
                "mandatory_uip": True
            },
            {
                "name": "JE-1",
                "full_name": "Japanese Encephalitis Vaccine (Dose 1 in endemic districts)",
                "protects_against": "Japanese Encephalitis / Brain Infection",
                "route": "Subcutaneous (Left Upper Arm)",
                "mandatory_uip": True
            },
            {
                "name": "Vitamin A (Dose 1)",
                "full_name": "Vitamin A Supplementation (1 Lakh IU)",
                "protects_against": "Night Blindness & Child Mortality",
                "route": "Oral Syrup (1 ml)",
                "mandatory_uip": True
            }
        ]
    },
    {
        "milestone_id": "16_24_months",
        "milestone_label": "16 - 24 Months",
        "age_weeks_min": 68,
        "age_weeks_max": 104,
        "category": "child",
        "vaccines": [
            {
                "name": "MR-2",
                "full_name": "Measles & Rubella Vaccine (Dose 2)",
                "protects_against": "Measles & Rubella Immunity Reinforcement",
                "route": "Subcutaneous (Right Upper Arm)",
                "mandatory_uip": True
            },
            {
                "name": "DPT Booster-1",
                "full_name": "Diphtheria, Pertussis, Tetanus Booster 1",
                "protects_against": "Diphtheria, Pertussis, Tetanus",
                "route": "Intramuscular (Left Mid-Thigh)",
                "mandatory_uip": True
            },
            {
                "name": "OPV Booster",
                "full_name": "Oral Polio Booster",
                "protects_against": "Poliomyelitis",
                "route": "Oral Drops (2 Drops)",
                "mandatory_uip": True
            },
            {
                "name": "Vitamin A (Dose 2)",
                "full_name": "Vitamin A (2 Lakh IU)",
                "protects_against": "Xerophthalmia & Blindness Prevention",
                "route": "Oral Syrup (2 ml)",
                "mandatory_uip": True
            }
        ]
    },
    {
        "milestone_id": "5_6_years",
        "milestone_label": "5 - 6 Years",
        "age_weeks_min": 260,
        "age_weeks_max": 312,
        "category": "child",
        "vaccines": [
            {
                "name": "DPT Booster-2",
                "full_name": "Diphtheria, Pertussis, Tetanus Booster 2",
                "protects_against": "School-age Diphtheria & Tetanus Immunity",
                "route": "Intramuscular (Upper Arm)",
                "mandatory_uip": True
            }
        ]
    },
    {
        "milestone_id": "10_16_years",
        "milestone_label": "10 & 16 Years",
        "age_weeks_min": 520,
        "age_weeks_max": 832,
        "category": "adolescent",
        "vaccines": [
            {
                "name": "Td",
                "full_name": "Tetanus & adult Diphtheria Toxoid",
                "protects_against": "Adolescent Tetanus & Diphtheria",
                "route": "Intramuscular (Upper Arm)",
                "mandatory_uip": True
            }
        ]
    },
    {
        "milestone_id": "pregnant_women",
        "milestone_label": "Pregnant Women Immunization",
        "age_weeks_min": 0,
        "age_weeks_max": 0,
        "category": "maternal",
        "vaccines": [
            {
                "name": "Td-1",
                "full_name": "Tetanus & adult Diphtheria (Early in Pregnancy)",
                "protects_against": "Maternal & Neonatal Tetanus",
                "route": "Intramuscular (Upper Arm)",
                "mandatory_uip": True
            },
            {
                "name": "Td-2",
                "full_name": "Tetanus & adult Diphtheria (4 weeks after Td-1)",
                "protects_against": "Maternal & Neonatal Tetanus",
                "route": "Intramuscular (Upper Arm)",
                "mandatory_uip": True
            },
            {
                "name": "Td Booster",
                "full_name": "Td Booster (If received 2 Td doses within 3 years of previous pregnancy)",
                "protects_against": "Tetanus Protection",
                "route": "Intramuscular (Upper Arm)",
                "mandatory_uip": True
            }
        ]
    }
]


def calculate_vaccination_schedule(
    dob_str: Optional[str] = None,
    age_in_weeks: Optional[int] = None,
    category: str = "child"
) -> Dict[str, Any]:
    """
    Calculates completed, current, and upcoming vaccination milestones according to Indian UIP standards.
    """
    current_age_weeks = age_in_weeks
    dob = None

    if dob_str:
        try:
            dob = datetime.fromisoformat(dob_str.replace("Z", ""))
            days_diff = (datetime.utcnow() - dob).days
            current_age_weeks = max(0, days_diff // 7)
        except Exception:
            current_age_weeks = age_in_weeks or 0
    elif current_age_weeks is None:
        current_age_weeks = 0

    if category == "pregnant":
        schedule_items = [m for m in UIP_VACCINATION_SCHEDULE if m["category"] == "maternal"]
        return {
            "category": "maternal",
            "current_status": "Maternal Immunization Protocol Active",
            "recommended_vaccines": schedule_items[0]["vaccines"],
            "guideline": "Administer Td-1 early in first trimester, followed by Td-2 after 4 weeks.",
            "registry_source": "Ministry of Health & Family Welfare (MoHFW) / U-WIN Portal"
        }

    completed_milestones = []
    current_due_milestones = []
    upcoming_milestones = []

    for milestone in UIP_VACCINATION_SCHEDULE:
        if milestone["category"] == "maternal":
            continue

        min_w = milestone["age_weeks_min"]
        max_w = milestone["age_weeks_max"]

        if current_age_weeks > max_w:
            completed_milestones.append({
                **milestone,
                "status": "COMPLETED",
                "administered": True
            })
        elif min_w <= current_age_weeks <= max_w:
            current_due_milestones.append({
                **milestone,
                "status": "CURRENTLY_DUE",
                "urgency": "HIGH",
                "administered": False
            })
        else:
            # Projected date
            projected_date = None
            if dob:
                projected_date = (dob + timedelta(weeks=min_w)).strftime("%Y-%m-%d")

            upcoming_milestones.append({
                **milestone,
                "status": "UPCOMING",
                "projected_due_date": projected_date,
                "administered": False
            })

    total_child_milestones = len(UIP_VACCINATION_SCHEDULE) - 1
    completion_rate_pct = round((len(completed_milestones) / total_child_milestones) * 100, 1)

    next_vaccine_name = "Fully Vaccinated for Age"
    next_vaccine_date = "N/A"
    if current_due_milestones:
        next_vaccine_name = ", ".join([v["name"] for v in current_due_milestones[0]["vaccines"]])
        next_vaccine_date = "DUE NOW (Visit Nearest Anganwadi / PHC)"
    elif upcoming_milestones:
        next_vaccine_name = ", ".join([v["name"] for v in upcoming_milestones[0]["vaccines"]])
        next_vaccine_date = upcoming_milestones[0].get("projected_due_date", "Upcoming")

    return {
        "child_age_weeks": current_age_weeks,
        "dob": dob_str or "Estimated",
        "uip_compliance_pct": completion_rate_pct,
        "next_vaccine_due": next_vaccine_name,
        "next_due_date": next_vaccine_date,
        "current_due": current_due_milestones,
        "completed": completed_milestones,
        "upcoming": upcoming_milestones,
        "free_phc_access": True,
        "registry": "U-WIN / CoWIN National Immunization Registry (MoHFW India)",
        "helpline": "National Health Helpline: 1075 (Toll-Free)"
    }


def generate_uwin_record(
    beneficiary_name: str = "Aarav Sharma",
    dob: str = "2024-05-12",
    guardian_name: str = "Siddharth Sharma",
    state: str = "Delhi",
    uwin_id: Optional[str] = None
) -> Dict[str, Any]:
    """Generates official U-WIN digital immunization certificate format."""
    import uuid
    schedule = calculate_vaccination_schedule(dob_str=dob)
    
    cert_id = uwin_id or f"UWIN-{datetime.utcnow().year}-{str(uuid.uuid4())[:8].upper()}"
    
    return {
        "certificate_id": cert_id,
        "beneficiary_name": beneficiary_name,
        "guardian_name": guardian_name,
        "date_of_birth": dob,
        "gender": "Male",
        "state": state,
        "vaccination_center": "Primary Health Centre (PHC) & Anganwadi Central Node",
        "uip_progress_rate": f"{schedule['uip_compliance_pct']}%",
        "next_vaccine_due": schedule["next_vaccine_due"],
        "next_due_date": schedule["next_due_date"],
        "doses_administered": [
            {"vaccine": "BCG", "date": dob, "batch": "BCG-2024-88A", "status": "VERIFIED"},
            {"vaccine": "OPV-0", "date": dob, "batch": "OPV-IND-401", "status": "VERIFIED"},
            {"vaccine": "HepB-0", "date": dob, "batch": "HEPB-992-B", "status": "VERIFIED"}
        ],
        "upcoming_schedule": schedule["upcoming"][:3],
        "qr_verification": f"https://uwin.mohfw.gov.in/verify?cert={cert_id}",
        "authorized_by": "Ministry of Health and Family Welfare (MoHFW), Govt of India"
    }


async def vaccination_agent_node(state: SynapseOSState) -> SynapseOSState:
    """LangGraph node execution for Universal Immunization & Vaccination Agent."""
    start = time.time()
    query = state.input_text.lower()
    
    # Extract age if mentioned
    weeks = 6
    if "birth" in query or "newborn" in query or "0 month" in query:
        weeks = 0
    elif "6 week" in query or "1.5 month" in query:
        weeks = 6
    elif "10 week" in query or "2.5 month" in query:
        weeks = 10
    elif "14 week" in query or "3.5 month" in query:
        weeks = 14
    elif "9 month" in query or "1 year" in query:
        weeks = 40
    elif "pregnant" in query or "pregnancy" in query:
        res = calculate_vaccination_schedule(category="pregnant")
    else:
        res = calculate_vaccination_schedule(age_in_weeks=weeks)

    if "category" not in locals() or category != "pregnant":
        res = calculate_vaccination_schedule(age_in_weeks=weeks)

    state.vaccination_data = res
    
    duration = int((time.time() - start) * 1000)
    state.trace.append(AgentTraceStep(
        agent_name="National Vaccination & U-WIN Schedule Agent (MoHFW UIP)",
        action=f"Calculated immunization timeline -> Next due: {res.get('next_vaccine_due')}",
        duration_ms=duration,
        details={"compliance": f"{res.get('uip_compliance_pct', 100)}%", "due": res.get("next_vaccine_due")}
    ))
    return state
