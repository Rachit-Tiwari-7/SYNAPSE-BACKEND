"""
SynapseOS — services/abdm_service.py
Ayushman Bharat Digital Mission (ABDM) & ABHA National Health ID Sandbox.
Ported and adapted from AI-Healthcare-System abdm.py for Indian National Healthcare Integration.
"""

import uuid
import random
from typing import Dict, Any
from datetime import datetime


def generate_abha_id(name: str = "Mausam Kar", year_of_birth: int = 2002, state_code: str = "DL") -> Dict[str, Any]:
    """
    Generates mock Indian ABHA (Ayushman Bharat Health Account) 14-digit number and virtual ABHA address.
    Compliant with NDHM / ABDM FHIR profile specifications.
    """
    # 14-digit format: XX-XXXX-XXXX-XXXX
    p1 = f"{random.randint(10, 99)}"
    p2 = f"{random.randint(1000, 9999)}"
    p3 = f"{random.randint(1000, 9999)}"
    p4 = f"{random.randint(1000, 9999)}"
    
    abha_number = f"{p1}-{p2}-{p3}-{p4}"
    clean_name = name.lower().replace(" ", "")
    abha_address = f"{clean_name}{random.randint(10, 99)}@abdm"
    
    consent_id = str(uuid.uuid4())

    return {
        "status": "ACTIVE",
        "abha_number": abha_number,
        "abha_address": abha_address,
        "name": name,
        "year_of_birth": year_of_birth,
        "state_code": state_code,
        "pm_jay_eligible": True,
        "pm_jay_benefit": "₹5,00,000 / Year Free Hospitalization Coverage (PM-JAY)",
        "consent_artefact_id": consent_id,
        "linked_hip": "All India Institute of Medical Sciences (AIIMS) - Central Node",
        "created_at": datetime.utcnow().isoformat() + "Z"
    }


def check_ayushman_bharat_schemes(condition: str = "general") -> Dict[str, Any]:
    """Returns matching Indian Government health schemes and subsidies."""
    return {
        "pmjay": {
            "name": "Ayushman Bharat Pradhan Mantri Jan Arogya Yojana (PM-JAY)",
            "coverage": "Up to ₹5 Lakh per family per year for secondary and tertiary care hospitalization.",
            "empanelled_hospitals": "28,000+ public and private hospitals across India.",
            "toll_free_helpline": "14555"
        },
        "jan_aushadhi": {
            "name": "Pradhan Mantri Bhartiya Janaushadhi Pariyojana (PMBJP)",
            "benefit": "Quality generic medicines available at 50% to 90% cheaper rates than branded equivalents.",
            "kendra_locator": "Over 10,000 Jan Aushadhi Kendras operating across India."
        },
        "tele_manas": {
            "name": "National Tele Mental Health Programme (Tele-MANAS)",
            "benefit": "Free 24/7 tele-counseling with clinical psychologists and psychiatrists in 20+ languages.",
            "helpline": "14416"
        },
        "nikshay": {
            "name": "Ni-kshay Poshan Yojana (TB Elimination)",
            "benefit": "Direct benefit transfer of ₹500/month for nutritional support to all TB patients."
        }
    }
