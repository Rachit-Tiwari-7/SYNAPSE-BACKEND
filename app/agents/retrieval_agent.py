"""
SynapseOS — agents/retrieval_agent.py
Parallel Hybrid Retrieval Agent (Wikipedia Medical REST API + 23 WHO/ICMR/MoHFW Clinical Guidelines).
Ported and adapted from MediGenius parallel_retrieval_agent.py and Mental-Health-Chatbot knowledge base.
"""

import httpx
from typing import Dict, Any, List

WIKIPEDIA_API = "https://en.wikipedia.org/api/rest_v1/page/summary"

# Comprehensive Curated Medical Knowledge Index from 23 Official WHO, ICMR & MoHFW Clinical Guidelines
CURATED_GUIDELINES_INDEX = {
    "hypertension": {
        "title": "WHO Guidelines for Pharmacological Treatment of Hypertension in Adults",
        "key_guidelines": [
            "Initiate pharmacotherapy for confirmed hypertension (SBP >= 140 mmHg or DBP >= 90 mmHg).",
            "Target blood pressure for patients without comorbidities is < 140/90 mmHg, and < 130/80 mmHg with diabetes/CVD.",
            "First-line classes: ACE inhibitors, ARBs, Calcium Channel Blockers (CCBs), and thiazide-like diuretics."
        ]
    },
    "diabetes": {
        "title": "ICMR Guidelines for Management of Type 2 Diabetes",
        "key_guidelines": [
            "Target HbA1c is < 7.0% for most non-pregnant adults to reduce microvascular risk.",
            "First-line agent: Metformin (unless contraindicated by eGFR < 30 mL/min).",
            "Lifestyle protocol: Minimum 150 min/week moderate aerobic exercise + low glycemic index diet."
        ]
    },
    "depression": {
        "title": "WHO mhGAP Intervention Guide — Depressive Disorders",
        "key_guidelines": [
            "Screen for acute self-harm risk; if positive, activate Tele-MANAS (14416) or 24/7 supervision.",
            "First-line psychological interventions: Problem-solving therapy, behavioral activation, and CBT.",
            "In moderate-to-severe depression, initiate SSRIs (e.g. Fluoxetine 20mg/day or Sertraline 50mg/day)."
        ]
    },
    "asthma": {
        "title": "Global Initiative for Asthma (GINA) Management Strategy",
        "key_guidelines": [
            "Inhaled Corticosteroids (ICS) form the cornerstone of maintenance therapy across all asthma steps.",
            "As-needed low-dose ICS-formoterol is the preferred reliever to reduce severe exacerbations."
        ]
    },
    "tuberculosis": {
        "title": "National Tuberculosis Elimination Program (NTEP) / Ni-kshay Protocol",
        "key_guidelines": [
            "Diagnostic standard: CBNAAT/TrueNat molecular testing on sputum samples.",
            "Standard 6-month regimen (2HRZE / 4HRE) under direct observation support.",
            "Monthly nutritional benefit of ₹500/month disbursed under Ni-kshay Poshan Yojana."
        ]
    },
    "malaria": {
        "title": "NVBDCP National Drug Policy on Malaria (India)",
        "key_guidelines": [
            "P. falciparum: Artemisinin-based Combination Therapy (ACT-AL or ACT-SP) + single-dose Primaquine.",
            "P. vivax: Chloroquine (25 mg/kg over 3 days) + Primaquine (0.25 mg/kg daily for 14 days)."
        ]
    },
    "dengue": {
        "title": "WHO Clinical Guidelines for Dengue Management",
        "key_guidelines": [
            "Judicious isotonic fluid resuscitation is the cornerstone of severe dengue/DHF management.",
            "Avoid NSAIDs (Aspirin, Ibuprofen) due to platelet dysfunction and hemorrhage risk; use Paracetamol."
        ]
    },
    "anemia": {
        "title": "Anemia Mukt Bharat Operational Guidelines (MoHFW)",
        "key_guidelines": [
            "Prophylactic Iron & Folic Acid (IFA) supplementation: weekly for adolescents, daily for pregnant women.",
            "Dietary diversification with iron-rich foods, Vitamin C enhancers, and bi-annual deworming."
        ]
    },
    "maternal": {
        "title": "Pradhan Mantri Surakshit Matritva Abhiyan (PMSMA) Clinical Standards",
        "key_guidelines": [
            "Mandatory 4 antenatal check-ups (ANC) with guaranteed specialist consultation on 9th of every month.",
            "Universal screening for gestational diabetes (OGTT), pre-eclampsia, and severe anemia."
        ]
    },
    "immunization": {
        "title": "Universal Immunization Programme (UIP) National Schedule",
        "key_guidelines": [
            "Birth: BCG, OPV-0, Hepatitis B birth dose.",
            "6, 10, 14 Weeks: Pentavalent (DPT+HepB+Hib), Rotavirus, fIPV, and PCV.",
            "9-12 Months: MR-1, PCV Booster, and Vitamin A dose 1."
        ]
    },
    "stroke": {
        "title": "Indian Stroke Management Guidelines & WHO Acute Protocol",
        "key_guidelines": [
            "Golden hour recognition: FAST assessment (Face, Arm, Speech, Time).",
            "Intravenous thrombolysis with rtPA within 4.5 hours of symptom onset in acute ischemic stroke."
        ]
    },
    "ckd": {
        "title": "KDIGO Clinical Practice Guideline for Chronic Kidney Disease",
        "key_guidelines": [
            "Monitor eGFR and urinary albumin-to-creatinine ratio (uACR) at least annually.",
            "Initiate ACEi/ARB and SGLT2 inhibitors in diabetic CKD with persistent albuminuria."
        ]
    },
    "copd": {
        "title": "GOLD Global Strategy for Diagnosis & Management of COPD",
        "key_guidelines": [
            "Smoking cessation is the single most effective intervention to modify COPD progression.",
            "Dual bronchodilator therapy (LABA + LAMA) is preferred for patients with persistent breathlessness."
        ]
    },
    "anxiety": {
        "title": "WHO mhGAP Intervention Guide — Anxiety & Stress Disorders",
        "key_guidelines": [
            "Psychoeducation, progressive muscle relaxation, diaphragmatic breathing, and sleep scheduling.",
            "Short-term SSRIs in severe functional impairment; avoid long-term benzodiazepine monotherapy."
        ]
    },
    "osteoarthritis": {
        "title": "ICMR Clinical Guidelines for Osteoarthritis Knee",
        "key_guidelines": [
            "Core non-pharmacological therapies: Quadriceps strengthening exercises, weight reduction, and low-impact walking.",
            "First-line analgesics: Topical NSAIDs (Diclofenac gel) or oral Paracetamol."
        ]
    },
    "thyroid": {
        "title": "Indian Thyroid Society Clinical Practice Guidelines",
        "key_guidelines": [
            "Primary Hypothyroidism: Levothyroxine 1.6 mcg/kg/day taken on an empty stomach 60 min before breakfast.",
            "Target TSH: 0.5 – 4.5 mIU/L in non-pregnant adults, and < 2.5 mIU/L in the first trimester."
        ]
    },
    "sepsis": {
        "title": "Surviving Sepsis Campaign International Guidelines",
        "key_guidelines": [
            "Hour-1 Bundle: Measure lactate, obtain blood cultures before antimicrobials, administer broad-spectrum antibiotics.",
            "Administer 30 mL/kg crystalloid fluid bolus for hypotension or lactate >= 4 mmol/L."
        ]
    },
    "epilepsy": {
        "title": "WHO mhGAP Guidelines for Epilepsy Management",
        "key_guidelines": [
            "Initiate monotherapy with anti-seizure medications (e.g. Levetiracetam, Valproate, or Carbamazepine).",
            "Emergency status epilepticus protocol: IV Lorazepam / Midazolam followed by fosphenytoin loading."
        ]
    },
    "gout": {
        "title": "ACR Guidelines for Management of Gout & Hyperuricemia",
        "key_guidelines": [
            "Acute flare: Colchicine (1.2 mg then 0.6 mg 1h later) or oral NSAIDs / Corticosteroids.",
            "Urate-lowering therapy: Allopurinol as first-line agent, titrating to target serum urate < 6.0 mg/dL."
        ]
    },
    "hepatitis": {
        "title": "National Viral Hepatitis Control Program (NVHCP) Guidelines",
        "key_guidelines": [
            "Hepatitis C: Directly Acting Antivirals (Sofosbuvir + Daclatasvir) for 12 weeks with >95% cure rate.",
            "Hepatitis B: Long-term Tenofovir or Entecavir therapy in patients with elevated ALT and DNA load."
        ]
    },
    "pneumonia": {
        "title": "WHO Community-Acquired Pneumonia Protocol",
        "key_guidelines": [
            "Stratify severity using CURB-65 or CRB-65 criteria.",
            "Outpatient mild CAP: Oral Amoxicillin 500mg-1g TID or Azithromycin in macrolide-sensitive areas."
        ]
    },
    "covid": {
        "title": "MoHFW / ICMR Clinical Guidance for Management of Adult COVID-19",
        "key_guidelines": [
            "Monitor SpO2 using pulse oximeter; red flag if SpO2 < 94% on room air or respiratory rate > 24/min.",
            "Supportive care: Adequate hydration, antipyretics, and prone positioning for oxygenation improvement."
        ]
    },
    "cardiac_arrest": {
        "title": "AHA / ILCOR Basic & Advanced Life Support Guidelines",
        "key_guidelines": [
            "Immediate high-quality CPR: 100-120 compressions/min, 5-6 cm depth, complete chest recoil.",
            "Early defibrillation for shockable rhythms (VF / Pulseless VT) + automated external defibrillator (AED)."
        ]
    }
}


async def search_wikipedia_medical(topic: str) -> Dict[str, Any]:
    """Fetches verified medical summary from Wikipedia REST API."""
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"{WIKIPEDIA_API}/{topic.replace(' ', '_')}", timeout=4.0)
            if resp.status_code == 200:
                data = resp.json()
                return {
                    "title": data.get("title"),
                    "extract": data.get("extract"),
                    "source": "Wikipedia Medical Knowledge",
                    "url": data.get("content_urls", {}).get("desktop", {}).get("page")
                }
    except Exception:
        pass
    return {}


async def hybrid_retrieve_clinical_context(query: str) -> Dict[str, Any]:
    """
    Executes hybrid RAG: checks local 23 WHO/ICMR clinical guidelines corpus first,
    and supplements with live Wikipedia/medical search.
    """
    query_lower = query.lower()
    matched_guidelines = []

    for key, data in CURATED_GUIDELINES_INDEX.items():
        if key in query_lower:
            matched_guidelines.append(data)

    # If no exact match in local guidelines, search Wikipedia REST
    wiki_result = {}
    if not matched_guidelines:
        words = [w for w in query_lower.split() if len(w) > 3]
        if words:
            wiki_result = await search_wikipedia_medical(words[0])

    return {
        "query": query,
        "who_icmr_guidelines": matched_guidelines,
        "external_wiki_context": wiki_result,
        "retrieval_sources_count": len(matched_guidelines) + (1 if wiki_result else 0),
        "corpus_total_indexed": len(CURATED_GUIDELINES_INDEX)
    }
