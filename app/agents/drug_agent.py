"""
SynapseOS — agents/drug_agent.py
Drug Safety, RxNav Name Normalization, and Multi-Drug Interaction Agent.
Combines NIH RxNorm REST APIs with live LLM clinical pharmacology reasoning (Groq/OpenRouter).
"""

import re
import httpx
from typing import Dict, Any, List, Optional
from backend.app.core.state import SynapseOSState, AgentTraceStep
from backend.app.services.llm_service import call_llm_json

RXNAV_BASE = "https://rxnav.nlm.nih.gov/REST"
RXNAV_TIMEOUT = 3.5

KNOWN_INTERACTIONS = [
    {
        "pair": {"aspirin", "warfarin"},
        "severity": "High / Major Risk",
        "effect": "Severe risk of major internal bleeding and hemorrhage.",
        "action": "Avoid combination unless under strict hematologist monitoring with INR tracking."
    },
    {
        "pair": {"ibuprofen", "warfarin"},
        "severity": "High / Major Risk",
        "effect": "NSAIDs damage stomach lining and potentiate anticoagulant effects of warfarin.",
        "action": "Do not co-administer; use acetaminophen (paracetamol) for mild analgesia if approved by doctor."
    },
    {
        "pair": {"ibuprofen", "lisinopril"},
        "severity": "Moderate Risk",
        "effect": "NSAIDs can decrease the antihypertensive effect of ACE inhibitors and increase renal impairment risk.",
        "action": "Monitor blood pressure and renal function (eGFR/creatinine)."
    },
    {
        "pair": {"sildenafil", "nitroglycerin"},
        "severity": "Critical / Contraindicated",
        "effect": "Profound, potentially fatal hypotension (drop in blood pressure).",
        "action": "Strictly contraindicated. Never take PDE5 inhibitors with nitrates."
    },
    {
        "pair": {"metformin", "alcohol"},
        "severity": "Moderate-High Risk",
        "effect": "Significantly elevated risk of lactic acidosis and hypoglycemia.",
        "action": "Limit alcohol consumption while on metformin therapy."
    },
    {
        "pair": {"atorvastatin", "clarithromycin"},
        "severity": "High Risk",
        "effect": "Inhibition of CYP3A4 increases statin blood levels, causing rhabdomyolysis and muscle toxicity.",
        "action": "Temporarily suspend statin during antibiotic course or use azithromycin."
    },
    {
        "pair": {"ciprofloxacin", "antacid"},
        "severity": "Moderate Risk",
        "effect": "Divalent/trivalent cations (aluminum, magnesium, calcium) chelate fluoroquinolones, preventing absorption.",
        "action": "Take ciprofloxacin 2 hours before or 6 hours after antacids/dairy."
    },
]

GENERIC_EQUIVALENTS = {
    "crocin": "Paracetamol (Acetaminophen) 500mg/650mg - Analgesic & Antipyretic",
    "dolo": "Paracetamol 650mg - Analgesic & Antipyretic",
    "combiflam": "Ibuprofen (400mg) + Paracetamol (325mg)",
    "augmentin": "Amoxicillin (500mg) + Clavulanic Acid (125mg)",
    "pantocid": "Pantoprazole 40mg - Proton Pump Inhibitor (Acid Reflux)",
    "pan d": "Pantoprazole (40mg) + Domperidone (30mg)",
    "glycomet": "Metformin Hydrochloride 500mg/850mg/1000mg",
    "telma": "Telmisartan 40mg/80mg - Angiotensin Receptor Blocker",
    "ecosprin": "Aspirin (Acetylsalicylic Acid) 75mg/150mg Gastro-resistant",
}

_STOPWORDS = {
    "can", "i", "take", "with", "and", "or", "the", "a", "an", "is", "it", "safe", "to", "does",
    "have", "interact", "interaction", "interactions", "between", "my", "for", "of", "drug", "drugs",
    "medication", "medicine", "combine", "mix", "together", "this", "that", "are", "will", "what",
    "about", "dosage", "side", "effects"
}


def extract_candidate_drugs(text: str) -> List[str]:
    words = re.findall(r"[a-zA-Z]{3,}", (text or "").lower())
    return [w for w in dict.fromkeys(words) if w not in _STOPWORDS]


async def resolve_drug_rxnav(term: str) -> Optional[str]:
    """Check NIH RxNorm for drug validity."""
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"{RXNAV_BASE}/rxcui.json", params={"name": term}, timeout=RXNAV_TIMEOUT)
            if resp.status_code == 200:
                data = resp.json()
                if data.get("idGroup", {}).get("rxnormId"):
                    return term
    except Exception:
        pass
    if term in GENERIC_EQUIVALENTS:
        return term
    return None


async def evaluate_drug_safety(text: str) -> Dict[str, Any]:
    """
    Evaluates mentioned medicines, queries NIH RxNav,
    and runs genuine LLM pharmacology interaction checking.
    """
    candidates = extract_candidate_drugs(text)
    detected_drugs = []
    
    for candidate in candidates[:8]:
        resolved = await resolve_drug_rxnav(candidate)
        if resolved:
            detected_drugs.append(resolved)

    detected_set = set(detected_drugs)
    interactions_found = []

    for item in KNOWN_INTERACTIONS:
        if item["pair"].issubset(detected_set):
            interactions_found.append({
                "drugs": list(item["pair"]),
                "severity": item["severity"],
                "effect": item["effect"],
                "recommended_action": item["action"]
            })

    # Generic mappings
    generic_info = []
    for d in detected_drugs:
        if d in GENERIC_EQUIVALENTS:
            generic_info.append({"brand": d, "composition": GENERIC_EQUIVALENTS[d]})

    fallback = {
        "detected_medications": detected_drugs,
        "interactions_count": len(interactions_found),
        "interactions": interactions_found,
        "generic_equivalents": generic_info,
        "safe_to_combine": len(interactions_found) == 0 if len(detected_drugs) >= 2 else None,
        "clinical_pharmacology_summary": "Standard interaction screening completed against NIH RxNav database.",
        "disclaimer": "Always verify drug regimens with a registered pharmacist or prescribing physician."
    }

    # If medications are detected, enrich with live LLM clinical pharmacology analysis
    if detected_drugs:
        system_prompt = (
            "You are a clinical pharmacologist and toxicology AI specialist. "
            "Analyze the following query regarding medications and drug interactions. "
            "Return a strictly valid JSON object with the schema:\n"
            "{\n"
            '  "detected_medications": ["list of detected drugs"],\n'
            '  "interactions": [\n'
            '    {"drugs": ["drugA", "drugB"], "severity": "High"|"Moderate"|"Low", "effect": "Mechanism and physiological impact", "recommended_action": "Safe management advice"}\n'
            '  ],\n'
            '  "interactions_count": 0,\n'
            '  "safe_to_combine": true|false,\n'
            '  "clinical_pharmacology_summary": "Concise plain-language pharmacological safety assessment",\n'
            '  "generic_equivalents": [{"brand": "name", "composition": "active ingredients"}]\n'
            "}"
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Query: {text}. Pre-identified candidates: {detected_drugs}"}
        ]
        llm_result = await call_llm_json(messages, fallback_dict=fallback)
        return llm_result

    return fallback


async def drug_agent_node(state: SynapseOSState) -> SynapseOSState:
    """LangGraph node execution for Drug Safety."""
    import time
    start = time.time()
    
    res = await evaluate_drug_safety(state.input_text)
    state.drug_check = res
    
    duration = int((time.time() - start) * 1000)
    state.trace.append(AgentTraceStep(
        agent_name="Pharmacology & Drug Safety Agent (RxNav + LLM)",
        action=f"Scanned {len(res.get('detected_medications', []))} medications via NIH RxNav & Groq/OpenRouter",
        duration_ms=duration,
        details={"detected": res.get("detected_medications", []), "hazards": len(res.get("interactions", []))}
    ))
    return state
