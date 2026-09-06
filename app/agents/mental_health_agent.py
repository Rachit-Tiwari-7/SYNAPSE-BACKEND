"""
SynapseOS — agents/mental_health_agent.py
Mental Health, Emotional Wellbeing & Women's Health Agent.
Grounded in WHO mhGAP, Tele-MANAS protocols, and empathetic LLM reasoning (Groq/OpenRouter).
"""

import time
from typing import Dict, Any, List
from backend.app.core.state import SynapseOSState, AgentTraceStep
from backend.app.services.llm_service import call_llm_json


async def evaluate_mental_wellbeing(text: str) -> Dict[str, Any]:
    """
    Evaluates emotional distress, stress indicators, and generates supportive counseling via Groq/OpenRouter.
    """
    fallback = {
        "domain": "MENTAL_WELLBEING",
        "support_title": "Emotional & Mental Health Support (WHO Grounded)",
        "wellbeing_score_assessment": "Emotional Stress Indicators Detected",
        "evidence_based_techniques": [
            "4-7-8 Breathing Technique: Inhale 4s, hold 7s, exhale 8s to downregulate nervous system.",
            "Progressive Muscle Relaxation: Systematically relax tension starting from shoulders down.",
            "Circadian Sleep Hygiene: Limit screens 1 hour prior to sleep."
        ],
        "personalized_coping_advice": "Acknowledge your feelings without self-judgment. Take small, manageable steps today.",
        "tele_manas_helpline": {
            "name": "Tele-MANAS (India National Mental Health Programme)",
            "number": "14416 (Toll-Free, 24/7, 20+ Languages)",
            "alt_number": "1800-891-4416"
        }
    }

    system_prompt = (
        "You are an empathetic, clinical psychologist assistant grounded in WHO mhGAP protocols. "
        "Analyze the user's emotional state or health inquiry and return a JSON object:\n"
        "{\n"
        '  "domain": "MENTAL_WELLBEING" | "WOMENS_HEALTH",\n'
        '  "support_title": "Title of support domain",\n'
        '  "wellbeing_score_assessment": "Clinical assessment of stress/affect state",\n'
        '  "personalized_coping_advice": "Empathetic, compassionate, actionable advice",\n'
        '  "evidence_based_techniques": ["Technique 1", "Technique 2", "Technique 3"],\n'
        '  "tele_manas_helpline": {"name": "Tele-MANAS", "number": "14416"}\n'
        "}"
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"User text: {text}"}
    ]

    return await call_llm_json(messages, fallback_dict=fallback)


async def mental_health_node(state: SynapseOSState) -> SynapseOSState:
    """LangGraph node execution for Mental & Women's Health."""
    start = time.time()
    res = await evaluate_mental_wellbeing(state.input_text)
    
    duration = int((time.time() - start) * 1000)
    state.trace.append(AgentTraceStep(
        agent_name="Mental & Women's Health Agent (Tele-MANAS + LLM)",
        action=f"Generated empathetic guidance -> {res.get('support_title', 'Support')}",
        duration_ms=duration,
        details={"domain": res.get("domain")}
    ))
    return state
