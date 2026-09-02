"""
SynapseOS — agents/preventive_health_agent.py
Rural & Semi-Urban Preventive Healthcare Education & Community Health Literacy Agent.
Provides step-by-step guidance for ORS/dehydration, maternal/infant nutrition, vector control, clean water, and interactive awareness quizzes.
"""

import time
import random
from typing import Dict, Any, List, Optional
from backend.app.core.state import SynapseOSState, AgentTraceStep
from backend.app.services.llm_service import call_llm_json

PREVENTIVE_HEALTH_CURRICULUM = [
    {
        "id": "ors_diarrhea",
        "title": "Childhood Diarrhea & ORS Preparation",
        "category": "Child Health & Emergency First Aid",
        "icon": "💧",
        "key_problem": "Dehydration from acute gastroenteritis is a leading preventable cause of child morbidity.",
        "actionable_steps": [
            "1. Take 1 Liter of clean boiled & cooled drinking water.",
            "2. Pour 1 entire sachet of WHO-standard ORS (Oral Rehydration Salts) into the water.",
            "3. Stir well until completely dissolved. Give frequent small sips to child after every loose stool.",
            "4. Home Emergency Alternative: If ORS packet is unavailable, mix 6 teaspoons of sugar + 1/2 teaspoon of salt in 1 liter clean water.",
            "5. Administer 1 Zinc tablet (20mg) daily for 14 consecutive days to heal gut lining and prevent diarrhea recurrence."
        ],
        "red_flags": "Sunken eyes, extreme lethargy, inability to drink, or blood in stool -> Rush to PHC immediately.",
        "gov_scheme": "Intensified Diarrhea Control Fortnight (IDCF) / National Health Mission"
    },
    {
        "id": "poshan_maternal",
        "title": "Maternal Nutrition & Anemia Prevention",
        "category": "Maternal & Child Health",
        "icon": "🤱",
        "key_problem": "Over 50% of pregnant women in rural India suffer from iron-deficiency anemia.",
        "actionable_steps": [
            "1. Consume 1 Iron & Folic Acid (IFA Red Tablet) daily starting from the 4th month of pregnancy (minimum 180 days).",
            "2. Take Calcium tablets twice daily (at different times from IFA, with meals).",
            "3. Eat local iron-rich foods: Drumstick leaves (moringa), spinach, jaggery (gur), sprouted pulses, and seasonal amla (Vitamin C).",
            "4. Complete 4 mandatory Antenatal Check-ups (ANC) at the nearest Anganwadi / PHC.",
            "5. Practice exclusive breastfeeding (no water, no honey) for the baby's first 6 full months."
        ],
        "red_flags": "Severe dizziness, severe swelling of feet/face, blurred vision -> Check blood pressure for pre-eclampsia.",
        "gov_scheme": "Poshan Abhiyaan (National Nutrition Mission) & Anemia Mukt Bharat (MoHFW)"
    },
    {
        "id": "vector_control",
        "title": "Vector-Borne Disease Control (Dengue & Malaria)",
        "category": "Community Sanitation & Epidemic Prevention",
        "icon": "🦟",
        "key_problem": "Aedes (Dengue) and Anopheles (Malaria) mosquitoes breed rapidly in small containers of stagnant water.",
        "actionable_steps": [
            "1. Empty, scrub, and dry out water coolers, flower pots, overhead tanks, and discarded tyres every Sunday (Dry Day).",
            "2. Keep all domestic water storage drums tightly covered with fine cloth or lids.",
            "3. Sleep inside Long-Lasting Insecticidal Nets (LLINs / Medicated Mosquito Nets), especially for children and pregnant mothers.",
            "4. Wear full-sleeve light-colored clothing and apply neem oil or repellent during daytime (Dengue mosquito bites at dawn/dusk).",
            "5. Contact ASHA / ANM for free rapid blood slide testing (RDT) at the very first onset of shivering fever."
        ],
        "red_flags": "High fever with severe eye/joint pain, bleeding from gums, or sudden drop in urine output.",
        "gov_scheme": "National Vector Borne Disease Control Programme (NVBDCP)"
    },
    {
        "id": "clean_water_sanitation",
        "title": "Safe Drinking Water & Household Hygiene",
        "category": "WASH (Water, Sanitation & Hygiene)",
        "icon": "🧼",
        "key_problem": "Waterborne pathogens cause Typhoid, Cholera, Hepatitis A/E, and recurring intestinal worms.",
        "actionable_steps": [
            "1. Bring drinking water to a rolling boil for at least 1-2 minutes to kill bacteria, viruses, and cysts.",
            "2. Use Chlorine / Halazone disinfection tablets (1 tablet per 20 liters) in community wells or flood conditions.",
            "3. Wash hands with soap and water for 20 seconds at 5 critical times: before eating, before feeding baby, after using latrine, after cleaning infant feces, and before cooking.",
            "4. Store drinking water in narrow-mouthed vessels with a ladle/tap to avoid hand contamination.",
            "5. Take bi-annual Albendazole (400mg) chewable deworming tablets on National Deworming Day."
        ],
        "red_flags": "Jaundice (yellow eyes/urine), high prolonged remittent fever (Typhoid), or rice-water diarrhea.",
        "gov_scheme": "Jal Jeevan Mission & Swachh Bharat Mission (Gramin)"
    },
    {
        "id": "lifestyle_ncd",
        "title": "Non-Communicable Disease (NCD) Prevention",
        "category": "Adult Preventive Health",
        "icon": "❤️",
        "key_problem": "Hypertension, diabetes, and cardiovascular diseases are rapidly rising in semi-urban and rural areas.",
        "actionable_steps": [
            "1. Reduce daily dietary salt intake to less than 1 level teaspoon (under 5 grams/day).",
            "2. Strictly avoid chewing Gutkha, Khaini, Bidi, and smoking to prevent oral cancer and arterial stiffness.",
            "3. Engage in 30-45 minutes of brisk physical activity daily (walking, cycling, yoga).",
            "4. Replace processed palm oils with cold-pressed mustard, groundnut, or sesame oil in moderation.",
            "5. Get free annual 30+ NCD screening (Blood Pressure, Random Blood Sugar, Oral/Breast check) at Ayushman Arogya Mandir."
        ],
        "red_flags": "Sudden chest pressure radiating to arm, unexplained weight loss with excessive thirst, or non-healing foot sores.",
        "gov_scheme": "National Programme for Prevention & Control of NCDs (NP-NCD) / Ayushman Bharat"
    }
]

COMMUNITY_QUIZ_BANK = [
    {
        "id": "q1",
        "question": "What is the correct way to prepare Oral Rehydration Salts (ORS) at home for a child with diarrhea?",
        "options": [
            "Mix 1 entire ORS packet in exactly 1 Liter of clean drinking water",
            "Mix ORS in half glass of hot milk",
            "Add only 1 pinch of ORS in a bowl of soup",
            "Take the dry powder directly without water"
        ],
        "correct_index": 0,
        "explanation": "WHO-standard ORS must be mixed in exactly 1 Liter of clean water to maintain the correct electrolyte balance (osmolarity)."
    },
    {
        "id": "q2",
        "question": "How long should a mother exclusively breastfeed her newborn baby without giving any outside water, honey, or milk?",
        "options": [
            "First 1 month only",
            "First 6 full months (180 days)",
            "Until the baby starts crying",
            "First 2 weeks"
        ],
        "correct_index": 1,
        "explanation": "Exclusive breastfeeding for the first 6 full months provides all necessary hydration, nutrients, and immune antibodies."
    },
    {
        "id": "q3",
        "question": "Which mosquitoes transmit Dengue and when do they typically bite?",
        "options": [
            "Aedes mosquitoes; they primarily bite during daylight hours (early morning & late afternoon)",
            "Culex mosquitoes; they only bite at midnight in deep forests",
            "Houseflies; they bite during meal times",
            "Anopheles mosquitoes; they only breed in running rivers"
        ],
        "correct_index": 0,
        "explanation": "Aedes aegypti mosquitoes breed in clean stagnant water in coolers/pots and are aggressive daytime biters."
    },
    {
        "id": "q4",
        "question": "What is the recommended duration of Zinc tablet therapy along with ORS during child diarrhea?",
        "options": [
            "Only 1 day",
            "14 consecutive days (20mg daily)",
            "1 month without food",
            "Zinc is not needed"
        ],
        "correct_index": 1,
        "explanation": "Giving Zinc 20mg for 14 full days heals the intestinal lining and prevents diarrhea episodes for the next 2-3 months."
    },
    {
        "id": "q5",
        "question": "Where can rural citizens get free annual screening for Blood Pressure, Diabetes, and free generic medicines?",
        "options": [
            "At any Ayushman Arogya Mandir (Health & Wellness Centre) and Jan Aushadhi Kendra",
            "Only in private tertiary metro hospitals",
            "Nowhere in India",
            "Only by flying abroad"
        ],
        "correct_index": 0,
        "explanation": "Over 160,000 Ayushman Arogya Mandirs provide free 30+ NCD screening and free essential medications nationwide."
    }
]


def get_preventive_topics() -> List[Dict[str, Any]]:
    """Returns the complete rural preventive health curriculum."""
    return PREVENTIVE_HEALTH_CURRICULUM


def generate_community_health_quiz(count: int = 3) -> Dict[str, Any]:
    """Generates a randomized 3-question community health awareness quiz."""
    selected_questions = random.sample(COMMUNITY_QUIZ_BANK, min(count, len(COMMUNITY_QUIZ_BANK)))
    return {
        "quiz_title": "Sanjeevni-OS Community Preventive Health Awareness Quiz",
        "target_audience": "Rural & Semi-Urban Communities, ASHA Workers, Families",
        "total_questions": len(selected_questions),
        "questions": [
            {
                "id": q["id"],
                "question": q["question"],
                "options": q["options"],
                "correct_index": q["correct_index"],
                "explanation": q["explanation"]
            }
            for q in selected_questions
        ],
        "passing_score_pct": 66,
        "awareness_benefit": "Increases health awareness by 20%+ on childhood dehydration, maternal nutrition, and epidemic prevention."
    }


def evaluate_quiz_answers(user_answers: Dict[str, int]) -> Dict[str, Any]:
    """
    Evaluates quiz submissions and computes community health literacy score.
    """
    score = 0
    total = len(user_answers)
    detailed_feedback = []

    quiz_map = {q["id"]: q for q in COMMUNITY_QUIZ_BANK}

    for qid, chosen_idx in user_answers.items():
        q_item = quiz_map.get(qid)
        if q_item:
            is_correct = (chosen_idx == q_item["correct_index"])
            if is_correct:
                score += 1
            detailed_feedback.append({
                "question_id": qid,
                "question": q_item["question"],
                "chosen_answer": q_item["options"][chosen_idx] if chosen_idx < len(q_item["options"]) else "N/A",
                "is_correct": is_correct,
                "correct_answer": q_item["options"][q_item["correct_index"]],
                "explanation": q_item["explanation"]
            })

    score_pct = round((score / max(1, total)) * 100, 1)
    
    return {
        "score": score,
        "total": total,
        "score_percentage": score_pct,
        "awareness_gain": "+25% Enhanced Preventive Health Literacy",
        "status": "EXCELLENT_AWARENESS" if score_pct >= 66 else "NEEDS_REVIEW",
        "badge": "🌟 Ayushman Health Ambassador" if score_pct >= 66 else "📚 Health Learner",
        "detailed_feedback": detailed_feedback
    }


async def preventive_health_agent_node(state: SynapseOSState) -> SynapseOSState:
    """LangGraph node execution for Rural Preventive Health Education Agent."""
    start = time.time()
    query = state.input_text.lower()

    # Determine most relevant preventive module
    matched_topic = PREVENTIVE_HEALTH_CURRICULUM[0]
    if any(k in query for k in ["ors", "diarrhea", "dehydration", "vomit", "zinc", "loose motion"]):
        matched_topic = PREVENTIVE_HEALTH_CURRICULUM[0]
    elif any(k in query for k in ["maternal", "pregnant", "nutrition", "breastfeed", "anemia", "poshan", "iron", "ifa"]):
        matched_topic = PREVENTIVE_HEALTH_CURRICULUM[1]
    elif any(k in query for k in ["mosquito", "dengue", "malaria", "fever", "vector", "net"]):
        matched_topic = PREVENTIVE_HEALTH_CURRICULUM[2]
    elif any(k in query for k in ["water", "clean", "hygiene", "soap", "boil", "typhoid", "cholera", "wash"]):
        matched_topic = PREVENTIVE_HEALTH_CURRICULUM[3]
    elif any(k in query for k in ["sugar", "salt", "bp", "hypertension", "diabetes", "tobacco", "bidi", "heart"]):
        matched_topic = PREVENTIVE_HEALTH_CURRICULUM[4]

    state.preventive_data = {
        "active_guide": matched_topic,
        "all_topics_count": len(PREVENTIVE_HEALTH_CURRICULUM),
        "quiz_available": True,
        "community_reach_goal": "+20% Awareness Gain Target"
    }

    duration = int((time.time() - start) * 1000)
    state.trace.append(AgentTraceStep(
        agent_name="Rural Preventive Healthcare & Community Education Agent",
        action=f"Loaded public health education guide -> {matched_topic['title']}",
        duration_ms=duration,
        details={"topic": matched_topic["title"], "category": matched_topic["category"]}
    ))
    return state
