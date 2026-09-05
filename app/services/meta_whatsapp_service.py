"""
SynapseOS — services/meta_whatsapp_service.py
Meta Official WhatsApp Cloud API Webhook Ingestion, Multilingual Language Selection,
Interactive Menu Engine, FSM, and Multi-Agent Dispatcher.
"""

import base64
import json
import logging
import re
import time
from typing import Dict, Any, Optional, List, Set
import httpx

from backend.app.core.config import settings
from backend.app.services.meta_whatsapp_client import (
    send_whatsapp_message,
    send_whatsapp_image,
    send_whatsapp_interactive_buttons,
    download_meta_media
)
from backend.app.core.session_manager import session_manager
from backend.app.agents.orchestrator import orchestrate_health_request
from backend.app.agents.triage_agent import analyze_symptoms
from backend.app.agents.drug_agent import evaluate_drug_safety
from backend.app.agents.scan_agent import analyze_medical_image
from backend.app.agents.appointment_agent import find_doctors_by_specialty
from backend.app.agents.vaccination_agent import calculate_vaccination_schedule
from backend.app.agents.preventive_health_agent import (
    get_preventive_topics,
    generate_community_health_quiz,
    evaluate_quiz_answers
)
from backend.app.agents.outbreak_agent import get_district_outbreak_risk
from backend.app.services.abdm_service import generate_abha_id, check_ayushman_bharat_schemes
from backend.app.services.i18n_service import translate_clinical_message, get_supported_languages

logger = logging.getLogger(__name__)

# =========================================================
# 1. Multilingual Menus & Language Onboarding
# =========================================================

LANGUAGE_SELECTION_MENU = (
    "🌿 *Welcome to SYNAPSE-OS AI Health Assistant* 🌿\n"
    "_Multi-Agent Clinical Intelligence & Public Health Platform_\n\n"
    "🌐 *Please select your preferred language / अपनी भाषा चुनें:*\n\n"
    "1️⃣ *English* (Default)\n"
    "2️⃣ *हिन्दी* (Hindi)\n"
    "3️⃣ *বাংলা* (Bengali)\n"
    "4️⃣ *தமிழ்* (Tamil)\n"
    "5️⃣ *తెలుగు* (Telugu)\n"
    "6️⃣ *मराठी* (Marathi)\n"
    "7️⃣ *ગુજરાતી* (Gujarati)\n"
    "8️⃣ *ಕನ್ನಡ* (Kannada)\n"
    "9️⃣ *മലയാളം* (Malayalam)\n"
    "🔟 *ਪੰਜਾਬੀ* (Punjabi)\n"
    "1️⃣1️⃣ *ଓଡ଼ିଆ* (Odia)\n\n"
    "👉 _Reply with the number (e.g. `1` for English, `2` for हिन्दी) or name of the language._\n"
    "_You can type 'lang' at any time to switch languages._"
)

LOCALIZED_MENUS: Dict[str, str] = {
    "en": (
        "🌿 SANJEEVNI-OS — Rural & Public Health AI\n"
        "Multilingual Healthcare, Vaccination & Outbreak Assistant\n\n"
        "Reply with a number or simply text your question:\n\n"
        "1 🩺 Symptom Triage — Type symptoms or ask any health question\n"
        "2 💊 Drug Safety — e.g. 2 Aspirin with Ibuprofen\n"
        "3 📷 Scan / Rx — Send an X-ray or Prescription photo\n"
        "4 🧠 Mental Health — Tele-MANAS 24x7 support\n"
        "5 👨‍⚕️ Find Doctor — e.g. 5 General Physician\n"
        "6 🪪 ABHA ID — PM-JAY ₹5 Lakh health card\n"
        "7 💉 Vaccination — e.g. 7 6 weeks\n"
        "8 🚨 Outbreak Alerts — e.g. 8 Delhi\n"
        "9 📝 Rural Health / Quiz — ORS, Nutrition, Quiz\n"
        "sos 🆘 Emergency — Instant ambulance & hospital guide\n\n"
        "Type 'lang' to change language anytime."
    ),
    "hi": (
        "🌿 संजीवनी-ओएस — ग्रामीण एवं जन स्वास्थ्य एआई\n"
        "बहुभाषी स्वास्थ्य सेवा, टीकाकरण एवं महामारी सहायक\n\n"
        "कोई भी स्वास्थ्य प्रश्न पूछें या नंबर चुनें:\n\n"
        "1 🩺 लक्षण जांच — लक्षण लिखें या कोई भी स्वास्थ्य सवाल पूछें\n"
        "2 💊 दवा सुरक्षा — उदा: 2 पैरासिटामोल के साथ एस्पिरिन\n"
        "3 📷 एक्स-रे व पर्ची — फोटो भेजें\n"
        "4 🧠 मानसिक स्वास्थ्य — टेली-मानस 24x7 सहायता\n"
        "5 👨‍⚕️ डॉक्टर खोजें — उदा: 5 जनरल फिजिशियन\n"
        "6 🪪 आभा कार्ड — ₹5 लाख मुफ्त इलाज कार्ड\n"
        "7 💉 टीकाकरण — उदा: 7 6 हफ्ते\n"
        "8 🚨 महामारी अलर्ट — उदा: 8 दिल्ली या पटना\n"
        "9 📝 स्वास्थ्य शिक्षा — ओआरएस, पोषण व क्विज़\n"
        "sos 🆘 आपातकालीन — तत्काल एम्बुलेंस सहायता\n\n"
        "भाषा बदलने के लिए 'lang' लिखें।"
    ),
    "bn": (
        "🌿 *সঞ্জীবনী-ওএস (Sanjeevni-OS) — গ্রামীণ ও জনস্বাস্থ্য এআই* 🌿\n"
        "_বহুভাষিক স্বাস্থ্য পরিষেবা ও টিকাদান নির্দেশিকা_\n\n"
        "স্বাগতম! আমি আপনাকে কীভাবে সাহায্য করতে পারি? একটি *নম্বর* লিখুন:\n\n"
        "1️⃣ *লক্ষণ পরীক্ষা* — `1` লিখে আপনার লক্ষণ জানান\n"
        "2️⃣ *ওষুধ নিরাপত্তা* — `2` লিখে ওষুধের নাম লিখুন\n"
        "3️⃣ *এক্স-রে ও প্রেসক্রিপশন AI* — ছবি বা এক্স-রে পাঠান 📷\n"
        "4️⃣ *মানসিক স্বাস্থ্য (Tele-MANAS)* — `4` লিখে সমস্যা জানান\n"
        "5️⃣ *ডাক্তার খুঁজুন* — `5` লিখে স্পেশালিস্ট খুঁজুন\n"
        "6️⃣ *আভা আইডি (ABHA Card)* — `6` লিখে ₹৫ লাখ স্বাস্থ্য কার্ড জানুন\n"
        "7️⃣ *টিকাদান সময়সূচী (UIP)* — `7` লিখে শিশুর বয়স জানান\n"
        "8️⃣ *মহামারী সতর্কতা* — `8` লিখে জেলার নাম জানান\n"
        "9️⃣ *স্বাস্থ্য শিক্ষা ও কুইজ* — `9` ওআরএস ও পুষ্টির জন্য\n"
        "🚨 *জরুরি সহায়তা* — তৎক্ষণাৎ *SOS* পাঠান"
    ),
    "ta": (
        "🌿 *சஞ்சீவனி-ஓஎஸ் (Sanjeevni-OS) — கிராமப்புற சுகாதார ஏஐ* 🌿\n"
        "_பன்மொழி சுகாதார பராமரிப்பு & தடுப்பூசி வழிகாட்டி_\n\n"
        "வணக்கம்! நான் உங்களுக்கு எப்படி உதவ முடியும்? எண்ணைத் தேர்ந்தெடுக்கவும்:\n\n"
        "1️⃣ *அறிகுறி பரிசோதனை* — `1` எழுதி அறிகுறிகளை அனுப்பவும்\n"
        "2️⃣ *மருந்து பாதுகாப்பு* — `2` எழுதி மருந்துகளின் பெயர்களை அனுப்பவும்\n"
        "3️⃣ *எக்ஸ்ரே மற்றும் மருந்து சீட்டு* — புகைப்படத்தை அனுப்பவும் 📷\n"
        "4️⃣ *மனநலம் (Tele-MANAS)* — `4` எழுதி ஆலோசனை பெறவும்\n"
        "5️⃣ *மருத்துவரை கண்டறிய* — `5` எழுதி மருத்துவரை தேடவும்\n"
        "6️⃣ *ஆபா அட்டை (ABHA)* — `6` எழுதி ₹5 லட்சம் காப்பீடு அறியவும்\n"
        "7️⃣ *தடுப்பூசி அட்டவணை (UIP)* — `7` எழுதி குழந்தையின் வயதை அனுப்பவும்\n"
        "8️⃣ *நோய் தொற்று எச்சரிக்கை* — `8` எழுதி மாவட்ட பெயரை அனுப்பவும்\n"
        "9️⃣ *சுகாதார விழிப்புணர்வு & வினாடி வினா* — `9` ஐ அனுப்பவும்\n"
        "🚨 *அவசர உதவி* — *SOS* அனுப்பவும்"
    ),
    "te": (
        "🌿 *సంజీవని-ఓఎస్ (Sanjeevni-OS) — గ్రామీణ ప్రజారోగ్య ఏఐ* 🌿\n"
        "_బహుభాషా ఆరోగ్య సంరక్షణ & వ్యాక్సినేషన్ ఇంటెలిజెన్స్_\n\n"
        "నమస్కారం! నేను మీకు ఎలా సహాయపడగలను? నంబర్ ఎంచుకోండి:\n\n"
        "1️⃣ *లక్షణాల నిర్ధారణ* — `1` మరియు లక్షణాలను టైప్ చేయండి\n"
        "2️⃣ *ఔషధ భద్రత* — `2` మరియు మందుల పేర్లు టైప్ చేయండి\n"
        "3️⃣ *ఎక్స్-రే & ప్రిస్క్రిప్షన్* — ఫోటో పంపండి 📷\n"
        "4️⃣ *మానసిక ఆరోగ్యం (Tele-MANAS)* — `4` టైప్ చేయండి\n"
        "5️⃣ *వైద్యుడిని కనుగొనండి* — `5` టైప్ చేయండి\n"
        "6️⃣ *ఆభా హెల్త్ కార్డ్ (ABHA)* — `6` టైప్ చేయండి\n"
        "7️⃣ *టీకా షెడ్యూల్ (UIP)* — `7` మరియు వయస్సు టైప్ చేయండి\n"
        "8️⃣ *వ్యాధి వ్యాప్తి హెచ్చరికలు* — `8` మరియు జిల్లా పేరు టైప్ చేయండి\n"
        "9️⃣ *ఆరోగ్య అవగాహన & క్విజ్* — `9` టైప్ చేయండి\n"
        "🚨 *అత్యవసర సహాయం* — *SOS* టైప్ చేయండి"
    ),
    "mr": (
        "🌿 *संजीवनी-ओएस (Sanjeevni-OS) — ग्रामीण व सार्वजनिक आरोग्य एआय* 🌿\n"
        "_बहुभाषिक आरोग्य सेवा आणि लसीकरण मार्गदर्शक_\n\n"
        "नमस्कार! मी आपली काय मदत करू शकतो? खालील पर्याय निवडा:\n\n"
        "1️⃣ *लक्षणे तपासणी* — `1` लिहून लक्षणे सांगा\n"
        "2️⃣ *औषध सुरक्षा* — `2` लिहून औषधांची नावे सांगा\n"
        "3️⃣ *क्ष-किरण (X-Ray) तपासणी* — फोटो पाठवा 📷\n"
        "4️⃣ *मानसिक आरोग्य* — `4` लिहून सल्ला घ्या\n"
        "5️⃣ *डॉक्टर शोधा* — `5` लिहून तज्ज्ञ शोधा\n"
        "6️⃣ *आभा कार्ड (ABHA)* — `6` लिहून माहिती मिळवा\n"
        "7️⃣ *लसीकरण वेळापत्रक (UIP)* — `7` लिहून वय सांगा\n"
        "8️⃣ *साथरोग सतर्कता* — `8` लिहून जिल्ह्याचे नाव सांगा\n"
        "9️⃣ *आरोग्य शिक्षण व प्रश्नमंजुषा* — `9` पाठवा\n"
        "🚨 *तातडीची मदत* — *SOS* पाठवा"
    ),
    "gu": (
        "🌿 *સંજીવની-ઓએસ (Sanjeevni-OS) — ગ્રામીણ આરોગ્ય એઆઈ* 🌿\n"
        "_બહુભાષી આરોગ્ય સેવા અને રસીકરણ માર્ગદર્શિકા_\n\n"
        "નમસ્તે! હું તમારી શું મદદ કરી શકું? નંબર પસંદ કરો:\n\n"
        "1️⃣ *લક્ષણો તપાસ* — `1` લખીને લક્ષણો જણાવો\n"
        "2️⃣ *દવા સુરક્ષા* — `2` લખીને દવાઓના નામ જણાવો\n"
        "3️⃣ *એક્સ-રે અને પ્રિસ્ક્રિપ્શન* — ફોટો મોકલો 📷\n"
        "4️⃣ *માનસિક સ્વાસ્થ્ય* — `4` લખીને સલાહ લો\n"
        "5️⃣ *ડૉક્ટર શોધો* — `5` લખીને નિષ્ણાત શોધો\n"
        "6️⃣ *આભા કાર્ડ (ABHA)* — `6` લખીને લાભ જાણો\n"
        "7️⃣ *રસીકરણ શેડ્યૂલ (UIP)* — `7` લખીને ઉંમર જણાવો\n"
        "8️⃣ *રોગચાળો ચેતવણી* — `8` લખીને જિલ્લો જણાવો\n"
        "9️⃣ *આરોગ્ય જાગૃતિ અને ક્વિઝ* — `9` મોકલો\n"
        "🚨 *ઇમરજન્સી સહાય* — *SOS* મોકલો"
    ),
    "kn": (
        "🌿 *ಸಂಜೀವನಿ-ಓಎಸ್ (Sanjeevni-OS) — ಗ್ರಾಮೀಣ ಆರೋಗ್ಯ ಎಐ* 🌿\n"
        "_ಬಹುಭಾಷಾ ಆರೋಗ್ಯ ಸೇವೆ ಮತ್ತು ಲಸಿಕೆ ಮಾರ್ಗದರ್ಶಿ_\n\n"
        "ನಮಸ್ಕಾರ! ನಾನು ನಿಮಗೆ ಹೇಗೆ ಸಹಾಯ ಮಾಡಲಿ? ಸಂಖ್ಯೆಯನ್ನು ಆರಿಸಿ:\n\n"
        "1️⃣ *ರೋಗಲಕ್ಷಣ ಪರೀಕ್ಷೆ* — `1` ಬರೆದು ಕಳುಹಿಸಿ\n"
        "2️⃣ *ಔಷಧಿ ಸುರಕ್ಷತೆ* — `2` ಬರೆದು ಕಳುಹಿಸಿ\n"
        "3️⃣ *ಎಕ್ಸ್‌ರೇ ತಪಾಸಣೆ* — ಫೋಟೋ ಕಳುಹಿಸಿ 📷\n"
        "4️⃣ *ಮಾನಸಿಕ ಆರೋಗ್ಯ* — `4` ಬರೆದು ಕಳುಹಿಸಿ\n"
        "5️⃣ *ವೈದ್ಯರನ್ನು ಹುಡುಕಿ* — `5` ಬರೆದು ಕಳುಹಿಸಿ\n"
        "6️⃣ *ಆಭಾ ಕಾರ್ಡ್ (ABHA)* — `6` ಬರೆದು ಕಳುಹಿಸಿ\n"
        "7️⃣ *ಲಸಿಕೆ ವೇಳಾಪಟ್ಟಿ (UIP)* — `7` ಬರೆದು ಕಳುಹಿಸಿ\n"
        "8️⃣ *ಸಾಂಕ್ರಾಮಿಕ ರೋಗ ಎಚ್ಚರಿಕೆ* — `8` ಬರೆದು ಕಳುಹಿಸಿ\n"
        "9️⃣ *ಆರೋಗ್ಯ ಜಾಗೃತಿ ಮತ್ತು ರಸಪ್ರಶ್ನೆ* — `9` ಕಳುಹಿಸಿ\n"
        "🚨 *ತುರ್ತು ಸೇವೆ* — *SOS* ಕಳುಹಿಸಿ"
    ),
    "ml": (
        "🌿 *സഞ്ജീവനി-ഒഎസ് (Sanjeevni-OS) — ആരോഗ്യ എഐ* 🌿\n"
        "_ബഹുഭാഷാ ആരോഗ്യ പരിപാലനം & പ്രതിരോധ കുത്തിവയ്പ്പ്_\n\n"
        "നമസ്കാരം! ഞാൻ നിങ്ങളെ എങ്ങനെ സഹായിക്കണം? നമ്പർ തിരഞ്ഞെടുക്കുക:\n\n"
        "1️⃣ *രോഗലക്ഷണ പരിശോധന* — `1` ടൈപ്പ് ചെയ്യുക\n"
        "2️⃣ *മരുന്ന് സുരക്ഷ* — `2` ടൈപ്പ് ചെയ്യുക\n"
        "3️⃣ *എക്സ്-റേ പരിശോധന* — ഫോട്ടോ അയക്കുക 📷\n"
        "4️⃣ *മാനസികാരോഗ്യം* — `4` ടൈപ്പ് ചെയ്യുക\n"
        "5️⃣ *ഡോക്ടറെ കണ്ടെത്തുക* — `5` ടൈപ്പ് ചെയ്യുക\n"
        "6️⃣ *ആഭാ കാർഡ് (ABHA)* — `6` ടൈപ്പ് ചെയ്യുക\n"
        "7️⃣ *കുത്തിവയ്പ്പ് വിവരങ്ങൾ (UIP)* — `7` ടൈപ്പ് ചെയ്യുക\n"
        "8️⃣ *പകർച്ചവ്യാധി ജാഗ്രത* — `8` ടൈപ്പ് ചെയ്യുക\n"
        "9️⃣ *ആരോഗ്യ ക്വിസ്* — `9` ടൈപ്പ് ചെയ്യുക\n"
        "🚨 *അടിയന്തര സഹായം* — *SOS* ടൈപ്പ് ചെയ്യുക"
    ),
    "pa": (
        "🌿 *ਸੰਜੀਵਨੀ-ਓਐਸ (Sanjeevni-OS) — ਸਿਹਤ ਏਆਈ* 🌿\n"
        "_ਬਹੁ-ਭਾਸ਼ਾਈ ਸਿਹਤ ਸੰਭਾਲ ਅਤੇ ਟੀਕਾਕਰਨ ਗਾਈਡ_\n\n"
        "ਸਤਿ ਸ੍ਰੀ ਅਕਾਲ! ਮੈਂ ਤੁਹਾਡੀ ਕੀ ਮਦਦ ਕਰ ਸਕਦਾ ਹਾਂ? ਨੰਬਰ ਚੁਣੋ:\n\n"
        "1️⃣ *ਲੱਛਣ ਜਾਂਚ* — `1` ਲਿਖ ਕੇ ਲੱਛਣ ਦੱਸੋ\n"
        "2️⃣ *ਦਵਾਈ ਸੁਰੱਖਿਆ* — `2` ਲਿਖ ਕੇ ਦਵਾਈਆਂ ਦੇ ਨਾਮ ਦੱਸੋ\n"
        "3️⃣ *ਐਕਸ-ਰੇ ਅਤੇ ਪਰਚੀ* — ਫੋਟੋ ਭੇਜੋ 📷\n"
        "4️⃣ *ਮਾਨਸਿਕ ਸਿਹਤ* — `4` ਲਿਖ ਕੇ ਸਲਾਹ ਲਓ\n"
        "5️⃣ *ਡਾਕਟਰ ਲੱਭੋ* — `5` ਲਿਖ ਕੇ ਡਾਕਟਰ ਲੱਭੋ\n"
        "6️⃣ *ਆਭਾ ਕਾਰਡ (ABHA)* — `6` ਲਿਖ ਕੇ ਜਾਣਕਾਰੀ ਲਓ\n"
        "7️⃣ *ਟੀਕਾਕਰਨ ਸ਼ਡਿਊਲ (UIP)* — `7` ਲਿਖ ਕੇ ਉਮਰ ਦੱਸੋ\n"
        "8️⃣ *ਮਹਾਂਮਾਰੀ ਚੇਤਾਵਨੀ* — `8` ਲਿਖ ਕੇ ਜ਼ਿਲ੍ਹੇ ਦਾ ਨਾਮ ਦੱਸੋ\n"
        "9️⃣ *ਸਿਹਤ ਜਾਗਰੂਕਤਾ ਤੇ ਕੁਇਜ਼* — `9` ਭੇਜੋ\n"
        "🚨 *ਐਮਰਜੈਂਸੀ ਸਹਾਇਤਾ* — *SOS* ਭੇਜੋ"
    ),
    "or": (
        "🌿 *ସଞ୍ଜୀବନୀ-ଓଏସ୍ (Sanjeevni-OS) — ସ୍ୱାସ୍ଥ୍ୟ ଏଆଇ* 🌿\n"
        "_ବହୁଭାଷୀ ସ୍ୱାସ୍ଥ୍ୟ ସେବା ଏବଂ ଟୀକାକରଣ ନିର୍ଦ୍ଦେଶିକା_\n\n"
        "ନମସ୍କାର! ମୁଁ ଆପଣଙ୍କୁ କିପରି ସାହାଯ୍ୟ କରିପାରିବି? ନମ୍ବର ବାଛନ୍ତୁ:\n\n"
        "1️⃣ *ଲକ୍ଷଣ ପରୀକ୍ଷା* — `1` ଲେଖି ଲକ୍ଷଣ ଜଣାନ୍ତୁ\n"
        "2️⃣ *ଔଷଧ ସୁରକ୍ଷା* — `2` ଲେଖି ଔଷଧ ନାମ ଜଣାନ୍ତୁ\n"
        "3️⃣ *ଏକ୍ସ-ରେ ଯାଞ୍ଚ* — ଫଟୋ ପଠାନ୍ତୁ 📷\n"
        "4️⃣ *ମାନସିକ ସ୍ୱାସ୍ଥ୍ୟ* — `4` ଲେଖି ପରାମର୍ଶ ନିଅନ୍ତୁ\n"
        "5️⃣ *ଡାକ୍ତର ଖୋଜନ୍ତୁ* — `5` ଲେଖି ଡାକ୍ତର ଖୋଜନ୍ତୁ\n"
        "6️⃣ *ଆଭା କାର୍ଡ (ABHA)* — `6` ଲେଖି ଜାଣନ୍ତୁ\n"
        "7️⃣ *ଟୀକାକରଣ ତାଲିକା (UIP)* — `7` ଲେଖି ବୟସ ଜଣାନ୍ତୁ\n"
        "8️⃣ *ମହାମାରୀ ସତର୍କତା* — `8` ଲେଖି ଜିଲ୍ଲା ଜଣାନ୍ତୁ\n"
        "9️⃣ *ସ୍ୱାସ୍ଥ୍ୟ ଶିକ୍ଷା ଓ କୁଇଜ୍* — `9` ପଠାନ୍ତୁ\n"
        "🚨 *ଜରୁରୀକାଳୀନ ସେବା* — *SOS* ପଠାନ୍ତୁ"
    )
}

MAIN_MENU_TEXT = LOCALIZED_MENUS["en"]


def parse_language_selection(text: str) -> Optional[str]:
    """Maps user input to supported language code."""
    t = text.strip().lower()
    mapping = {
        "1": "en", "en": "en", "english": "en",
        "2": "hi", "hi": "hi", "hindi": "hi", "हिंदी": "hi", "हिन्दी": "hi",
        "3": "bn", "bn": "bn", "bengali": "bn", "বাংলা": "bn", "bangla": "bn",
        "4": "ta", "ta": "ta", "tamil": "ta", "தமிழ்": "ta",
        "5": "te", "te": "te", "telugu": "te", "తెలుగు": "te",
        "6": "mr", "mr": "mr", "marathi": "mr", "मराठी": "mr",
        "7": "gu", "gu": "gu", "gujarati": "gu", "ગુજરાતી": "gu",
        "8": "kn", "kn": "kn", "kannada": "kn", "ಕನ್ನಡ": "kn",
        "9": "ml", "ml": "ml", "malayalam": "ml", "മലയാളം": "ml",
        "10": "pa", "pa": "pa", "punjabi": "pa", "ਪੰਜਾਬੀ": "pa",
        "11": "or", "or": "or", "odia": "or", "ଓଡ଼ିଆ": "or", "oriya": "or"
    }
    return mapping.get(t)


def detect_language_script(text: str) -> str:
    """Detects Indian regional language script from unicode code points."""
    for char in text:
        cp = ord(char)
        if 0x0900 <= cp <= 0x097F:
            return "hi"  # Devanagari (Hindi/Marathi)
        elif 0x0980 <= cp <= 0x09FF:
            return "bn"  # Bengali / Assamese
        elif 0x0B80 <= cp <= 0x0BFF:
            return "ta"  # Tamil
        elif 0x0C00 <= cp <= 0x0C7F:
            return "te"  # Telugu
        elif 0x0A80 <= cp <= 0x0AFF:
            return "gu"  # Gujarati
        elif 0x0C80 <= cp <= 0x0CFF:
            return "kn"  # Kannada
        elif 0x0D00 <= cp <= 0x0D7F:
            return "ml"  # Malayalam
        elif 0x0A00 <= cp <= 0x0A7F:
            return "pa"  # Punjabi
        elif 0x0B00 <= cp <= 0x0B7F:
            return "or"  # Odia
    return "en"

import re


def strip_markdown_to_plain_text(text: str) -> str:
    """Converts markdown text to clean, readable normal text without markdown asterisks, hashes, or syntax."""
    if not text:
        return ""
    # 1. Clean code blocks and inline code ticks
    s = re.sub(r'```[a-zA-Z]*\n?', '', text)
    s = s.replace('```', '').replace('`', '')
    # 2. Normalize markdown headers (#, ##, ###, ####)
    s = re.sub(r'^[#]+\s*', '', s, flags=re.MULTILINE)
    # 3. Remove horizontal lines (---, ===, ***)
    s = re.sub(r'^[-=*]{3,}\s*$', '', s, flags=re.MULTILINE)
    # 4. Normalize list bullets (* item or - item -> • item)
    s = re.sub(r'^\s*[-*]\s+', '• ', s, flags=re.MULTILINE)
    # 5. Remove bold and italic markers (* and _)
    s = s.replace('**', '').replace('*', '').replace('__', '').replace('_', '')
    # 6. Normalize multiple consecutive newlines
    s = re.sub(r'\n{3,}', '\n\n', s)
    return s.strip()


_IGNORED_DIAGNOSIS_SUBSTRINGS = [
    "triage & outbreak", "specialist findings", "clinical rationale",
    "drug safety", "ai council", "immediate action", "what to expect",
    "assessment & findings", "clinical assessment", "audit finding",
    "disclaimer", "patient status", "vitals to monitor", "safety protocol",
    "status: consensus reached", "findings", "general informational inquiry",
    "informational inquiry", "inquiry regarding", "general inquiry", "pharmacological"
]

_INDIAN_RELIEF_DATABASE = [
    {
        "keywords": ["fever", "pyrexia", "temperature", "body ache", "headache"],
        "primary": "• Dolo 650 / Crocin (Paracetamol 650mg): 1 tab after meals (with water) for fever/pain (max 3/day).",
        "secondary": "• Electral ORS: 1 sachet dissolved in 1L water; sip throughout day for hydration."
    },
    {
        "keywords": ["vomit", "nausea", "dehydrat", "loose motion", "diarrhea", "stool", "gastric"],
        "primary": "• Electral ORS: 1 packet in 1L clean drinking water; sip slowly after every episode.",
        "secondary": "• Pan-D (Pantoprazole + Domperidone): 1 cap 30 min before breakfast (empty stomach) for nausea/acidity."
    },
    {
        "keywords": ["acid", "heartburn", "reflux", "stomach burn", "indigestion", "gerd"],
        "primary": "• Pan-40 / Pantocid (Pantoprazole 40mg): 1 tab 30 min before breakfast on empty stomach.",
        "secondary": "• Gelusil / Digene gel: 2 tsp liquid after meals as needed."
    },
    {
        "keywords": ["cold", "sneeze", "runny nose", "congestion", "throat", "cough", "rhinitis"],
        "primary": "• Cetirizine 10mg (Cetzine / Okacet): 1 tab at bedtime after food for runny nose/sneezes.",
        "secondary": "• Warm saline gargles & steam: 3 times daily; lozenges (Strepsils) after meals."
    }
]

def _clean_card_text(s: str) -> str:
    s = re.sub(r'[*_#`]', '', s)
    s = re.sub(r'^[•\-\d\.\s]+', '', s)
    return s.strip()

def _truncate_clean(text: str, max_len: int = 100) -> str:
    if len(text) <= max_len:
        return text
    truncated = text[:max_len].rsplit(' ', 1)[0]
    return truncated.rstrip(' ,;:-') + '...'

def _extract_diagnosis_condition(text: str) -> str:
    # 1. Match explicit Risk Profile / Diagnosis / Suspected Condition with markdown resilience
    m = re.search(
        r'(?:\*\*|\*|•|\-)?\s*(?:Risk Profile|Suspected (?:Condition|Diagnosis|Infection)|Primary Diagnosis|Diagnosis)\s*(?:\*\*|\*)?\s*[:\-]\s*([^\n]+)',
        text,
        re.IGNORECASE
    )
    if m:
        val = _clean_card_text(m.group(1))
        if '.' in val:
            val = val.split('.')[0].strip()
        if not any(ign in val.lower() for ign in _IGNORED_DIAGNOSIS_SUBSTRINGS) and len(val) > 4:
            return val

    # 2. Match Executive Summary & Suspected Diagnosis section
    m_exec_sec = re.search(r'(?:Executive Summary & Suspected Diagnosis|Suspected Diagnosis)[^\n]*\n+([^\n]+)', text, re.IGNORECASE)
    if m_exec_sec:
        val = _clean_card_text(m_exec_sec.group(1))
        if '.' in val:
            val = val.split('.')[0].strip()
        if not any(ign in val.lower() for ign in _IGNORED_DIAGNOSIS_SUBSTRINGS) and len(val) > 4:
            m_cond = re.search(r'(?:consistent with|indicative of|suggestive of|including|risk for)\s+([^.\n]+)', val, re.IGNORECASE)
            if m_cond:
                return _clean_card_text(m_cond.group(1))
            return val

    # 3. Match within Executive Summary text
    exec_m = re.search(r'(?:Executive Summary|Summary)[^\n]*\n+([^\n]+(?:\n[^\n]+)?)', text, re.IGNORECASE)
    if exec_m:
        exec_text = exec_m.group(1)
        m_cond = re.search(r'(?:consistent with|indicative of|suggestive of|including|risk for)\s+([^.\n]+)', exec_text, re.IGNORECASE)
        if m_cond:
            c = _clean_card_text(m_cond.group(1))
            if len(c) > 4 and not any(ign in c.lower() for ign in _IGNORED_DIAGNOSIS_SUBSTRINGS):
                return c
        first_sentence = _clean_card_text(exec_text.split('.')[0])
        if len(first_sentence) > 10 and not any(ign in first_sentence.lower() for ign in _IGNORED_DIAGNOSIS_SUBSTRINGS):
            return first_sentence

    # 4. Fallback on Critical Flags Detected
    m_flags = re.search(r'Critical Flags Detected:\*?\s*([^\n]+)', text, re.IGNORECASE)
    if m_flags:
        flags = _clean_card_text(m_flags.group(1))
        return f"Acute Clinical Presentation ({flags})"

    return "Multi-agent clinical audit completed by AI Council."

def _extract_medications_guidance(text: str, is_emergency: bool) -> List[str]:
    meds: List[str] = []
    text_lower = text.lower()

    # Check for explicit medical prohibition / withholding rule (e.g., in CNS or surgical emergency)
    if re.search(r'(?:do not self-medicate|strictly avoid|withhold(?:ing)?\s+symptomatic relief|avoid taking|withhold self-medication)', text_lower):
        meds.append("• ⚠️ Withhold self-medication: Do not take painkillers or anti-emetics (masks neurological & abdominal signs).")
        meds.append("• At Hospital: IV fluids & targeted emergency medications will be administered.")
        return meds

    # Check if text contains structured Indian medicine section
    med_sec = re.search(
        r'(?:Recommended Medications & Relief \(India\)|Medications & Symptom Relief \(India\)|Medications & Relief|Medications|Drug Safety)[^\n]*\n+([\s\S]*?)(?=\n\n|\n\*[0-9]|\n[#*•]{1,3}\s+[A-Z]|\Z)',
        text,
        re.IGNORECASE
    )
    if med_sec:
        lines = med_sec.group(1).split('\n')
        for l in lines:
            cl = _clean_card_text(l)
            if len(cl) > 12 and any(term in cl.lower() for term in ["dolo", "crocin", "paracetamol", "electral", "ors", "cetzine", "pan", "pantoprazole", "tablet", "tab", "mg", "gargle"]):
                meds.append(f"• {cl}")
                if len(meds) >= 2:
                    break

    if meds:
        return meds

    # Fallback to Indian OTC relief database based on symptoms in text
    for entry in _INDIAN_RELIEF_DATABASE:
        if any(k in text_lower for k in entry["keywords"]):
            meds.append(entry["primary"])
            if entry.get("secondary"):
                meds.append(entry["secondary"])
            break

    if not meds:
        if is_emergency:
            meds.append("• ⚠️ Withhold oral medicines until in-person doctor examination.")
            meds.append("• Hospital team will establish immediate IV access & therapy.")
        else:
            meds.append("• Dolo 650 (Paracetamol 650mg): 1 tab after meals (with water) for fever/pain (max 3/day).")
            meds.append("• Electral ORS: Sip for hydration; consult doctor before taking antibiotics.")

    return meds[:2]


async def classify_medical_image_type(image_base64: Optional[str], caption: Optional[str] = None) -> str:
    """
    Uses OpenRouter Vision to accurately classify incoming WhatsApp media into:
    - 'prescription' (Handwritten/printed doctor prescription, medicine receipt, lab report)
    - 'chest_xray' (Chest radiograph / lung scan)
    - 'bone_fracture' (Limb, wrist, hand, leg, foot, joint orthopedic X-ray / CT)
    Falls back gracefully to caption heuristics if OpenRouter vision is unavailable.
    """
    caption_lower = (caption or "").lower().strip()
    if any(k in caption_lower for k in ["rx", "prescription", "medicine", "doctor note", "pills", "syrup", "slip", "parcha", "dawa", "report"]):
        return "prescription"
    if any(k in caption_lower for k in ["chest", "lung", "pneumonia", "covid", "cough", "breath", "thorax"]):
        return "chest_xray"
    if any(k in caption_lower for k in ["fracture", "bone", "wrist", "hand", "leg", "arm", "knee", "foot", "ankle", "joint", "ortho"]):
        return "bone_fracture"

    if not image_base64 or not settings.OPENROUTER_API_KEY:
        return "bone_fracture"

    try:
        data_url = image_base64 if image_base64.startswith("data:") else f"data:image/jpeg;base64,{image_base64}"
        headers = {
            "Authorization": f"Bearer {settings.OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
            "HTTP-Referer": settings.OPENROUTER_REFERER or "https://synapseos.health",
            "X-Title": settings.OPENROUTER_APP_TITLE or "SynapseOS Image Classifier"
        }
        vision_model = settings.OPENROUTER_PRIMARY_MODEL or "google/gemini-2.0-flash-001"
        payload = {
            "model": vision_model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "Analyze this image and classify it into exactly one of three categories:\n"
                                "1. 'prescription' if it is a doctor's prescription, medical slip, medicine bill, or lab test report.\n"
                                "2. 'chest_xray' if it is a chest radiograph or lung X-ray.\n"
                                "3. 'bone_fracture' if it is an orthopedic bone X-ray, fracture scan, or limb scan.\n\n"
                                "Return JSON only in this exact format: {\"category\": \"prescription\" | \"chest_xray\" | \"bone_fracture\"}"
                            )
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": data_url}
                        }
                    ]
                }
            ],
            "temperature": 0.0,
            "max_tokens": 80
        }
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload)
            if resp.status_code == 200:
                raw_content = resp.json()["choices"][0]["message"]["content"]
                clean_json = raw_content.replace("```json", "").replace("```", "").strip()
                parsed = json.loads(clean_json)
                cat = parsed.get("category", "").lower().strip()
                if cat in ("prescription", "lab_report", "report"):
                    return "prescription"
                elif cat in ("chest_xray", "chest", "lung"):
                    return "chest_xray"
                elif cat in ("bone_fracture", "bone", "fracture", "orthopedic"):
                    return "bone_fracture"
    except Exception as exc:
        logger.warning(f"[OpenRouter Vision Classifier] Fallback triggered: {exc}")

    return "bone_fracture"


def format_compact_generic_qa_card(text: str) -> str:
    """
    Formats generic health inquiries into a clean, concise, 3-section WhatsApp advice card.
    Avoids unsolicited clinical consensus or fake triage headings.
    """
    cleaned = strip_markdown_to_plain_text(text)
    paragraphs = [p.strip() for p in cleaned.split("\n\n") if p.strip()]
    
    # Extract main answer
    main_ans = "\n\n".join(paragraphs[:2]) if paragraphs else cleaned
    
    lines = [
        "🟢 SANJEEVNI HEALTH ADVISORY",
        "━━━━━━━━━━━━━━━━━━━━",
        f"💡 Key Information:\n{main_ans}",
        "",
        "⚠️ Medical Note: Educational guidance only. Consult a doctor for personal evaluation.",
        "━━━━━━━━━━━━━━━━━━━━",
        "🌿 Powered by Sanjeevni-OS"
    ]
    return "\n".join(lines).strip()


def format_compact_whatsapp_card(text: str) -> str:
    """
    Transforms verbose multi-agent diagnostic audit into a punchy, normal plain-text WhatsApp card (no markdown).
    Extracts: Triage Badge, Suspected Diagnosis, Council Consensus, Top 2 Actions, Medications & Relief (India), Red Flags, and Quick Shortcuts.
    """
    if not text or len(text.strip()) < 180:
        return format_response_for_whatsapp(text, compact=False)

    # 1. Determine Triage Status Badge (Normal Text, No Markdown)
    text_lower = text.lower()
    is_emergency = False
    badge = "🟡 SYNAPSE-OS CLINICAL ASSESSMENT"
    if "🔴" in text or "patient status: emergency" in text_lower or "emergency care" in text_lower or "emergency triage" in text_lower:
        badge = "🔴 SYNAPSE-OS EMERGENCY TRIAGE — CRITICAL"
        is_emergency = True
    elif "🟢" in text or "patient status: home care" in text_lower or "home self-care" in text_lower or ("home care" in text_lower and "doctor consult" not in text_lower):
        badge = "🟢 SYNAPSE-OS HOME CARE & MONITORING"
    elif "🟡" in text or "doctor consult" in text_lower:
        badge = "🟡 SYNAPSE-OS DOCTOR CONSULTATION RECOMMENDED"
    elif "emergency" in text_lower and any(k in text_lower for k in ["emergency room", "emergency department", "call 112", "call 108"]):
        badge = "🔴 SYNAPSE-OS EMERGENCY TRIAGE — CRITICAL"
        is_emergency = True

    lines = [f"{badge}\n━━━━━━━━━━━━━━━━━━━━"]

    # 2. Suspected Diagnosis
    condition = _extract_diagnosis_condition(text)
    condition = _truncate_clean(condition, 110)
    lines.append(f"🩺 Suspected Diagnosis: {condition}")

    # Extract Council confidence
    conf_match = (
        re.search(r'([0-9]{1,3}%)\s*(?:Consensus|Confidence|Agreement)', text, re.IGNORECASE) or
        re.search(r'(?:Confidence|Consensus|Agreement)[:\-]?\*?\s*([0-9]{1,3}%)', text, re.IGNORECASE) or
        re.search(r'Confidence:\*?\s*(\w+)', text, re.IGNORECASE)
    )
    if conf_match:
        lines.append(f"📊 Council Consensus: {conf_match.group(1)} Agreement")

    # 3. Immediate Actions (Max 2 concise steps)
    actions = []
    bullet_items = re.findall(r'\*\s+\*?([A-Za-z\s]+)[:\-]\*?\s*([^\n]+)', text)
    for title, desc in bullet_items:
        clean_title = title.strip()
        if any(k in clean_title.lower() for k in ["seek", "emergency", "care", "rest", "hydration", "neck", "consult"]):
            clean_d = _clean_card_text(desc)
            clean_d = _truncate_clean(clean_d, 95)
            actions.append(f"{clean_title}: {clean_d}")
            if len(actions) >= 2:
                break

    if not actions:
        action_match = re.search(r'(?:Immediate Action Plan|Action Plan)[^\n]*\n+([^\n]+)', text, re.IGNORECASE)
        if action_match:
            act_line = _clean_card_text(action_match.group(1))
            if len(act_line) > 8:
                actions.append(_truncate_clean(act_line, 95))

    if actions:
        lines.append("\n📋 Immediate Actions:")
        for i, act in enumerate(actions[:2], 1):
            lines.append(f"{i}. {act}")
    else:
        if is_emergency:
            lines.append("\n📋 Immediate Actions:\n1. Call 112 / 108 or proceed to nearest hospital emergency immediately.\n2. Do not drive yourself; have a companion accompany you.")
        else:
            lines.append("\n📋 Immediate Actions:\n1. Schedule consultation with a General Physician within 24–48 hours.\n2. Maintain complete rest and active fluid hydration.")

    # 4. Medications & Relief Available in India (with how to take)
    meds = _extract_medications_guidance(text, is_emergency)
    lines.append("\n💊 Medications & Relief (India):")
    for m_line in meds:
        lines.append(m_line)

    # 5. Red Flags
    red_flags = []
    for title, desc in bullet_items:
        clean_title = title.strip()
        if any(k in clean_title.lower() for k in ["respiratory", "breath", "chest", "oxygen", "spo2", "neurological", "neck", "fever"]):
            clean_d = _clean_card_text(desc)
            clean_d = _truncate_clean(clean_d, 80)
            red_flags.append(f"{clean_title}: {clean_d}")
            if len(red_flags) >= 2:
                break

    if red_flags:
        lines.append("\n🚨 Seek Emergency Care / Call 108 If:")
        for rf in red_flags:
            lines.append(f"• {rf}")
    else:
        lines.append("\n🚨 Seek Emergency Care If:\n• Shortness of breath, SpO2 < 92%, neck stiffness, or fever > 103°F")

    # 6. Interactive Quick Action Buttons / Shortcuts (Normal Text, No Markdown)
    lines.append("\n━━━━━━━━━━━━━━━━━━━━")
    lines.append("👉 Quick Shortcuts:")
    lines.append("• Reply 5 to find PM-JAY doctors & book slot")
    lines.append("• Reply sos for instant emergency ambulance (108)")
    lines.append("• Reply full for the complete clinical report")
    lines.append("\n🌿 Powered by Synapse-OS Multi-Agent Swarm")

    card_str = "\n".join(lines)
    return strip_markdown_to_plain_text(card_str)


def format_response_for_whatsapp(text: str, compact: bool = True) -> str:
    """Formats clinical response as clean, normal plain text without markdown for WhatsApp."""
    if not text:
        return "Thank you for consulting Synapse-OS. Please monitor your health and consult a physician if needed."

    text_lower = text.lower()
    # Check if this is a general informational Q&A without active symptoms
    is_general_inquiry = any(k in text_lower for k in [
        "informational inquiry", "general inquiry", "inquiry regarding",
        "what is ", "what is calpol", "benefits of ", "how does ", "why is "
    ]) and not any(k in text_lower for k in ["patient status: emergency", "fever", "severe pain", "vomiting", "diarrhea"])

    if is_general_inquiry:
        plain = strip_markdown_to_plain_text(text)
        footer = "🌿 Powered by Synapse-OS Multi-Agent Swarm"
        if not plain.endswith(footer):
            plain += f"\n\n{footer}"
        return plain

    # Only formal multi-agent diagnostic audits get transformed into compact triage cards
    is_triage_audit = (
        any(k in text_lower for k in [
            "patient status:", "triage level:", "triage & outbreak", 
            "clinical assessment & care guidance", "symptom triage", "suspected condition"
        ]) or ("🔴" in text and "emergency" in text_lower)
    )

    if compact and is_triage_audit:
        return format_compact_whatsapp_card(text)

    plain = strip_markdown_to_plain_text(text)
    footer = "🌿 Powered by Synapse-OS Multi-Agent Swarm"
    if not plain.endswith(footer):
        plain += f"\n\n{footer}"
    return plain


# =========================================================
# 2. Inbound Webhook Processor & Deduplication Cache
# =========================================================

class WhatsAppMessageDeduplicator:
    """
    Thread/Asyncio-safe in-memory cache to prevent processing duplicate messages 
    caused by Meta WhatsApp webhook retries, network glitches, or rapid multi-delivery.
    """
    def __init__(self, ttl_seconds: float = 600.0, max_entries: int = 5000):
        self.ttl_seconds = ttl_seconds
        self.max_entries = max_entries
        self._processed: Dict[str, float] = {}
        self._in_flight: Set[str] = set()

    def _cleanup(self, now: float):
        expired = [k for k, ts in self._processed.items() if now - ts > self.ttl_seconds]
        for k in expired:
            self._processed.pop(k, None)
        if len(self._processed) > self.max_entries:
            sorted_keys = sorted(self._processed.keys(), key=lambda k: self._processed[k])
            for k in sorted_keys[:len(self._processed) - self.max_entries]:
                self._processed.pop(k, None)

    def is_duplicate(self, msg_id: Optional[str], sender_phone: Optional[str] = None, text: Optional[str] = None) -> bool:
        now = time.time()
        self._cleanup(now)

        key = msg_id
        if not key and sender_phone:
            clean_txt = (text or "").strip().lower()
            key = f"fingerprint:{sender_phone}:{clean_txt}"

        if not key:
            return False

        if key in self._in_flight or key in self._processed:
            return True

        return False

    def mark_in_flight(self, key: Optional[str]):
        if key:
            self._in_flight.add(key)

    def mark_processed(self, key: Optional[str]):
        if key:
            self._in_flight.discard(key)
            self._processed[key] = time.time()

    def remove_in_flight(self, key: Optional[str]):
        if key:
            self._in_flight.discard(key)

    def clear(self):
        self._processed.clear()
        self._in_flight.clear()


message_deduplicator = WhatsAppMessageDeduplicator()


def clear_deduplication_cache():
    """Clears deduplication cache (primarily for test isolation)."""
    message_deduplicator.clear()


async def process_whatsapp_inbound_webhook(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Unified Inbound WhatsApp Webhook Processor for Meta WhatsApp Cloud API.
    Handles Language Selection Onboarding, Greetings, Multi-Turn FSM, and Specialist Agent Routing.
    Protects against duplicate webhook dispatches and retries.
    """
    logger.info(f"[Meta Webhook Inbound] Received payload keys: {list(payload.keys())}")

    sender_phone = "unknown"
    msg_type = "text"
    message_text = ""
    image_base64 = None
    media_id = None
    caption = ""
    msg_id = None

    # 1. Parse Official Meta Graph API Format
    if "entry" in payload:
        try:
            entry = payload["entry"][0]
            change = entry.get("changes", [{}])[0]
            value = change.get("value", {})

            # Check if this is a message delivery status update (sent, delivered, read)
            if "statuses" in value and "messages" not in value:
                status_info = value["statuses"][0]
                logger.info(f"[Meta Delivery Status] Msg {status_info.get('id')}: {status_info.get('status')}")
                return {"status": "status_acknowledged", "delivery_status": status_info.get("status")}

            messages = value.get("messages", [])
            if not messages:
                return {"status": "ignored", "reason": "no_messages_in_payload"}

            msg = messages[0]
            msg_id = msg.get("id")
            sender_phone = msg.get("from", "unknown")
            msg_type = msg.get("type", "text")

            if msg_type == "text":
                message_text = msg.get("text", {}).get("body", "").strip()
            elif msg_type == "interactive":
                interactive = msg.get("interactive", {})
                if interactive.get("type") == "button_reply":
                    message_text = interactive.get("button_reply", {}).get("title", "")
                elif interactive.get("type") == "list_reply":
                    message_text = interactive.get("list_reply", {}).get("title", "")
            elif msg_type == "image":
                image_info = msg.get("image", {})
                media_id = image_info.get("id")
                caption = image_info.get("caption", "").strip()
                message_text = caption or "Medical scan uploaded"
            elif msg_type == "location":
                loc = msg.get("location", {})
                message_text = f"location:{loc.get('latitude')},{loc.get('longitude')}"
        except Exception as e:
            logger.error(f"[Meta Payload Parsing Error] {e}")
            return {"status": "error", "detail": f"Failed parsing Meta payload: {str(e)}"}

    # 2. Parse Simulation / Fallback Format
    else:
        data = payload.get("data", payload)
        msg_id = data.get("id") or payload.get("id") or payload.get("message_id")
        sender_phone = (
            data.get("from") or
            data.get("sender_phone") or
            payload.get("sender_phone") or
            "919876543210"
        )
        msg_type = data.get("type", data.get("message_type", "text"))
        message_text = (
            data.get("text") or
            data.get("message") or
            data.get("body") or
            payload.get("message") or
            ""
        ).strip()
        caption = data.get("caption", "")
        image_base64 = data.get("image_base64") or (data.get("body") if isinstance(data.get("body"), str) and data.get("body").startswith("data:image") else None)

    sender_phone = str(sender_phone).replace("@c.us", "").replace("+", "").strip()

    # Deduplication check: Drop duplicate/retried messages
    dedup_key = msg_id or (f"fingerprint:{sender_phone}:{message_text}" if message_text else None)
    if dedup_key and message_deduplicator.is_duplicate(msg_id, sender_phone, message_text):
        logger.warning(f"[Meta Webhook Deduplication] Ignored duplicate/retried message (key={dedup_key}) from {sender_phone}")
        return {
            "status": "duplicate_ignored",
            "id": msg_id,
            "sender": sender_phone,
            "reason": "Message already processed or currently in-flight."
        }

    if dedup_key:
        message_deduplicator.mark_in_flight(dedup_key)

    try:
        return await _dispatch_inbound_whatsapp_message(
            sender_phone=sender_phone,
            message_text=message_text,
            msg_type=msg_type,
            image_base64=image_base64,
            media_id=media_id,
            caption=caption
        )
    finally:
        if dedup_key:
            message_deduplicator.mark_processed(dedup_key)


async def _dispatch_inbound_whatsapp_message(
    sender_phone: str,
    message_text: str,
    msg_type: str = "text",
    image_base64: Optional[str] = None,
    media_id: Optional[str] = None,
    caption: str = ""
) -> Dict[str, Any]:
    session = session_manager.get_session(sender_phone)
    user_lang = session["context"].get("lang") or detect_language_script(message_text)

    # 3. Handle Medical Image Upload (FractureNet YOLOv8 / MONAI / TrOCR)
    if msg_type == "image" or image_base64 or media_id:
        if media_id and not image_base64:
            media_bytes = await download_meta_media(media_id)
            if media_bytes:
                image_base64 = f"data:image/jpeg;base64,{base64.b64encode(media_bytes).decode('utf-8')}"

        img_type = await classify_medical_image_type(image_base64, caption=caption or message_text)

        if img_type == "prescription":
            try:
                from backend.app.services.prescription_ocr_service import (
                    validate_image_bytes,
                    normalize_and_resize_image,
                    run_prescription_ocr,
                    interpret_prescription,
                    format_prescription_for_whatsapp
                )
                if image_base64:
                    clean_b64 = image_base64.split(",")[-1] if "," in image_base64 else image_base64
                    raw_bytes = base64.b64decode(clean_b64)
                    valid, err_code, err_msg, pil_img = validate_image_bytes(raw_bytes)
                    if valid and pil_img:
                        data_url = normalize_and_resize_image(pil_img)
                        ok, err_obj, ocr_data = await run_prescription_ocr(data_url)
                        if ok and ocr_data:
                            # Run downstream clinical interpretation & triage
                            interp = await interpret_prescription(ocr_data=ocr_data, lang=user_lang or "en")
                            reply_text = format_prescription_for_whatsapp(ocr_data=ocr_data, interpretation=interp)
                            dispatch_res = await send_whatsapp_message(to_phone=sender_phone, text=reply_text)
                            return {
                                "status": "processed",
                                "type": "prescription_ocr_interpretation",
                                "sender": sender_phone,
                                "dispatch": dispatch_res,
                                "reply_dispatched": dispatch_res,
                                "scan_summary": interp.get("likely_condition", "Prescription Interpreted")
                            }
            except Exception as e:
                logger.error(f"[WhatsApp Prescription OCR/Triage Error] {e}")

        scan_result = analyze_medical_image(
            image_type=img_type,
            filename="whatsapp_meta_scan.jpg",
            image_base64=image_base64
        )

        boxes = scan_result.get("visual_bounding_boxes", [])
        box_str = f"• Detections: {len(boxes)} anomaly zone(s) localized." if boxes else "• Findings: No acute displaced fracture."

        reply_lines = [
            "📷 SANJEEVNI X-RAY SCAN ANALYSIS",
            "━━━━━━━━━━━━━━━━━━━━",
            f"• Modality: {img_type.replace('_', ' ').title()}",
            f"• Status: {scan_result.get('urgency_badge', 'Standard Review')}",
            f"• AI Impression: {scan_result.get('ai_diagnosis_summary', 'Analysis Completed')}",
            box_str,
            "",
            "💡 Recommended Next Step:",
            f"{scan_result.get('recommended_clinical_action', 'Consult a registered orthopedic specialist.')}",
            "",
            "⚠️ Preliminary AI analysis. Must be confirmed by a radiologist.",
            "━━━━━━━━━━━━━━━━━━━━",
            "🌿 Powered by Sanjeevni-OS"
        ]

        reply_text = "\n".join(reply_lines)
        dispatch_res = await send_whatsapp_message(to_phone=sender_phone, text=reply_text)

        # Dispatch the actual annotated YOLOv8 JPG and Grad-CAM JPG straight into the chat bubble
        result_img_url = scan_result.get("remote_result_image")
        gradcam_img_url = scan_result.get("remote_gradcam_image")
        image_dispatches = []

        if result_img_url:
            img_res = await send_whatsapp_image(
                to_phone=sender_phone,
                image_url=result_img_url,
                caption="🎯 FractureNet YOLOv8: Anomaly Localization Overlay"
            )
            image_dispatches.append({"type": "yolo_overlay", "result": img_res})

        if gradcam_img_url:
            cam_res = await send_whatsapp_image(
                to_phone=sender_phone,
                image_url=gradcam_img_url,
                caption="🔥 Grad-CAM: Neural Attention Heatmap"
            )
            image_dispatches.append({"type": "gradcam_heatmap", "result": cam_res})

        return {
            "status": "processed",
            "type": "medical_image",
            "sender": sender_phone,
            "dispatch": dispatch_res,
            "reply_dispatched": dispatch_res,
            "images_dispatched": image_dispatches,
            "scan_summary": scan_result.get("ai_diagnosis_summary")
        }

    # 4. Handle Empty Text
    if not message_text:
        return {"status": "ignored", "reason": "empty_text"}

    text_lower = message_text.lower().strip()

    # 5. Language Change Trigger ("lang", "language", "bhasha", "भाषा")
    if text_lower in ("lang", "language", "bhasha", "भाषा", "change language", "select language"):
        session_manager.set_flow(sender_phone, "LANG_SELECT")
        dispatch_res = await send_whatsapp_message(to_phone=sender_phone, text=LANGUAGE_SELECTION_MENU)
        return {
            "status": "processed",
            "type": "language_menu_dispatched",
            "sender": sender_phone,
            "dispatch": dispatch_res,
            "reply_dispatched": dispatch_res
        }

    # 6. Handle Active Language Selection Flow (LANG_SELECT)
    if session.get("active_flow") == "LANG_SELECT":
        selected_code = parse_language_selection(text_lower)
        if selected_code:
            session["context"]["lang"] = selected_code
            session_manager.reset_flow(sender_phone)

            lang_names = {
                "en": "English", "hi": "हिन्दी (Hindi)", "bn": "বাংলা (Bengali)",
                "ta": "தமிழ் (Tamil)", "te": "తెలుగు (Telugu)", "mr": "मराठी (Marathi)",
                "gu": "ગુજરાતી (Gujarati)", "kn": "ಕನ್ನಡ (Kannada)", "ml": "മലയാളം (Malayalam)",
                "pa": "ਪੰਜਾਬੀ (Punjabi)", "or": "ଓଡ଼ିଆ (Odia)"
            }
            confirm_msg = f"🌐 Language Selected: {lang_names.get(selected_code, 'English')}\n━━━━━━━━━━━━━━━━━━━━\n\n"
            menu_text = confirm_msg + LOCALIZED_MENUS.get(selected_code, LOCALIZED_MENUS["en"])

            dispatch_res = await send_whatsapp_message(to_phone=sender_phone, text=menu_text)
            return {
                "status": "processed",
                "type": "language_selected",
                "language": selected_code,
                "sender": sender_phone,
                "dispatch": dispatch_res,
                "reply_dispatched": dispatch_res
            }
        elif any(text_lower.startswith(prefix) for prefix in ["1 ", "2 ", "3 ", "4 ", "5 ", "6", "7 ", "8 ", "9 ", "sos", "emergency"]):
            session_manager.reset_flow(sender_phone)
        elif len(text_lower) > 12:
            session_manager.reset_flow(sender_phone)
        else:
            retry_msg = "⚠️ Invalid selection. Please reply with a number from 1 to 11:\n\n" + LANGUAGE_SELECTION_MENU
            dispatch_res = await send_whatsapp_message(to_phone=sender_phone, text=retry_msg)
            return {"status": "processed", "type": "language_retry"}

    # 7. Greeting / Main Menu Trigger
    if text_lower in ("hi", "hello", "hey", "menu", "help", "start", "guide", "synapse", "synapseos", "sanjeevni", "options"):
        if not session["context"].get("lang"):
            session_manager.set_flow(sender_phone, "LANG_SELECT")
            dispatch_res = await send_whatsapp_message(to_phone=sender_phone, text=LANGUAGE_SELECTION_MENU)
            return {
                "status": "processed",
                "type": "menu_dispatched",
                "sender": sender_phone,
                "dispatch": dispatch_res,
                "reply_dispatched": dispatch_res
            }
        else:
            chosen_lang = session["context"].get("lang", "en")
            active_menu = LOCALIZED_MENUS.get(chosen_lang, LOCALIZED_MENUS["en"])
            dispatch_res = await send_whatsapp_message(to_phone=sender_phone, text=active_menu)
            return {
                "status": "processed",
                "type": "menu_dispatched",
                "sender": sender_phone,
                "dispatch": dispatch_res,
                "reply_dispatched": dispatch_res
            }

    # 7b. Full Diagnostic Report Request ("full", "report", "details")
    if text_lower in ("full", "report", "details", "full report", "detailed report", "audit"):
        last_report = session["context"].get("last_full_report")
        if last_report:
            clean_full = strip_markdown_to_plain_text(last_report)
            footer = "🌿 Powered by Sanjeevni-OS Multi-Agent Swarm"
            if not clean_full.endswith(footer):
                clean_full += f"\n\n{footer}"
            dispatch_res = await send_whatsapp_message(to_phone=sender_phone, text=clean_full)
            return {
                "status": "processed",
                "type": "full_report_dispatched",
                "sender": sender_phone,
                "dispatch": dispatch_res,
                "reply_dispatched": dispatch_res
            }
        else:
            no_rep = (
                "ℹ️ No previous diagnostic report found in this session.\n\n"
                "Please describe your symptoms or text '1 <symptoms>' (e.g. '1 High fever, headache and dry cough') to start an AI clinical triage!"
            )
            dispatch_res = await send_whatsapp_message(to_phone=sender_phone, text=no_rep)
            return {"status": "processed", "type": "no_prior_report", "dispatch": dispatch_res, "reply_dispatched": dispatch_res}

    # 8. Emergency SOS Trigger
    if text_lower in ("sos", "emergency", "112", "108", "save me", "help me"):
        sos_res = (
            "🔴 SANJEEVNI EMERGENCY DISPATCH\n"
            "━━━━━━━━━━━━━━━━━━━━\n"
            "🚨 Immediate Emergency Call:\n"
            "• Ambulance: 108 (Direct Emergency)\n"
            "• National Emergency: 112\n"
            "• Tele-MANAS Mental Crisis: 14416 (24x7)\n"
            "• Poison Control: 1800-116-117\n\n"
            "🏥 First-Aid Protocol:\n"
            "1. Place patient in recovery position (on their side).\n"
            "2. Keep airway clear and loosen tight clothes.\n"
            "3. DO NOT give oral food/water if unconscious or dizzy.\n\n"
            "📍 Share WhatsApp Location to find nearest Emergency Room."
        )
        dispatch_res = await send_whatsapp_message(to_phone=sender_phone, text=sos_res)
        return {
            "status": "processed",
            "type": "emergency_sos",
            "sender": sender_phone,
            "dispatch": dispatch_res,
            "reply_dispatched": dispatch_res
        }

    # 9. Handle Active Multi-Turn FSM (Health Literacy Quiz Flow)
    if session.get("active_flow") == "QUIZ_FLOW":
        questions = session["context"].get("questions", [])
        answers = session["context"].get("answers", {})
        step = session.get("step", 0)

        choice_map = {"a": 0, "b": 1, "c": 2, "d": 3, "1": 0, "2": 1, "3": 2, "4": 3}
        ans_idx = choice_map.get(text_lower[:1], 0)
        answers[f"q{step + 1}"] = ans_idx

        session_manager.advance_step(sender_phone, {"answers": answers})
        next_step = step + 1

        if next_step < len(questions):
            q = questions[next_step]
            opts = "\n".join([f"{chr(65+i)}. {opt}" for i, opt in enumerate(q["options"])])
            msg = f"📝 Health Quiz — Question {next_step + 1}/{len(questions)}:\n\n{q['question']}\n\n{opts}\n\nReply with A, B, C, or D."
            await send_whatsapp_message(to_phone=sender_phone, text=msg)
            return {"status": "processed", "type": "quiz_step", "step": next_step}
        else:
            eval_res = evaluate_quiz_answers(answers)
            session_manager.reset_flow(sender_phone)
            quiz_summary = (
                "🏆 HEALTH LITERACY QUIZ COMPLETED!\n"
                "━━━━━━━━━━━━━━━━━━━━\n"
                f"• Score: {eval_res.get('score', '3/3')} ({eval_res.get('percentage', 100)}%)\n"
                f"• Grade: {eval_res.get('grade', 'Health Champion')}\n\n"
                "💡 Key Takeaway: Always combine ORS + Zinc for diarrhea and empty standing water to prevent dengue!"
            )
            await send_whatsapp_message(to_phone=sender_phone, text=quiz_summary)
            return {"status": "processed", "type": "quiz_completed", "score": eval_res.get("score")}

    # 10. Command Option 2: Drug Safety & RxNav
    if text_lower.startswith("2 ") or text_lower == "2":
        query = message_text[2:].strip() if text_lower.startswith("2 ") else ""
        if not query:
            reply_text = (
                "💊 DRUG SAFETY & RXNAV CHECKER\n"
                "━━━━━━━━━━━━━━━━━━━━\n"
                "Please reply with medicine names (e.g. '2 Aspirin with Ibuprofen' or '2 Paracetamol and Alcohol')."
            )
        else:
            drug_res = await evaluate_drug_safety(query)
            status = drug_res.get("overall_status", "Evaluated")
            badge_icon = "🔴" if "risk" in status.lower() or "danger" in status.lower() or "severe" in status.lower() else "🟢"
            reply_parts = [
                "⚠️ SANJEEVNI DRUG SAFETY CHECK",
                "━━━━━━━━━━━━━━━━━━━━",
                f"💊 Query: {query}",
                f"{badge_icon} Status: {status}"
            ]
            interactions = drug_res.get("interactions", [])
            if interactions:
                reply_parts.append("\n🔍 Interactions Detected:")
                for item in interactions[:2]:
                    reply_parts.append(f"• {item.get('severity', 'Risk')} Risk: {item.get('effect')}\n  Action: {item.get('recommended_action')}")
            else:
                reply_parts.append("✅ No severe drug-drug interactions detected.")

            if drug_res.get("safe_alternatives"):
                reply_parts.append(f"\n💡 Safe Alternative: {', '.join(drug_res['safe_alternatives'][:2])}")

            reply_parts.append("\n━━━━━━━━━━━━━━━━━━━━\n🌿 Powered by Sanjeevni-OS")
            reply_text = "\n".join(reply_parts)

        dispatch_res = await send_whatsapp_message(to_phone=sender_phone, text=reply_text)
        return {
            "status": "processed",
            "type": "drug_check",
            "sender": sender_phone,
            "dispatch": dispatch_res,
            "reply_dispatched": dispatch_res
        }

    # 11. Command Option 5: Doctor Lookup
    if text_lower.startswith("5 ") or text_lower == "5":
        specialty = message_text[2:].strip() if text_lower.startswith("5 ") else "General Physician"
        doctors_res = find_doctors_by_specialty(specialty or "General Physician")
        reply_parts = [
            f"🏥 EMPANELLED PM-JAY DOCTORS ({specialty.title()})",
            "━━━━━━━━━━━━━━━━━━━━"
        ]
        doctors_list = doctors_res if isinstance(doctors_res, list) else doctors_res.get("available_doctors", [])
        if doctors_list:
            for doc in doctors_list[:3]:
                slots = doc.get("available_slots", [])
                next_slot = slots[0] if slots else "Available Today"
                hospital = doc.get("hospital", "AIIMS / Empanelled Center")
                reply_parts.append(
                    f"👨‍⚕️ {doc.get('name')} — {doc.get('specialty')}\n"
                    f"   🏥 {hospital}\n"
                    f"   💳 Fee: {doc.get('fee', '₹0 (PM-JAY)')} | Next: {next_slot}"
                )
            reply_parts.append("\n👉 Reply with preferred slot to schedule.")
        else:
            reply_parts.append(f"No specific doctors found for '{specialty}'. Please consult a General Physician.")

        reply_parts.append("━━━━━━━━━━━━━━━━━━━━\n🌿 Powered by Sanjeevni-OS")
        reply_text = "\n".join(reply_parts)
        dispatch_res = await send_whatsapp_message(to_phone=sender_phone, text=reply_text)
        return {
            "status": "processed",
            "type": "doctor_lookup",
            "sender": sender_phone,
            "dispatch": dispatch_res,
            "reply_dispatched": dispatch_res
        }

    # 12. Command Option 6: ABHA Health Card & PM-JAY
    if text_lower == "6" or text_lower.startswith("6 "):
        abha_data = generate_abha_id(name="Sanjeevni User", year_of_birth=1998, state_code="MP")
        abha_num = abha_data.get("abha_number") or abha_data.get("abha_id")
        schemes_data = check_ayushman_bharat_schemes()
        pmjay_cov = schemes_data.get("pmjay", {}).get("coverage", "₹5 Lakh / family / year free care")
        jan_benefit = schemes_data.get("jan_aushadhi", {}).get("benefit", "Generic medicines at 50-90% savings")
        reply_parts = [
            "🪪 AYUSHMAN BHARAT DIGITAL MISSION (ABDM)",
            "━━━━━━━━━━━━━━━━━━━━",
            f"• ABHA ID: {abha_num}",
            f"• ABHA Address: {abha_data.get('abha_address')}",
            "• PM-JAY Wallet: ₹5,00,000 / Family / Year",
            f"• Card Status: {abha_data.get('status', 'ACTIVE')}\n",
            "📜 Active National Health Schemes:",
            f"• PM-JAY: {pmjay_cov}",
            f"• Jan Aushadhi (PMBJP): {jan_benefit}"
        ]

        reply_parts.append("\n━━━━━━━━━━━━━━━━━━━━\n🌿 Powered by Sanjeevni-OS")
        reply_text = "\n".join(reply_parts)
        dispatch_res = await send_whatsapp_message(to_phone=sender_phone, text=reply_text)
        return {
            "status": "processed",
            "type": "abha_info",
            "sender": sender_phone,
            "dispatch": dispatch_res,
            "reply_dispatched": dispatch_res
        }

    # 13. Command Option 7: UIP Vaccination Schedule
    if text_lower == "7" or text_lower.startswith("7 "):
        param = message_text[2:].strip().lower() if text_lower.startswith("7 ") else ""
        if "preg" in param:
            v_res = calculate_vaccination_schedule(category="pregnant")
            reply_parts = [
                "💉 MATERNAL VACCINATION SCHEDULE (UIP)",
                "━━━━━━━━━━━━━━━━━━━━",
                "• Protocol: Universal Maternal Immunization",
                "• Vaccines Due:"
            ]
            for v in v_res.get("recommended_vaccines", [])[:2]:
                reply_parts.append(f"  • {v['name']}: {v['protects_against']}")
            reply_parts.append(f"\n💡 Guidance: {v_res.get('guideline')}")
        else:
            weeks = 6
            if "birth" in param or "0" in param:
                weeks = 0
            elif "10" in param:
                weeks = 10
            elif "14" in param:
                weeks = 14
            elif "9" in param or "month" in param:
                weeks = 40

            v_res = calculate_vaccination_schedule(age_in_weeks=weeks)
            reply_parts = [
                "💉 UNIVERSAL IMMUNIZATION PROGRAMME (UIP)",
                "━━━━━━━━━━━━━━━━━━━━",
                f"• UIP Progress: {v_res.get('uip_compliance_pct')}% Covered",
                f"• Next Due: {v_res.get('next_vaccine_due')}",
                f"• Milestone: {v_res.get('next_due_date')}\n"
            ]
            if v_res.get("current_due"):
                reply_parts.append("📋 Due at this age:")
                for m in v_res["current_due"][:1]:
                    for v in m["vaccines"][:3]:
                        reply_parts.append(f"  • {v['name']}: {v['protects_against']}")
            elif v_res.get("upcoming"):
                next_m = v_res["upcoming"][0]
                reply_parts.append(f"📋 Upcoming ({next_m['milestone_label']}):")
                for v in next_m["vaccines"][:3]:
                    reply_parts.append(f"  • {v['name']}: {v['protects_against']}")

            reply_parts.append("\n🏥 Free at all Anganwadis & Primary Health Centres.")

        reply_parts.append("━━━━━━━━━━━━━━━━━━━━\n🌿 Powered by Sanjeevni-OS")
        reply_text = "\n".join(reply_parts)
        dispatch_res = await send_whatsapp_message(to_phone=sender_phone, text=reply_text)
        return {
            "status": "processed",
            "type": "vaccination_schedule",
            "sender": sender_phone,
            "dispatch": dispatch_res,
            "reply_dispatched": dispatch_res
        }

    # 14. Command Option 8: District Outbreak Alerts
    if text_lower == "8" or text_lower.startswith("8 "):
        district_query = message_text[2:].strip() if text_lower.startswith("8 ") else "Delhi"
        outbreak_res = get_district_outbreak_risk(district_query or "Delhi")["data"]
        reply_parts = [
            f"🚨 DISTRICT OUTBREAK SURVEILLANCE ({outbreak_res['district']})",
            "━━━━━━━━━━━━━━━━━━━━",
            f"• Active Infection: {outbreak_res['primary_outbreak']}",
            f"• Risk Level: {outbreak_res['risk_badge']}",
            f"• Weekly Cases: {outbreak_res['weekly_cases']} ({outbreak_res['velocity_pct']})",
            f"\n📋 Advisory: {outbreak_res['preventive_advisory']}",
            f"📞 Helpdesk: {outbreak_res['helpline']}",
            "━━━━━━━━━━━━━━━━━━━━\n🌿 Powered by Sanjeevni-OS"
        ]
        reply_text = "\n".join(reply_parts)
        dispatch_res = await send_whatsapp_message(to_phone=sender_phone, text=reply_text)
        return {
            "status": "processed",
            "type": "outbreak_alert",
            "sender": sender_phone,
            "dispatch": dispatch_res,
            "reply_dispatched": dispatch_res
        }

    # 15. Command Option 9: Rural Preventive Health & Quiz Trigger
    if text_lower == "9" or text_lower.startswith("9 "):
        if "quiz" in text_lower:
            quiz_data = generate_community_health_quiz(count=3)
            questions = quiz_data.get("questions", [])
            if questions:
                session_manager.set_flow(sender_phone, "QUIZ_FLOW", {"questions": questions, "answers": {}})
                q1 = questions[0]
                opts = "\n".join([f"{chr(65+i)}. {opt}" for i, opt in enumerate(q1["options"])])
                msg = (
                    "📝 COMMUNITY HEALTH AWARENESS QUIZ (1/3)\n"
                    "━━━━━━━━━━━━━━━━━━━━\n"
                    f"{q1['question']}\n\n"
                    f"{opts}\n\n"
                    "Reply with A, B, C, or D to submit your answer!"
                )
                dispatch_res = await send_whatsapp_message(to_phone=sender_phone, text=msg)
                return {
                    "status": "processed",
                    "type": "quiz_started",
                    "sender": sender_phone,
                    "dispatch": dispatch_res,
                    "reply_dispatched": dispatch_res
                }

        reply_parts = [
            "🌿 RURAL PREVENTIVE HEALTHCARE GUIDES",
            "━━━━━━━━━━━━━━━━━━━━",
            "1. 💧 ORS & Diarrhea: Mix 1 packet in 1L clean water + Zinc 20mg for 14 days.",
            "2. 🤱 Poshan Nutrition: Daily IFA iron tablets + 6 months exclusive breastfeeding.",
            "3. 🦟 Dengue Control: Empty water coolers every Sunday; sleep under mosquito nets.",
            "4. 🧼 Safe Water: Boil water for 2 mins; 20-second handwashing before food.\n",
            "👉 Reply '9 quiz' to take the 3-question awareness quiz!",
            "━━━━━━━━━━━━━━━━━━━━\n🌿 Powered by Sanjeevni-OS"
        ]
        reply_text = "\n".join(reply_parts)
        dispatch_res = await send_whatsapp_message(to_phone=sender_phone, text=reply_text)
        return {
            "status": "processed",
            "type": "preventive_education",
            "sender": sender_phone,
            "dispatch": dispatch_res,
            "reply_dispatched": dispatch_res
        }

    # 16. Command Option 1 & 4, or General Natural Clinical Language
    if text_lower.startswith("1 "):
        clean_text = message_text[2:].strip()
    elif text_lower.startswith("4 "):
        clean_text = f"Mental health check: {message_text[2:].strip()}"
    else:
        clean_text = message_text

    # Execute Swarm Orchestrator with Clinical Fail-Safe Protection
    try:
        agent_result = await orchestrate_health_request(
            message=clean_text,
            channel="whatsapp",
            user_id=sender_phone
        )
        session["context"]["last_full_report"] = agent_result.final_response
        response_text = format_response_for_whatsapp(agent_result.final_response, compact=True)
        trace_steps = len(agent_result.trace)
        intent = agent_result.detected_intent
    except Exception as exc:
        logger.error(f"[WhatsApp Swarm Exception] Fallback triggered: {exc}", exc_info=True)
        response_text = (
            "⚠️ SANJEEVNI CLINICAL ADVISORY\n"
            "━━━━━━━━━━━━━━━━━━━━\n"
            "We encountered a temporary processing delay with our live clinical reasoning swarm. Your symptom query has been recorded.\n\n"
            "🚨 Immediate Emergency Guidance:\n"
            "If you or the patient have severe symptoms (intense chest pain, breathlessness, high fever, or confusion):\n"
            "• Call 112 (National Emergency) or 108 (Ambulance) immediately.\n"
            "• Mental Health: Call 14416 (Tele-MANAS 24x7).\n\n"
            "━━━━━━━━━━━━━━━━━━━━\n"
            "👉 Quick Shortcuts:\n"
            "• Reply menu for directory and schedules\n"
            "• Reply sos for instant ambulance dispatch\n\n"
            "🌿 Powered by Sanjeevni-OS"
        )
        trace_steps = 0
        intent = "fallback_emergency_advisory"

    dispatch_res = await send_whatsapp_message(to_phone=sender_phone, text=response_text)

    return {
        "status": "processed",
        "sender": sender_phone,
        "dispatch": dispatch_res,
        "reply_dispatched": dispatch_res,
        "agent_trace_steps": trace_steps,
        "intent": intent
    }


async def trigger_emergency_sos_whatsapp(
    emergency_contact: str,
    patient_name: str,
    location_coords: str,
    blood_group: str,
    critical_symptoms: str
) -> Dict[str, Any]:
    """Dispatches 1-click Emergency SOS alert to pre-set emergency contact via WhatsApp."""
    sos_message = (
        f"🚨 *SYNAPSE-OS — EMERGENCY SOS ALERT* 🚨\n\n"
        f"Patient *{patient_name}* has triggered an urgent emergency medical alert.\n\n"
        f"• *Reported Condition:* {critical_symptoms}\n"
        f"• *Blood Group:* {blood_group}\n"
        f"• *Live GPS Coordinates:* {location_coords}\n"
        f"• *Google Maps Navigation:* https://maps.google.com/?q={location_coords}\n\n"
        f"Automated alert dispatched. Call national emergency services 112 / 108 directly."
    )
    delivery_result = await send_whatsapp_message(to_phone=emergency_contact, text=sos_message)
    return {
        "emergency_alert_dispatched": delivery_result.get("delivered", False),
        "dispatch_mode": delivery_result.get("mode", "SANDBOX_SIMULATION"),
        "contact_notified": emergency_contact,
        "patient": patient_name,
        "delivery_details": delivery_result
    }
