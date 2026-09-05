"""
SynapseOS — services/i18n_service.py
Multilingual Translation & Audio Accessibility Service.
Comprehensive support for 11+ Indian Regional Languages & International standard.
"""

from typing import Dict, Any, List

SUPPORTED_LANGUAGES = [
    {"code": "en", "name": "English", "native": "English", "flag": "🌐"},
    {"code": "hi", "name": "Hindi", "native": "हिन्दी", "flag": "🇮🇳"},
    {"code": "bn", "name": "Bengali", "native": "বাংলা", "flag": "🇮🇳"},
    {"code": "ta", "name": "Tamil", "native": "தமிழ்", "flag": "🇮🇳"},
    {"code": "te", "name": "Telugu", "native": "తెలుగు", "flag": "🇮🇳"},
    {"code": "mr", "name": "Marathi", "native": "मराठी", "flag": "🇮🇳"},
    {"code": "gu", "name": "Gujarati", "native": "ગુજરાતી", "flag": "🇮🇳"},
    {"code": "kn", "name": "Kannada", "native": "ಕನ್ನಡ", "flag": "🇮🇳"},
    {"code": "ml", "name": "Malayalam", "native": "മലയാളം", "flag": "🇮🇳"},
    {"code": "pa", "name": "Punjabi", "native": "ਪੰਜਾਬੀ", "flag": "🇮🇳"},
    {"code": "or", "name": "Odia", "native": "ଓଡ଼ିଆ", "flag": "🇮🇳"},
    {"code": "es", "name": "Spanish", "native": "Español", "flag": "🇪🇸"}
]

# Multi-Language Pre-Translated Clinical Emergency, Triage, and Wearables Telemetry Messages
CLINICAL_I18N_STRINGS = {
    "en": {
        "emergency_alert": "⚠️ EMERGENCY ALERT: Call 112 or 108 immediately or visit the nearest emergency room.",
        "triage_home": "🟢 Home Care Advice: Maintain hydration, rest adequately, and monitor temperature & vitals.",
        "triage_doctor": "🟡 Clinical Consult: Schedule an in-person consultation or telemedicine slot within 24-48h.",
        "tele_manas": "National Tele-MANAS Mental Health Helpline: 14416 (Toll-Free 24/7)",
        "abha_badge": "Ayushman Bharat Digital Mission (ABDM) Verified",
        "wearable_sync_success": "Synced successfully with Apple Health & Google Health Connect.",
        "ecg_normal": "Sinus Rhythm: No signs of Atrial Fibrillation detected by Apple Watch ECG sensor."
    },
    "hi": {
        "emergency_alert": "⚠️ आपातकालीन सूचना: कृपया तुरंत 112 या 108 पर कॉल करें या नजदीकी अस्पताल जाएं।",
        "triage_home": "🟢 प्राथमिक सलाह: पर्याप्त आराम करें, तरल पदार्थ पिएं और लक्षणों पर नजर रखें।",
        "triage_doctor": "🟡 डॉक्टर परामर्श: 24-48 घंटे के भीतर डॉक्टर से परामर्श लें।",
        "tele_manas": "राष्ट्रीय मानसिक स्वास्थ्य हेल्पलाइन (Tele-MANAS): 14416 (निःशुल्क 24/7 सेवा)",
        "abha_badge": "आयुष्मान भारत डिजिटल मिशन (ABDM) सत्यापित",
        "wearable_sync_success": "एप्पल हेल्थ और गूगल हेल्थ कनेक्ट के साथ सफलतापूर्वक सिंक हुआ।",
        "ecg_normal": "साइनस रिदम: ईसीजी सेंसर द्वारा कोई एट्रियल फिब्रिलेशन नहीं पाया गया।"
    },
    "bn": {
        "emergency_alert": "⚠️ জরুরি সতর্কতা: অবিলম্বে ১১২ বা ১০৮ নম্বরে যোগাযোগ করুন অথবা নিকটস্থ হাসপাতালে যান।",
        "triage_home": "🟢 প্রাথমিক পরামর্শ: প্রচুর জল পান করুন, বিশ্রাম নিন এবং লক্ষণগুলি লক্ষ্য রাখুন।",
        "triage_doctor": "🟡 ডাক্তারি পরামর্শ: ২৪-৪৮ ঘণ্টার মধ্যে ডাক্তারের সঙ্গে যোগাযোগ করুন।",
        "tele_manas": "টেলি-মানস হেল্পলাইন: ১৪৪১৬ (২৪/৭ বিনামূল্যে সেবা)",
        "abha_badge": "আয়ুষ্মান ভারত ডিজিটাল মিশন দ্বারা যাচাইকৃত",
        "wearable_sync_success": "অ্যাপল হেলথ ও গুগল হেলথ কানেক্টের সাথে সফলভাবে সিঙ্ক হয়েছে।",
        "ecg_normal": "স্বাভাবিক সাইনাস রিদম: কোনো অলিন্দ ফিব্রিলেশন সনাক্ত হয়নি।"
    },
    "ta": {
        "emergency_alert": "⚠️ அவசர அறிவிப்பு: உடனடியாக 112 அல்லது 108 ஐ அழைக்கவும் அல்லது மருத்துவமனைக்குச் செல்லவும்.",
        "triage_home": "🟢 முதலுதவி ஆலோசனை: போதுமான ஓய்வு எடுத்துக் கொள்ளுங்கள், நிறைய தண்ணீர் குடிக்கவும்.",
        "triage_doctor": "🟡 மருத்துவ ஆலோசனை: 24-48 மணி நேரத்திற்குள் மருத்துவரை அணுகவும்.",
        "tele_manas": "டெலி-மானாஸ் உதவி எண்: 14416 (24/7 இலவச சேவை)",
        "abha_badge": "ஆயுஷ்மான் பாரத் சரிபார்க்கப்பட்டது",
        "wearable_sync_success": "ஆப்பிள் ஹெல்த் மற்றும் கூகுள் ஹெல்த் உடன் வெற்றிகரமாக ஒத்திசைக்கப்பட்டது.",
        "ecg_normal": "இயல்பான சைனஸ் ரிதம்: இதயத் துடிப்பு ஒழுங்காக உள்ளது."
    },
    "te": {
        "emergency_alert": "⚠️ అత్యవసర హెచ్చరిక: దయచేసి వెంటనే 112 లేదా 108 కి కాల్ చేయండి.",
        "triage_home": "🟢 ప్రాథమిక సలహా: తగినంత విశ్రాంతి తీసుకోండి మరియు ద్రవాలు త్రాగండి.",
        "triage_doctor": "🟡 వైద్య సంప్రదింపులు: 24-48 గంటల్లో వైద్యుడిని సంప్రదించండి.",
        "tele_manas": "టెలి-మానస్ హెల్ప్‌లైన్: 14416 (24/7 ఉచిత సేవ)",
        "abha_badge": "ఆయుష్మాన్ భారత్ ధృవీకరించబడింది",
        "wearable_sync_success": "ఆపిల్ హెల్త్ మరియు గూగుల్ హెల్త్ కనెక్ట్‌తో విజయవంతంగా సింక్ చేయబడింది.",
        "ecg_normal": "సాధారణ సైనస్ రిథమ్: గుండె లయ సాధారణ స్థితిలో ఉంది."
    },
    "mr": {
        "emergency_alert": "⚠️ आणीबाणी सूचना: कृपया ताबडतोब 112 किंवा 108 वर कॉल करा किंवा जवळच्या रुग्णालयात जा.",
        "triage_home": "🟢 घरगुती काळजी सल्ला: भरपूर पाणी प्या, विश्रांती घ्या आणि लक्षणांवर लक्ष ठेवा.",
        "triage_doctor": "🟡 वैद्यकीय सल्ला: 24-48 तासांच्या आत डॉक्टरांचा सल्ला घ्या.",
        "tele_manas": "राष्ट्रीय मानसिक आरोग्य हेल्पलाइन: 14416 (24/7 विनामूल्य)",
        "abha_badge": "आयुष्मान भारत डिजिटल मिशन (ABDM) सत्यापित",
        "wearable_sync_success": "ॲपल हेल्थ आणि गुगल हेल्थ कनेक्टसह यशस्वीरित्या सिंक झाले.",
        "ecg_normal": "सामान्य सायनस रिदम: ईसीजीमध्ये कोणतीही अनियमितता नाही."
    },
    "gu": {
        "emergency_alert": "⚠️ કટોકટી ચેતવણી: તાત્કાલિક 112 અથવા 108 પર કૉલ કરો અથવા નજીકની હૉસ્પિટલ જાઓ.",
        "triage_home": "🟢 ઘરેલું સંભાળ: પૂરતો આરામ કરો, પ્રવાહી લો અને લક્ષણો પર નજર રાખો.",
        "triage_doctor": "🟡 ડૉક્ટરની સલાહ: 24-48 કલાકની અંદર ડૉક્ટરનો સંપર્ક કરો.",
        "tele_manas": "ટેલી-માનસ હેલ્પલાઇન: 14416 (24/7 મફત સેવા)",
        "abha_badge": "આયુષ્માન ભારત ડિજિટલ મિશન પ્રમાણિત",
        "wearable_sync_success": "એપલ હેલ્થ અને ગૂગલ હેલ્થ સાથે સફળતાપૂર્વક સિંક થયું.",
        "ecg_normal": "સામાન્ય સાયનસ રિધમ: સામાન્ય હૃદય લય."
    },
    "kn": {
        "emergency_alert": "⚠️ ತುರ್ತು ಎಚ್ಚರಿಕೆ: ತಕ್ಷಣ 112 ಅಥವಾ 108 ಗೆ ಕರೆ ಮಾಡಿ ಅಥವಾ ಹತ್ತಿರದ ಆಸ್ಪತ್ರೆಗೆ ಭೇಟಿ ನೀಡಿ.",
        "triage_home": "🟢 ಮನೆಯ ಆರೈಕೆ ಸಲಹೆ: ಸಾಕಷ್ಟು ವಿಶ್ರಾಂತಿ ಪಡೆಯಿರಿ, ನೀರು ಕುಡಿಯಿರಿ ಮತ್ತು ರೋಗಲಕ್ಷಣಗಳನ್ನು ಗಮನಿಸಿ.",
        "triage_doctor": "🟡 ವೈದ್ಯರ ಸಮಾಲೋಚನೆ: 24-48 ಗಂಟೆಗಳ ಒಳಗೆ ವೈದ್ಯರನ್ನು ಸಂಪರ್ಕಿಸಿ.",
        "tele_manas": "ಟೆಲಿ-ಮಾನಸ್ ಮಾನಸಿಕ ಆರೋಗ್ಯ ಸಹಾಯವಾಣಿ: 14416 (24/7 ಉಚಿತ)",
        "abha_badge": "ಆಯುಷ್ಮಾನ್ ಭಾರತ್ ಡಿಜಿಟಲ್ ಮಿಷನ್ ಪರಿಶೀಲಿಸಲಾಗಿದೆ",
        "wearable_sync_success": "ಆಪಲ್ ಹೆಲ್ತ್ ಮತ್ತು ಗೂಗಲ್ ಹೆಲ್ತ್ ಕನೆಕ್ಟ್ ಜೊತೆಗೆ ಯಶಸ್ವಿಯಾಗಿ ಸಿಂಕ್ ಆಗಿದೆ.",
        "ecg_normal": "ಸಾಮಾನ್ಯ ಸೈನಸ್ ರಿದಮ್: ಇಸಿಜಿಯಲ್ಲಿ ಯಾವುದೇ ತೊಂದರೆ ಕಂಡುಬಂದಿಲ್ಲ."
    },
    "ml": {
        "emergency_alert": "⚠️ അടിയന്തര മുന്നറിയിപ്പ്: ഉടൻ 112 അല്ലെങ്കിൽ 108 ലേക്ക് വിളിക്കുക അല്ലെങ്കിൽ അടുത്തുള്ള ആശുപത്രി സന്ദർശിക്കുക.",
        "triage_home": "🟢 ഗൃഹ പരിചരണ ഉപദേശം: ആവശ്യത്തിന് വിശ്രമിക്കുക, ധാരാളം വെള്ളം കുടിക്കുക, ലക്ഷണങ്ങൾ ശ്രദ്ധിക്കുക.",
        "triage_doctor": "🟡 ഡോക്ടറുടെ ഉപദേശം: 24-48 മണിക്കൂറിനുള്ളിൽ ഡോക്ടറെ കാണുക.",
        "tele_manas": "ടെലി-മാനസ് ഹെൽപ്പ് ലൈൻ: 14416 (24/7 സൗജന്യം)",
        "abha_badge": "ആയുഷ്മാൻ ഭാരത് ഡിജിറ്റൽ മിഷൻ പരിശോധിച്ചുറപ്പിച്ചത്",
        "wearable_sync_success": "ആപ്പിൾ ഹെൽത്ത് & ഗൂഗിൾ ഹെൽത്തുമായി സമന്വയിപ്പിച്ചു.",
        "ecg_normal": "സാധാരണ സൈനസ് റിഥം: ഹൃദയ താളം സാധാരണ നിലയിലാണ്."
    },
    "pa": {
        "emergency_alert": "⚠️ ਐਮਰਜੈਂਸੀ ਅਲਰਟ: ਤੁਰੰਤ 112 ਜਾਂ 108 'ਤੇ ਕਾਲ ਕਰੋ ਜਾਂ ਨਜ਼ਦੀਕੀ ਹਸਪਤਾਲ ਜਾਓ।",
        "triage_home": "🟢 ਘਰੇਲੂ ਦੇਖਭਾਲ: ਆਰਾਮ ਕਰੋ, ਤਰਲ ਪਦਾਰਥ ਪੀਓ ਅਤੇ ਲੱਛਣਾਂ 'ਤੇ ਨਜ਼ਰ ਰੱਖੋ।",
        "triage_doctor": "🟡 ਡਾਕਟਰ ਦੀ ਸਲਾਹ: 24-48 ਘੰਟਿਆਂ ਦੇ ਅੰਦਰ ਡਾਕਟਰ ਨਾਲ ਸੰਪਰਕ ਕਰੋ।",
        "tele_manas": "ਟੈਲੀ-ਮਾਨਸ ਹੈਲਪਲਾਈਨ: 14416 (24/7 ਮੁਫ਼ਤ ਸੇਵਾ)",
        "abha_badge": "ਆਯੁਸ਼ਮਾਨ ਭਾਰਤ ਡਿਜੀਟਲ ਮਿਸ਼ਨ ਪ੍ਰਮਾਣਿਤ",
        "wearable_sync_success": "ਐਪਲ ਹੈਲਥ ਅਤੇ ਗੂਗਲ ਹੈਲਥ ਕਨੈਕਟ ਨਾਲ ਸਫਲਤਾਪੂਰਵਕ ਸਿੰਕ ਹੋਇਆ।",
        "ecg_normal": "ਆਮ ਸਾਈਨਸ ਰਿਦਮ: ਈਸੀਜੀ ਵਿੱਚ ਕੋਈ ਗੜਬੜੀ ਨਹੀਂ।"
    },
    "or": {
        "emergency_alert": "⚠️ ଜରୁରୀକାଳୀନ ସତର୍କତା: ତୁରନ୍ତ ୧୧୨ କିମ୍ବା ୧୦୮ କୁ କଲ୍ କରନ୍ତୁ କିମ୍ବା ନିକଟସ୍ଥ ଡାକ୍ତରଖାନାକୁ ଯାଆନ୍ତୁ।",
        "triage_home": "🟢 ଘରୋଇ ଯତ୍ନ ପରାମର୍ଶ: ପର୍ଯ୍ୟାପ୍ତ ବିଶ୍ରାମ ନିଅନ୍ତୁ, ପ୍ରଚୁର ପାଣି ପିଅନ୍ତୁ ଏବଂ ଲକ୍ଷଣ ଉପରେ ନଜର ରଖନ୍ତୁ।",
        "triage_doctor": "🟡 ଡାକ୍ତରୀ ପରାମର୍ଶ: ୨୪-୪୮ ଘଣ୍ଟା ମଧ୍ୟରେ ଡାକ୍ତରଙ୍କ ସହିତ ଯୋଗାଯୋଗ କରନ୍ତୁ।",
        "tele_manas": "ଟେଲି-ମାନସ ହେଲ୍ପଲାଇନ୍: ୧୪୪୧୬ (୨୪/୭ ମାଗଣା ସେବା)",
        "abha_badge": "ଆୟୁଷ୍ମାନ ଭାରତ ଯାଞ୍ଚ ହୋଇଛି",
        "wearable_sync_success": "ଆପଲ୍ ହେଲ୍ଥ ଏବଂ ଗୁଗଲ୍ ହେଲ୍ଥ ସହିତ ସଫଳତାର ସହିତ ସିଙ୍କ୍ ହୋଇଛି।",
        "ecg_normal": "ସ୍ୱାଭାବିକ ସାଇନସ୍ ରିଦମ୍: ହୃଦସ୍ପନ୍ଦନ ସ୍ୱାଭାବିକ ଅଛି।"
    },
    "es": {
        "emergency_alert": "⚠️ ALERTA MÉDICA: Llame inmediatamente al número de emergencias (911/112).",
        "triage_home": "🟢 Cuidado en el hogar: Descanse adecuadamente y beba abundantes líquidos.",
        "triage_doctor": "🟡 Consulta Médica: Programe una consulta médica dentro de 24 a 48 horas.",
        "tele_manas": "Línea de Ayuda Psicológica Gratuita: 988",
        "abha_badge": "Registro Médico Certificado",
        "wearable_sync_success": "Sincronizado con éxito con Apple Health y Google Health Connect.",
        "ecg_normal": "Ritmo sinusal normal: no se detectaron signos de fibrilación auricular."
    }
}


def translate_clinical_message(key: str, lang: str = "hi") -> str:
    """Translates key clinical warnings into regional Indian & global languages."""
    lang_code = lang.lower()
    lang_dict = CLINICAL_I18N_STRINGS.get(lang_code, CLINICAL_I18N_STRINGS["hi"])
    return lang_dict.get(key, CLINICAL_I18N_STRINGS["en"].get(key, ""))


def get_supported_languages() -> List[Dict[str, str]]:
    """Returns list of supported Indian and international languages."""
    return SUPPORTED_LANGUAGES
