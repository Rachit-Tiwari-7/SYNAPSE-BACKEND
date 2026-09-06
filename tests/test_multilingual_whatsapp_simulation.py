"""
SynapseOS — tests/test_multilingual_whatsapp_simulation.py
Simulates realistic WhatsApp end-to-end interactions across multiple Indian regional languages.
Tests:
1. Language Selection Flow (Hindi, Bengali, Tamil, Telugu, Marathi, Gujarati)
2. Symptom Triage in Native Script (Devanagari Hindi, Bengali, Tamil, Telugu)
3. Symptom Triage with Transliterated Hinglish
4. Prescription OCR Image Upload with Active Language Session
5. Informational Health Inquiries in Regional Languages
6. Verification that WhatsApp response cards comply with plain text guidelines (no broken markdown).
"""

import pytest
import base64
import io
from PIL import Image
from unittest.mock import patch, AsyncMock
from backend.app.services.meta_whatsapp_service import (
    process_whatsapp_inbound_webhook,
    format_compact_whatsapp_card,
    format_response_for_whatsapp,
    LOCALIZED_MENUS,
    LANGUAGE_SELECTION_MENU
)
from backend.app.core.session_manager import session_manager
from backend.app.services.prescription_ocr_service import format_prescription_for_whatsapp
from backend.app.agents.orchestrator import orchestrate_health_request
from backend.app.core.state import SynapseOSState
from backend.app.services.i18n_service import detect_text_language


def _make_dummy_image_b64() -> str:
    img = Image.new("RGB", (200, 200), color=(240, 240, 240))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode('utf-8')}"


@pytest.fixture(autouse=True)
def clear_sessions():
    session_manager._sessions.clear()
    yield
    session_manager._sessions.clear()


# ======================================================================
# 1. TEST LANGUAGE SELECTION ONBOARDING
# ======================================================================

@pytest.mark.asyncio
async def test_simulate_whatsapp_onboarding_and_hindi_selection():
    phone = "919876543210"
    
    with patch("backend.app.services.meta_whatsapp_service.send_whatsapp_message", new_callable=AsyncMock) as mock_send:
        mock_send.return_value = {"delivered": True, "mode": "MOCK"}

        # User sends initial greeting "hi"
        res1 = await process_whatsapp_inbound_webhook({
            "sender_phone": phone,
            "message": "hi"
        })
        assert res1["status"] == "processed"
        assert res1["type"] == "menu_dispatched"
        
        # Verify session is in LANG_SELECT flow
        session = session_manager.get_session(phone)
        assert session["active_flow"] == "LANG_SELECT"

        # User replies with '2' to select Hindi
        res2 = await process_whatsapp_inbound_webhook({
            "sender_phone": phone,
            "message": "2"
        })
        assert res2["status"] == "processed"
        assert res2["type"] == "language_selected"
        assert res2["language"] == "hi"

        # Session language is preserved as 'hi'
        session = session_manager.get_session(phone)
        assert session["context"]["lang"] == "hi"


@pytest.mark.asyncio
async def test_simulate_whatsapp_language_switching():
    phone = "919876543211"
    
    with patch("backend.app.services.meta_whatsapp_service.send_whatsapp_message", new_callable=AsyncMock) as mock_send:
        mock_send.return_value = {"delivered": True, "mode": "MOCK"}

        # User selects Bengali ('3')
        await process_whatsapp_inbound_webhook({"sender_phone": phone, "message": "hi"})
        res = await process_whatsapp_inbound_webhook({"sender_phone": phone, "message": "3"})
        assert res["language"] == "bn"
        assert session_manager.get_session(phone)["context"]["lang"] == "bn"

        # User sends 'lang' to change language to Tamil ('4')
        res_lang = await process_whatsapp_inbound_webhook({"sender_phone": phone, "message": "lang"})
        assert res_lang["type"] == "language_menu_dispatched"

        res_tamil = await process_whatsapp_inbound_webhook({"sender_phone": phone, "message": "4"})
        assert res_tamil["language"] == "ta"
        assert session_manager.get_session(phone)["context"]["lang"] == "ta"
# ======================================================================
# 2. TEST HINDI SYMPTOM TRIAGE SIMULATION
# ======================================================================

@pytest.mark.asyncio
async def test_simulate_hindi_symptom_triage_flow():
    phone = "919876543212"
    
    with patch("backend.app.services.meta_whatsapp_service.send_whatsapp_message", new_callable=AsyncMock) as mock_send:
        mock_send.return_value = {"delivered": True, "mode": "MOCK"}

        # 1. Onboarding & Select Hindi
        await process_whatsapp_inbound_webhook({"sender_phone": phone, "message": "hi"})
        await process_whatsapp_inbound_webhook({"sender_phone": phone, "message": "2"})

        # Mock orchestrate_health_request to return Hindi clinical council state
        mock_state = SynapseOSState(
            session_id="test_session",
            user_id="test_user",
            input_text="मुझे 3 दिन से तेज बुखार, खांसी और बदन में तेज दर्द है",
            language="hi",
            final_response=(
                "🟡 SANJEEVNI CLINICAL ASSESSMENT\n"
                "━━━━━━━━━━━━━━━━━━━━\n"
                "Suspected Diagnosis: वायरल बुखार एवं ऊपरी श्वसन संक्रमण (Viral Pyrexia & URTI)\n"
                "Council Consensus: 92% Consensus\n\n"
                "* Immediate Action: 24 घंटे के भीतर डॉक्टर से परामर्श लें और पूर्ण विश्राम करें।\n"
                "* Hydration & Care: ओआरएस (ORS) और पर्याप्त गुनगुने तरल पदार्थ का सेवन करें।\n\n"
                "Medications: Paracetamol (Dolo 650) 650mg दिन में दो बार भोजन के बाद लें।\n\n"
                "Seek Emergency Care If: सांस लेने में अत्यधिक तकलीफ या तेज बुखार (>103°F) हो।"
            )
        )

        with patch("backend.app.services.meta_whatsapp_service.orchestrate_health_request", new_callable=AsyncMock) as mock_orch:
            mock_orch.return_value = mock_state

            res = await process_whatsapp_inbound_webhook({
                "sender_phone": phone,
                "message": "1 मुझे 3 दिन से तेज बुखार, खांसी और बदन में तेज दर्द है"
            })

            assert res["status"] == "processed"
            assert mock_send.called
            
            sent_text = mock_send.call_args[1]["text"]
            
            # Verify output is formatted in Hindi
            assert "संजीवनी" in sent_text
            assert "━━━━━━━━━━━━━━━━━━━━" in sent_text
            assert "संभावित निदान" in sent_text
            assert "दवाइयां एवं राहत" in sent_text
            assert "त्वरित शॉर्टकट" in sent_text
            assert "संजीवनी-ओएस मल्टी-एजेंट द्वारा संचालित" in sent_text
            
            # Verify NO markdown asterisks or backticks
            assert "**" not in sent_text
            assert "```" not in sent_text
            assert "#" not in sent_text


@pytest.mark.asyncio
async def test_simulate_hinglish_symptom_triage_flow():
    phone = "919876543213"
    
    with patch("backend.app.services.meta_whatsapp_service.send_whatsapp_message", new_callable=AsyncMock) as mock_send:
        mock_send.return_value = {"delivered": True, "mode": "MOCK"}

        mock_state = SynapseOSState(
            session_id="test_session",
            user_id="test_user",
            input_text="mujhe 2 din se tez bukhar aur sar dard hai, kya karu?",
            language="hi",
            final_response=(
                "🟡 SANJEEVNI CLINICAL ASSESSMENT\n"
                "━━━━━━━━━━━━━━━━━━━━\n"
                "Suspected Diagnosis: वायरल बुखार एवं सिरदर्द (Viral Pyrexia)\n"
                "Council Consensus: 90% Consensus\n\n"
                "* Immediate Action: पर्याप्त आराम करें और ओआरएस घोल का सेवन करें।\n"
                "* Fever Monitoring: तापमान की नियमित निगरानी रखें।\n\n"
                "Medications: Dolo 650mg भोजन के बाद 1 गोली ले सकते हैं।\n\n"
                "Seek Emergency Care If: सांस लेने में तकलीफ या बेहोशी के लक्षण हों।"
            )
        )

        with patch("backend.app.services.meta_whatsapp_service.orchestrate_health_request", new_callable=AsyncMock) as mock_orch:
            mock_orch.return_value = mock_state

            res = await process_whatsapp_inbound_webhook({
                "sender_phone": phone,
                "message": "mujhe 2 din se tez bukhar aur sar dard hai, kya karu?"
            })

            assert res["status"] == "processed"
            assert mock_send.called
            sent_text = mock_send.call_args[1]["text"]
            
            # Automatic language detection detects Hinglish and renders in Hindi
            assert "संजीवनी" in sent_text
            assert "━━━━━━━━━━━━━━━━━━━━" in sent_text
            assert "संभावित निदान" in sent_text
            assert "**" not in sent_text


# ======================================================================
# 3. TEST BENGALI, TAMIL & TELUGU SCRIPT DETECTION & SWARM ROUTING
# ======================================================================

@pytest.mark.asyncio
async def test_simulate_bengali_symptom_query():
    bengali_query = "আমার ৩ দিন ধরে খুব জ্বর ও শুকনো কাশি হচ্ছে"
    detected = detect_text_language(bengali_query)
    assert detected == "bn"


@pytest.mark.asyncio
async def test_simulate_tamil_symptom_query():
    tamil_query = "எனக்கு 2 நாட்களாக கடுமையான காய்ச்சல் மற்றும் தலைவலி உள்ளது"
    detected = detect_text_language(tamil_query)
    assert detected == "ta"


@pytest.mark.asyncio
async def test_simulate_telugu_symptom_query():
    telugu_query = "నాకు తీవ్రమైన జ్వరం మరియు తలనొప్పి ఉంది"
    detected = detect_text_language(telugu_query)
    assert detected == "te"


# ======================================================================
# 4. TEST WHATSAPP PRESCRIPTION OCR SIMULATION IN HINDI
# ======================================================================

@pytest.mark.asyncio
async def test_simulate_whatsapp_prescription_image_in_hindi():
    phone = "919876543214"
    
    with patch("backend.app.services.meta_whatsapp_service.send_whatsapp_message", new_callable=AsyncMock) as mock_send:
        mock_send.return_value = {"delivered": True, "mode": "MOCK"}

        # 1. Onboarding & Select Hindi
        await process_whatsapp_inbound_webhook({"sender_phone": phone, "message": "hi"})
        await process_whatsapp_inbound_webhook({"sender_phone": phone, "message": "2"})

        dummy_b64 = _make_dummy_image_b64()

        mock_ocr = {
            "success": True,
            "document_type": "medical_prescription",
            "doctor": {"name": "Dr. Sharma"},
            "diagnosis": "तीव्र श्वसन संक्रमण (Acute Bronchitis)",
            "medications": [
                {
                    "name": "Augmentin 625",
                    "raw_name": "Augmentin 625mg",
                    "strength": "625mg",
                    "frequency": "1-0-1",
                    "timing": "भोजन के बाद",
                    "confidence": 0.95,
                    "is_uncertain": False
                }
            ]
        }

        mock_interp = {
            "likely_condition": "तीव्र ब्रोंकाइटिस (श्वसन संक्रमण)",
            "plain_language_summary": "श्वसन नली में संक्रमण के उपचार हेतु एंटीबायोटिक और सहायक दवाएं।",
            "medication_guide": [
                {
                    "medicine": "Augmentin 625",
                    "purpose": "बैक्टीरियल संक्रमण नियंत्रण",
                    "timing": "1 गोली सुबह-शाम भोजन के बाद",
                    "generic_alternative": "Amoxycillin + Clavulanic Acid 625mg (जन औषधि)"
                }
            ],
            "generic_savings_tip": "जन औषधि केंद्र से जेनेरिक दवा लेकर 60% बचत करें।",
            "precautions_and_rules": ["पूरा 5 दिन का कोर्स पूरा करें।", "समय पर दवा लें।"],
            "red_flag_warnings": ["सांस लेने में अत्यधिक कठिनाई होने पर तुरंत 108 पर संपर्क करें।"]
        }

        dummy_img = Image.new("RGB", (200, 200), color=(240, 240, 240))

        with patch("backend.app.services.prescription_ocr_service.validate_image_bytes") as mock_val, \
             patch("backend.app.services.prescription_ocr_service.run_prescription_ocr", new_callable=AsyncMock) as mock_ocr_call, \
             patch("backend.app.services.prescription_ocr_service.interpret_prescription", new_callable=AsyncMock) as mock_interp_call:

            mock_val.return_value = (True, None, None, dummy_img)
            mock_ocr_call.return_value = (True, None, mock_ocr)
            mock_interp_call.return_value = mock_interp

            res = await process_whatsapp_inbound_webhook({
                "sender_phone": phone,
                "type": "image",
                "image_base64": dummy_b64,
                "caption": "डॉक्टर की पर्ची"
            })

            assert res["status"] == "processed"
            assert res["type"] == "prescription_ocr_interpretation"
            assert mock_send.called

            sent_text = mock_send.call_args[1]["text"]
            
            # Verify Hindi plain text card
            assert "📋 संजीवनी पर्ची एवं स्वास्थ्य सारांश" in sent_text
            assert "🩺 संभावित निदान: तीव्र ब्रोंकाइटिस (श्वसन संक्रमण)" in sent_text
            assert "📊 काउंसिल सहमति: 94% सहमति" in sent_text
            assert "💊 दवाइयां एवं सेवन विधि (भारत):" in sent_text
            assert "Augmentin 625" in sent_text
            assert "💰 जन औषधि बचत विकल्प:" in sent_text
            assert "🚨 तुरंत आपातकालीन सहायता लें / 108 पर कॉल करें यदि:" in sent_text
            assert "🌿 संजीवनी-ओएस मल्टी-एजेंट द्वारा संचालित" in sent_text
            
            # Check strict plain text rule
            assert "**" not in sent_text
            assert "```" not in sent_text
            assert "*" not in sent_text

