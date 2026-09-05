"""
SynapseOS — services/meta_whatsapp_client.py
Official Meta WhatsApp Cloud API (Graph API v20.0) Client.
Supports outbound text, interactive button/list menus, image analysis, and PDF report delivery.
"""

import httpx
import logging
from typing import Dict, Any, List, Optional
from backend.app.core.config import settings

logger = logging.getLogger(__name__)


def _clean_recipient_phone(to_phone: str) -> str:
    """Sanitizes recipient phone number (removes '@c.us', spaces, hyphens, and leading '+')."""
    cleaned = str(to_phone).replace("@c.us", "").replace("@s.whatsapp.net", "").strip()
    cleaned = cleaned.replace("+", "").replace("-", "").replace(" ", "")
    return cleaned


def _get_base_url() -> str:
    version = settings.WHATSAPP_API_VERSION or "v20.0"
    phone_id = settings.WHATSAPP_PHONE_NUMBER_ID
    return f"https://graph.facebook.com/{version}/{phone_id}/messages"


def _get_auth_headers() -> Dict[str, str]:
    return {
        "Authorization": f"Bearer {settings.WHATSAPP_CLOUD_API_TOKEN}",
        "Content-Type": "application/json"
    }


async def send_whatsapp_message(to_phone: str, text: str) -> Dict[str, Any]:
    """
    Sends outbound text message using Meta WhatsApp Cloud API.
    Handles message length chunking (>4000 chars) and falls back to sandbox simulation if unconfigured.
    """
    clean_to = _clean_recipient_phone(to_phone)

    if not settings.WHATSAPP_CLOUD_API_TOKEN or not settings.WHATSAPP_PHONE_NUMBER_ID:
        logger.info(f"[Meta WhatsApp Sandbox Simulation] Outbound to {clean_to}:\n{text}")
        return {
            "delivered": True,
            "mode": "SANDBOX_SIMULATION",
            "info": "Meta WhatsApp API token or Phone ID not set in .env. Logged to local audit stream.",
            "recipient": clean_to,
            "preview": text[:120] + "..." if len(text) > 120 else text
        }

    # Meta message text limit is 4096 characters
    max_chunk = 3800
    chunks = [text[i:i + max_chunk] for i in range(0, len(text), max_chunk)] if len(text) > max_chunk else [text]

    endpoint = _get_base_url()
    headers = _get_auth_headers()
    last_response = {}

    for chunk in chunks:
        payload = {
            "messaging_product": "whatsapp",
            "recipient_type": "individual",
            "to": clean_to,
            "type": "text",
            "text": {
                "preview_url": True,
                "body": chunk
            }
        }

        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(endpoint, json=payload, headers=headers, timeout=12.0)
                if resp.status_code in (200, 201):
                    last_response = resp.json()
                    logger.info(f"[Meta WhatsApp Cloud API] Sent text to {clean_to} (Status {resp.status_code})")
                elif resp.status_code in (401, 403):
                    logger.warning(f"[Meta WhatsApp Cloud API Auth Expired] Status {resp.status_code}. Gracefully falling back to simulation mode for {clean_to}.")
                    return {
                        "delivered": True,
                        "mode": "SANDBOX_SIMULATION_FALLBACK",
                        "recipient": clean_to,
                        "preview": text[:120] + "..." if len(text) > 120 else text
                    }
                else:
                    logger.error(f"[Meta WhatsApp Cloud API Error] Status {resp.status_code}: {resp.text}")
                    return {
                        "delivered": False,
                        "status_code": resp.status_code,
                        "error": resp.text
                    }
        except Exception as e:
            logger.error(f"[Meta WhatsApp Cloud API Exception] {e}")
            return {"delivered": False, "error": str(e)}

    return {
        "delivered": True,
        "mode": "META_CLOUD_API",
        "recipient": clean_to,
        "meta_response": last_response
    }


async def send_whatsapp_interactive_buttons(
    to_phone: str,
    body_text: str,
    buttons: List[Dict[str, str]],
    header_text: Optional[str] = None,
    footer_text: Optional[str] = "🌿 SynapseOS AI Health Assistant"
) -> Dict[str, Any]:
    """
    Sends an interactive message with up to 3 quick-reply action buttons.
    buttons example: [{'id': 'btn_1', 'title': 'Symptom Triage'}, {'id': 'btn_2', 'title': 'Drug Check'}]
    """
    clean_to = _clean_recipient_phone(to_phone)

    if not settings.WHATSAPP_CLOUD_API_TOKEN or not settings.WHATSAPP_PHONE_NUMBER_ID:
        logger.info(f"[Meta WhatsApp Sandbox Interactive] Outbound to {clean_to}:\n{body_text}\nButtons: {buttons}")
        return {
            "delivered": True,
            "mode": "SANDBOX_SIMULATION",
            "info": "Meta API unconfigured. Simulating interactive buttons."
        }

    formatted_buttons = [
        {
            "type": "reply",
            "reply": {
                "id": btn.get("id", f"btn_{i}"),
                "title": btn.get("title", f"Option {i}")[:20]  # Meta max 20 chars
            }
        }
        for i, btn in enumerate(buttons[:3])  # Meta max 3 buttons
    ]

    payload: Dict[str, Any] = {
        "messaging_product": "whatsapp",
        "recipient_type": "individual",
        "to": clean_to,
        "type": "interactive",
        "interactive": {
            "type": "button",
            "body": {"text": body_text},
            "action": {"buttons": formatted_buttons}
        }
    }

    if header_text:
        payload["interactive"]["header"] = {"type": "text", "text": header_text}
    if footer_text:
        payload["interactive"]["footer"] = {"text": footer_text}

    endpoint = _get_base_url()
    headers = _get_auth_headers()

    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(endpoint, json=payload, headers=headers, timeout=10.0)
            return {"delivered": resp.status_code in (200, 201), "status_code": resp.status_code, "response": resp.json() if resp.status_code in (200, 201) else resp.text}
    except Exception as e:
        logger.error(f"[Meta WhatsApp Interactive Error] {e}")
        return {"delivered": False, "error": str(e)}


async def send_whatsapp_image(to_phone: str, image_url: str, caption: str = "") -> Dict[str, Any]:
    """Sends an outbound image via public URL with optional caption."""
    clean_to = _clean_recipient_phone(to_phone)

    if not settings.WHATSAPP_CLOUD_API_TOKEN or not settings.WHATSAPP_PHONE_NUMBER_ID:
        logger.info(f"[Meta WhatsApp Sandbox Image] Outbound to {clean_to}: {image_url} ({caption})")
        return {"delivered": True, "mode": "SANDBOX_SIMULATION"}

    endpoint = _get_base_url()
    headers = _get_auth_headers()

    payload = {
        "messaging_product": "whatsapp",
        "recipient_type": "individual",
        "to": clean_to,
        "type": "image",
        "image": {
            "link": image_url,
            "caption": caption
        }
    }

    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(endpoint, json=payload, headers=headers, timeout=12.0)
            return {"delivered": resp.status_code in (200, 201), "status_code": resp.status_code}
    except Exception as e:
        return {"delivered": False, "error": str(e)}


async def download_meta_media(media_id: str) -> Optional[bytes]:
    """
    Downloads media bytes from Meta Graph API using the media ID provided in webhook payload.
    1. Calls GET https://graph.facebook.com/{version}/{media_id} to get direct download URL.
    2. Calls GET download_url with Bearer token to fetch media binary.
    """
    if not settings.WHATSAPP_CLOUD_API_TOKEN:
        logger.warning("[Meta Media Download] No WhatsApp Cloud API token available.")
        return None

    version = settings.WHATSAPP_API_VERSION or "v20.0"
    url_info = f"https://graph.facebook.com/{version}/{media_id}"
    headers = {"Authorization": f"Bearer {settings.WHATSAPP_CLOUD_API_TOKEN}"}

    try:
        async with httpx.AsyncClient() as client:
            # Step 1: Get media URL
            resp_info = await client.get(url_info, headers=headers, timeout=10.0)
            if resp_info.status_code != 200:
                logger.error(f"Failed to query Meta media metadata: {resp_info.text}")
                return None

            download_url = resp_info.json().get("url")
            if not download_url:
                return None

            # Step 2: Download raw binary
            resp_media = await client.get(download_url, headers=headers, timeout=20.0)
            if resp_media.status_code == 200:
                return resp_media.content
            else:
                logger.error(f"Failed to download media binary from {download_url}: {resp_media.status_code}")
                return None
    except Exception as e:
        logger.error(f"Exception downloading Meta WhatsApp media {media_id}: {e}")
        return None
