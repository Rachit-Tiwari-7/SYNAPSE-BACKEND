"""
SynapseOS — services/whatsapp_service.py
Unified facade re-exporting official Meta WhatsApp Cloud API services and handlers.
"""

from backend.app.services.meta_whatsapp_service import (
    process_whatsapp_inbound_webhook,
    trigger_emergency_sos_whatsapp,
    MAIN_MENU_TEXT,
    format_response_for_whatsapp,
    format_compact_whatsapp_card
)
from backend.app.services.meta_whatsapp_client import (
    send_whatsapp_message,
    send_whatsapp_image,
    send_whatsapp_interactive_buttons,
    download_meta_media
)

__all__ = [
    "process_whatsapp_inbound_webhook",
    "send_whatsapp_message",
    "send_whatsapp_image",
    "send_whatsapp_interactive_buttons",
    "download_meta_media",
    "trigger_emergency_sos_whatsapp",
    "MAIN_MENU_TEXT",
    "format_response_for_whatsapp",
    "format_compact_whatsapp_card"
]


