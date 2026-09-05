"""
SynapseOS — core/session_manager.py
Thread-safe in-memory session and conversation state tracker for WhatsApp multi-turn workflows.
"""

import time
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class WhatsAppSessionManager:
    """Tracks per-user active conversation flows (e.g. Quizzes, Triage follow-ups, Doctor bookings)."""

    def __init__(self, ttl_seconds: int = 1800):
        self._sessions: Dict[str, Dict[str, Any]] = {}
        self.ttl_seconds = ttl_seconds

    def get_session(self, user_phone: str) -> Dict[str, Any]:
        """Fetches active session or creates a fresh one."""
        now = time.time()
        phone_key = str(user_phone).replace("+", "").strip()

        if phone_key not in self._sessions or (now - self._sessions[phone_key].get("last_active", 0) > self.ttl_seconds):
            self._sessions[phone_key] = {
                "phone": phone_key,
                "active_flow": "IDLE",
                "step": 0,
                "context": {},
                "last_active": now
            }
        self._sessions[phone_key]["last_active"] = now
        return self._sessions[phone_key]

    def set_flow(self, user_phone: str, flow_name: str, initial_context: Optional[Dict[str, Any]] = None):
        """Sets the active conversational flow and initializes context."""
        session = self.get_session(user_phone)
        session["active_flow"] = flow_name
        session["step"] = 0
        session["context"] = initial_context or {}
        session["last_active"] = time.time()
        logger.info(f"[Session] User {user_phone} entered flow: {flow_name}")

    def advance_step(self, user_phone: str, step_data: Optional[Dict[str, Any]] = None):
        """Advances the step counter and merges new context data."""
        session = self.get_session(user_phone)
        session["step"] += 1
        if step_data:
            session["context"].update(step_data)
        session["last_active"] = time.time()

    def reset_flow(self, user_phone: str):
        """Resets user state to IDLE."""
        phone_key = str(user_phone).replace("+", "").strip()
        if phone_key in self._sessions:
            self._sessions[phone_key]["active_flow"] = "IDLE"
            self._sessions[phone_key]["step"] = 0
            self._sessions[phone_key]["context"] = {}
            self._sessions[phone_key]["last_active"] = time.time()
            logger.info(f"[Session] User {user_phone} reset to IDLE")


# Global singleton instance
session_manager = WhatsAppSessionManager()
