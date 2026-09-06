"""
SynapseOS — core/state.py
Shared agent state schema across multi-agent workflows.
"""

from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
from datetime import datetime


class AgentTraceStep(BaseModel):
    agent_name: str
    action: str
    status: str = "completed"
    duration_ms: int = 0
    details: Optional[Dict[str, Any]] = None


class SynapseOSState(BaseModel):
    session_id: str
    user_id: str = "guest_user"
    channel: str = "web"  # web, voice, whatsapp, telegram, discord
    input_text: str
    language: str = "en"
    detected_intent: Optional[str] = None
    
    # Sub-agent outputs
    safety_cleared: bool = True
    safety_message: Optional[str] = None
    
    triage_data: Optional[Dict[str, Any]] = None
    drug_check: Optional[Dict[str, Any]] = None
    verification: Optional[Dict[str, Any]] = None
    scan_analysis: Optional[Dict[str, Any]] = None
    digital_twin: Optional[Dict[str, Any]] = None
    logistics: Optional[Dict[str, Any]] = None
    vaccination_data: Optional[Dict[str, Any]] = None
    preventive_data: Optional[Dict[str, Any]] = None
    outbreak_data: Optional[Dict[str, Any]] = None
    
    # Execution trace for UI visualization
    trace: List[AgentTraceStep] = Field(default_factory=list)
    
    # Final consolidated output
    final_response: str = ""
    suggested_actions: List[str] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    @property
    def user_message(self) -> str:
        return self.input_text

