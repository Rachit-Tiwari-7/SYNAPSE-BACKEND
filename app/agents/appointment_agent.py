"""
SynapseOS — agents/appointment_agent.py
Automated Clinical Appointment Scheduling & Calendar Dispatch Agent.
Ported from MedAgent-main appointment scheduling workflow.
"""

import uuid
from typing import Dict, Any, List
from datetime import datetime, timedelta

# Sample Available Specialist Roster
DOCTORS_ROSTER = [
    {
        "doctor_id": "DOC-AIIMS-101",
        "name": "Dr. Rajesh K. Varma",
        "specialty": "Cardiologist",
        "hospital": "AIIMS New Delhi",
        "experience": "18 Years",
        "fee": "₹0 (PM-JAY Empanelled)",
        "available_slots": ["Tomorrow at 10:30 AM", "Tomorrow at 02:00 PM", "Friday at 11:00 AM"]
    },
    {
        "doctor_id": "DOC-MAX-102",
        "name": "Dr. Ananya Sen",
        "specialty": "General Physician / Internal Medicine",
        "hospital": "Max Super Speciality Hospital",
        "experience": "12 Years",
        "fee": "₹0 (PM-JAY Empanelled)",
        "available_slots": ["Today at 04:30 PM", "Tomorrow at 11:15 AM", "Thursday at 03:00 PM"]
    },
    {
        "doctor_id": "DOC-APOLLO-103",
        "name": "Dr. Sameer Deshmukh",
        "specialty": "Pulmonologist",
        "hospital": "Apollo Hospitals",
        "experience": "15 Years",
        "fee": "₹0 (PM-JAY Empanelled)",
        "available_slots": ["Tomorrow at 09:00 AM", "Friday at 01:30 PM"]
    }
]


def find_doctors_by_specialty(specialty: str = "General Physician") -> List[Dict[str, Any]]:
    """Returns available doctors matching the needed clinical specialty."""
    specialty_lower = specialty.lower()
    matches = [d for d in DOCTORS_ROSTER if specialty_lower in d["specialty"].lower()]
    return matches if matches else DOCTORS_ROSTER


def book_appointment_slot(
    patient_name: str,
    doctor_id: str,
    slot_time: str,
    abha_id: str = "91-4829-1029-4821",
    symptoms_brief: str = "General consultation"
) -> Dict[str, Any]:
    """Books and confirms an appointment slot with digital token generation."""
    booking_id = f"APT-{uuid.uuid4().hex[:6].upper()}"
    doctor = next((d for d in DOCTORS_ROSTER if d["doctor_id"] == doctor_id), DOCTORS_ROSTER[0])

    return {
        "status": "CONFIRMED",
        "booking_id": booking_id,
        "patient_name": patient_name,
        "abha_id": abha_id,
        "doctor_name": doctor["name"],
        "specialty": doctor["specialty"],
        "hospital": doctor["hospital"],
        "appointment_time": slot_time,
        "symptoms_brief": symptoms_brief,
        "consultation_mode": "In-Person Clinic / Tele-Consultation",
        "calendar_ics_payload": {
            "title": f"Medical Consultation with {doctor['name']}",
            "description": f"SynapseOS Booking Ref: {booking_id} for {patient_name} ({symptoms_brief})",
            "location": doctor["hospital"]
        },
        "notification_dispatched": True,
        "created_at": datetime.utcnow().isoformat() + "Z"
    }
