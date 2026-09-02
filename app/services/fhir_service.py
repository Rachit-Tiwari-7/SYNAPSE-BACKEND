"""
SynapseOS — services/fhir_service.py
HL7 / FHIR R4 (Fast Healthcare Interoperability Resources) Standard Data Models.
Ported from AI-Healthcare-System fhir.py for SynapseOS EHR Interoperability.
Generates compliant FHIR R4 Bundles for Patient, Observations (Vitals), Conditions, and DiagnosticReports.
"""

from typing import Dict, Any, List
from datetime import datetime
import uuid


def build_fhir_r4_bundle(
    patient_id: str,
    name: str,
    gender: str = "male",
    birth_date: str = "1995-05-12",
    vitals: Dict[str, Any] = None,
    conditions: List[str] = None
) -> Dict[str, Any]:
    """Generates an official HL7 FHIR R4 Bundle containing Patient, Observations, and Conditions."""
    bundle_id = str(uuid.uuid4())
    timestamp = datetime.utcnow().isoformat() + "Z"

    entries = []

    # 1. FHIR Patient Resource
    patient_resource = {
        "resourceType": "Patient",
        "id": patient_id,
        "identifier": [
            {
                "system": "https://healthid.ndhm.gov.in",
                "value": patient_id,
                "type": {"coding": [{"system": "http://terminology.hl7.org/CodeSystem/v2-0203", "code": "MR"}]}
            }
        ],
        "active": True,
        "name": [{"use": "official", "text": name}],
        "gender": gender.lower(),
        "birthDate": birth_date
    }
    entries.append({
        "fullUrl": f"urn:uuid:{uuid.uuid4()}",
        "resource": patient_resource
    })

    # 2. FHIR Observation Resources (Vitals)
    if vitals:
        if "systolic_bp" in vitals:
            entries.append({
                "fullUrl": f"urn:uuid:{uuid.uuid4()}",
                "resource": {
                    "resourceType": "Observation",
                    "status": "final",
                    "category": [{"coding": [{"system": "http://terminology.hl7.org/CodeSystem/observation-category", "code": "vital-signs"}]}],
                    "code": {"coding": [{"system": "http://loinc.org", "code": "8480-6", "display": "Systolic blood pressure"}]},
                    "subject": {"reference": f"Patient/{patient_id}"},
                    "valueQuantity": {"value": vitals["systolic_bp"], "unit": "mmHg", "system": "http://unitsofmeasure.org", "code": "mm[Hg]"}
                }
            })
        if "fasting_glucose" in vitals:
            entries.append({
                "fullUrl": f"urn:uuid:{uuid.uuid4()}",
                "resource": {
                    "resourceType": "Observation",
                    "status": "final",
                    "category": [{"coding": [{"system": "http://terminology.hl7.org/CodeSystem/observation-category", "code": "laboratory"}]}],
                    "code": {"coding": [{"system": "http://loinc.org", "code": "1558-6", "display": "Fasting Glucose [Mass/volume] in Blood"}]},
                    "subject": {"reference": f"Patient/{patient_id}"},
                    "valueQuantity": {"value": vitals["fasting_glucose"], "unit": "mg/dL", "system": "http://unitsofmeasure.org", "code": "mg/dL"}
                }
            })

    # 3. FHIR Condition Resources
    if conditions:
        for cond in conditions:
            entries.append({
                "fullUrl": f"urn:uuid:{uuid.uuid4()}",
                "resource": {
                    "resourceType": "Condition",
                    "clinicalStatus": {"coding": [{"system": "http://terminology.hl7.org/CodeSystem/condition-clinical", "code": "active"}]},
                    "verificationStatus": {"coding": [{"system": "http://terminology.hl7.org/CodeSystem/condition-ver-status", "code": "confirmed"}]},
                    "code": {"text": cond},
                    "subject": {"reference": f"Patient/{patient_id}"},
                    "recordedDate": timestamp
                }
            })

    return {
        "resourceType": "Bundle",
        "id": bundle_id,
        "type": "collection",
        "timestamp": timestamp,
        "total": len(entries),
        "entry": entries
    }


def build_wearable_fhir_bundle(
    payload: Dict[str, Any],
    patient_id: str = "PAT-91-4829",
    patient_name: str = "Siddharth Sharma"
) -> Dict[str, Any]:
    """
    Transforms wearable telemetry payload into a compliant HL7 FHIR R4 Bundle
    mapped with standardized LOINC codes and UCUM units.
    """
    bundle_id = str(uuid.uuid4())
    timestamp = datetime.utcnow().isoformat() + "Z"
    source_device = payload.get("device_name", "Wearable Device")
    source_type = payload.get("source", "apple_health")

    entries = []

    # 1. SpO2 (Pulse Oximetry) -> LOINC 2708-6 / 59408-5
    if payload.get("spo2_percent") is not None:
        entries.append({
            "fullUrl": f"urn:uuid:{uuid.uuid4()}",
            "resource": {
                "resourceType": "Observation",
                "status": "final",
                "category": [{"coding": [{"system": "http://terminology.hl7.org/CodeSystem/observation-category", "code": "vital-signs"}]}],
                "code": {"coding": [{"system": "http://loinc.org", "code": "59408-5", "display": "Oxygen saturation in Arterial blood by Pulse oximetry"}]},
                "subject": {"reference": f"Patient/{patient_id}", "display": patient_name},
                "effectiveDateTime": timestamp,
                "valueQuantity": {"value": float(payload["spo2_percent"]), "unit": "%", "system": "http://unitsofmeasure.org", "code": "%"},
                "device": {"display": f"{source_device} ({source_type})"}
            }
        })

    # 2. Current Heart Rate -> LOINC 8867-4
    if payload.get("heart_rate_bpm") is not None:
        entries.append({
            "fullUrl": f"urn:uuid:{uuid.uuid4()}",
            "resource": {
                "resourceType": "Observation",
                "status": "final",
                "category": [{"coding": [{"system": "http://terminology.hl7.org/CodeSystem/observation-category", "code": "vital-signs"}]}],
                "code": {"coding": [{"system": "http://loinc.org", "code": "8867-4", "display": "Heart rate"}]},
                "subject": {"reference": f"Patient/{patient_id}", "display": patient_name},
                "effectiveDateTime": timestamp,
                "valueQuantity": {"value": int(payload["heart_rate_bpm"]), "unit": "beats/min", "system": "http://unitsofmeasure.org", "code": "/min"},
                "device": {"display": f"{source_device} ({source_type})"}
            }
        })

    # 3. Resting Heart Rate -> LOINC 40443-4
    if payload.get("resting_heart_rate") is not None:
        entries.append({
            "fullUrl": f"urn:uuid:{uuid.uuid4()}",
            "resource": {
                "resourceType": "Observation",
                "status": "final",
                "category": [{"coding": [{"system": "http://terminology.hl7.org/CodeSystem/observation-category", "code": "vital-signs"}]}],
                "code": {"coding": [{"system": "http://loinc.org", "code": "40443-4", "display": "Resting heart rate"}]},
                "subject": {"reference": f"Patient/{patient_id}", "display": patient_name},
                "effectiveDateTime": timestamp,
                "valueQuantity": {"value": int(payload["resting_heart_rate"]), "unit": "beats/min", "system": "http://unitsofmeasure.org", "code": "/min"},
                "device": {"display": f"{source_device} ({source_type})"}
            }
        })

    # 4. Heart Rate Variability (HRV / SDNN) -> LOINC 80404-7
    if payload.get("hrv_ms") is not None:
        entries.append({
            "fullUrl": f"urn:uuid:{uuid.uuid4()}",
            "resource": {
                "resourceType": "Observation",
                "status": "final",
                "category": [{"coding": [{"system": "http://terminology.hl7.org/CodeSystem/observation-category", "code": "vital-signs"}]}],
                "code": {"coding": [{"system": "http://loinc.org", "code": "80404-7", "display": "R-R interval.standard deviation (HRV)"}]},
                "subject": {"reference": f"Patient/{patient_id}", "display": patient_name},
                "effectiveDateTime": timestamp,
                "valueQuantity": {"value": int(payload["hrv_ms"]), "unit": "ms", "system": "http://unitsofmeasure.org", "code": "ms"},
                "device": {"display": f"{source_device} ({source_type})"}
            }
        })

    # 5. Respiratory Rate -> LOINC 9279-1
    if payload.get("respiratory_rate") is not None:
        entries.append({
            "fullUrl": f"urn:uuid:{uuid.uuid4()}",
            "resource": {
                "resourceType": "Observation",
                "status": "final",
                "category": [{"coding": [{"system": "http://terminology.hl7.org/CodeSystem/observation-category", "code": "vital-signs"}]}],
                "code": {"coding": [{"system": "http://loinc.org", "code": "9279-1", "display": "Respiratory rate"}]},
                "subject": {"reference": f"Patient/{patient_id}", "display": patient_name},
                "effectiveDateTime": timestamp,
                "valueQuantity": {"value": int(payload["respiratory_rate"]), "unit": "breaths/min", "system": "http://unitsofmeasure.org", "code": "/min"},
                "device": {"display": f"{source_device} ({source_type})"}
            }
        })

    # 6. Step Count -> LOINC 41950-7
    if payload.get("steps") is not None:
        entries.append({
            "fullUrl": f"urn:uuid:{uuid.uuid4()}",
            "resource": {
                "resourceType": "Observation",
                "status": "final",
                "category": [{"coding": [{"system": "http://terminology.hl7.org/CodeSystem/observation-category", "code": "activity"}]}],
                "code": {"coding": [{"system": "http://loinc.org", "code": "41950-7", "display": "Number of steps in 24 hour Measured"}]},
                "subject": {"reference": f"Patient/{patient_id}", "display": patient_name},
                "effectiveDateTime": timestamp,
                "valueQuantity": {"value": int(payload["steps"]), "unit": "steps", "system": "http://unitsofmeasure.org", "code": "1"},
                "device": {"display": f"{source_device} ({source_type})"}
            }
        })

    # 7. Sleep Duration -> LOINC 93832-4
    if payload.get("sleep_duration_hrs") is not None:
        entries.append({
            "fullUrl": f"urn:uuid:{uuid.uuid4()}",
            "resource": {
                "resourceType": "Observation",
                "status": "final",
                "category": [{"coding": [{"system": "http://terminology.hl7.org/CodeSystem/observation-category", "code": "activity"}]}],
                "code": {"coding": [{"system": "http://loinc.org", "code": "93832-4", "display": "Sleep duration"}]},
                "subject": {"reference": f"Patient/{patient_id}", "display": patient_name},
                "effectiveDateTime": timestamp,
                "valueQuantity": {"value": float(payload["sleep_duration_hrs"]), "unit": "hours", "system": "http://unitsofmeasure.org", "code": "h"},
                "device": {"display": f"{source_device} ({source_type})"}
            }
        })

    # 8. Single-Lead ECG Interpretation -> LOINC 11524-6
    if payload.get("ecg_classification"):
        entries.append({
            "fullUrl": f"urn:uuid:{uuid.uuid4()}",
            "resource": {
                "resourceType": "Observation",
                "status": "final",
                "category": [{"coding": [{"system": "http://terminology.hl7.org/CodeSystem/observation-category", "code": "exam"}]}],
                "code": {"coding": [{"system": "http://loinc.org", "code": "11524-6", "display": "EKG study"}]},
                "subject": {"reference": f"Patient/{patient_id}", "display": patient_name},
                "effectiveDateTime": timestamp,
                "valueString": str(payload["ecg_classification"]),
                "device": {"display": f"{source_device} ({source_type})"}
            }
        })

    return {
        "resourceType": "Bundle",
        "id": bundle_id,
        "type": "collection",
        "timestamp": timestamp,
        "total": len(entries),
        "entry": entries
    }
