"""
SynapseOS — services/pinata_service.py
Decentralized medical record & clinical report storage engine via Pinata IPFS.
Includes zero-config simulation fallback when PINATA_JWT is not configured.
"""

import hashlib
import json
import logging
from typing import Dict, Any, Optional
import httpx

from backend.app.core.config import settings

logger = logging.getLogger("synapseos.pinata")

PINATA_PIN_JSON_URL = "https://api.pinata.cloud/pinning/pinJSONToIPFS"
PINATA_PIN_FILE_URL = "https://api.pinata.cloud/pinning/pinFileToIPFS"

# In-memory storage for offline / simulation mode
_SIMULATED_IPFS_STORE: Dict[str, Dict[str, Any]] = {}


def _generate_simulated_cid(content_bytes: bytes) -> str:
    """Generates a deterministic simulated IPFS CID (Qm...) from content hash."""
    sha256 = hashlib.sha256(content_bytes).hexdigest()
    return f"Qm{sha256[:44]}"


def get_ipfs_gateway_url(cid: str) -> str:
    """Returns the full public or dedicated gateway URL for an IPFS CID."""
    base = settings.PINATA_GATEWAY_URL.rstrip("/")
    return f"{base}/{cid}"


async def upload_json_to_ipfs(data: Dict[str, Any], record_name: str = "medical_record.json") -> Dict[str, Any]:
    """
    Pins arbitrary JSON data (FHIR bundle, triage summary, diagnosis report) to IPFS via Pinata.
    Falls back to deterministic local simulation if PINATA_JWT is not set.
    """
    jwt = settings.PINATA_JWT.strip()
    json_bytes = json.dumps(data, sort_keys=True).encode("utf-8")

    if not jwt:
        cid = _generate_simulated_cid(json_bytes)
        gateway_url = get_ipfs_gateway_url(cid)
        _SIMULATED_IPFS_STORE[cid] = {
            "name": record_name,
            "data": data,
            "type": "application/json",
            "simulated": True
        }
        logger.info(f"[Pinata SIMULATED] Pinned JSON to simulated IPFS. CID: {cid}")
        return {
            "cid": cid,
            "ipfs_hash": cid,
            "gateway_url": gateway_url,
            "simulated": True,
            "status": "pinned"
        }

    # Real Pinata HTTP Request
    payload = {
        "pinataMetadata": {
            "name": record_name,
            "keyvalues": {"type": "medical-json", "service": "Sanjeevni-OS"}
        },
        "pinataContent": data
    }
    headers = {
        "Authorization": f"Bearer {jwt}",
        "Content-Type": "application/json"
    }

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(PINATA_PIN_JSON_URL, json=payload, headers=headers)
            if resp.status_code in (200, 201):
                res_data = resp.json()
                cid = res_data.get("IpfsHash", "")
                return {
                    "cid": cid,
                    "ipfs_hash": cid,
                    "gateway_url": get_ipfs_gateway_url(cid),
                    "simulated": False,
                    "status": "pinned"
                }
            else:
                logger.warning(f"Pinata API returned status {resp.status_code}: {resp.text}. Falling back to simulation.")
    except Exception as e:
        logger.error(f"Pinata upload exception: {e}. Falling back to simulation.")

    # Fallback if network/token failure
    cid = _generate_simulated_cid(json_bytes)
    _SIMULATED_IPFS_STORE[cid] = {"name": record_name, "data": data, "type": "application/json", "simulated": True}
    return {
        "cid": cid,
        "ipfs_hash": cid,
        "gateway_url": get_ipfs_gateway_url(cid),
        "simulated": True,
        "status": "pinned_fallback"
    }


async def upload_file_to_ipfs(file_bytes: bytes, filename: str, content_type: str = "application/pdf") -> Dict[str, Any]:
    """
    Pins raw file bytes (PDF, Image, ECG) to IPFS via Pinata.
    Falls back to deterministic local simulation if PINATA_JWT is not set.
    """
    jwt = settings.PINATA_JWT.strip()

    if not jwt:
        cid = _generate_simulated_cid(file_bytes)
        gateway_url = get_ipfs_gateway_url(cid)
        _SIMULATED_IPFS_STORE[cid] = {
            "name": filename,
            "bytes_len": len(file_bytes),
            "type": content_type,
            "simulated": True
        }
        logger.info(f"[Pinata SIMULATED] Pinned file {filename} ({len(file_bytes)} bytes) -> CID: {cid}")
        return {
            "cid": cid,
            "ipfs_hash": cid,
            "gateway_url": gateway_url,
            "simulated": True,
            "status": "pinned"
        }

    headers = {"Authorization": f"Bearer {jwt}"}
    files = {
        "file": (filename, file_bytes, content_type),
        "pinataMetadata": (None, json.dumps({"name": filename, "keyvalues": {"service": "Sanjeevni-OS"}}))
    }

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(PINATA_PIN_FILE_URL, files=files, headers=headers)
            if resp.status_code in (200, 201):
                res_data = resp.json()
                cid = res_data.get("IpfsHash", "")
                return {
                    "cid": cid,
                    "ipfs_hash": cid,
                    "gateway_url": get_ipfs_gateway_url(cid),
                    "simulated": False,
                    "status": "pinned"
                }
            else:
                logger.warning(f"Pinata File API returned {resp.status_code}: {resp.text}. Using simulation fallback.")
    except Exception as e:
        logger.error(f"Pinata File upload failed: {e}. Using simulation fallback.")

    cid = _generate_simulated_cid(file_bytes)
    _SIMULATED_IPFS_STORE[cid] = {"name": filename, "bytes_len": len(file_bytes), "type": content_type, "simulated": True}
    return {
        "cid": cid,
        "ipfs_hash": cid,
        "gateway_url": get_ipfs_gateway_url(cid),
        "simulated": True,
        "status": "pinned_fallback"
    }


def get_simulated_record(cid: str) -> Optional[Dict[str, Any]]:
    """Retrieves simulated record from in-memory cache."""
    return _SIMULATED_IPFS_STORE.get(cid)
