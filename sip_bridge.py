import base64
import hashlib
import hmac
import json
import os
import re
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import PlainTextResponse, Response

router = APIRouter(prefix="/twilio", tags=["twilio-sip-bridge"])


@dataclass
class SipBridgeSettings:
    # Public base URL for webhooks.
    public_base_url: str

    # LiveKit SIP bridge settings.
    livekit_sip_domain: str
    livekit_sip_username: str
    livekit_sip_password: str
    livekit_inbound_trunk_id: str
    livekit_outbound_trunk_id: str
    allowed_source_numbers: list[str]  # List of allowed source numbers
    room_prefix: str = "coldcall-"


class TwilioSipBridge:
    def __init__(self, settings: SipBridgeSettings) -> None:
        self.settings = settings

    @staticmethod
    def _normalize_number(number: str) -> str:
        return re.sub(r"[^0-9+]", "", number or "")

    def is_allowed_source(self, from_number: str) -> bool:
        source = self._normalize_number(from_number)
        for allowed in self.settings.allowed_source_numbers:
            if source == self._normalize_number(allowed):
                return True
        return False

    def generate_room_name(self, from_number: str, call_sid: str) -> str:
        # A deterministic, SIP-safe room slug for dispatch rules and diagnostics.
        normalized = re.sub(r"[^0-9]", "", from_number or "unknown")[-8:] or "anon"
        sid_tail = re.sub(r"[^A-Za-z0-9]", "", call_sid or "")[-6:] or "nosid"
        return f"{self.settings.room_prefix}{normalized}-{sid_tail}-{int(time.time())}"

    def build_livekit_sip_uri(self, room_name: str) -> str:
        # SIP URI user part is derived from room name so dispatch rules can map to rooms.
        sip_user = urllib.parse.quote(room_name, safe="")
        return f"sip:{sip_user}@{self.settings.livekit_sip_domain};transport=tls"

    def build_dispatch_rule_payload(self) -> dict[str, Any]:
        # LiveKit dispatch rule: route SIP INVITE to a unique room under room_prefix.
        return {
            "name": "twilio-inbound-dynamic-room",
            "trunk_ids": [self.settings.livekit_inbound_trunk_id],
            "rule": {
                "dispatchRuleIndividual": {
                    "roomPrefix": self.settings.room_prefix,
                }
            },
            "roomConfig": {
                "agents": [{"agentName": "calling-agent"}],
            },
            "metadata": {
                "allowed_from": self.settings.allowed_twilio_source_number,
                "notes": "Webhook verifies Twilio source and maps caller into prefixed room.",
            },
        }

    def build_outbound_trunk_payload(self) -> dict[str, Any]:
        # Outbound trunk payload that LiveKit uses for PSTN/SIP egress via Twilio.
        return {
            "name": "twilio-outbound-trunk",
            "trunk": {
                "destination_country": "US",
                "numbers": [self.settings.twilio_caller_id],
                "auth_username": self.settings.livekit_sip_username,
                "auth_password": self.settings.livekit_sip_password,
            },
            "provider": "twilio",
            "source": "script-generated",
            "trunk_id": self.settings.livekit_outbound_trunk_id,
        }

    def build_sip_participant_payload(self, room_name: str, to_number: str) -> dict[str, Any]:
        sip_uri = self.build_livekit_sip_uri(room_name)
        return {
            "identity": f"pstn-{re.sub(r'[^0-9]', '', to_number or '') or 'prospect'}",
            "name": "Prospect PSTN Participant",
            "room_name": room_name,
            "sip": {
                "sip_uri": sip_uri,
                "username": self.settings.livekit_sip_username,
                "password": self.settings.livekit_sip_password,
                "outbound_trunk_id": self.settings.livekit_outbound_trunk_id,
            },
            "webrtc_bridge": {
                "audio_codec": "opus",
                "transport": "SRTP over DTLS",
                "notes": "SIP leg is bridged by LiveKit SFU into WebRTC room media.",
            },
        }

    def _twiml_dial_sip(self, room_name: str) -> str:
        sip_uri = self.build_livekit_sip_uri(room_name)
        # Twilio sends SIP INVITE to LiveKit URI using digest credentials.
        return (
            '<?xml version="1.0" encoding="UTF-8"?>'
            "<Response>"
            "<Dial answerOnBridge=\"true\">"
            f"<Sip username=\"{self.settings.livekit_sip_username}\" password=\"{self.settings.livekit_sip_password}\">"
            f"{sip_uri}</Sip>"
            "</Dial>"
            "</Response>"
        )

    def _twiml_reject(self, reason: str) -> str:
        safe_reason = (reason or "Unauthorized").replace("<", "").replace(">", "")
        return (
            '<?xml version="1.0" encoding="UTF-8"?>'
            "<Response>"
            f"<Say>{safe_reason}</Say>"
            "<Hangup/>"
            "</Response>"
        )

    # PTCL integration would use a different approach for outbound calls
    # This would be implemented based on PTCL's SIP API


bridge = TwilioSipBridge(
    SipBridgeSettings(
        public_base_url=os.getenv("PUBLIC_BASE_URL", "https://example.com"),
        livekit_sip_domain=os.getenv("LIVEKIT_SIP_DOMAIN", "sip.livekit.cloud"),
        livekit_sip_username=os.getenv("LIVEKIT_SIP_USERNAME", ""),
        livekit_sip_password=os.getenv("LIVEKIT_SIP_PASSWORD", ""),
        livekit_inbound_trunk_id=os.getenv("LIVEKIT_INBOUND_TRUNK_ID", ""),
        livekit_outbound_trunk_id=os.getenv("LIVEKIT_OUTBOUND_TRUNK_ID", ""),
        allowed_source_numbers=json.loads(os.getenv("ALLOWED_SOURCE_NUMBERS", "[]")),
        room_prefix=os.getenv("LIVEKIT_ROOM_PREFIX", "coldcall-"),
    )
)


@router.post("/webhook/incoming")
async def incoming_sip_webhook(request: Request) -> Response:
    form = await request.form()
    form_data = {k: str(v) for k, v in form.items()}

    # For PTCL, we would implement PTCL-specific validation here
    # For now, we'll just check if the source is allowed

    from_number = form_data.get("From", "")
    call_sid = form_data.get("CallSid", "")

    if not bridge.is_allowed_source(from_number):
        return Response(content=bridge._twiml_reject("Caller not authorized for this route."), media_type="application/xml")

    room_name = bridge.generate_room_name(from_number=from_number, call_sid=call_sid)
    return Response(content=bridge._twiml_dial_sip(room_name=room_name), media_type="application/xml")


@router.post("/webhook/outbound/answer")
async def outbound_answer_webhook(request: Request) -> Response:
    room_name = request.query_params.get("room") or bridge.generate_room_name(
        from_number=request.query_params.get("to", ""),
        call_sid=request.query_params.get("sid", ""),
    )
    # For PTCL, we would implement PTCL-specific response here
    # For now, we'll just return a basic response
    return Response(content="<?xml version=\"1.0\" encoding=\"UTF-8\"?><Response><Say>Connecting you now</Say></Response>", media_type="application/xml")


@router.post("/webhook/outbound/status")
async def outbound_status_webhook(request: Request) -> PlainTextResponse:
    # Twilio status callbacks can be used for analytics, retries, CRM events, etc.
    _ = await request.form()
    return PlainTextResponse("ok")


@router.post("/api/outbound/call")
async def create_outbound_call(payload: dict[str, str]) -> dict[str, Any]:
    to_number = payload.get("to", "").strip()
    if not to_number:
        raise HTTPException(status_code=400, detail="Field 'to' is required")

    room_name = bridge.generate_room_name(from_number=to_number, call_sid="outbound")

    # For PTCL, we would implement PTCL-specific outbound call creation here
    # For now, we'll just return a mock response
    return {
        "call": {
            "status": "initiated",
            "to": to_number,
            "from": "PTCL",
        },
        "livekit_room": room_name,
        "sip_participant_payload": bridge.build_sip_participant_payload(room_name, to_number),
    }


@router.get("/api/config/payloads")
async def get_sip_payloads() -> dict[str, Any]:
    example_room = bridge.generate_room_name(from_number=bridge.settings.allowed_twilio_source_number, call_sid="example")
    return {
        "dispatch_rule_payload": bridge.build_dispatch_rule_payload(),
        "outbound_trunk_payload": bridge.build_outbound_trunk_payload(),
        "sip_participant_payload": bridge.build_sip_participant_payload(example_room, bridge.settings.allowed_twilio_source_number),
    }


def print_configuration_payloads() -> None:
    example_room = bridge.generate_room_name(from_number=bridge.settings.allowed_twilio_source_number, call_sid="example")
    payload = {
        "dispatch_rule_payload": bridge.build_dispatch_rule_payload(),
        "outbound_trunk_payload": bridge.build_outbound_trunk_payload(),
        "sip_participant_payload": bridge.build_sip_participant_payload(example_room, bridge.settings.allowed_twilio_source_number),
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    print_configuration_payloads()
