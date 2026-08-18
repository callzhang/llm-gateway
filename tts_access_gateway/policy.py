"""TTS-only route and payload policy."""

from __future__ import annotations

from typing import Mapping


MODEL = "qwen3-tts-1.7b-customvoice"
VOICES = frozenset(
    {
        "Vivian",
        "Serena",
        "Uncle_Fu",
        "Dylan",
        "Eric",
        "Ryan",
        "Aiden",
        "Ono_Anna",
        "Sohee",
    }
)
ALLOWED_FIELDS = frozenset(
    {
        "model",
        "input",
        "voice",
        "instructions",
        "response_format",
    }
)


class PolicyDenied(Exception):
    """The authenticated request is outside the TTS service contract."""


def validate_route(method: str, path: str) -> None:
    if (method.upper(), path) not in {
        ("POST", "/v1/audio/speech"),
        ("GET", "/v1/models"),
    }:
        raise PolicyDenied("route not allowed")


def validate_speech_payload(
    payload: Mapping[str, object],
) -> dict[str, object]:
    unsupported = sorted(set(payload) - ALLOWED_FIELDS)
    if unsupported:
        raise PolicyDenied(
            f"unsupported fields: {', '.join(unsupported)}"
        )
    if payload.get("model") != MODEL:
        raise PolicyDenied("model not allowed")
    if payload.get("voice") not in VOICES:
        raise PolicyDenied("voice not allowed")
    text = payload.get("input")
    if not isinstance(text, str) or not text.strip() or len(text) > 3000:
        raise PolicyDenied("input must contain 1-3000 characters")
    instructions = payload.get("instructions")
    if instructions is not None and not isinstance(instructions, str):
        raise PolicyDenied("instructions must be text")
    response_format = payload.get("response_format", "mp3")
    if response_format != "mp3":
        raise PolicyDenied("only MP3 is allowed")

    normalized = dict(payload)
    normalized["response_format"] = "mp3"
    return normalized


def model_listing() -> dict[str, object]:
    return {
        "object": "list",
        "data": [
            {
                "id": MODEL,
                "object": "model",
                "owned_by": "stardust",
            }
        ],
    }
