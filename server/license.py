"""
Air-gapped license-key gate.

Design goals (deliberate, read before "hardening" this):

  1. TIME-SENSITIVE. A key carries an ``exp`` (unix seconds). After that
     instant the gate returns 403 on every inference endpoint. This is the
     whole point: give field/PM a knob that visibly expires without any
     phone-home.

  2. FULLY OFFLINE. Validation is a local HMAC check. No network, no clock
     server, no license file to sync. Works inside a disconnected container.

  3. ZERO REBUILD TO ISSUE / RENEW. The key travels at run time:
        docker run -e JINA_LICENSE_KEY=<key> ...
     Minting a new key (``jina-airgap.py keygen``) never touches the image.
     The signing secret is a build-arg with a public default, so even the
     secret does not require a rebuild to change.

  4. 防君子不防小人 ("keep honest folks honest, not a real lock"). The signing
     secret ships inside the image and defaults to a known constant, so
     anyone who reads this file can forge a key or set JINA_LICENSE_ENFORCE=0.
     That is intentional. This is a speed-bump for process compliance, NOT a
     cryptographic entitlement system. Do not sell it as DRM.

Key format (compact, copy-pasteable, no dots-in-the-middle surprises):

    JINA-<base64url(payload_json)>.<base64url(hmac_sha256(payload_json))>

payload_json = {"sub": "<customer>", "exp": <unix>, "iat": <unix>,
                "model": "<model-id or *>", "v": 1}
"""

import base64
import hashlib
import hmac
import json
import os
import time
from typing import Optional

# Public default secret. Overridable at build time (--build-arg
# LICENSE_SECRET=...) or run time (-e JINA_LICENSE_SECRET=...). Being public
# is a feature here, not a leak — see module docstring point 4.
DEFAULT_SECRET = "jina-airgap-symbolic-license-v1"

PREFIX = "JINA-"


def _secret() -> bytes:
    # Empty env (e.g. an unset --build-arg baked as "") falls back to the
    # public default so the CPU/GPU images stay consistent with keygen.
    return (os.environ.get("JINA_LICENSE_SECRET") or DEFAULT_SECRET).encode("utf-8")


def _b64e(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def _b64d(s: str) -> bytes:
    pad = "=" * (-len(s) % 4)
    return base64.urlsafe_b64decode(s + pad)


def _sign(payload_bytes: bytes) -> str:
    return _b64e(hmac.new(_secret(), payload_bytes, hashlib.sha256).digest())


def issue(sub: str, days: int, model: str = "*", secret: Optional[str] = None) -> str:
    """Mint a license key valid for ``days`` from now.

    Kept dependency-free and importable so ``jina-airgap.py keygen`` can call
    it directly without spinning up the server stack.
    """
    now = int(time.time())
    payload = {
        "sub": sub,
        "iat": now,
        "exp": now + int(days) * 86400,
        "model": model,
        "v": 1,
    }
    payload_bytes = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    if secret is not None:
        # Local override path for keygen --secret without mutating env.
        sig = _b64e(hmac.new(secret.encode("utf-8"), payload_bytes, hashlib.sha256).digest())
    else:
        sig = _sign(payload_bytes)
    return f"{PREFIX}{_b64e(payload_bytes)}.{sig}"


def inspect(key: str) -> dict:
    """Decode a key WITHOUT verifying the signature (for humans / logging)."""
    body = key[len(PREFIX):] if key.startswith(PREFIX) else key
    token, _, _sig = body.partition(".")
    return json.loads(_b64d(token))


def validate(key: Optional[str], model_id: str = "") -> tuple[bool, str, dict]:
    """Validate a key against secret, expiry, and (optionally) model scope.

    Returns (ok, reason, payload). ``reason`` is a short machine-ish string
    that doubles as the human-facing message. ``payload`` is the decoded
    claims dict (possibly empty on malformed input).
    """
    if not key:
        return False, "no_license", {}

    body = key[len(PREFIX):] if key.startswith(PREFIX) else key
    token, sep, sig = body.partition(".")
    if not sep or not token or not sig:
        return False, "malformed_license", {}

    try:
        payload_bytes = _b64d(token)
        payload = json.loads(payload_bytes)
    except Exception:
        return False, "malformed_license", {}

    expected = _sign(payload_bytes)
    if not hmac.compare_digest(expected, sig):
        return False, "bad_signature", payload

    exp = int(payload.get("exp", 0))
    now = int(time.time())
    if now >= exp:
        return False, "license_expired", payload

    scope = payload.get("model", "*")
    short = model_id.split("/")[-1] if model_id else ""
    if scope not in ("*", "", model_id, short):
        return False, "model_not_licensed", payload

    return True, "ok", payload


def status(key: Optional[str], model_id: str = "") -> dict:
    """Compact status block for /health and startup logs."""
    enforced = os.environ.get("JINA_LICENSE_ENFORCE", "1") == "1"
    ok, reason, payload = validate(key, model_id)
    out = {"enforced": enforced, "valid": ok, "reason": reason}
    if payload:
        out["licensed_to"] = payload.get("sub")
        exp = payload.get("exp")
        if exp:
            out["expires"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(int(exp)))
            out["days_left"] = max(0, round((int(exp) - time.time()) / 86400, 1))
    return out
