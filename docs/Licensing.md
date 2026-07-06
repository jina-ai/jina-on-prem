Time-sensitive license keys for air-gapped deployments.

## What this is

A lightweight, offline entitlement gate. It gives you a **visible, expiring key** that field engineers and customers must set before the inference endpoints answer. It is designed to satisfy a "the deployment must be licensed and time-bound" process requirement without any phone-home, license server, or clock sync.

It is **not** DRM. The signing secret ships inside the image and a determined operator can bypass it (mint their own key, or set `JINA_LICENSE_ENFORCE=0`). That trade-off is deliberate: a hard cryptographic entitlement system would break the air-gap promise (no callbacks) and add operational weight nobody wants. Treat this as a compliance speed-bump, not a lock.

## How it works

- The key is a signed, self-contained token carrying an expiry (`exp`), issued-to (`sub`), and optional model scope.
- Validation is a local HMAC-SHA256 check. No network. Works fully disconnected.
- The key is injected **at run time**, so issuing or renewing a key **never requires rebuilding the image**.
- `/health` and the docs stay open (so Docker's healthcheck and a quick "is my key ok?" probe always work). All inference `POST` endpoints require a valid key.

## Issue a key

```bash
python jina-airgap.py keygen --sub acme-corp --days 30
# restrict to one model:
python jina-airgap.py keygen --sub acme --days 90 --model jina-embeddings-v5-text-nano
# machine-readable:
python jina-airgap.py keygen --sub trial --days 7 --json
```

The key prints to stdout. Hand it to the operator.

## Deploy with a key

```bash
docker run -e JINA_LICENSE_KEY=JINA-xxxxx.yyyyy -p 8080:8080 jina/MODEL:cpu
```

Or via the deploy helper / compose by exporting `JINA_LICENSE_KEY` in the environment.

Check status any time (no key needed for /health):

```bash
curl -s http://localhost:8080/health | python3 -m json.tool
# -> "license": {"enforced": true, "valid": true, "licensed_to": "acme-corp",
#                "expires": "...", "days_left": 27.4}
```

A gated request without a valid key returns HTTP 403:

```json
{"error": {"code": "license_expired",
           "message": "License expired. Request a renewed key (no rebuild needed).",
           "type": "license_error"}}
```

## Renew

Mint a new key and restart the container with the new `JINA_LICENSE_KEY`. No rebuild, no re-transfer of the image.

## Runtime knobs

| Env var | Default | Meaning |
|---|---|---|
| `JINA_LICENSE_KEY` | `""` | The key to present. Empty = gate fails closed. |
| `JINA_LICENSE_ENFORCE` | `1` | Set `0` to disable the gate entirely. |
| `JINA_LICENSE_SECRET` | public constant | HMAC signing secret. Rotate with `keygen --secret` + matching env / `--build-arg LICENSE_SECRET=`. |

## Rotating the secret (optional)

The default secret is public (see `server/license.py`). To use your own:

```bash
# bundle with a custom secret baked in
python jina-airgap.py bundle --model jina-embeddings-v5-text-nano --cpu-only \
  # (secret via docker --build-arg LICENSE_SECRET=... if you build directly)
# mint keys with the same secret
python jina-airgap.py keygen --sub acme --days 30 --secret my-secret
# or override at run time
docker run -e JINA_LICENSE_SECRET=my-secret -e JINA_LICENSE_KEY=... ...
```

This raises the bar slightly (a reader of the public image can't mint keys without the secret) but is still not a security boundary.
