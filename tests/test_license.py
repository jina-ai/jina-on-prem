"""
Unit tests for the license-key gate. No server, no Docker, no network.

Run:
  python tests/test_license.py
"""

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "server"))
import license as L  # noqa: E402

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"

_results = []


def check(name, cond):
    _results.append(cond)
    print(f"  [{PASS if cond else FAIL}] {name}")


def main():
    print("License gate unit tests\n")

    # Valid key round-trips.
    k = L.issue("acme", 30, "*")
    ok, reason, payload = L.validate(k)
    check("valid key -> ok", ok and reason == "ok")
    check("valid key claims sub", payload.get("sub") == "acme")

    # Expired key (0 days => exp == iat == now => now >= exp).
    ke = L.issue("acme", 0, "*")
    ok, reason, _ = L.validate(ke)
    check("expired key -> license_expired", (not ok) and reason == "license_expired")

    # Tampered signature.
    ok, reason, _ = L.validate(k[:-3] + "AAA")
    check("tampered sig -> bad_signature", (not ok) and reason == "bad_signature")

    # Tampered payload (flip a char in the token body) must fail signature.
    body = k[len(L.PREFIX):]
    tok, _, sig = body.partition(".")
    mangled = L.PREFIX + tok[:-1] + ("A" if tok[-1] != "A" else "B") + "." + sig
    ok, reason, _ = L.validate(mangled)
    check("tampered payload -> not valid", not ok)

    # Malformed input.
    check("garbage -> malformed", L.validate("JINA-notbase64.sig")[1] in ("malformed_license", "bad_signature"))
    check("no dot -> malformed", L.validate("JINA-abc")[1] == "malformed_license")
    check("empty -> no_license", L.validate("")[1] == "no_license")

    # Model scoping.
    km = L.issue("acme", 30, "jina-embeddings-v5-text-nano")
    check("scope match (short)", L.validate(km, "jinaai/jina-embeddings-v5-text-nano")[0])
    check("scope match (full)", L.validate(km, "jina-embeddings-v5-text-nano")[0])
    check("scope mismatch", L.validate(km, "jinaai/jina-reranker-v3")[1] == "model_not_licensed")
    check("wildcard scope any model", L.validate(k, "jinaai/anything")[0])

    # Secret rotation: a key minted with a custom secret must fail the default,
    # and pass only when the env secret matches.
    kc = L.issue("acme", 30, "*", secret="custom-secret")
    check("custom-secret key fails default", not L.validate(kc)[0])
    os.environ["JINA_LICENSE_SECRET"] = "custom-secret"
    check("custom-secret key passes matching env", L.validate(kc)[0])
    os.environ.pop("JINA_LICENSE_SECRET", None)

    # Empty env secret falls back to default (mirrors Dockerfile ENV="" case).
    os.environ["JINA_LICENSE_SECRET"] = ""
    check("empty env secret uses default", L.validate(k)[0])
    os.environ.pop("JINA_LICENSE_SECRET", None)

    # status() shape.
    st = L.status(k, "jinaai/jina-embeddings-v5-text-nano")
    check("status valid + days_left", st["valid"] and st["days_left"] > 29)

    # inspect() decodes without verifying.
    check("inspect decodes sub", L.inspect(k)["sub"] == "acme")

    print()
    total, passed = len(_results), sum(_results)
    if passed == total:
        print(f"{PASS}: {passed}/{total}")
        sys.exit(0)
    print(f"{FAIL}: {passed}/{total}")
    sys.exit(1)


if __name__ == "__main__":
    main()
