"""
End-to-end tests for jina-airgapped inference server.
Run against a live server: TEST_URL=http://localhost:8080 python tests/test_e2e.py
"""

import os
import sys
import json
import time
import urllib.request
import urllib.error

BASE_URL = os.environ.get("TEST_URL", "http://localhost:8080")

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"


def request(method, path, data=None, headers=None):
    url = f"{BASE_URL}{path}"
    body = json.dumps(data).encode() if data else None
    h = {"Content-Type": "application/json", **(headers or {})}
    req = urllib.request.Request(url, data=body, headers=h, method=method)
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())
    except Exception as e:
        return 0, {"error": str(e)}


def test_health():
    status, body = request("GET", "/health")
    assert status == 200, f"Expected 200, got {status}"
    assert body.get("status") == "ok", f"Expected ok, got {body}"
    assert body.get("ready"), "Model not ready"
    print(f"  {PASS} /health -> model={body.get('model', '?')} device={body.get('device', '?')}")


def test_embeddings_basic():
    status, body = request("POST", "/v1/embeddings", {
        "input": ["Hello world", "Jina AI embeddings"],
        "model": "test",
    })
    assert status == 200, f"Expected 200, got {status}: {body}"
    data = body["data"]
    assert len(data) == 2, f"Expected 2 embeddings, got {len(data)}"
    emb = data[0]["embedding"]
    assert isinstance(emb, list) and len(emb) > 0, "Empty embedding"
    assert abs(sum(x**2 for x in emb) - 1.0) < 0.01, "Embedding not normalized"
    print(f"  {PASS} /v1/embeddings -> dim={len(emb)}, normalized")


def test_embeddings_matryoshka():
    target_dim = 128
    status, body = request("POST", "/v1/embeddings", {
        "input": ["Test matryoshka truncation"],
        "dimensions": target_dim,
    })
    assert status == 200, f"Expected 200, got {status}"
    emb = body["data"][0]["embedding"]
    assert len(emb) == target_dim, f"Expected dim {target_dim}, got {len(emb)}"
    print(f"  {PASS} /v1/embeddings Matryoshka -> dim={len(emb)}")


def test_embeddings_tasks():
    """Test v5 text task parameters (text models support: retrieval, text-matching, clustering, classification)."""
    for task in ["retrieval", "text-matching", "clustering", "classification"]:
        status, body = request("POST", "/v1/embeddings", {
            "input": ["Test task parameter"],
            "task": task,
        })
        assert status == 200, f"Task {task} failed: {status}: {body}"
    print(f"  {PASS} /v1/embeddings task parameter works (all 4 text tasks)")


def test_throughput_reported():
    """Test that tok/s is reported in usage and health endpoint."""
    # Run a batch to populate stats
    texts = [f"This is sentence number {i} for throughput testing." for i in range(10)]
    status, body = request("POST", "/v1/embeddings", {"input": texts})
    assert status == 200, f"Expected 200, got {status}"

    usage = body.get("usage", {})
    assert "tok_per_s" in usage, f"tok_per_s missing from usage: {usage}"
    tok_per_s = usage["tok_per_s"]
    assert isinstance(tok_per_s, (int, float)) and tok_per_s > 0, \
        f"tok_per_s should be positive, got {tok_per_s}"

    # Check prompt_tokens uses real tokenizer counts (not just word split)
    assert "prompt_tokens" in usage, "prompt_tokens missing"
    assert usage["prompt_tokens"] > 0, "prompt_tokens should be > 0"

    print(f"  {PASS} tok/s reported in usage -> {tok_per_s:.1f} tok/s | {usage['prompt_tokens']} tokens")

    # Check health also reports throughput
    status2, health = request("GET", "/health")
    assert status2 == 200
    assert "throughput" in health, f"throughput missing from /health: {health}"
    tp = health["throughput"]
    assert "avg_tok_per_s" in tp and tp["avg_tok_per_s"] > 0, \
        f"avg_tok_per_s missing or zero: {tp}"
    print(f"  {PASS} /health throughput -> avg={tp['avg_tok_per_s']:.1f} tok/s peak={tp['peak_tok_per_s']:.1f} tok/s")


def test_throughput_gpu_vs_cpu():
    """Benchmark: embed 100 texts and report tok/s."""
    texts = [
        "The quick brown fox jumps over the lazy dog. " * 5 + f" Sentence {i}."
        for i in range(100)
    ]
    t0 = time.time()
    status, body = request("POST", "/v1/embeddings", {"input": texts})
    wall_time = time.time() - t0
    assert status == 200, f"Expected 200, got {status}"

    tok_per_s = body["usage"].get("tok_per_s", 0)
    n_tokens = body["usage"].get("prompt_tokens", 0)
    print(f"  {PASS} Batch 100 texts: {n_tokens} tokens in {wall_time:.2f}s wall -> {tok_per_s:.1f} tok/s (encode only)")


def wait_for_ready(timeout=180):
    print(f"  Waiting for server at {BASE_URL}...")
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            status, body = request("GET", "/health")
            if status == 200 and body.get("ready"):
                return True
        except Exception:
            pass
        time.sleep(3)
    return False


def main():
    print(f"\n=== Jina AI v5 Air-Gapped E2E Tests ===")
    print(f"URL: {BASE_URL}\n")

    if not wait_for_ready():
        print(f"{FAIL} Server not ready after 180s")
        sys.exit(1)

    tests = [
        test_health,
        test_embeddings_basic,
        test_embeddings_matryoshka,
        test_embeddings_tasks,
        test_throughput_reported,
        test_throughput_gpu_vs_cpu,
    ]

    print("Running tests...\n")
    failed = 0
    for test in tests:
        try:
            test()
        except AssertionError as e:
            print(f"  {FAIL} {test.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"  {FAIL} {test.__name__}: unexpected error: {e}")
            failed += 1

    print(f"\n{'='*40}")
    total = len(tests)
    passed = total - failed
    if failed == 0:
        print(f"\033[32mAll {total} tests passed!\033[0m")
    else:
        print(f"\033[31m{failed}/{total} tests failed\033[0m")
        sys.exit(1)


if __name__ == "__main__":
    main()
