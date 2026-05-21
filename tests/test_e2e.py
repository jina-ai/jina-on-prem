"""
End-to-end tests for jina-airgapped inference server.
Run against a live container: TEST_URL=http://localhost:8080 python tests/test_e2e.py
"""

import os
import sys
import json
import time
import urllib.request
import urllib.error

BASE_URL = os.environ.get("TEST_URL", "http://localhost:8080")
MODEL_TYPE = os.environ.get("JINA_MODEL_TYPE", "embedding")

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"


def request(method, path, data=None, headers=None):
    url = f"{BASE_URL}{path}"
    body = json.dumps(data).encode() if data else None
    h = {"Content-Type": "application/json", **(headers or {})}
    req = urllib.request.Request(url, data=body, headers=h, method=method)
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
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
    print(f"  {PASS} /health -> {body.get('model', '?')} on {body.get('device', '?')}")


def test_embeddings():
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
    for task in ["retrieval.query", "retrieval.passage", "text-matching"]:
        status, body = request("POST", "/v1/embeddings", {
            "input": ["Test task parameter"],
            "task": task,
        })
        assert status == 200, f"Task {task} failed: {status}"
    print(f"  {PASS} /v1/embeddings task parameter works")


def test_rerank():
    status, body = request("POST", "/v1/rerank", {
        "query": "What is machine learning?",
        "documents": [
            "Machine learning is a subset of artificial intelligence.",
            "Python is a programming language.",
            "Deep learning uses neural networks.",
        ],
        "top_n": 2,
    })
    assert status == 200, f"Expected 200, got {status}: {body}"
    results = body["results"]
    assert len(results) == 2, f"Expected 2 results, got {len(results)}"
    assert "relevance_score" in results[0], "Missing relevance_score"
    assert results[0]["relevance_score"] >= results[1]["relevance_score"], "Not sorted by score"
    print(f"  {PASS} /v1/rerank -> top score={results[0]['relevance_score']:.3f}")


def test_chat_completions():
    status, body = request("POST", "/v1/chat/completions", {
        "messages": [
            {"role": "user", "content": "<html><body><h1>Hello</h1><p>World</p></body></html>"}
        ],
        "max_tokens": 64,
    })
    assert status == 200, f"Expected 200, got {status}: {body}"
    choices = body["choices"]
    assert len(choices) > 0, "No choices returned"
    content = choices[0]["message"]["content"]
    assert content, "Empty response"
    print(f"  {PASS} /v1/chat/completions -> {repr(content[:60])}")


def wait_for_ready(timeout=120):
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
    print(f"\n=== Jina AI Air-Gapped E2E Tests ===")
    print(f"URL: {BASE_URL} | Model type: {MODEL_TYPE}\n")

    if not wait_for_ready():
        print(f"{FAIL} Server not ready after 120s")
        sys.exit(1)

    print("Running tests...\n")

    tests = {
        "embedding": [test_health, test_embeddings, test_embeddings_matryoshka, test_embeddings_tasks],
        "reranker": [test_health, test_rerank],
        "reader": [test_health, test_chat_completions],
    }

    run_tests = tests.get(MODEL_TYPE, [test_health])

    failed = 0
    for test in run_tests:
        try:
            test()
        except AssertionError as e:
            print(f"  {FAIL} {test.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"  {FAIL} {test.__name__}: unexpected error: {e}")
            failed += 1

    print(f"\n{'='*40}")
    total = len(run_tests)
    passed = total - failed
    if failed == 0:
        print(f"\033[32mAll {total} tests passed!\033[0m")
    else:
        print(f"\033[31m{failed}/{total} tests failed\033[0m")
        sys.exit(1)


if __name__ == "__main__":
    main()
