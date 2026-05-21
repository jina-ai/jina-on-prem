#!/usr/bin/env python3
"""
Jina v5 Throughput Benchmark
Measures real tok/s for all v5 embedding models on CPU and GPU.
"""

import os
import time
import argparse
import platform
import subprocess
import sys
from typing import Optional

# Sentences of varying lengths to simulate real-world mixed input
BENCHMARK_CORPUS = [
    "The quick brown fox jumps over the lazy dog.",
    "Semantic search enables finding relevant content based on meaning rather than exact keyword matching.",
    "In recent years, transformer-based models have revolutionized natural language processing tasks.",
    "Jina AI provides state-of-the-art embedding models optimized for retrieval and semantic similarity.",
    "Air-gapped environments require all dependencies to be bundled and deployed without internet access.",
    "Hello world.",
    "This is a test.",
    "Machine learning models can be deployed on edge devices with limited computational resources.",
    "The NVIDIA L4 GPU offers 30.3 TFLOPS of FP16 Tensor performance in a low-power form factor suitable for inference workloads.",
    "Embeddings are dense vector representations that capture semantic meaning of text.",
    "Fine-tuning large language models requires significant computational resources and carefully curated datasets.",
    "Python is a versatile programming language widely used in data science and machine learning applications.",
    "Docker containers provide a consistent environment for deploying applications across different platforms.",
    "Vector databases enable efficient similarity search over millions of high-dimensional embeddings.",
    "The attention mechanism in transformer models allows the model to focus on relevant parts of the input sequence.",
    "Deep learning has achieved remarkable results in computer vision, speech recognition, and natural language processing.",
    "Retrieval-augmented generation combines the strengths of retrieval systems and generative language models.",
    "OK.",
    "Batch processing improves GPU utilization by processing multiple inputs simultaneously.",
    "The softmax function converts raw logits into probability distributions over the vocabulary.",
    "Quantization reduces model size and inference latency by using lower-precision arithmetic.",
    "Knowledge distillation transfers knowledge from a large teacher model to a smaller student model.",
    "Contrastive learning trains models to embed similar items close together and dissimilar items far apart.",
    "Sparse attention mechanisms reduce the quadratic complexity of standard self-attention.",
    "Multi-modal models process and align information from different modalities such as text and images.",
    "Yes.",
    "No.",
    "The embedding dimension determines the capacity of the model to encode semantic information.",
    "Normalized embeddings have unit length and enable cosine similarity to be computed efficiently.",
    "Cross-encoder models score pairs of texts jointly for higher accuracy at the cost of throughput.",
]

MODELS = [
    ("jinaai/jina-embeddings-v5-text-nano",  "239M"),
    ("jinaai/jina-embeddings-v5-text-small", "677M"),
    ("jinaai/jina-embeddings-v5-omni-nano",  "1.04B"),
    ("jinaai/jina-embeddings-v5-omni-small", "1.74B"),
]


def get_cpu_info():
    try:
        result = subprocess.run(
            ["lscpu"],
            capture_output=True, text=True
        )
        for line in result.stdout.splitlines():
            if "Model name" in line:
                return line.split(":")[1].strip()
    except Exception:
        pass
    return platform.processor() or "Unknown CPU"


def get_gpu_info():
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return "No GPU detected"


def benchmark_model(model_id: str, device: str, batch_size: int = 32, duration_s: float = 10.0):
    """Benchmark a model, returns tok/s."""
    print(f"\n  Loading {model_id} on {device}...")

    from sentence_transformers import SentenceTransformer
    from transformers import AutoTokenizer

    model = SentenceTransformer(
        model_id,
        trust_remote_code=True,
        device=device,
    )

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    except Exception as e:
        print(f"  Warning: tokenizer load failed ({e}), using word-split fallback")
        tokenizer = None

    def count_tokens(texts):
        if tokenizer is not None:
            enc = tokenizer(texts, add_special_tokens=True, truncation=True, max_length=512)
            return sum(len(ids) for ids in enc["input_ids"])
        return sum(len(t.split()) for t in texts)

    # Build batches from the corpus (cycle to fill batch)
    corpus = BENCHMARK_CORPUS
    batches = []
    for i in range(0, max(batch_size * 10, len(corpus)), batch_size):
        batch = [corpus[j % len(corpus)] for j in range(i, i + batch_size)]
        batches.append(batch)

    # Warm up: 3 batches
    print(f"  Warming up (3 batches)...")
    for b in batches[:3]:
        model.encode(b, task="retrieval", convert_to_numpy=True, normalize_embeddings=True)

    # Steady-state measurement: run for exactly duration_s seconds
    print(f"  Measuring for {duration_s}s...")
    total_tokens = 0
    batch_idx = 0
    t_start = time.perf_counter()
    t_end = t_start + duration_s

    while time.perf_counter() < t_end:
        batch = batches[batch_idx % len(batches)]
        model.encode(batch, task="retrieval", convert_to_numpy=True, normalize_embeddings=True)
        total_tokens += count_tokens(batch)
        batch_idx += 1

    elapsed = time.perf_counter() - t_start
    tok_per_s = total_tokens / elapsed

    avg_tokens = total_tokens / batch_idx if batch_idx > 0 else 0
    print(f"  Done: {batch_idx} batches | {total_tokens:,} tokens | {elapsed:.1f}s | {tok_per_s:,.0f} tok/s")
    print(f"  Avg tokens/batch: {avg_tokens:.1f}")

    # Cleanup
    del model
    import gc
    gc.collect()
    try:
        import torch
        if device == "cuda":
            torch.cuda.empty_cache()
    except Exception:
        pass

    return tok_per_s


def main():
    parser = argparse.ArgumentParser(description="Jina v5 Throughput Benchmark")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--duration", type=float, default=10.0, help="Steady-state measurement duration in seconds")
    parser.add_argument("--device", choices=["cpu", "cuda", "both"], default="both")
    parser.add_argument("--models", nargs="+", default=None, help="Subset of models to benchmark")
    args = parser.parse_args()

    print("=" * 70)
    print("Jina v5 Embedding Throughput Benchmark")
    print("=" * 70)

    cpu_info = get_cpu_info()
    gpu_info = get_gpu_info()
    print(f"CPU: {cpu_info}")
    print(f"GPU: {gpu_info}")
    print(f"Batch size: {args.batch_size}")
    print(f"Duration: {args.duration}s steady-state")

    import torch
    has_gpu = torch.cuda.is_available()
    print(f"CUDA available: {has_gpu}")

    models_to_run = MODELS
    if args.models:
        models_to_run = [(m, p) for m, p in MODELS if any(x in m for x in args.models)]

    results = {}  # model_id -> {"cpu": tok_s, "gpu": tok_s}

    devices = []
    if args.device in ("cpu", "both"):
        devices.append("cpu")
    if args.device in ("cuda", "both") and has_gpu:
        devices.append("cuda")

    for model_id, params in models_to_run:
        results[model_id] = {"params": params, "cpu": None, "gpu": None}
        for device in devices:
            print(f"\n[{model_id}] ({params}) on {device.upper()}")
            try:
                tok_s = benchmark_model(model_id, device, args.batch_size, args.duration)
                results[model_id][device] = tok_s
            except Exception as e:
                print(f"  ERROR: {e}")
                results[model_id][device] = None

    # Print summary table
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    # CPU info short
    cpu_short = cpu_info
    if "Intel" in cpu_info:
        # Extract model number
        parts = cpu_info.split()
        try:
            idx = parts.index("Platinum") if "Platinum" in parts else (parts.index("Gold") if "Gold" in parts else -1)
            if idx >= 0:
                cpu_short = f"Intel Xeon {parts[idx]} {parts[idx+1]}"
        except Exception:
            pass

    gpu_short = "NVIDIA L4"
    cpu_tflops = "?"
    gpu_tflops = "30.3"

    # Try to get actual CPU TFLOPS
    # For Intel Xeon 8481C: ~3.6 TFLOPS (FP32 peak ~4.4, but for reference)
    # We'll report as measured

    print(f"\nHardware:")
    print(f"  CPU: {cpu_info}")
    print(f"  GPU: {gpu_info}")

    print(f"\n{'Model':<45} {'CPU tok/s':>12} {'GPU tok/s':>12}")
    print("-" * 70)
    for model_id, data in results.items():
        cpu_val = f"{data['cpu']:,.0f}" if data['cpu'] is not None else "N/A"
        gpu_val = f"{data['gpu']:,.0f}" if data['gpu'] is not None else "N/A"
        print(f"{model_id:<45} {cpu_val:>12} {gpu_val:>12}")

    # Output machine-readable JSON
    import json
    output = {
        "hardware": {
            "cpu": cpu_info,
            "gpu": gpu_info,
        },
        "config": {
            "batch_size": args.batch_size,
            "duration_s": args.duration,
        },
        "results": results,
    }
    with open("/tmp/benchmark_results.json", "w") as f:
        json.dump(output, f, indent=2)
    print("\nResults saved to /tmp/benchmark_results.json")

    # Also print raw numbers for README update
    print("\n--- README TABLE FORMAT ---")
    print(f"CPU: {cpu_short} | GPU: NVIDIA L4 (30.3 TFLOPS FP16)")
    print()
    print("| Model | CPU tok/s | GPU tok/s |")
    print("|-------|-----------|-----------|")
    print(f"|       | {cpu_short} | NVIDIA L4 (30.3 TFLOPS FP16) |")
    for model_id, data in results.items():
        cpu_val = f"{data['cpu']:,.0f}" if data['cpu'] is not None else "N/A"
        gpu_val = f"{data['gpu']:,.0f}" if data['gpu'] is not None else "N/A"
        print(f"| {model_id} | {cpu_val} | {gpu_val} |")


if __name__ == "__main__":
    main()
