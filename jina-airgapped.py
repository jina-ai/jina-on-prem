#!/usr/bin/env python3
"""
jina-airgapped - Air-Gapped Deployment Toolkit for Jina AI Models

Two phases:

  Phase 1 - BUNDLE (requires network):
    Build a self-contained Docker image bundle with model weights baked in.
    Run this on a machine with internet access.

  Phase 2 - DEPLOY (no network required):
    Load and run the bundle on a fully disconnected air-gapped machine.
    Zero external dependencies at runtime.

Commands:
  bundle  - [Phase 1, network] Build Docker image, bake in weights, save as .tar.gz
  deploy  - [Phase 2, offline] Load saved .tar.gz and start container
  serve   - Serve a model directly (no Docker, requires model deps installed)
  list    - List available models

Usage:
  python jina-airgapped.py bundle [--model MODEL_ID] [--output OUTPUT] [--hf-token TOKEN] [--cpu-only]
  python jina-airgapped.py deploy --image IMAGE_FILE [--port PORT] [--gpu]
  python jina-airgapped.py serve --model MODEL_ID [--port PORT] [--cpu-only]
  python jina-airgapped.py list
"""

import os
import sys
import json
import argparse
import subprocess
import textwrap
import tempfile
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()
CATALOG_PATH = SCRIPT_DIR / "models" / "catalog.json"

# ANSI colors (no deps)
BOLD = "\033[1m"
GREEN = "\033[32m"
CYAN = "\033[36m"
YELLOW = "\033[33m"
RED = "\033[31m"
RESET = "\033[0m"
DIM = "\033[2m"


def c(text, color):
    if sys.stdout.isatty():
        return f"{color}{text}{RESET}"
    return text


def load_catalog():
    with open(CATALOG_PATH) as f:
        return json.load(f)["models"]


def print_banner():
    banner = r"""
     _ _             _    ___  ___                      _
    | (_)           | |   |  \/  |                     | |
    | |_ _ __   __ _| |   | .  . |  ___  __| | ___| |___
    | | | '_ \ / _` | |   | |\/| | / _ \/ _` |/ _ \ / __|
    | | | | | | (_| |_|   | |  | ||  __/ (_| |  __/ \__ \
    |_|_|_| |_|\__,_(_)   \_|  |_/ \___|\__,_|\___|_|___/

      Air-Gapped Deployment Toolkit for Jina AI Models
    """
    if sys.stdout.isatty():
        print(c(banner, CYAN))
    else:
        print("=== Jina AI Air-Gapped Toolkit ===")


def modality_badge(m):
    mod = m.get("modality", "text")
    if mod == "multimodal":
        return c("[OMNI]", CYAN)
    elif mod == "code":
        return c("[CODE]", YELLOW)
    return c("[TEXT]", GREEN)


def list_models(models, verbose=False):
    print(f"\n{c('Available Models', BOLD)} ({len(models)} total, newest first)\n")
    for i, m in enumerate(models, 1):
        badge = modality_badge(m)
        release = m.get("release", "")
        print(f"  {c(str(i).rjust(2), DIM)}. {badge} {c(m['id'], BOLD)}  {c(release, DIM)}")
        print(f"       {m['description']}")
        if verbose:
            dim_info = f" | Dim: {m['output_dim']}" if "output_dim" in m else ""
            vram = m.get("vram_gb", "?")
            print(
                f"       HF: {m['hf_repo']} | Params: {m['parameters']} "
                f"| VRAM: ~{vram}GB{dim_info} | License: {m['license']}"
            )
        print()


def select_model_interactive(models):
    list_models(models, verbose=True)
    while True:
        try:
            choice = input(c("Select model (number): ", BOLD)).strip()
            idx = int(choice) - 1
            if 0 <= idx < len(models):
                return models[idx]
            print(c(f"  Invalid choice. Enter 1-{len(models)}", RED))
        except (ValueError, EOFError):
            print(c("  Invalid input", RED))
        except KeyboardInterrupt:
            print("\nCancelled.")
            sys.exit(0)


def select_runtime_interactive():
    print(f"\n{c('Runtime', BOLD)}")
    print(f"  {c('1', DIM)}. GPU (CUDA) - recommended, faster")
    print(f"  {c('2', DIM)}. CPU only - slower but no GPU required")
    while True:
        try:
            choice = input(c("Select runtime (1/2): ", BOLD)).strip()
            if choice == "1":
                return "gpu"
            elif choice == "2":
                return "cpu"
            print(c("  Enter 1 or 2", RED))
        except (ValueError, EOFError, KeyboardInterrupt):
            print("\nCancelled.")
            sys.exit(0)


def get_dockerfile_path(model):
    return SCRIPT_DIR / "docker" / "embeddings" / "Dockerfile"


def build_model_requirements(model: dict) -> str:
    """Generate a requirements.txt string from the model's deps field."""
    deps = model.get("deps", {})
    lines = []
    for pkg, spec in deps.items():
        if spec is None:
            # Optional dep - skip (not required)
            continue
        lines.append(f"{pkg}{spec}")
    return "\n".join(lines) + "\n" if lines else ""


def cmd_list(args):
    models = load_catalog()
    print_banner()
    list_models(models, verbose=True)


def cmd_bundle(args):
    """
    Phase 1 (network required): Build a self-contained Docker image with model
    weights baked in, then save to a portable .tar.gz bundle.
    Transfer this file to the air-gapped machine and run `deploy`.
    """
    models = load_catalog()
    print_banner()

    print(f"\n{c('Phase 1: BUNDLE', BOLD)} {c('(requires network)', YELLOW)}")
    print(f"  Downloads model weights and bakes them into a Docker image.")
    print(f"  Output: self-contained .tar.gz - no network needed at runtime.\n")

    # Select model
    if args.model:
        matches = [m for m in models if m["id"] == args.model]
        if not matches:
            print(c(f"Model '{args.model}' not found. Run 'list' to see available models.", RED))
            sys.exit(1)
        model = matches[0]
    else:
        model = select_model_interactive(models)

    print(f"\n{c('Selected:', BOLD)} {model['id']}")
    print(f"  Type: {model['type']} | Params: {model['parameters']} | VRAM: ~{model.get('vram_gb', '?')}GB")

    # Show model-specific deps
    deps = model.get("deps", {})
    if deps:
        required_deps = {k: v for k, v in deps.items() if v is not None}
        if required_deps:
            print(f"  Model deps: {', '.join(f'{k}{v}' for k, v in required_deps.items())}")

    # Select runtime
    if args.cpu_only:
        runtime = "cpu"
    else:
        runtime = select_runtime_interactive()

    # Docker image tag
    image_tag = f"jina/{model['id'].lower()}:{runtime}"
    output_file = args.output or f"jina-{model['id'].lower()}-{runtime}.tar.gz"

    print(f"\n{c('Bundle Plan:', BOLD)}")
    print(f"  Model:   {model['hf_repo']}")
    print(f"  Runtime: {runtime}")
    print(f"  Tag:     {image_tag}")
    print(f"  Output:  {output_file}")

    if not args.yes:
        try:
            confirm = input(c("\nProceed? [y/N]: ", BOLD)).strip().lower()
        except KeyboardInterrupt:
            print("\nCancelled.")
            sys.exit(0)
        if confirm not in ("y", "yes"):
            print("Cancelled.")
            sys.exit(0)

    dockerfile = get_dockerfile_path(model)
    if not dockerfile.exists():
        print(c(f"Dockerfile not found: {dockerfile}", RED))
        sys.exit(1)

    # Write model-specific requirements to Dockerfile context
    model_reqs_path = dockerfile.parent / "model-requirements.txt"
    model_reqs_content = build_model_requirements(model)
    if model_reqs_content:
        model_reqs_path.write_text(model_reqs_content)
        print(f"\n{c('Model-specific deps written to:', DIM)} {model_reqs_path.name}")
        print(c(f"  {model_reqs_content.strip()}", DIM))
    else:
        # Write empty file so Dockerfile COPY doesn't fail
        model_reqs_path.write_text("# no model-specific deps\n")

    # Build Docker image
    print(f"\n{c('Building Docker image...', CYAN)} (this downloads model weights, may take 10-30 min)")

    build_args = [
        "docker", "build",
        "-f", str(dockerfile),
        "--build-arg", f"MODEL_ID={model['hf_repo']}",
        "-t", image_tag,
    ]

    if args.hf_token:
        build_args += ["--build-arg", f"HF_TOKEN={args.hf_token}"]

    if runtime == "cpu":
        build_args += ["--build-arg", "BASE_IMAGE=python:3.11-slim"]

    build_args.append(str(SCRIPT_DIR))

    print(c(f"  Running: {' '.join(build_args[:6])} ...", DIM))

    result = subprocess.run(build_args)

    # Cleanup temp model-requirements.txt
    if model_reqs_path.exists():
        model_reqs_path.unlink()

    if result.returncode != 0:
        print(c("\nDocker build failed!", RED))
        sys.exit(1)

    print(c("\nBuild successful!", GREEN))

    # Save image to tar.gz
    print(f"\n{c('Saving bundle to:', CYAN)} {output_file}")
    print("  This may take a few minutes...")

    save_cmd = f"docker save {image_tag} | gzip > {output_file}"
    result = subprocess.run(save_cmd, shell=True)
    if result.returncode != 0:
        print(c("\nFailed to save bundle!", RED))
        sys.exit(1)

    size_mb = Path(output_file).stat().st_size / (1024 * 1024)
    print(c(f"\nBundle ready! Saved {size_mb:.0f} MB to: {output_file}", GREEN))

    print(f"\n{c('Transfer to air-gapped machine, then deploy:', BOLD)}")
    gpu_flag = "--gpus all" if runtime == "gpu" else ""
    print(f"\n  Option A - using this toolkit:")
    print(f"    python jina-airgapped.py deploy --image {output_file}{' --gpu' if runtime == 'gpu' else ''}")
    print(f"\n  Option B - raw Docker:")
    print(f"    docker load < {output_file}")
    print(f"    docker run {gpu_flag} -p 8080:8080 {image_tag}")
    print(f"\n  Test: curl http://localhost:8080/health")


def cmd_deploy(args):
    """
    Phase 2 (no network required): Load a .tar.gz bundle and start the container.
    Everything is self-contained - zero internet access needed.
    """
    image_file = args.image
    if not Path(image_file).exists():
        print(c(f"File not found: {image_file}", RED))
        sys.exit(1)

    print(f"\n{c('Phase 2: DEPLOY', BOLD)} {c('(no network required)', GREEN)}")
    print(f"  Loading air-gapped bundle and starting container.\n")

    print(f"{c('Loading bundle:', CYAN)} {image_file}")
    result = subprocess.run(["docker", "load", "-i", image_file], capture_output=True, text=True)
    if result.returncode != 0:
        print(c(f"Failed to load bundle:\n{result.stderr}", RED))
        sys.exit(1)

    image_name = None
    for line in result.stdout.splitlines():
        if "Loaded image:" in line:
            image_name = line.split("Loaded image:")[-1].strip()
            break

    if not image_name:
        print(c("Could not determine image name from load output.", YELLOW))
        print(result.stdout)
        image_name = input(c("Enter image name to run: ", BOLD)).strip()

    print(c(f"\nLoaded: {image_name}", GREEN))

    port = args.port or 8080
    gpu_flag = ["--gpus", "all"] if args.gpu else []

    run_cmd = ["docker", "run", "--rm", "-p", f"{port}:8080"] + gpu_flag + [image_name]
    print(f"\n{c('Starting container:', CYAN)}")
    print(c(f"  {' '.join(run_cmd)}", DIM))
    print(f"\n  API ready at: http://localhost:{port}/health")
    print(f"  Schemas: OpenAI (/v1/embeddings), Voyage (/v1/embeddings), Gemini (/v1/models/...),")
    print(f"           Cohere (/v1/embed), Reranker (/v1/rerank)\n")

    subprocess.run(run_cmd)


def cmd_serve(args):
    """Serve a model directly without Docker (requires local model files and deps)."""
    models = load_catalog()

    if args.model:
        matches = [m for m in models if m["id"] == args.model]
        if not matches:
            model_info = {
                "id": args.model,
                "type": "embedding",
                "hf_repo": args.model,
                "family": "embeddings",
            }
        else:
            model_info = matches[0]
    else:
        model_info = select_model_interactive(models)

    port = args.port or 8080

    env = os.environ.copy()
    env["JINA_MODEL_ID"] = args.local_path or model_info["hf_repo"]
    env["PORT"] = str(port)

    if args.cpu_only:
        env["CUDA_VISIBLE_DEVICES"] = ""

    server_script = SCRIPT_DIR / "server" / "app.py"
    if not server_script.exists():
        print(c(f"Server script not found: {server_script}", RED))
        sys.exit(1)

    print(f"\n{c('Starting server (no Docker):', CYAN)}")
    print(f"  Model: {env['JINA_MODEL_ID']}")
    print(f"  Port:  {port}")
    print(f"  Schemas: OpenAI, Voyage AI, Gemini, Cohere\n")

    subprocess.run([sys.executable, str(server_script)], env=env)


def main():
    parser = argparse.ArgumentParser(
        description="Jina AI Air-Gapped Deployment Toolkit",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
        Two-phase air-gap workflow:

          Phase 1 - BUNDLE (connected machine, needs internet):
            python jina-airgapped.py bundle
            python jina-airgapped.py bundle --model jina-embeddings-v5-text-nano --output jina-v5-nano.tar.gz
            python jina-airgapped.py bundle --model jina-embeddings-v5-text-small --cpu-only

          Phase 2 - DEPLOY (air-gapped machine, no internet):
            python jina-airgapped.py deploy --image jina-v5-nano.tar.gz --gpu
            python jina-airgapped.py deploy --image jina-v5-nano.tar.gz  # CPU

          Serve directly (no Docker, deps must be installed):
            python jina-airgapped.py serve --model jinaai/jina-embeddings-v5-text-nano --port 8080
            python jina-airgapped.py serve --local-path /data/models/jina-v5-nano

          List all available models:
            python jina-airgapped.py list
        """),
    )

    subparsers = parser.add_subparsers(dest="command")

    # bundle (Phase 1)
    bundle_p = subparsers.add_parser(
        "bundle",
        help="[Phase 1 - network required] Build and save Docker image bundle",
    )
    bundle_p.add_argument("--model", help="Model ID (from catalog)")
    bundle_p.add_argument("--output", "-o", help="Output .tar.gz file path")
    bundle_p.add_argument("--hf-token", dest="hf_token", help="HuggingFace token for gated models")
    bundle_p.add_argument("--cpu-only", action="store_true", help="Build CPU-only image")
    bundle_p.add_argument("-y", "--yes", action="store_true", help="Skip confirmation")

    # deploy (Phase 2)
    deploy_p = subparsers.add_parser(
        "deploy",
        help="[Phase 2 - no network] Load saved bundle and start container",
    )
    deploy_p.add_argument("--image", "-i", required=True, help="Path to .tar.gz bundle file")
    deploy_p.add_argument("--port", "-p", type=int, default=8080, help="Host port (default: 8080)")
    deploy_p.add_argument("--gpu", action="store_true", help="Enable GPU passthrough")

    # serve
    serve_p = subparsers.add_parser("serve", help="Serve model directly (no Docker)")
    serve_p.add_argument("--model", help="Model ID or HuggingFace repo")
    serve_p.add_argument("--local-path", dest="local_path", help="Local path to model files")
    serve_p.add_argument("--port", "-p", type=int, default=8080, help="Port (default: 8080)")
    serve_p.add_argument("--cpu-only", action="store_true", help="Force CPU inference")

    # list
    subparsers.add_parser("list", help="List available models")

    # Backward-compat aliases (hidden)
    pack_p = subparsers.add_parser("pack", help=argparse.SUPPRESS)
    pack_p.add_argument("--model")
    pack_p.add_argument("--output", "-o")
    pack_p.add_argument("--hf-token", dest="hf_token")
    pack_p.add_argument("--cpu-only", action="store_true")
    pack_p.add_argument("-y", "--yes", action="store_true")

    load_p = subparsers.add_parser("load", help=argparse.SUPPRESS)
    load_p.add_argument("--image", "-i", required=True)
    load_p.add_argument("--port", "-p", type=int, default=8080)
    load_p.add_argument("--gpu", action="store_true")

    args = parser.parse_args()

    if args.command in ("bundle", "pack"):
        cmd_bundle(args)
    elif args.command in ("deploy", "load"):
        cmd_deploy(args)
    elif args.command == "serve":
        cmd_serve(args)
    elif args.command == "list":
        cmd_list(args)
    else:
        print_banner()
        parser.print_help()
        print(f"\n{c('Quick start:', BOLD)}")
        print("  python jina-airgapped.py list    # see all 28 available models")
        print("  python jina-airgapped.py bundle  # Phase 1: build bundle (network required)")
        print("  python jina-airgapped.py deploy  # Phase 2: run on air-gapped machine")


if __name__ == "__main__":
    main()
