"""
MobileLLM Command-Line Interface (mobilellm / mobilellm-cli).
"""

import argparse
import os
import sys
import time

from mobilellm.engine import Engine
from mobilellm.exceptions import SecurityViolationError
from mobilellm.models import ModelConfig, SamplingParams
from mobilellm.server.app import start_server


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mobilellm",
        description="MobileLLM Engine CLI — Mobile-First Edge AI Inference Framework",
    )
    parser.add_argument(
        "--endpoint",
        type=str,
        default=os.environ.get("MOBILELLM_ENDPOINT", "http://127.0.0.1:8080"),
        help="HTTP endpoint of MobileLLM daemon (default: http://127.0.0.1:8080)",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=os.environ.get("MOBILELLM_API_KEY"),
        help="Bearer API key for authenticated server connections",
    )
    parser.add_argument(
        "--native",
        action="store_true",
        help="Execute inference directly via in-process C-ABI (no HTTP server required)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="Request timeout in seconds (default: 60.0)",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # status
    subparsers.add_parser("status", help="Query engine health, runtime state, and hardware")

    # models
    subparsers.add_parser("models", help="List models loaded in the engine")

    # load
    load_parser = subparsers.add_parser("load", help="Load a GGUF model into runtime memory")
    load_parser.add_argument("model_path", type=str, help="Filesystem path to .gguf model")
    load_parser.add_argument("--context-len", type=int, default=2048, help="Context length tokens")
    load_parser.add_argument("--threads", type=int, default=4, help="CPU threads to allocate")

    # unload
    unload_parser = subparsers.add_parser("unload", help="Unload a model from runtime memory")
    unload_parser.add_argument("model_id", type=str, help="Model ID to unload")

    # generate
    gen_parser = subparsers.add_parser("generate", help="Generate text from a prompt")
    gen_parser.add_argument("prompt", type=str, help="Input prompt text")
    gen_parser.add_argument("--model", type=str, default=None, help="Model path or ID")
    gen_parser.add_argument("--max-tokens", type=int, default=128, help="Max tokens to generate")
    gen_parser.add_argument("--temp", type=float, default=0.7, help="Sampling temperature")
    gen_parser.add_argument("--top-p", type=float, default=0.9, help="Top-p sampling")
    gen_parser.add_argument("--no-stream", action="store_true", help="Disable streaming output")

    # benchmark
    bench_parser = subparsers.add_parser(
        "benchmark", help="Benchmark latency, TTFT, and throughput"
    )
    bench_parser.add_argument(
        "--model", type=str, required=True, help="Path or ID of model to benchmark"
    )
    bench_parser.add_argument(
        "--prompt",
        type=str,
        default="Explain the difference between compiled and interpreted languages.",
        help="Prompt text for benchmarking",
    )
    bench_parser.add_argument("--tokens", type=int, default=64, help="Tokens to generate per run")
    bench_parser.add_argument("--runs", type=int, default=3, help="Number of benchmark iterations")

    # serve
    serve_parser = subparsers.add_parser(
        "serve", help="Start local OpenAI-compatible loopback server"
    )
    serve_parser.add_argument(
        "--model", type=str, help="Path to initial GGUF model to load on startup"
    )
    serve_parser.add_argument(
        "--host", type=str, default="127.0.0.1", help="Host address (default: 127.0.0.1)"
    )
    serve_parser.add_argument(
        "--port", type=int, default=8080, help="Port to listen on (default: 8080)"
    )
    serve_parser.add_argument(
        "--allow-remote-connections",
        action="store_true",
        help="Allow binding to non-loopback interface (strictly requires --api-key)",
    )

    return parser


def get_engine(args: argparse.Namespace) -> Engine:
    if args.native:
        return Engine(use_native=True)
    return Engine(
        endpoint=args.endpoint,
        api_key=args.api_key,
        timeout=args.timeout,
    )


def cmd_status(engine: Engine) -> int:
    try:
        health = engine.health()
        hw = engine.capabilities()
        print("MobileLLM Runtime Status")
        print("========================")
        print(f"  Status:          {health.status.upper()}")
        print(f"  Engine Version:  {health.version}")
        print(f"  Backend:         {health.backend}")
        print(f"  Uptime:          {health.uptime_seconds:.1f}s")
        print(f"  Models Loaded:   {health.models_loaded}")
        print(f"  Hardware:        {hw}")
        return 0
    except Exception as e:
        print(f"Error checking status: {e}", file=sys.stderr)
        return 1


def cmd_models(engine: Engine) -> int:
    try:
        models = engine.list_models()
        if not models:
            print("No models currently loaded.")
            return 0

        print(f"{'ID':<32} {'ARCH':<10} {'QUANT':<10} {'CONTEXT':<8} {'SIZE (MB)':<10}")
        print("-" * 75)
        for m in models:
            size_mb = f"{m.file_size_bytes / (1024 * 1024):.1f}" if m.file_size_bytes else "N/A"
            print(
                f"{m.id:<32} {m.architecture:<10} {m.quantization:<10} {m.context_length:<8} {size_mb:<10}"
            )
        return 0
    except Exception as e:
        print(f"Error querying models: {e}", file=sys.stderr)
        return 1


def cmd_load(engine: Engine, args: argparse.Namespace) -> int:
    try:
        cfg = ModelConfig(
            model_path=args.model_path,
            context_length=args.context_len,
            n_threads=args.threads,
        )
        print(f"Loading model from '{args.model_path}'...")
        start = time.perf_counter()
        model = engine.load_model(cfg)
        elapsed = time.perf_counter() - start
        print(f"Model successfully loaded in {elapsed:.2f}s!")
        print(f"  Model ID:     {model.id}")
        print(f"  Architecture: {model.info.architecture}")
        print(f"  Quantization: {model.info.quantization}")
        print(f"  Context Len:  {model.info.context_length}")
        return 0
    except Exception as e:
        print(f"Error loading model: {e}", file=sys.stderr)
        return 1


def cmd_unload(engine: Engine, args: argparse.Namespace) -> int:
    try:
        res = engine.unload_model(args.model_id)
        if res:
            print(f"Model '{args.model_id}' successfully unloaded.")
            return 0
        else:
            print(f"Failed to unload model '{args.model_id}'.", file=sys.stderr)
            return 1
    except Exception as e:
        print(f"Error unloading model: {e}", file=sys.stderr)
        return 1


def cmd_generate(engine: Engine, args: argparse.Namespace) -> int:
    try:
        # Determine model
        model = None
        if args.model:
            # Check if it's a file path or model ID
            if os.path.isfile(args.model):
                model = engine.load_model(args.model)
            else:
                models = engine.list_models()
                match = next((m for m in models if m.id == args.model), None)
                if match:
                    model = engine._loaded_models.get(match.id) or engine.load_model(match.path)
                else:
                    model = engine.load_model(args.model)
        else:
            loaded = engine.list_models()
            if loaded:
                model = engine._loaded_models.get(loaded[0].id) or engine.load_model(loaded[0].path)
            else:
                print("Error: No model loaded and no --model specified.", file=sys.stderr)
                return 1

        params = SamplingParams(
            temperature=args.temp,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
        )

        if args.no_stream:
            resp = model.generate(args.prompt, params)
            print(resp.text)
        else:
            for token in model.generate_stream(args.prompt, params):
                print(token, end="", flush=True)
            print()
        return 0
    except Exception as e:
        print(f"Generation error: {e}", file=sys.stderr)
        return 1


def cmd_benchmark(engine: Engine, args: argparse.Namespace) -> int:
    try:
        print(f"Initializing benchmark with model: {args.model}")
        print(f"Runs: {args.runs} | Tokens per run: {args.tokens}")
        print("-" * 55)

        # Load model
        model = None
        if os.path.isfile(args.model):
            model = engine.load_model(args.model)
        else:
            model = engine.load_model(ModelConfig(model_path=args.model))

        params = SamplingParams(max_tokens=args.tokens, temperature=0.0)  # greedy for stability

        ttft_list: list[float] = []
        tps_list: list[float] = []
        total_times: list[float] = []

        for i in range(args.runs):
            t0 = time.perf_counter()
            first_token_time: float | None = None
            token_count = 0

            for _ in model.generate_stream(args.prompt, params):
                if first_token_time is None:
                    first_token_time = time.perf_counter()
                token_count += 1

            t_end = time.perf_counter()
            total_duration = t_end - t0
            ttft_ms = ((first_token_time - t0) * 1000.0) if first_token_time else 0.0
            decode_time = (t_end - first_token_time) if first_token_time else total_duration
            tps = (token_count / decode_time) if decode_time > 0 else 0.0

            ttft_list.append(ttft_ms)
            tps_list.append(tps)
            total_times.append(total_duration)
            print(
                f"Run {i + 1}: TTFT = {ttft_ms:.1f}ms | TPS = {tps:.2f} tok/s | Tokens = {token_count}"
            )

        avg_ttft = sum(ttft_list) / len(ttft_list)
        avg_tps = sum(tps_list) / len(tps_list)
        print("-" * 55)
        print("Benchmark Results Summary:")
        print(f"  Avg Time To First Token (TTFT): {avg_ttft:.2f} ms")
        print(f"  Avg Throughput:                 {avg_tps:.2f} tokens/sec")
        print(f"  Total Duration:                 {sum(total_times):.2f} s")
        return 0
    except Exception as e:
        print(f"Benchmark error: {e}", file=sys.stderr)
        return 1


def cmd_serve(args: argparse.Namespace) -> int:
    try:
        print(f"Starting MobileLLM loopback server on http://{args.host}:{args.port}")
        server = start_server(
            host=args.host,
            port=args.port,
            api_key=args.api_key,
            allow_remote=args.allow_remote_connections,
            initial_model=args.model,
        )
        print("Endpoints available:")
        print("  GET  /v1/health")
        print("  GET  /v1/models")
        print("  POST /v1/chat/completions")
        print("  POST /v1/models/load")
        print("  POST /v1/models/unload")
        print("Press Ctrl+C to terminate.")
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down MobileLLM server...")
        finally:
            server.shutdown_server()
        return 0
    except SecurityViolationError as e:
        print(f"Security Policy Violation: {e}", file=sys.stderr)
        return 2
    except Exception as e:
        print(f"Server startup error: {e}", file=sys.stderr)
        return 1


def main(argv: list[str] | None = None) -> int:
    parser = create_parser()
    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        return 0

    if args.command == "serve":
        return cmd_serve(args)

    engine = get_engine(args)
    try:
        if args.command == "status":
            return cmd_status(engine)
        elif args.command == "models":
            return cmd_models(engine)
        elif args.command == "load":
            return cmd_load(engine, args)
        elif args.command == "unload":
            return cmd_unload(engine, args)
        elif args.command == "generate":
            return cmd_generate(engine, args)
        elif args.command == "benchmark":
            return cmd_benchmark(engine, args)
        else:
            parser.print_help()
            return 1
    finally:
        engine.close()


if __name__ == "__main__":
    sys.exit(main())
