"""
Example 1: Synchronous Model Loading and Generation with MobileLLM.
"""

from mobilellm import Engine, SamplingParams


def main() -> None:
    # Connect to local HTTP server or native runtime directly
    # To connect to an active HTTP daemon (e.g. mobilellm serve or Android via adb forward):
    #   engine = Engine(endpoint="http://127.0.0.1:8080")
    # To run zero-network direct native inference on macOS/Linux:
    engine = Engine(use_native=True)
    engine.connect()

    print("MobileLLM Runtime Health:")
    print(engine.health())

    # Load model weights into memory
    print("\nLoading model...")
    model = engine.load_model("models/smollm2-135m-instruct-q4_k_m.gguf")
    print(f"Loaded: {model.id} ({model.info.architecture}, {model.info.quantization})")

    # 1. Unary Generation
    prompt = "Explain why the sky appears blue in one sentence."
    print(f"\nPrompt: {prompt}")
    response = model.generate(prompt, SamplingParams(max_tokens=64, temperature=0.7))
    print(f"Response: {response.text}")
    if response.metrics:
        print(f"Metrics: {response.metrics.completion_tokens} tokens @ {response.metrics.tokens_per_second:.1f} tok/s")

    # 2. Streaming Generation
    prompt_stream = "Write 3 bullet points about Python programming language."
    print(f"\nStreaming Prompt: {prompt_stream}\nOutput: ", end="")
    for token in model.generate_stream(prompt_stream, SamplingParams(max_tokens=128)):
        print(token, end="", flush=True)
    print("\n")

    # Clean unload
    model.unload()
    engine.close()


if __name__ == "__main__":
    main()

