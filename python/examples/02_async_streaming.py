"""
Example 2: Asynchronous Model Loading and Streaming Generation.
"""

import asyncio
from mobilellm import Engine, SamplingParams


async def main() -> None:
    # Use asynchronous client context manager
    async with Engine.async_client(use_native=True) as engine:
        print("Connected to MobileLLM asynchronously.")
        health = await engine.health()
        print(f"Backend: {health.backend} | Version: {health.version}")

        # Asynchronously load model weights
        model = await engine.load_model("models/smollm2-135m-instruct-q4_k_m.gguf")
        print(f"Loaded: {model.id}")

        # Async unary generation
        res = await model.generate("What is quantum computing?")
        print(f"Unary response:\n{res.text}\n")

        # Async streaming token generation
        print("Async Streaming:\n", end="")
        async for token in model.generate_stream(
            "List 3 features of async programming in Python.",
            SamplingParams(max_tokens=96, temperature=0.6),
        ):
            print(token, end="", flush=True)
        print("\n")

        await model.unload()


if __name__ == "__main__":
    asyncio.run(main())

