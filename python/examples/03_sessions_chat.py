"""
Example 3: Stateful Multi-Turn Conversational Sessions.
"""

from mobilellm import Engine, SamplingParams, SessionConfig


def main() -> None:
    with Engine(use_native=True) as engine:
        model = engine.load_model("models/smollm2-135m-instruct-q4_k_m.gguf")

        # Create session with KV-cache preservation across turns
        session_config = SessionConfig(max_context_tokens=1024, enable_prefix_caching=True)
        session = model.create_session(session_config)

        # Turn 1
        print("User: My favorite fruit is mango. Remember this.")
        r1 = session.generate("My favorite fruit is mango. Remember this.", SamplingParams(max_tokens=32))
        print(f"Assistant: {r1.text.strip()}\n")

        # Turn 2
        print("User: What is my favorite fruit?")
        r2 = session.generate("What is my favorite fruit?", SamplingParams(max_tokens=32))
        print(f"Assistant: {r2.text.strip()}\n")

        session.close()
        model.unload()


if __name__ == "__main__":
    main()

