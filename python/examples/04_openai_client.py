"""
Example 4: Using MobileLLM with the Official OpenAI Python SDK.

Prerequisites:
  1. Start local MobileLLM loopback server:
       mobilellm serve --model models/smollm2-135m-instruct-q4_k_m.gguf --port 8080
  2. Install openai client:
       pip install openai
"""

from openai import OpenAI


def main() -> None:
    # Point standard OpenAI client to local loopback server
    client = OpenAI(
        base_url="http://127.0.0.1:8080/v1",
        api_key="not-needed",  # Default loopback binding requires no key
    )

    print("Sending request to local MobileLLM OpenAI-compatible server...")
    response = client.chat.completions.create(
        model="smollm2-135m-instruct-q4_k_m",
        messages=[
            {"role": "system", "content": "You are a concise AI assistant."},
            {"role": "user", "content": "Explain edge AI in two sentences."},
        ],
        stream=True,
    )

    for chunk in response:
        if chunk.choices and chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="", flush=True)
    print()


if __name__ == "__main__":
    main()

