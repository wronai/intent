import asyncio
import os
import sys
import time

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from intentforge.llm import LLMConfig, OllamaProvider

MODELS_TO_TEST = [
    "llama3.1:8b",
    "mistral:7b",
    "qwen2.5:7b",
    "deepseek-r1:8b",
    "llama3.2-vision:11b",
]


async def test_model(model_name):
    print(f"\n[{model_name}] Initializing...", end="", flush=True)

    # Check if model needs to be pulled (simple check by listing)
    # Note: listing all models might be slow, so we'll just try to use it and catch errors
    # or rely on Ollama to pull it if configured, but by default OllamaProvider might fail if not present.
    # For this script we assume they might need pulling or are present.
    # Let's just try to generate.

    try:
        config = LLMConfig(ollama_host=os.getenv("OLLAMA_HOST", "http://localhost:11434"))
        # We need to hack a bit because get_llm_provider handles the factory.
        # But we can instantiate OllamaProvider directly as in example_list_models

        provider = OllamaProvider(config)

        # There isn't a direct way to set the model on the provider instance after init generally,
        # but looking at the code in ollama_example.py:
        # llm = get_llm_provider("ollama", model="llama3")
        # So we should use get_llm_provider or just pass model to generate if supported.
        # Let's check how the provider uses the model.
        # Typically model is passed in generate or config.

        # Let's import get_llm_provider
        from intentforge.llm import get_llm_provider

        llm = get_llm_provider("ollama", model=model_name)

        print(" Generating...", end="", flush=True)
        start_time = time.time()

        # Simple generation
        response = await llm.generate(
            "Say hello and state your name.", system="You are a helpful assistant."
        )

        duration = time.time() - start_time

        print(f" Done ({duration:.2f}s)")
        print(f"    Response: {response.content.strip()[:100].replace('\n', ' ')}...")
        print(f"    Tokens: {response.total_tokens}")
        print(f"    Latency: {response.latency_ms:.0f}ms")
        return True

    except Exception as e:
        print(f" FAILED: {e}")
        return False


async def main():
    print("=" * 60)
    print("Testing Ollama Models")
    print("=" * 60)

    results = {}

    for model in MODELS_TO_TEST:
        success = await test_model(model)
        results[model] = "PASS" if success else "FAIL"

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    for model, status in results.items():
        print(f"{model:<25} {status}")


if __name__ == "__main__":
    asyncio.run(main())
