#!/usr/bin/env python3
"""
IntentForge + Ollama Example
============================

Demonstruje użycie IntentForge z lokalnym LLM (Ollama).
Nie wymaga klucza API - wszystko działa lokalnie.

Setup:
    1. Install Ollama:
       curl -fsSL https://ollama.com/install.sh | sh

    2. Pull models:
       ollama pull llama3
       ollama pull codellama

    3. Run this example:
       python ollama_example.py

Or with Docker:
    docker-compose --profile ollama up -d
    docker exec intentforge-ollama ollama pull llama3
    python ollama_example.py
"""

import asyncio
import os
import sys

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from intentforge.llm import LLMConfig, OllamaProvider, generate, generate_code, get_llm_provider


async def example_direct_ollama():
    """Direct Ollama usage"""
    print("\n" + "=" * 60)
    print("1. Direct Ollama Provider")
    print("=" * 60)

    # Create Ollama provider
    llm = get_llm_provider("ollama", model="llama3")

    # Simple generation
    print("\n[Generating response...]")
    response = await llm.generate(
        "Explain microservices architecture in 3 sentences.",
        system="You are a software architect. Be concise.",
    )

    print(f"\nModel: {response.model}")
    print(f"Tokens: {response.total_tokens}")
    print(f"Latency: {response.latency_ms:.0f}ms")
    print(f"\nResponse:\n{response.content}")

    return response


async def example_code_generation():
    """Code generation with Ollama"""
    print("\n" + "=" * 60)
    print("2. Code Generation with CodeLlama")
    print("=" * 60)

    # Use CodeLlama for code
    llm = get_llm_provider("ollama", model="codellama")

    print("\n[Generating FastAPI endpoint...]")
    response = await llm.generate_code(
        "Create a REST API endpoint for user registration with email validation", language="python"
    )

    print(f"\nGenerated {len(response.content)} characters")
    print(f"Latency: {response.latency_ms:.0f}ms")
    print(f"\nCode:\n{response.content[:1000]}...")

    return response


async def example_via_litellm():
    """Using Ollama through LiteLLM"""
    print("\n" + "=" * 60)
    print("3. Ollama via LiteLLM (Universal API)")
    print("=" * 60)

    # LiteLLM with Ollama backend
    # Model format: "ollama/model_name"
    try:
        llm = get_llm_provider("litellm", model="ollama/llama3")

        print("\n[Generating via LiteLLM...]")
        response = await llm.generate(
            "Write a haiku about Python programming.", system="You are a poet who loves coding."
        )

        print(f"\nResponse:\n{response.content}")
        return response

    except ImportError:
        print("\n⚠️  LiteLLM not installed. Install with: pip install litellm")
        return None


async def example_streaming():
    """Streaming response from Ollama"""
    print("\n" + "=" * 60)
    print("4. Streaming Response")
    print("=" * 60)

    llm = get_llm_provider("ollama", model="llama3")

    print("\n[Streaming response...]")
    print("-" * 40)

    async for chunk in llm.generate_stream(
        "Write a short story about a robot learning to code. Make it 3 paragraphs.",
        system="You are a creative writer.",
    ):
        print(chunk, end="", flush=True)

    print("\n" + "-" * 40)
    print("[Stream complete]")


async def example_list_models():
    """List available Ollama models"""
    print("\n" + "=" * 60)
    print("5. Available Ollama Models")
    print("=" * 60)

    config = LLMConfig(ollama_host=os.getenv("OLLAMA_HOST", "http://localhost:11434"))
    ollama = OllamaProvider(config)

    try:
        models = await ollama.list_models()

        print("\nInstalled models:")
        for model in models:
            print(f"  - {model}")

        if not models:
            print("  (no models installed)")
            print("\n  To install models, run:")
            print("    ollama pull llama3")
            print("    ollama pull codellama")
            print("    ollama pull mistral")

        return models

    except Exception as e:
        print(f"\n⚠️  Could not connect to Ollama: {e}")
        print("  Make sure Ollama is running: ollama serve")
        return []


async def example_quick_helpers():
    """Using quick helper functions"""
    print("\n" + "=" * 60)
    print("6. Quick Helper Functions")
    print("=" * 60)

    # Set environment for helpers
    os.environ["LLM_PROVIDER"] = "ollama"
    os.environ["LLM_MODEL"] = "llama3"

    # Quick generate
    print("\n[Using generate() helper...]")
    result = await generate("What is Docker? One sentence only.")
    print(f"Result: {result}")

    # Quick code generation
    print("\n[Using generate_code() helper...]")
    code = await generate_code(
        "function to validate email address",
        language="python",
        model="codellama",  # Override model
    )
    print(f"Generated code:\n{code[:300]}...")


async def main():
    """Run all examples"""
    print("\n" + "#" * 60)
    print("  IntentForge + Ollama Examples")
    print("#" * 60)

    # Check Ollama connection
    models = await example_list_models()

    if not models:
        print("\n" + "!" * 60)
        print("  Ollama not available. Please install and start Ollama first.")
        print("!" * 60)
        print("\nInstallation:")
        print("  curl -fsSL https://ollama.com/install.sh | sh")
        print("  ollama pull llama3")
        print("  ollama pull codellama")
        return

    # Run examples
    await example_direct_ollama()

    if "codellama" in models or "codellama:latest" in models:
        await example_code_generation()
    else:
        print("\n⚠️  Skipping code generation - codellama not installed")
        print("  Run: ollama pull codellama")

    await example_via_litellm()
    await example_streaming()
    await example_quick_helpers()

    print("\n" + "=" * 60)
    print("✅ All examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
