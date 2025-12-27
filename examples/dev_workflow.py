#!/usr/bin/env python3
"""
Developer Workflow Example: Caching & Reuse
===========================================

This example demonstrates how to use IntentForge in a developer workflow:
1. Generate code (caching enabled)
2. Save it to a local file
3. Dynamically import and use it

This allows "Generative Caching" - once you generate a good result,
it's cached instantly for subsequent runs, making the dev loop fast.
"""

import asyncio
import importlib.util
import os
import sys

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from intentforge import Intent, IntentForge, IntentType, TargetPlatform


async def generate_and_save_module(forge: IntentForge, name: str, description: str):
    """Generates code and saves it as a reusable module"""

    print(f"\n[1] Processing module: '{name}'")
    print(f"    Intent: {description}")

    intent = Intent(
        description=description,
        intent_type=IntentType.WORKFLOW,
        target_platform=TargetPlatform.GENERIC_PYTHON,
        context={"function_name": name},
    )

    import time

    start = time.time()
    result = await forge.process_intent(intent)
    duration = time.time() - start

    if not result.success or not result.validation_passed:
        print(f"❌ Failed: {result.validation_errors}")
        print(f"FAILED CODE:\n---\n{result.generated_code}\n---")
        return None

    is_cached = result.metadata.get("from_cache", False)
    status = "⚡ CACHE HIT" if is_cached else "🤖 GENERATED"
    print(f"    Status: {status} ({duration:.3f}s)")

    # Save to file
    os.makedirs("generated_libs", exist_ok=True)
    file_path = f"generated_libs/{name}.py"

    with open(file_path, "w") as f:
        f.write(result.generated_code)

    print(f"    Saved to: {file_path}")
    return file_path


def import_module_from_path(module_name, file_path):
    """Dynamically imports a module from a file path"""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


async def main():
    print("=" * 60)
    print("IntentForge Dev Workflow Example")
    print("=" * 60)

    # Initialize Once (Cache lives here)
    # Initialize Once (Cache lives here)
    # Default to Ollama for local dev if not specified
    provider = os.getenv("LLM_PROVIDER", "ollama")
    model = os.getenv("LLM_MODEL", "llama3.1:8b")

    print(f"Using Provider: {provider}, Model: {model}")

    forge = IntentForge(enable_auto_deploy=False, sandbox_mode=True, provider=provider, model=model)

    # Define a task
    func_name = "fibonacci"
    description = "Function to calculate the nth fibonacci number recursively"

    # --- Run 1: Generation ---
    print("\n--- RUN 1: Generating Code ---")
    path = await generate_and_save_module(forge, func_name, description)

    if path:
        # Use it
        lib = import_module_from_path(func_name, path)
        print("\n[2] Testing generated module:")
        try:
            # Usage depends on generated code
            # Note: The LLM might name the function differently, so we try a few guesses or inspect
            func = getattr(lib, "fibonacci", getattr(lib, "fib", None))
            if func:
                result = func(10)
                print("    Input: 10")
                print(f"    Result: {result}")
            else:
                print("    Could not find expected function name in generated code.")
                print(f"    Available: {dir(lib)}")
        except Exception as e:
            print(f"    Runtime Error: {e}")
            print(
                "    (Code might need manual tweak if LLM messed up, but caching allows quick iteration)"
            )

    # --- Run 2: Simulation of next dev cycle ---
    print("\n--- RUN 2: Re-running (Expect Cache Hit) ---")
    # This should be instant
    await generate_and_save_module(forge, func_name, description)

    print("\n✅ Workflow Complete")


if __name__ == "__main__":
    asyncio.run(main())
