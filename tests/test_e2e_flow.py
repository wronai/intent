import os
import shutil
import sys

import pytest

# Add parent to path for importing intentforge
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from intentforge import Intent, IntentType, TargetPlatform
from intentforge.core import IntentForge


# Helper to clean up any generated artifacts after tests
@pytest.fixture
def cleanup_generated():
    yield
    if os.path.exists("test_generated"):
        shutil.rmtree("test_generated")


@pytest.mark.asyncio
async def test_e2e_generation_and_caching(cleanup_generated):
    """
    Test the full flow:
    1. Submit intent
    2. Generate code
    3. Verify result
    4. Submit same intent
    5. Verify cache hit
    """

    api_key = os.getenv("LLM_API_KEY", "dummy")

    # Initialize IntentForge with in-memory cache
    forge = IntentForge(api_key=api_key, enable_auto_deploy=False, sandbox_mode=True)

    # 1. Define Intent
    # Using generic intent type and platform
    intent = Intent(
        description="Create a function that adds two numbers",
        intent_type=IntentType.API_ENDPOINT,  # API_ENDPOINT is heavily tested/supported
        target_platform=TargetPlatform.PYTHON_FASTAPI,
        context={"function_name": "add_numbers"},
    )

    print("\n[Test] Submitting first intent...")
    result1 = await forge.process_intent(intent)

    # Assertions for first run
    assert result1.success, f"First run failed: {result1.validation_errors}"
    assert result1.generated_code is not None
    assert "add_numbers" in result1.generated_code or "add" in result1.generated_code
    # accessing metadata for cache status since .cached attribute doesn't exist on IntentResult
    assert result1.metadata.get("from_cache", False) is False

    # 3. Process Intent (Second Run - Should be Cached)
    print("\n[Test] Submitting second intent (expecting cache hit)...")
    result2 = await forge.process_intent(intent)

    # Assertions for second run
    assert result2.success
    assert result2.generated_code == result1.generated_code
    # This is the critical E2E check for caching
    assert result2.metadata.get("from_cache", False) is True

    print("[Test] E2E Flow passed!")
