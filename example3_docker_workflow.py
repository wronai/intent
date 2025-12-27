#!/usr/bin/env python3
"""
Example 3: Complete Docker Workflow
====================================

Demonstruje pełny workflow z Docker:
1. Uruchomienie wszystkich serwisów
2. Generowanie kodu przez API REST
3. Generowanie przez MQTT
4. Walidacja i cache
5. Zapisywanie do bazy danych

Uruchomienie:
    # 1. Uruchom Docker
    docker-compose up -d

    # 2. Uruchom przykład
    docker-compose exec intentforge python examples/example3_docker_workflow.py

    # Lub lokalnie:
    python examples/example3_docker_workflow.py
"""

import asyncio
import json
import os
import sys
import time
from dataclasses import dataclass

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@dataclass
class WorkflowConfig:
    """Configuration for the workflow"""

    api_url: str = "http://localhost:8000"
    mqtt_host: str = "localhost"
    mqtt_port: int = 1883
    redis_url: str = "redis://localhost:6379/0"
    db_host: str = "localhost"
    db_port: int = 5432


def print_section(title: str):
    """Print formatted section header"""
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print("=" * 60)


def print_step(step: int, description: str):
    """Print step indicator"""
    print(f"\n[Step {step}] {description}")
    print("-" * 40)


# =============================================================================
# 1. REST API Example
# =============================================================================


def example_rest_api(config: WorkflowConfig):
    """Generate code using REST API"""
    import httpx

    print_section("1. REST API Code Generation")

    # Prepare request
    intent_data = {
        "description": "Create REST API for blog posts with categories, tags, and comments",
        "intent_type": "api_endpoint",
        "target_platform": "python_fastapi",
        "context": {
            "tables": ["posts", "categories", "tags", "comments"],
            "auth_required": True,
            "pagination": True,
        },
        "constraints": [
            "Use async SQLAlchemy",
            "Include proper error handling",
            "Add OpenAPI documentation",
        ],
    }

    print_step(1, "Sending intent to API")
    print(f"URL: {config.api_url}/api/generate")
    print(f"Intent: {intent_data['description'][:50]}...")

    try:
        response = httpx.post(f"{config.api_url}/api/generate", json=intent_data, timeout=60.0)
        response.raise_for_status()

        result = response.json()

        print_step(2, "Response received")
        print(f"Status: {'✓ Success' if result.get('success') else '✗ Failed'}")
        print(f"Language: {result.get('language', 'unknown')}")
        print(f"Code length: {len(result.get('generated_code', ''))} chars")
        print(f"Validation: {'✓ Passed' if result.get('validation_passed') else '✗ Failed'}")
        print(f"Processing time: {result.get('processing_time_ms', 0):.2f}ms")

        if result.get("generated_code"):
            print("\n📝 Generated code (first 500 chars):")
            print("-" * 40)
            print(result["generated_code"][:500] + "...")

        return result

    except httpx.ConnectError:
        print("⚠️  Could not connect to API. Is Docker running?")
        print("   Run: docker-compose up -d")
        return None
    except Exception as e:
        print(f"❌ Error: {e}")
        return None


# =============================================================================
# 2. MQTT Example
# =============================================================================


def example_mqtt(config: WorkflowConfig):
    """Generate code using MQTT"""
    import uuid

    import paho.mqtt.client as mqtt

    print_section("2. MQTT Code Generation")

    result = {"received": False, "data": None}
    request_id = str(uuid.uuid4())
    client_id = f"example-{int(time.time())}"

    def on_connect(client, userdata, flags, rc):
        print(f"Connected to MQTT broker (rc={rc})")
        client.subscribe(f"intentforge/intent/response/{client_id}")

    def on_message(client, userdata, msg):
        try:
            payload = json.loads(msg.payload.decode())
            if payload.get("request_id") == request_id:
                result["received"] = True
                result["data"] = payload
        except Exception as e:
            print(f"Error parsing message: {e}")

    # Create client
    client = mqtt.Client(client_id=client_id)
    client.on_connect = on_connect
    client.on_message = on_message

    print_step(1, "Connecting to MQTT broker")
    print(f"Host: {config.mqtt_host}:{config.mqtt_port}")

    try:
        client.connect(config.mqtt_host, config.mqtt_port, 60)
        client.loop_start()
        time.sleep(1)  # Wait for connection

        # Send intent
        intent = {
            "request_id": request_id,
            "description": "Create IoT temperature sensor handler for ESP32 with DHT22",
            "intent_type": "firmware_function",
            "target_platform": "esp32_micropython",
            "context": {"sensor": "DHT22", "gpio_pin": 4, "mqtt_topic": "sensors/temperature"},
        }

        print_step(2, "Publishing intent")
        print(f"Topic: intentforge/intent/request/{client_id}")
        print(f"Intent: {intent['description'][:50]}...")

        client.publish(f"intentforge/intent/request/{client_id}", json.dumps(intent), qos=1)

        # Wait for response
        print_step(3, "Waiting for response...")
        timeout = 30
        start = time.time()

        while not result["received"] and (time.time() - start) < timeout:
            time.sleep(0.1)

        client.loop_stop()
        client.disconnect()

        if result["received"]:
            data = result["data"]
            print(f"Status: {'✓ Success' if data.get('success') else '✗ Failed'}")

            if data.get("generated_code"):
                print("\n📝 Generated firmware code (first 500 chars):")
                print("-" * 40)
                print(data["generated_code"][:500] + "...")
        else:
            print("⚠️  Timeout waiting for response")
            print("   Make sure IntentForge worker is running")

        return result["data"]

    except Exception as e:
        print(f"❌ MQTT Error: {e}")
        print("   Is Mosquitto running? docker-compose up -d mqtt")
        return None


# =============================================================================
# 3. Cache Example
# =============================================================================


def example_cache(config: WorkflowConfig):
    """Demonstrate caching behavior"""
    import httpx

    print_section("3. Caching Demonstration")

    intent_data = {
        "description": "Simple hello world endpoint",
        "intent_type": "api_endpoint",
        "target_platform": "python_fastapi",
    }

    print_step(1, "First request (cache miss)")

    try:
        start = time.time()
        response = httpx.post(f"{config.api_url}/api/generate", json=intent_data, timeout=60.0)
        first_time = (time.time() - start) * 1000
        first_result = response.json()

        print(f"Time: {first_time:.2f}ms")
        print(f"Cache hit: {first_result.get('metadata', {}).get('from_cache', False)}")

        print_step(2, "Second request (cache hit expected)")

        start = time.time()
        response = httpx.post(f"{config.api_url}/api/generate", json=intent_data, timeout=60.0)
        second_time = (time.time() - start) * 1000
        second_result = response.json()

        print(f"Time: {second_time:.2f}ms")
        print(f"Cache hit: {second_result.get('metadata', {}).get('from_cache', False)}")

        print_step(3, "Comparison")
        speedup = first_time / second_time if second_time > 0 else 0
        print(f"First request:  {first_time:.2f}ms")
        print(f"Second request: {second_time:.2f}ms")
        print(f"Speedup: {speedup:.1f}x faster with cache")

        return {"first": first_result, "second": second_result}

    except Exception as e:
        print(f"❌ Error: {e}")
        return None


# =============================================================================
# 4. Batch Generation Example
# =============================================================================


def example_batch(config: WorkflowConfig):
    """Generate multiple related endpoints"""
    import httpx

    print_section("4. Batch Generation (CRUD for E-commerce)")

    entities = [
        ("products", "product catalog with search, filtering, and categories"),
        ("orders", "order management with status tracking and payment integration"),
        ("customers", "customer profiles with addresses and preferences"),
        ("inventory", "stock management with low-stock alerts"),
    ]

    results = []

    for i, (entity, description) in enumerate(entities, 1):
        print_step(i, f"Generating {entity} API")

        try:
            response = httpx.post(
                f"{config.api_url}/api/generate",
                json={
                    "description": f"Create REST API for {description}",
                    "intent_type": "api_endpoint",
                    "target_platform": "python_fastapi",
                    "context": {"table": entity},
                },
                timeout=60.0,
            )

            result = response.json()
            code_lines = len(result.get("generated_code", "").split("\n"))

            print(f"  Entity: {entity}")
            print(f"  Status: {'✓' if result.get('success') else '✗'}")
            print(f"  Lines of code: {code_lines}")

            results.append(
                {"entity": entity, "success": result.get("success"), "lines": code_lines}
            )

        except Exception as e:
            print(f"  ❌ Error: {e}")
            results.append({"entity": entity, "success": False, "error": str(e)})

    # Summary
    print("\n" + "-" * 40)
    print("📊 Batch Generation Summary:")
    total_lines = sum(r.get("lines", 0) for r in results)
    successful = sum(1 for r in results if r.get("success"))
    print(f"  Successful: {successful}/{len(results)}")
    print(f"  Total lines generated: {total_lines}")

    return results


# =============================================================================
# 5. Full Workflow Demo
# =============================================================================


async def full_workflow_demo():
    """Run complete workflow demonstration"""

    print("\n" + "=" * 60)
    print("  IntentForge - Complete Docker Workflow Demo")
    print("=" * 60)
    print("\nThis demo shows the full capabilities of IntentForge:")
    print("  1. REST API code generation")
    print("  2. MQTT-based generation")
    print("  3. Caching behavior")
    print("  4. Batch generation")

    # Configuration from environment or defaults
    config = WorkflowConfig(
        api_url=os.getenv("API_URL", "http://localhost:8000"),
        mqtt_host=os.getenv("MQTT_HOST", "localhost"),
        mqtt_port=int(os.getenv("MQTT_PORT", "1883")),
        redis_url=os.getenv("REDIS_URL", "redis://localhost:6379/0"),
    )

    print("\n📍 Configuration:")
    print(f"  API URL:    {config.api_url}")
    print(f"  MQTT:       {config.mqtt_host}:{config.mqtt_port}")
    print(f"  Redis:      {config.redis_url}")

    input("\nPress Enter to start the demo...")

    # Run examples
    results = {}

    # 1. REST API
    results["rest"] = example_rest_api(config)
    input("\nPress Enter to continue to MQTT example...")

    # 2. MQTT
    results["mqtt"] = example_mqtt(config)
    input("\nPress Enter to continue to cache example...")

    # 3. Cache
    results["cache"] = example_cache(config)
    input("\nPress Enter to continue to batch example...")

    # 4. Batch
    results["batch"] = example_batch(config)

    # Final summary
    print_section("Demo Complete!")
    print("\n📊 Summary:")
    print(f"  REST API:  {'✓' if results.get('rest') else '✗'}")
    print(f"  MQTT:      {'✓' if results.get('mqtt') else '✗'}")
    print(f"  Caching:   {'✓' if results.get('cache') else '✗'}")
    print(f"  Batch:     {'✓' if results.get('batch') else '✗'}")

    print("\n🎉 Thank you for trying IntentForge!")
    print("   Documentation: https://intentforge.readthedocs.io")
    print("   GitHub: https://github.com/wronai/intent")

    return results


# =============================================================================
# Quick Tests (without full services)
# =============================================================================


def quick_local_test():
    """Test without Docker - just library functionality"""

    print_section("Quick Local Test (no Docker required)")

    from intentforge.patterns import FullstackPatterns, PatternConfig, PatternType
    from intentforge.validator import CodeValidator

    # Test 1: Validator
    print_step(1, "Code Validation")

    validator = CodeValidator()

    good_code = '''
def hello(name: str) -> str:
    """Say hello"""
    return f"Hello, {name}!"
'''

    bad_code = """
import os
os.system("rm -rf /")  # Dangerous!
"""

    good_result = validator.validate_sync(good_code, "python")
    bad_result = validator.validate_sync(bad_code, "python")

    print(f"Good code valid: {good_result.is_valid} (security: {good_result.security_score})")
    print(f"Bad code valid: {bad_result.is_valid} (security: {bad_result.security_score})")

    if bad_result.errors:
        print(f"  Detected: {bad_result.errors[0].message[:60]}...")

    # Test 2: Pattern generation
    print_step(2, "Pattern Generation")

    config = PatternConfig(
        pattern_type=PatternType.CRUD_API,
        target_table="tasks",
        fields=[
            {"name": "title", "type": "text", "required": True},
            {"name": "description", "type": "textarea"},
            {"name": "status", "type": "select", "options": ["todo", "doing", "done"]},
            {"name": "due_date", "type": "date"},
        ],
    )

    files = FullstackPatterns.form_to_database(config)

    print(f"Generated {len(files)} files:")
    for name, content in files.items():
        lines = len(content.split("\n"))
        print(f"  - {name}: {lines} lines")

    # Test 3: Schema validation
    print_step(3, "Schema Validation")

    from intentforge.schema_registry import SchemaType, get_registry

    registry = get_registry()

    form_data = {
        "form_id": "test-form",
        "fields": [{"name": "email", "type": "email", "required": True}],
    }

    result = registry.validate(form_data, SchemaType.FORM_DATA)
    print(f"Form schema valid: {result.is_valid}")

    print("\n✓ All local tests passed!")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="IntentForge Docker Workflow Example")
    parser.add_argument("--quick", action="store_true", help="Run quick local test only")
    parser.add_argument("--rest", action="store_true", help="Run REST API example only")
    parser.add_argument("--mqtt", action="store_true", help="Run MQTT example only")
    parser.add_argument("--cache", action="store_true", help="Run cache example only")
    parser.add_argument("--batch", action="store_true", help="Run batch example only")

    args = parser.parse_args()

    config = WorkflowConfig()

    if args.quick:
        quick_local_test()
    elif args.rest:
        example_rest_api(config)
    elif args.mqtt:
        example_mqtt(config)
    elif args.cache:
        example_cache(config)
    elif args.batch:
        example_batch(config)
    else:
        # Run full demo
        asyncio.run(full_workflow_demo())
