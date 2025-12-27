"""
IntentForge - Complete Usage Examples
Demonstrates form-to-database, MQTT integration, and NLP code generation
"""

# ============================================================================
# Example 1: Basic Intent to Code
# ============================================================================

from intentforge import Intent, IntentForge, IntentType, TargetPlatform


async def basic_example():
    """Generate API endpoint from natural language"""

    # Initialize forge
    forge = IntentForge(
        mqtt_broker="localhost", mqtt_port=1883, enable_auto_deploy=False, sandbox_mode=True
    )

    # Create intent
    intent = Intent(
        description="Create REST API endpoint to list users with pagination and search",
        intent_type=IntentType.API_ENDPOINT,
        target_platform=TargetPlatform.PYTHON_FASTAPI,
        context={
            "table": "users",
            "fields": ["id", "name", "email", "created_at"],
            "auth_required": True,
        },
        constraints=["Use async/await", "Include proper error handling", "Add rate limiting"],
    )

    # Process intent
    result = await forge.process_intent(intent)

    if result.success:
        print("Generated Code:")
        print(result.generated_code)
        print(f"\nValidation passed: {result.validation_passed}")
    else:
        print(f"Errors: {result.validation_errors}")


# ============================================================================
# Example 2: Form to Database Pattern
# ============================================================================

from intentforge import FullstackPatterns, PatternConfig, PatternType


def form_to_database_example():
    """Generate complete form-to-database integration"""

    config = PatternConfig(
        pattern_type=PatternType.FORM_TO_DATABASE,
        target_table="contacts",
        fields=[
            {"name": "name", "type": "text", "required": True},
            {"name": "email", "type": "email", "required": True},
            {"name": "phone", "type": "text", "required": False},
            {"name": "company", "type": "text", "required": False},
            {"name": "message", "type": "textarea", "required": True},
            {"name": "priority", "type": "select", "options": ["low", "medium", "high"]},
        ],
        auth_required=False,
        use_validation=True,
        framework="fastapi",
        include_tests=True,
    )

    # Generate all components
    result = FullstackPatterns.form_to_database(config)

    # Write generated files
    import os

    output_dir = "generated/contacts"
    os.makedirs(output_dir, exist_ok=True)

    for filename, content in result.items():
        ext = ".html" if "html" in filename else ".js" if "js" in filename else ".py"
        filepath = os.path.join(output_dir, f"{filename}{ext}")
        with open(filepath, "w") as f:
            f.write(content)
        print(f"Generated: {filepath}")

    return result


# ============================================================================
# Example 3: Schema Validation
# ============================================================================

from intentforge import SchemaType, get_registry


def schema_validation_example():
    """Validate data before processing"""

    registry = get_registry()

    # Validate form data
    form_data = {
        "form_id": "contact-form",
        "action": "/api/contacts",
        "method": "POST",
        "fields": [
            {"name": "name", "type": "text", "required": True},
            {"name": "email", "type": "email", "required": True},
        ],
        "database": {"table": "contacts", "operation": "insert"},
    }

    result = registry.validate(form_data, SchemaType.FORM_DATA)

    print(f"Form data valid: {result.is_valid}")
    if not result.is_valid:
        print(f"Errors: {result.errors}")

    # Validate and generate SQL
    is_valid, sql, params = registry.validate_form_to_sql(form_data)

    if is_valid:
        print(f"\nGenerated SQL: {sql}")
        print(f"Parameters: {params}")


# ============================================================================
# Example 4: Environment Configuration
# ============================================================================

from intentforge import EnvConfig, EnvHandler


def env_configuration_example():
    """Configure database from .env file"""

    # Create custom config
    config = EnvConfig(
        env_file=".env",
        prefix="APP_",
        required_vars=["DB_HOST", "DB_PASSWORD"],
        default_values={"DB_PORT": "5432", "DB_NAME": "intentforge"},
    )

    # Load environment
    env = EnvHandler(config)

    try:
        env.load()

        # Access values
        print(f"Database: {env.get_database_url()}")
        print(f"Debug mode: {env.get_bool('DEBUG', False)}")
        print(f"Pool size: {env.get_int('DB_POOL_SIZE', 5)}")

    except Exception as e:
        print(f"Config error: {e}")
        print("Run: make env-init to generate template")


# ============================================================================
# Example 5: MQTT Event-Driven Code Generation
# ============================================================================

import asyncio


async def mqtt_event_example():
    """Handle code generation requests via MQTT"""

    from intentforge import IntentForge

    forge = IntentForge(mqtt_broker="localhost", mqtt_port=1883, enable_auto_deploy=False)

    # Register event handlers
    @forge.register_observer("on_intent_received")
    def on_intent(intent):
        print(f"Received intent: {intent.description[:50]}...")

    @forge.register_observer("on_code_generated")
    def on_code(data):
        print(f"Generated {len(data['code'])} chars of {data['language']} code")

    @forge.register_observer("on_validation_complete")
    def on_validation(data):
        status = "✓" if data["valid"] else "✗"
        print(f"Validation {status}")

    # Start listening for MQTT messages
    forge.start()

    print("Listening for intents on MQTT...")
    print("Publish to: intentforge/intent/request/<client_id>")

    # Keep running
    try:
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        forge.stop()


# ============================================================================
# Example 6: Code Validation Pipeline
# ============================================================================

from intentforge import CodeValidator, ValidationLevel


def validation_example():
    """Validate generated code before use"""

    validator = CodeValidator(sandbox_mode=True)

    # Test Python code
    python_code = '''
import os
from sqlalchemy import select

async def get_users(session, limit: int = 50):
    """Get users with pagination"""
    query = select(User).limit(limit)
    result = await session.execute(query)
    return result.scalars().all()
'''

    result = validator.validate_sync(python_code, "python", ValidationLevel.FULL)

    print(f"Valid: {result.is_valid}")
    print(f"Security score: {result.security_score}/100")
    print(f"Complexity: {result.complexity_score}")

    if result.errors:
        print(f"Errors: {result.error_messages}")

    if result.warnings:
        print(f"Warnings: {[w.message for w in result.warnings]}")


# ============================================================================
# Example 7: Complete Workflow - Static HTML to Database
# ============================================================================

COMPLETE_WORKFLOW_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>IntentForge Demo</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
</head>
<body>
    <div class="container mt-5">
        <h1>Contact Form</h1>

        <!-- Form with intent declarations -->
        <form id="contact-form" data-intent="Handle contact form submission and store in database">
            <div class="mb-3">
                <label class="form-label">Name</label>
                <input type="text" name="name" class="form-control" required>
            </div>

            <div class="mb-3">
                <label class="form-label">Email</label>
                <input type="email" name="email" class="form-control" required>
            </div>

            <div class="mb-3">
                <label class="form-label">Message</label>
                <textarea name="message" class="form-control" rows="4" required></textarea>
            </div>

            <button type="submit" class="btn btn-primary">Send</button>
        </form>

        <!-- Code generation button -->
        <div class="mt-4">
            <button
                class="btn btn-secondary"
                data-intent="Generate Python API endpoint for this form"
                data-intent-event="click"
                data-intent-target="#generated-code"
                data-intent-platform="python_fastapi">
                Generate API Code
            </button>
        </div>

        <!-- Generated code output -->
        <div id="generated-code" class="mt-4"></div>
    </div>

    <!-- IntentForge Client -->
    <script data-intentforge-config='{"broker": "ws://localhost:9001", "autoConnect": true}'>
    </script>
    <script src="/static/js/intentforge-client.js"></script>

    <script>
        // Form submission via MQTT
        document.getElementById('contact-form').addEventListener('submit', async (e) => {
            e.preventDefault();

            const formData = new FormData(e.target);
            const data = Object.fromEntries(formData.entries());

            try {
                // Send via MQTT
                const result = await window.intentForge.publish(
                    'intentforge/forms/contacts',
                    {
                        action: 'create',
                        data: data,
                        request_id: crypto.randomUUID()
                    }
                );

                alert('Message sent successfully!');
                e.target.reset();

            } catch (error) {
                alert('Error: ' + error.message);
            }
        });
    </script>
</body>
</html>
"""


def generate_complete_example():
    """Generate complete working example"""
    import os

    os.makedirs("examples/complete", exist_ok=True)

    # Write HTML
    with open("examples/complete/index.html", "w") as f:
        f.write(COMPLETE_WORKFLOW_HTML)

    # Generate backend
    config = PatternConfig(
        pattern_type=PatternType.FORM_TO_DATABASE,
        target_table="contacts",
        fields=[
            {"name": "name", "type": "text", "required": True},
            {"name": "email", "type": "email", "required": True},
            {"name": "message", "type": "textarea", "required": True},
        ],
        auth_required=False,
        framework="fastapi",
    )

    result = FullstackPatterns.form_to_database(config)

    # Write backend files
    for filename, content in result.items():
        ext = (
            ".py"
            if "backend" in filename or "test" in filename
            else ".js"
            if "js" in filename
            else ".html"
            if "html" in filename
            else ".sql"
        )
        filepath = f"examples/complete/{filename}{ext}"
        with open(filepath, "w") as f:
            f.write(content)

    # Generate MQTT handler
    mqtt_handler = FullstackPatterns.mqtt_handler("contacts")
    with open("examples/complete/mqtt_handler.py", "w") as f:
        f.write(mqtt_handler)

    print("Complete example generated in examples/complete/")
    print("\nFiles created:")
    for f in os.listdir("examples/complete"):
        print(f"  - {f}")


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    import sys

    examples = {
        "basic": basic_example,
        "form": form_to_database_example,
        "schema": schema_validation_example,
        "env": env_configuration_example,
        "mqtt": mqtt_event_example,
        "validate": validation_example,
        "complete": generate_complete_example,
    }

    if len(sys.argv) < 2:
        print("IntentForge Examples")
        print("=" * 50)
        print("\nUsage: python examples.py <example_name>")
        print("\nAvailable examples:")
        for name, func in examples.items():
            doc = func.__doc__ or "No description"
            print(f"  {name:12} - {doc.strip().split(chr(10))[0]}")
        print("\nOr run: make demo")
    else:
        example_name = sys.argv[1]
        if example_name in examples:
            func = examples[example_name]
            if asyncio.iscoroutinefunction(func):
                asyncio.run(func())
            else:
                func()
        else:
            print(f"Unknown example: {example_name}")
            print(f"Available: {', '.join(examples.keys())}")
