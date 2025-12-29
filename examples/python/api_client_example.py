#!/usr/bin/env python3
"""
IntentForge API Client Example
==============================

Demonstrates how to use IntentForge as a Python client:
1. REST API calls
2. MQTT real-time communication
3. Code generation workflows
4. Data operations

Usage:
    python api_client_example.py

    # Or with specific examples:
    python api_client_example.py --rest
    python api_client_example.py --mqtt
    python api_client_example.py --generate
"""

import argparse
import asyncio
import json
import os
import sys
from dataclasses import dataclass
from typing import Any

import httpx

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


@dataclass
class IntentForgeConfig:
    """Configuration for IntentForge client"""

    api_url: str = "http://localhost:8001"
    mqtt_host: str = "localhost"
    mqtt_port: int = 1883
    timeout: float = 60.0


class IntentForgeClient:
    """
    Python client for IntentForge API.

    Example:
        client = IntentForgeClient()

        # Generate code
        code = await client.generate("Create REST API for products")

        # Execute intent
        result = await client.intent("Send email to user@example.com")

        # CRUD operations
        users = await client.data("users").list()
    """

    def __init__(self, config: IntentForgeConfig | None = None):
        self.config = config or IntentForgeConfig()
        self._http = httpx.AsyncClient(timeout=self.config.timeout)

    async def close(self):
        await self._http.aclose()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.close()

    # =========================================================================
    # Health & Status
    # =========================================================================

    async def health(self) -> dict:
        """Check API health status"""
        response = await self._http.get(f"{self.config.api_url}/health")
        response.raise_for_status()
        return response.json()

    async def status(self) -> dict:
        """Get detailed system status"""
        response = await self._http.get(f"{self.config.api_url}/api/status")
        response.raise_for_status()
        return response.json()

    # =========================================================================
    # Code Generation
    # =========================================================================

    async def generate(
        self,
        description: str,
        platform: str = "python_fastapi",
        intent_type: str = "api_endpoint",
        context: dict | None = None,
        constraints: list | None = None,
    ) -> dict:
        """
        Generate code from natural language description.

        Args:
            description: What to generate (e.g., "REST API for user management")
            platform: Target platform (python_fastapi, python_flask, nodejs_express, etc.)
            intent_type: Type of intent (api_endpoint, database_schema, workflow, etc.)
            context: Additional context for generation
            constraints: List of constraints (e.g., ["Use async/await", "Include tests"])

        Returns:
            dict with generated_code, language, validation_passed, etc.
        """
        payload = {
            "description": description,
            "target_platform": platform,
            "intent_type": intent_type,
            "context": context or {},
            "constraints": constraints or [],
        }

        response = await self._http.post(f"{self.config.api_url}/api/generate", json=payload)
        response.raise_for_status()
        return response.json()

    async def generate_crud(
        self, entity: str, fields: list[dict], platform: str = "python_fastapi"
    ) -> dict:
        """
        Generate complete CRUD operations for an entity.

        Args:
            entity: Entity name (e.g., "products", "users")
            fields: List of field definitions
            platform: Target platform

        Example:
            result = await client.generate_crud("products", [
                {"name": "title", "type": "str", "required": True},
                {"name": "price", "type": "float", "required": True},
                {"name": "description", "type": "str"}
            ])
        """
        return await self.generate(
            description=f"Create complete CRUD API for {entity} management",
            platform=platform,
            intent_type="api_endpoint",
            context={
                "entity": entity,
                "fields": fields,
                "operations": ["create", "read", "update", "delete", "list"],
            },
            constraints=[
                "Include input validation",
                "Add proper error handling",
                "Include pagination for list endpoint",
            ],
        )

    # =========================================================================
    # Intent Processing
    # =========================================================================

    async def intent(self, description: str, context: dict | None = None) -> dict:
        """
        Process a natural language intent.

        Args:
            description: What to do (e.g., "Send welcome email to new user")
            context: Additional context data

        Returns:
            Result of intent processing
        """
        payload = {"description": description, "context": context or {}, "intent_type": "workflow"}

        response = await self._http.post(f"{self.config.api_url}/api/intent", json=payload)
        response.raise_for_status()
        return response.json()

    # =========================================================================
    # Data Operations
    # =========================================================================

    def data(self, table: str) -> "DataHandler":
        """
        Get a data handler for CRUD operations.

        Example:
            users = await client.data("users").list(limit=10)
            user = await client.data("users").get(123)
            await client.data("users").create({"name": "John", "email": "john@example.com"})
        """
        return DataHandler(self, table)

    # =========================================================================
    # Form Handling
    # =========================================================================

    async def submit_form(self, form_id: str, data: dict, intent: str | None = None) -> dict:
        """
        Submit form data with optional intent processing.

        Args:
            form_id: Form identifier
            data: Form data
            intent: Optional intent description for backend generation
        """
        payload = {"form_id": form_id, "data": data, "intent": intent}

        response = await self._http.post(f"{self.config.api_url}/api/form", json=payload)
        response.raise_for_status()
        return response.json()

    # =========================================================================
    # Email
    # =========================================================================

    async def send_email(
        self,
        to: str,
        subject: str | None = None,
        template: str | None = None,
        data: dict | None = None,
    ) -> dict:
        """
        Send email using IntentForge.

        Args:
            to: Recipient email
            subject: Email subject (optional if using template)
            template: Email template name
            data: Template data
        """
        payload = {
            "action": "send",
            "to": to,
            "subject": subject,
            "template": template,
            "data": data or {},
        }

        response = await self._http.post(f"{self.config.api_url}/api/email", json=payload)
        response.raise_for_status()
        return response.json()


class DataHandler:
    """Handler for data/CRUD operations"""

    def __init__(self, client: IntentForgeClient, table: str):
        self.client = client
        self.table = table

    async def list(
        self,
        limit: int = 100,
        offset: int = 0,
        order_by: str | None = None,
        filters: dict | None = None,
    ) -> dict:
        """List records with pagination and filtering"""
        payload = {
            "action": "list",
            "table": self.table,
            "limit": limit,
            "offset": offset,
            "order_by": order_by,
            "filters": filters or {},
        }

        response = await self.client._http.post(
            f"{self.client.config.api_url}/api/data", json=payload
        )
        response.raise_for_status()
        return response.json()

    async def get(self, id: Any) -> dict:
        """Get a single record by ID"""
        payload = {"action": "get", "table": self.table, "id": id}

        response = await self.client._http.post(
            f"{self.client.config.api_url}/api/data", json=payload
        )
        response.raise_for_status()
        return response.json()

    async def create(self, data: dict) -> dict:
        """Create a new record"""
        payload = {"action": "create", "table": self.table, "data": data}

        response = await self.client._http.post(
            f"{self.client.config.api_url}/api/data", json=payload
        )
        response.raise_for_status()
        return response.json()

    async def update(self, id: Any, data: dict) -> dict:
        """Update an existing record"""
        payload = {"action": "update", "table": self.table, "id": id, "data": data}

        response = await self.client._http.post(
            f"{self.client.config.api_url}/api/data", json=payload
        )
        response.raise_for_status()
        return response.json()

    async def delete(self, id: Any) -> dict:
        """Delete a record"""
        payload = {"action": "delete", "table": self.table, "id": id}

        response = await self.client._http.post(
            f"{self.client.config.api_url}/api/data", json=payload
        )
        response.raise_for_status()
        return response.json()


# =============================================================================
# Example Functions
# =============================================================================


async def example_health_check():
    """Check API health"""
    print("\n" + "=" * 60)
    print("1. Health Check")
    print("=" * 60)

    async with IntentForgeClient() as client:
        try:
            health = await client.health()
            print(f"✅ API is healthy: {health}")
        except Exception as e:
            print(f"❌ API not available: {e}")
            print("   Make sure IntentForge is running: make start")


async def example_code_generation():
    """Generate code examples"""
    print("\n" + "=" * 60)
    print("2. Code Generation")
    print("=" * 60)

    async with IntentForgeClient() as client:
        # Simple generation
        print("\n[2.1] Simple API generation:")
        print("-" * 40)

        try:
            result = await client.generate(
                "Create REST API endpoint for product search with filters",
                platform="python_fastapi",
                context={"filters": ["category", "price_min", "price_max", "in_stock"]},
            )

            print(f"Status: {'✅ Success' if result.get('success') else '❌ Failed'}")
            print(f"Language: {result.get('language', 'unknown')}")
            print(f"Validation: {'✅ Passed' if result.get('validation_passed') else '❌ Failed'}")

            if result.get("generated_code"):
                print(f"\nGenerated code ({len(result['generated_code'])} chars):")
                print("-" * 40)
                print(result["generated_code"][:500] + "...")

        except Exception as e:
            print(f"❌ Error: {e}")

        # CRUD generation
        print("\n[2.2] CRUD generation:")
        print("-" * 40)

        try:
            result = await client.generate_crud(
                "orders",
                [
                    {"name": "customer_id", "type": "int", "required": True},
                    {"name": "total", "type": "float", "required": True},
                    {"name": "status", "type": "str", "default": "pending"},
                    {"name": "items", "type": "list"},
                ],
            )

            print(f"Status: {'✅ Success' if result.get('success') else '❌ Failed'}")
            if result.get("generated_code"):
                lines = result["generated_code"].count("\n")
                print(f"Generated {lines} lines of code")

        except Exception as e:
            print(f"❌ Error: {e}")


async def example_intent_processing():
    """Process natural language intents"""
    print("\n" + "=" * 60)
    print("3. Intent Processing")
    print("=" * 60)

    async with IntentForgeClient() as client:
        intents = [
            ("Return a list of 5 active users", {"format": "json"}),
            ("Calculate the total revenue for last month", {"table": "orders"}),
            ("Generate a report of top selling products", {"limit": 10}),
        ]

        for description, context in intents:
            print(f"\n[Intent] {description}")
            print("-" * 40)

            try:
                result = await client.intent(description, context)
                print(f"Status: {'✅ Success' if result.get('success') else '❌ Failed'}")

                if result.get("result"):
                    print(f"Result: {json.dumps(result['result'], indent=2)[:300]}...")

            except Exception as e:
                print(f"❌ Error: {e}")


async def example_data_operations():
    """CRUD data operations"""
    print("\n" + "=" * 60)
    print("4. Data Operations")
    print("=" * 60)

    async with IntentForgeClient() as client:
        # List
        print("\n[4.1] List users:")
        try:
            result = await client.data("users").list(limit=5)
            print(f"Found {len(result.get('data', []))} users")
        except Exception as e:
            print(f"❌ Error: {e}")

        # Create
        print("\n[4.2] Create user:")
        try:
            result = await client.data("users").create(
                {"name": "Test User", "email": "test@example.com"}
            )
            print(f"Created: {result}")
        except Exception as e:
            print(f"❌ Error: {e}")


async def example_form_submission():
    """Form submission example"""
    print("\n" + "=" * 60)
    print("5. Form Submission")
    print("=" * 60)

    async with IntentForgeClient() as client:
        try:
            result = await client.submit_form(
                form_id="contact",
                data={
                    "name": "Jan Kowalski",
                    "email": "jan@example.com",
                    "message": "Hello from Python client!",
                },
                intent="Save contact form and send confirmation email",
            )
            print(f"Form submitted: {result}")
        except Exception as e:
            print(f"❌ Error: {e}")


async def main():
    """Run all examples"""
    print("\n" + "#" * 60)
    print("  IntentForge Python Client Examples")
    print("#" * 60)

    await example_health_check()
    await example_code_generation()
    await example_intent_processing()
    await example_data_operations()
    await example_form_submission()

    print("\n" + "=" * 60)
    print("✅ All examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IntentForge Python Client Examples")
    parser.add_argument("--health", action="store_true", help="Run health check only")
    parser.add_argument("--generate", action="store_true", help="Run code generation examples")
    parser.add_argument("--intent", action="store_true", help="Run intent processing examples")
    parser.add_argument("--data", action="store_true", help="Run data operation examples")
    parser.add_argument("--form", action="store_true", help="Run form submission examples")

    args = parser.parse_args()

    if args.health:
        asyncio.run(example_health_check())
    elif args.generate:
        asyncio.run(example_code_generation())
    elif args.intent:
        asyncio.run(example_intent_processing())
    elif args.data:
        asyncio.run(example_data_operations())
    elif args.form:
        asyncio.run(example_form_submission())
    else:
        asyncio.run(main())
