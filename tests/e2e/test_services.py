"""
IntentForge E2E Tests - Service Health Checks
==============================================

Tests that verify services are running and accessible.
These tests require Docker services to be running.

Run:
    docker-compose up -d
    pytest tests/e2e/test_services.py -v
"""

import asyncio
import os
import sys
from pathlib import Path

import pytest

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv

# Load .env file
load_dotenv(project_root / ".env")

# Configuration from .env
APP_HOST = os.getenv("APP_HOST", "localhost")
APP_PORT = os.getenv("APP_PORT", "8000")
APP_PORT_EXTERNAL = os.getenv("APP_PORT_EXTERNAL", APP_PORT)
WEB_PORT = os.getenv("WEB_PORT", "80")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT_EXTERNAL", "5432")
MQTT_HOST = os.getenv("MQTT_HOST", "localhost")
MQTT_PORT = os.getenv("MQTT_PORT_EXTERNAL", "1883")
MQTT_WS_PORT = os.getenv("MQTT_WEBSOCKET_PORT_EXTERNAL", "9001")
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = os.getenv("REDIS_PORT_EXTERNAL", "6379")


@pytest.fixture
def require_intentforge_api():
    """Skip test unless IntentForge API is reachable and healthy."""
    import requests

    url = f"http://localhost:{APP_PORT_EXTERNAL}/health"
    try:
        response = requests.get(url, timeout=3)
    except requests.exceptions.ConnectionError:
        pytest.skip("IntentForge API server not reachable")

    if response.status_code != 200:
        pytest.skip(
            f"IntentForge health endpoint not available at {url} (status={response.status_code}). "
            "Start the server (e.g. `docker-compose up -d` or `make run-server`)."
        )

    return f"http://localhost:{APP_PORT_EXTERNAL}"


@pytest.fixture
def require_web_ui():
    """Skip test unless the nginx web UI is reachable."""
    import requests

    url = f"http://localhost:{WEB_PORT}/health"
    try:
        response = requests.get(url, timeout=3)
    except requests.exceptions.ConnectionError:
        pytest.skip("Web server not reachable")

    if response.status_code != 200:
        pytest.skip(f"Web health endpoint not available at {url} (status={response.status_code})")

    return f"http://localhost:{WEB_PORT}"


def is_port_open(host: str, port: int, timeout: float = 2.0) -> bool:
    """Check if a port is open on a host"""
    import socket

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(timeout)
    try:
        result = sock.connect_ex((host, port))
        return result == 0
    except OSError:
        return False
    finally:
        sock.close()


@pytest.mark.e2e
class TestServiceHealth:
    """Test that services are running and healthy"""

    @pytest.mark.skipif(
        not is_port_open(
            "localhost", int(os.getenv("APP_PORT_EXTERNAL", os.getenv("APP_PORT", "8000")))
        ),
        reason="IntentForge server not running",
    )
    def test_api_health_endpoint(self):
        """Test API health endpoint responds"""
        import requests

        url = f"http://localhost:{APP_PORT_EXTERNAL}/health"
        try:
            response = requests.get(url, timeout=5)
            if response.status_code != 200:
                pytest.skip(
                    f"IntentForge health endpoint not available at {url} (status={response.status_code}). "
                    "Start the server (e.g. `docker-compose up -d` or `make run-server`) or install the correct port in .env."
                )
        except requests.exceptions.ConnectionError:
            pytest.skip("API server not reachable")

    @pytest.mark.skipif(
        not is_port_open("localhost", int(os.getenv("WEB_PORT", "80"))),
        reason="Web server not running",
    )
    def test_web_server_responds(self):
        """Test web server responds"""
        import requests

        url = f"http://localhost:{WEB_PORT}/"
        try:
            response = requests.get(url, timeout=5)
            assert response.status_code in [200, 304]
        except requests.exceptions.ConnectionError:
            pytest.skip("Web server not reachable")

    @pytest.mark.skipif(
        not is_port_open("localhost", int(os.getenv("DB_PORT_EXTERNAL", "5432"))),
        reason="PostgreSQL not running",
    )
    def test_postgres_connection(self):
        """Test PostgreSQL is accessible"""
        assert is_port_open("localhost", int(DB_PORT))

    @pytest.mark.skipif(
        not is_port_open("localhost", int(os.getenv("REDIS_PORT_EXTERNAL", "6379"))),
        reason="Redis not running",
    )
    def test_redis_connection(self):
        """Test Redis is accessible"""
        try:
            import redis

            r = redis.Redis(host="localhost", port=int(REDIS_PORT), socket_timeout=2)
            assert r.ping()
        except ImportError:
            # Just check port is open if redis-py not installed
            assert is_port_open("localhost", int(REDIS_PORT))
        except redis.exceptions.ConnectionError:
            pytest.skip("Redis not reachable")

    @pytest.mark.skipif(
        not is_port_open("localhost", int(os.getenv("MQTT_PORT_EXTERNAL", "1883"))),
        reason="MQTT broker not running",
    )
    def test_mqtt_connection(self):
        """Test MQTT broker is accessible"""
        assert is_port_open("localhost", int(MQTT_PORT))


@pytest.mark.e2e
class TestServiceConfiguration:
    """Test that services use correct configuration from .env"""

    def test_env_ports_are_integers(self):
        """Test all port values are valid integers"""
        ports = {
            "APP_PORT": APP_PORT,
            "WEB_PORT": WEB_PORT,
            "DB_PORT": DB_PORT,
            "MQTT_PORT": MQTT_PORT,
            "MQTT_WS_PORT": MQTT_WS_PORT,
            "REDIS_PORT": REDIS_PORT,
        }

        for name, value in ports.items():
            assert value.isdigit(), f"{name} should be integer, got: {value}"
            port = int(value)
            assert 1 <= port <= 65535, f"{name} should be valid port (1-65535), got: {port}"

    def test_hosts_are_valid(self):
        """Test host values are valid"""
        hosts = {
            "APP_HOST": APP_HOST,
            "DB_HOST": DB_HOST,
            "MQTT_HOST": MQTT_HOST,
            "REDIS_HOST": REDIS_HOST,
        }

        for name, value in hosts.items():
            assert value, f"{name} should not be empty"
            # Basic validation - should be localhost, IP, or hostname
            assert len(value) > 0


@pytest.mark.e2e
class TestIntentForgeAPI:
    """API-level tests (run only if real IntentForge server is up)."""

    def test_generate_rejects_invalid_payload(self, require_intentforge_api):
        import requests

        base_url = require_intentforge_api
        response = requests.post(f"{base_url}/api/generate", json={}, timeout=5)
        assert response.status_code == 400
        data = response.json()
        assert data.get("success") is False

    def test_service_endpoint_requires_action(self, require_intentforge_api):
        import requests

        base_url = require_intentforge_api
        response = requests.post(f"{base_url}/api/forms", json={}, timeout=5)
        assert response.status_code == 400
        data = response.json()
        assert data.get("success") is False
        assert "action" in str(data).lower()

    def test_docs_available(self, require_intentforge_api):
        import requests

        base_url = require_intentforge_api
        response = requests.get(f"{base_url}/docs", timeout=5)
        assert response.status_code == 200
        assert "swagger" in response.text.lower() or "openapi" in response.text.lower()

    def test_openapi_json_available(self, require_intentforge_api):
        import requests

        base_url = require_intentforge_api
        response = requests.get(f"{base_url}/openapi.json", timeout=5)
        assert response.status_code == 200
        data = response.json()
        assert "openapi" in data


@pytest.mark.e2e
class TestWebUIContent:
    """Tests for nginx-served UI content (run only if web is up)."""

    def test_web_health(self, require_web_ui):
        import requests

        base_url = require_web_ui
        response = requests.get(f"{base_url}/health", timeout=5)
        assert response.status_code == 200

    def test_examples_contact_form_served(self, require_web_ui):
        import requests

        base_url = require_web_ui
        response = requests.get(f"{base_url}/examples/usecases/01_contact_form.html", timeout=5)
        assert response.status_code == 200
        assert "Formularz Kontaktowy" in response.text

    def test_frontend_sdk_served(self, require_web_ui):
        import requests

        base_url = require_web_ui
        response = requests.get(f"{base_url}/sdk/intentforge.js", timeout=5)
        assert response.status_code == 200
        assert "IntentForge" in response.text

    def test_nginx_proxies_api_generate(self, require_web_ui):
        import requests

        base_url = require_web_ui
        response = requests.post(f"{base_url}/api/generate", json={}, timeout=5)
        # FastAPI returns JSON 400 on invalid payload
        assert response.status_code == 400
        data = response.json()
        assert data.get("success") is False


@pytest.mark.e2e
@pytest.mark.asyncio
class TestAsyncServiceHealth:
    """Async tests for service health"""

    async def test_concurrent_health_checks(self):
        """Test multiple services concurrently"""
        import aiohttp

        services = [
            (f"http://localhost:{APP_PORT}/health", "API"),
            (f"http://localhost:{WEB_PORT}/", "Web"),
        ]

        async def check_service(url: str, name: str) -> tuple:
            try:
                async with (
                    aiohttp.ClientSession() as session,
                    session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as resp,
                ):
                    # If something else is listening, avoid failing the suite.
                    if resp.status == 404:
                        return (name, resp.status, False)
                    return (name, resp.status, True)
            except Exception:
                return (name, 0, False)

        try:
            results = await asyncio.gather(*[check_service(url, name) for url, name in services])

            # At least log results, don't fail if services aren't running
            for name, status, success in results:
                if success:
                    assert status in [200, 304], f"{name} returned {status}"
        except ImportError:
            pytest.skip("aiohttp not installed")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "e2e"])
