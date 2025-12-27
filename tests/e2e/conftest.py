"""
E2E Test Configuration and Fixtures
====================================

Shared fixtures and configuration for all E2E tests.
"""

import os
import sys
from pathlib import Path

import pytest

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv

# Load .env file
load_dotenv(project_root / ".env")


def pytest_configure(config):
    """Configure pytest markers"""
    config.addinivalue_line("markers", "e2e: mark test as end-to-end test")
    config.addinivalue_line("markers", "slow: mark test as slow running")


@pytest.fixture(scope="session")
def env_config():
    """Provide environment configuration for tests"""
    return {
        "APP_HOST": os.getenv("APP_HOST", "0.0.0.0"),
        "APP_PORT": int(os.getenv("APP_PORT", "8000")),
        "APP_PORT_EXTERNAL": int(os.getenv("APP_PORT_EXTERNAL", os.getenv("APP_PORT", "8000"))),
        "WEB_PORT": int(os.getenv("WEB_PORT", "80")),
        "DB_HOST": os.getenv("DB_HOST", "localhost"),
        "DB_PORT": int(os.getenv("DB_PORT", "5432")),
        "DB_PORT_EXTERNAL": int(os.getenv("DB_PORT_EXTERNAL", "5432")),
        "DB_NAME": os.getenv("DB_NAME", "intentforge"),
        "DB_USER": os.getenv("DB_USER", "postgres"),
        "MQTT_HOST": os.getenv("MQTT_HOST", "localhost"),
        "MQTT_PORT": int(os.getenv("MQTT_PORT", "1883")),
        "MQTT_PORT_EXTERNAL": int(os.getenv("MQTT_PORT_EXTERNAL", "1883")),
        "MQTT_WEBSOCKET_PORT": int(os.getenv("MQTT_WEBSOCKET_PORT", "9001")),
        "MQTT_WEBSOCKET_PORT_EXTERNAL": int(os.getenv("MQTT_WEBSOCKET_PORT_EXTERNAL", "9001")),
        "REDIS_HOST": os.getenv("REDIS_HOST", "localhost"),
        "REDIS_PORT": int(os.getenv("REDIS_PORT", "6379")),
        "REDIS_PORT_EXTERNAL": int(os.getenv("REDIS_PORT_EXTERNAL", "6379")),
        "TEST_BASE_URL": os.getenv(
            "TEST_BASE_URL", f"http://localhost:{os.getenv('WEB_PORT', '80')}"
        ),
        "TEST_API_URL": os.getenv(
            "TEST_API_URL",
            f"http://localhost:{os.getenv('APP_PORT_EXTERNAL', os.getenv('APP_PORT', '8000'))}",
        ),
    }


@pytest.fixture(scope="session")
def api_url(env_config):
    """API URL for tests"""
    return env_config["TEST_API_URL"]


@pytest.fixture(scope="session")
def base_url(env_config):
    """Base URL for frontend tests"""
    return env_config["TEST_BASE_URL"]


@pytest.fixture
def check_port():
    """Fixture to check if a port is open"""
    import socket

    def _check(host: str, port: int, timeout: float = 2.0) -> bool:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        try:
            result = sock.connect_ex((host, port))
            return result == 0
        except OSError:
            return False
        finally:
            sock.close()

    return _check


@pytest.fixture
def skip_if_no_docker(check_port, env_config):
    """Skip test if Docker services are not running"""
    if not check_port("localhost", env_config["APP_PORT"]):
        pytest.skip("Docker services not running (APP_PORT not accessible)")
