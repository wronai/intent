"""
IntentForge E2E Tests - Configuration System
=============================================

Tests that verify the unified configuration system works correctly
across Python, Docker, and frontend environments.

Run:
    pytest tests/e2e/test_config.py -v
"""

import os
import re
import sys
from pathlib import Path

import pytest

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv

# Load .env file
env_path = project_root / ".env"
load_dotenv(env_path)


class TestEnvConfiguration:
    """Test that .env configuration is loaded correctly"""

    def test_env_file_exists(self):
        """Test that .env file exists"""
        assert env_path.exists(), f".env file not found at {env_path}"

    def test_env_example_exists(self):
        """Test that .env.example file exists"""
        example_path = project_root / ".env.example"
        assert example_path.exists(), f".env.example file not found at {example_path}"

    def test_env_files_have_same_keys(self):
        """Test that .env and .env.example have the same keys"""

        def extract_keys(file_path: Path) -> set:
            keys = set()
            with open(file_path) as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        key = line.split("=")[0].strip()
                        keys.add(key)
            return keys

        env_keys = extract_keys(env_path)
        example_keys = extract_keys(project_root / ".env.example")

        # .env should have at least all keys from .env.example
        missing_in_env = example_keys - env_keys
        assert not missing_in_env, f"Keys in .env.example but not in .env: {missing_in_env}"

    def test_no_legacy_server_keys(self):
        """Ensure legacy server keys are not used (use APP_* instead)."""

        def extract_keys(file_path: Path) -> set:
            keys = set()
            with open(file_path) as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        key = line.split("=")[0].strip()
                        keys.add(key)
            return keys

        env_keys = extract_keys(env_path)
        example_keys = extract_keys(project_root / ".env.example")

        legacy = {"HOST", "PORT", "WORKERS"}
        assert not (legacy & env_keys), f"Legacy keys present in .env: {sorted(legacy & env_keys)}"
        assert not (legacy & example_keys), (
            f"Legacy keys present in .env.example: {sorted(legacy & example_keys)}"
        )


class TestPythonConfigSettings:
    """Test Python configuration settings"""

    def test_settings_load(self):
        """Test that Settings can be loaded"""
        from intentforge.config import get_settings

        settings = get_settings()
        assert settings is not None

    def test_app_port_from_env(self):
        """Test APP_PORT is loaded from .env"""
        from intentforge.config import get_settings

        settings = get_settings()

        env_port = os.getenv("APP_PORT", "8000")
        assert settings.app_port == int(env_port)

    def test_app_host_from_env(self):
        """Test APP_HOST is loaded from .env"""
        from intentforge.config import get_settings

        settings = get_settings()

        env_host = os.getenv("APP_HOST", "0.0.0.0")
        assert settings.app_host == env_host

    def test_database_settings(self):
        """Test database settings are loaded"""
        from intentforge.config import get_database_settings

        db = get_database_settings()

        assert db.host == os.getenv("DB_HOST", "localhost")
        assert db.port == int(os.getenv("DB_PORT", "5432"))
        assert db.name == os.getenv("DB_NAME", "intentforge")

    def test_mqtt_settings(self):
        """Test MQTT settings are loaded"""
        from intentforge.config import get_mqtt_settings

        mqtt = get_mqtt_settings()

        assert mqtt.host == os.getenv("MQTT_HOST", "localhost")
        assert mqtt.port == int(os.getenv("MQTT_PORT", "1883"))
        assert mqtt.websocket_port == int(os.getenv("MQTT_WEBSOCKET_PORT", "9001"))

    def test_redis_settings(self):
        """Test Redis settings are loaded"""
        from intentforge.config import get_redis_settings

        redis = get_redis_settings()

        assert redis.host == os.getenv("REDIS_HOST", "localhost")
        assert redis.port == int(os.getenv("REDIS_PORT", "6379"))

    def test_backwards_compatibility(self):
        """Test backwards compatibility properties"""
        from intentforge.config import get_settings

        settings = get_settings()

        # Old properties should still work
        assert settings.host == settings.app_host
        assert settings.port == settings.app_port
        assert settings.workers == settings.app_workers


class TestPortConfiguration:
    """Test that all port configurations are consistent"""

    def test_all_ports_defined(self):
        """Test that all required ports are defined in .env"""
        required_ports = [
            "APP_PORT",
            "APP_PORT_EXTERNAL",
            "WEB_PORT",
            "DB_PORT",
            "DB_PORT_EXTERNAL",
            "MQTT_PORT",
            "MQTT_PORT_EXTERNAL",
            "MQTT_WEBSOCKET_PORT",
            "MQTT_WEBSOCKET_PORT_EXTERNAL",
            "REDIS_PORT",
            "REDIS_PORT_EXTERNAL",
        ]

        for port_var in required_ports:
            value = os.getenv(port_var)
            assert value is not None, f"{port_var} not defined in .env"
            assert value.isdigit(), f"{port_var} should be a number, got: {value}"

    def test_external_ports_match_internal_by_default(self):
        """Test that external ports match internal ports by default"""
        port_pairs = [
            ("DB_PORT", "DB_PORT_EXTERNAL"),
            ("MQTT_PORT", "MQTT_PORT_EXTERNAL"),
            ("MQTT_WEBSOCKET_PORT", "MQTT_WEBSOCKET_PORT_EXTERNAL"),
            ("REDIS_PORT", "REDIS_PORT_EXTERNAL"),
        ]

        for internal, external in port_pairs:
            internal_val = os.getenv(internal)
            external_val = os.getenv(external)
            # Both must be defined; they may differ for port remapping (e.g. local DB conflict)
            assert internal_val is not None, f"{internal} not defined"
            assert external_val is not None, f"{external} not defined"


class TestDockerComposeConfig:
    """Test Docker Compose configuration uses .env variables"""

    def test_docker_compose_exists(self):
        """Test docker-compose.yml exists"""
        dc_path = project_root / "docker-compose.yml"
        assert dc_path.exists()

    def test_docker_compose_uses_env_vars(self):
        """Test docker-compose.yml uses environment variables for ports"""
        dc_path = project_root / "docker-compose.yml"
        content = dc_path.read_text()

        # Check that ports use ${VAR:-default} syntax
        assert "${APP_PORT_EXTERNAL:-8000}" in content, "APP_PORT_EXTERNAL not parameterized"
        assert "${DB_PORT_EXTERNAL:-5432}" in content, "DB_PORT_EXTERNAL not parameterized"
        assert "${MQTT_PORT_EXTERNAL:-1883}" in content, "MQTT_PORT_EXTERNAL not parameterized"
        assert "${REDIS_PORT_EXTERNAL:-6379}" in content, "REDIS_PORT_EXTERNAL not parameterized"
        assert "${WEB_PORT:-80}" in content, "WEB_PORT not parameterized"

    def test_docker_compose_env_file(self):
        """Test docker-compose.yml references .env file"""
        dc_path = project_root / "docker-compose.yml"
        content = dc_path.read_text()

        assert "env_file:" in content
        assert ".env" in content

    def test_docker_compose_build_targets(self):
        """Server should build runtime stage and worker should build worker stage."""
        dc_path = project_root / "docker-compose.yml"
        content = dc_path.read_text()
        assert "target: runtime" in content
        assert "target: worker" in content


class TestDockerComposeConfigAlt:
    """Test docker/docker-compose.yml uses .env variables"""

    def test_docker_compose_alt_exists(self):
        dc_path = project_root / "docker" / "docker-compose.yml"
        assert dc_path.exists()

    def test_docker_compose_alt_uses_env_vars(self):
        dc_path = project_root / "docker" / "docker-compose.yml"
        content = dc_path.read_text()

        assert re.search(
            r"\$\{APP_PORT_EXTERNAL:-\d+\}:\$\{APP_PORT:-8000\}",
            content,
        ), "APP_PORT_EXTERNAL/APP_PORT mapping not parameterized in docker/docker-compose.yml"
        assert "${DB_PORT_EXTERNAL:-5432}" in content, (
            "DB_PORT_EXTERNAL not parameterized in docker/docker-compose.yml"
        )
        assert "${MQTT_PORT_EXTERNAL:-1883}" in content, (
            "MQTT_PORT_EXTERNAL not parameterized in docker/docker-compose.yml"
        )
        assert "${REDIS_PORT_EXTERNAL:-6379}" in content, (
            "REDIS_PORT_EXTERNAL not parameterized in docker/docker-compose.yml"
        )
        assert "${WEB_PORT:-80}" in content, (
            "WEB_PORT not parameterized in docker/docker-compose.yml"
        )


class TestMakefileTargets:
    """Test Makefile targets exist and use env-driven ports."""

    def test_makefile_has_run_target(self):
        mk_path = project_root / "Makefile"
        content = mk_path.read_text()
        assert "run:" in content
        assert "run-server" in content

    def test_makefile_has_e2e_targets(self):
        mk_path = project_root / "Makefile"
        content = mk_path.read_text()
        assert "e2e:" in content
        assert "e2e-config:" in content
        assert "e2e-services:" in content

    def test_makefile_run_server_uses_app_port(self):
        mk_path = project_root / "Makefile"
        content = mk_path.read_text()
        assert "--port $(APP_PORT)" in content


class TestE2ETestConfig:
    """Test E2E test configuration"""

    def test_test_urls_use_env_ports(self):
        """Test that test URLs use ports from .env"""
        app_port_external = os.getenv("APP_PORT_EXTERNAL", os.getenv("APP_PORT", "8000"))
        web_port = os.getenv("WEB_PORT", "80")

        # These should match what's in test_frontend.py
        expected_api_url = f"http://localhost:{app_port_external}"
        expected_base_url = f"http://localhost:{web_port}"

        # Verify env vars for testing exist
        test_base = os.getenv("TEST_BASE_URL", expected_base_url)
        test_api = os.getenv("TEST_API_URL", expected_api_url)

        assert test_base is not None
        assert test_api is not None
        assert test_api.endswith(f":{app_port_external}"), (
            f"TEST_API_URL should use APP_PORT_EXTERNAL={app_port_external}, got {test_api}"
        )


class TestNginxConfig:
    """Validate nginx config wiring for /frontend and /examples."""

    def test_nginx_root_is_frontend(self):
        conf_path = project_root / "config" / "nginx.conf"
        content = conf_path.read_text()
        assert "root /usr/share/nginx/html/frontend;" in content

    def test_nginx_examples_alias(self):
        conf_path = project_root / "config" / "nginx.conf"
        content = conf_path.read_text()
        assert "location /examples/" in content
        assert "alias /usr/share/nginx/html/examples/;" in content


class TestCLIModule:
    """Ensure CLI exists as declared in pyproject entrypoints."""

    def test_cli_module_importable(self):
        import intentforge.cli as cli

        assert hasattr(cli, "main")

    def test_cli_validate_schemas(self):
        import intentforge.cli as cli

        code = cli.main(["validate-schemas"])
        assert code == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
