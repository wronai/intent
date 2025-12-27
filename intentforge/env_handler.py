"""
Environment Handler - Secure configuration management
Supports .env files, environment variables, and secret managers
"""

import json
import logging
import os
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class SecretProvider(Enum):
    """Supported secret providers"""

    ENV_FILE = "env_file"  # .env files
    ENVIRONMENT = "environment"  # OS environment variables
    VAULT = "vault"  # HashiCorp Vault
    AWS_SECRETS = "aws_secrets"  # AWS Secrets Manager
    AZURE_KEYVAULT = "azure_keyvault"  # Azure Key Vault
    GCP_SECRETS = "gcp_secrets"  # Google Cloud Secret Manager


@dataclass
class EnvConfig:
    """Environment configuration"""

    env_file: str = ".env"
    env_file_encoding: str = "utf-8"
    case_sensitive: bool = True
    interpolate: bool = True  # Allow ${VAR} interpolation
    prefix: str = ""  # Prefix for all variables
    required_vars: list[str] = field(default_factory=list)
    default_values: dict[str, str] = field(default_factory=dict)
    secret_provider: SecretProvider = SecretProvider.ENV_FILE
    vault_url: str | None = None
    vault_token: str | None = None


class EnvParseError(Exception):
    """Error parsing environment file"""

    pass


class MissingEnvVar(Exception):
    """Required environment variable is missing"""

    pass


class EnvHandler:
    """
    Secure environment variable handler
    Loads from .env files, validates, and provides type-safe access
    """

    # Pattern for .env file parsing
    ENV_PATTERN = re.compile(
        r"""
        ^
        (?:export\s+)?          # Optional 'export' prefix
        (?P<key>[A-Za-z_][A-Za-z0-9_]*)  # Variable name
        \s*=\s*                 # Equals sign with optional whitespace
        (?:
            (?P<squoted>'[^']*')  |  # Single quoted value
            (?P<dquoted>"[^"]*")  |  # Double quoted value
            (?P<unquoted>[^\s#]*)    # Unquoted value
        )
        (?:\s*\#.*)?            # Optional comment
        $
        """,
        re.VERBOSE | re.MULTILINE,
    )

    # Pattern for variable interpolation
    INTERPOLATION_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")

    def __init__(self, config: EnvConfig | None = None):
        self.config = config or EnvConfig()
        self._values: dict[str, str] = {}
        self._loaded = False
        self._secret_cache: dict[str, str] = {}

    def load(self, env_file: str | None = None) -> "EnvHandler":
        """
        Load environment variables from .env file
        """
        env_path = Path(env_file or self.config.env_file)

        # Load from file if exists
        if env_path.exists():
            self._load_from_file(env_path)

        # Merge with OS environment
        self._merge_os_environ()

        # Apply defaults
        self._apply_defaults()

        # Interpolate variables
        if self.config.interpolate:
            self._interpolate_values()

        # Validate required
        self._validate_required()

        self._loaded = True
        return self

    def _load_from_file(self, path: Path) -> None:
        """Parse .env file"""
        try:
            content = path.read_text(encoding=self.config.env_file_encoding)

            for line_num, line in enumerate(content.splitlines(), 1):
                line = line.strip()

                # Skip empty lines and comments
                if not line or line.startswith("#"):
                    continue

                match = self.ENV_PATTERN.match(line)
                if match:
                    key = match.group("key")

                    # Get value from appropriate group
                    value = (
                        match.group("squoted")
                        or match.group("dquoted")
                        or match.group("unquoted")
                        or ""
                    )

                    # Remove quotes
                    if value.startswith(("'", '"')):
                        value = value[1:-1]

                    # Apply prefix
                    if self.config.prefix:
                        key = f"{self.config.prefix}{key}"

                    self._values[key] = value
                else:
                    logger.warning(f"Could not parse line {line_num}: {line}")

        except Exception as e:
            raise EnvParseError(f"Error parsing {path}: {e}")

    def _merge_os_environ(self) -> None:
        """Merge with OS environment variables (OS takes precedence)"""
        for key, value in os.environ.items():
            if self.config.prefix:
                if key.startswith(self.config.prefix):
                    self._values[key] = value
            else:
                self._values[key] = value

    def _apply_defaults(self) -> None:
        """Apply default values for missing keys"""
        for key, default in self.config.default_values.items():
            if key not in self._values:
                self._values[key] = default

    def _interpolate_values(self) -> None:
        """Interpolate ${VAR} references in values"""
        max_iterations = 10  # Prevent infinite loops

        for _ in range(max_iterations):
            changed = False

            for key, value in self._values.items():
                new_value = self.INTERPOLATION_PATTERN.sub(
                    lambda m: self._values.get(m.group(1), m.group(0)), value
                )
                if new_value != value:
                    self._values[key] = new_value
                    changed = True

            if not changed:
                break

    def _validate_required(self) -> None:
        """Validate that all required variables are present"""
        missing = []
        for var in self.config.required_vars:
            key = f"{self.config.prefix}{var}" if self.config.prefix else var
            if key not in self._values:
                missing.append(var)

        if missing:
            raise MissingEnvVar(f"Missing required environment variables: {', '.join(missing)}")

    def get(self, key: str, default: str | None = None) -> str | None:
        """Get environment variable value"""
        full_key = f"{self.config.prefix}{key}" if self.config.prefix else key
        return self._values.get(full_key, default)

    def get_int(self, key: str, default: int = 0) -> int:
        """Get environment variable as integer"""
        value = self.get(key)
        if value is None:
            return default
        try:
            return int(value)
        except ValueError:
            return default

    def get_bool(self, key: str, default: bool = False) -> bool:
        """Get environment variable as boolean"""
        value = self.get(key)
        if value is None:
            return default
        return value.lower() in ("true", "1", "yes", "on")

    def get_list(
        self, key: str, separator: str = ",", default: list[str] | None = None
    ) -> list[str]:
        """Get environment variable as list"""
        value = self.get(key)
        if value is None:
            return default or []
        return [v.strip() for v in value.split(separator) if v.strip()]

    def get_json(self, key: str, default: dict | None = None) -> dict | None:
        """Get environment variable as JSON object"""
        value = self.get(key)
        if value is None:
            return default
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return default

    def get_database_url(
        self,
        driver_key: str = "DB_DRIVER",
        host_key: str = "DB_HOST",
        port_key: str = "DB_PORT",
        user_key: str = "DB_USER",
        password_key: str = "DB_PASSWORD",
        database_key: str = "DB_NAME",
    ) -> str:
        """Build database URL from individual components"""
        driver = self.get(driver_key, "postgresql")
        host = self.get(host_key, "localhost")
        port = self.get(port_key, "5432")
        user = self.get(user_key, "postgres")
        password = self.get(password_key, "")
        database = self.get(database_key, "app")

        if password:
            return f"{driver}://{user}:{password}@{host}:{port}/{database}"
        return f"{driver}://{user}@{host}:{port}/{database}"

    def to_dict(self, include_prefix: bool = False) -> dict[str, str]:
        """Export all values as dictionary"""
        if include_prefix or not self.config.prefix:
            return self._values.copy()

        prefix_len = len(self.config.prefix)
        return {
            k[prefix_len:] if k.startswith(self.config.prefix) else k: v
            for k, v in self._values.items()
        }

    def generate_template(self, output_path: str = ".env.example") -> None:
        """Generate .env.example template from current config"""
        lines = [
            "# Environment Configuration",
            "# Generated by IntentForge",
            "",
            "# Database Configuration",
            "DB_DRIVER=postgresql",
            "DB_HOST=localhost",
            "DB_PORT=5432",
            "DB_USER=postgres",
            "DB_PASSWORD=your_secure_password",
            "DB_NAME=intentforge",
            "",
            "# MQTT Configuration",
            "MQTT_HOST=localhost",
            "MQTT_PORT=1883",
            "MQTT_USER=",
            "MQTT_PASSWORD=",
            "",
            "# LLM Configuration",
            "ANTHROPIC_API_KEY=your_api_key_here",
            "LLM_MODEL=claude-sonnet-4-5-20250929",
            "LLM_MAX_TOKENS=4096",
            "",
            "# Redis Configuration (for caching)",
            "REDIS_URL=redis://localhost:6379/0",
            "",
            "# Application Settings",
            "APP_ENV=development",
            "APP_DEBUG=true",
            "APP_SECRET_KEY=generate_a_secure_key",
            "",
            "# Logging",
            "LOG_LEVEL=INFO",
            "LOG_FORMAT=json",
        ]

        Path(output_path).write_text("\n".join(lines))
        logger.info(f"Generated template at {output_path}")


class DatabaseEnvMapper:
    """
    Maps form fields to database operations with .env configuration
    """

    def __init__(self, env_handler: EnvHandler):
        self.env = env_handler

    def get_connection_config(self) -> dict[str, Any]:
        """Get database connection configuration"""
        return {
            "driver": self.env.get("DB_DRIVER", "postgresql"),
            "host": self.env.get("DB_HOST", "localhost"),
            "port": self.env.get_int("DB_PORT", 5432),
            "user": self.env.get("DB_USER", "postgres"),
            "password": self.env.get("DB_PASSWORD", ""),
            "database": self.env.get("DB_NAME", "app"),
            "pool_size": self.env.get_int("DB_POOL_SIZE", 5),
            "ssl": self.env.get_bool("DB_SSL", False),
        }

    def generate_sqlalchemy_url(self) -> str:
        """Generate SQLAlchemy connection URL"""
        config = self.get_connection_config()

        driver_map = {
            "postgresql": "postgresql+asyncpg",
            "mysql": "mysql+aiomysql",
            "sqlite": "sqlite+aiosqlite",
        }

        driver = driver_map.get(config["driver"], config["driver"])

        if config["driver"] == "sqlite":
            return f"{driver}:///{config['database']}"

        auth = f"{config['user']}:{config['password']}" if config["password"] else config["user"]
        return f"{driver}://{auth}@{config['host']}:{config['port']}/{config['database']}"

    def generate_connection_code(self, framework: str = "sqlalchemy") -> str:
        """Generate database connection code"""
        if framework == "sqlalchemy":
            return '''"""
Database Connection - Auto-generated by IntentForge
Uses environment variables from .env file
"""

import os
from dotenv import load_dotenv
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

# Load environment variables
load_dotenv()

# Build connection URL from environment
DATABASE_URL = (
    f"{os.getenv('DB_DRIVER', 'postgresql')}+asyncpg://"
    f"{os.getenv('DB_USER', 'postgres')}:"
    f"{os.getenv('DB_PASSWORD', '')}@"
    f"{os.getenv('DB_HOST', 'localhost')}:"
    f"{os.getenv('DB_PORT', '5432')}/"
    f"{os.getenv('DB_NAME', 'app')}"
)

# Create async engine
engine = create_async_engine(
    DATABASE_URL,
    pool_size=int(os.getenv('DB_POOL_SIZE', '5')),
    echo=os.getenv('DB_ECHO', 'false').lower() == 'true'
)

# Session factory
async_session = sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False
)


async def get_session():
    """Dependency for FastAPI"""
    async with async_session() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()
'''

        elif framework == "psycopg2":
            return '''"""
Database Connection - Auto-generated by IntentForge
Uses environment variables from .env file
"""

import os
from dotenv import load_dotenv
import psycopg2
from psycopg2 import pool

# Load environment variables
load_dotenv()

# Connection pool
connection_pool = psycopg2.pool.ThreadedConnectionPool(
    minconn=1,
    maxconn=int(os.getenv('DB_POOL_SIZE', '10')),
    host=os.getenv('DB_HOST', 'localhost'),
    port=int(os.getenv('DB_PORT', '5432')),
    user=os.getenv('DB_USER', 'postgres'),
    password=os.getenv('DB_PASSWORD', ''),
    database=os.getenv('DB_NAME', 'app')
)


def get_connection():
    """Get connection from pool"""
    return connection_pool.getconn()


def release_connection(conn):
    """Return connection to pool"""
    connection_pool.putconn(conn)


def execute_query(query: str, params: dict = None):
    """Execute parameterized query safely"""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(query, params or {})
            if query.strip().upper().startswith('SELECT'):
                return cur.fetchall()
            conn.commit()
            return cur.rowcount
    finally:
        release_connection(conn)
'''

        return ""


# Global env handler instance
_env_handler: EnvHandler | None = None


def get_env() -> EnvHandler:
    """Get global environment handler"""
    global _env_handler
    if _env_handler is None:
        _env_handler = EnvHandler().load()
    return _env_handler


def configure_env(config: EnvConfig) -> EnvHandler:
    """Configure and load environment handler"""
    global _env_handler
    _env_handler = EnvHandler(config).load()
    return _env_handler
