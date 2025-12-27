"""
Configuration Management - Secure .env loading with validation
Supports multiple environments and secret management
"""

import os
from pathlib import Path
from typing import Optional, Dict, Any, List, Literal
from functools import lru_cache
from pydantic import Field, field_validator, SecretStr, AnyUrl
from pydantic_settings import BaseSettings, SettingsConfigDict


class DatabaseSettings(BaseSettings):
    """Database configuration with secure credential handling"""
    
    model_config = SettingsConfigDict(
        env_prefix="DB_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )
    
    # Connection settings
    driver: Literal["postgresql", "mysql", "sqlite", "mongodb"] = "postgresql"
    host: str = "localhost"
    port: int = 5432
    port_external: int = 5432
    name: str = "intentforge"
    user: str = "postgres"
    password: SecretStr = Field(default=SecretStr(""))
    
    # Pool settings
    pool_size: int = Field(default=10, ge=1, le=100)
    pool_timeout: int = Field(default=30, ge=5, le=300)
    pool_recycle: int = Field(default=3600, ge=60)
    
    # SSL settings
    ssl_mode: Optional[str] = None
    ssl_ca: Optional[Path] = None
    
    @property
    def connection_string(self) -> str:
        """Generate connection string without exposing password in logs"""
        password = self.password.get_secret_value()
        if self.driver == "sqlite":
            return f"sqlite:///{self.name}.db"
        elif self.driver == "mongodb":
            return f"mongodb://{self.user}:{password}@{self.host}:{self.port}/{self.name}"
        else:
            return f"{self.driver}://{self.user}:{password}@{self.host}:{self.port}/{self.name}"
    
    @property
    def safe_connection_string(self) -> str:
        """Connection string safe for logging (password masked)"""
        return self.connection_string.replace(
            self.password.get_secret_value(), 
            "****"
        )


class MQTTSettings(BaseSettings):
    """MQTT broker configuration"""
    
    model_config = SettingsConfigDict(
        env_prefix="MQTT_",
        env_file=".env",
        extra="ignore"
    )
    
    host: str = "localhost"
    port: int = 1883
    port_external: int = 1883
    websocket_port: int = 9001
    websocket_port_external: int = 9001
    username: Optional[str] = None
    password: Optional[SecretStr] = None
    
    # Topics
    topic_prefix: str = "intentforge"
    qos: int = Field(default=1, ge=0, le=2)
    
    # TLS
    use_tls: bool = False
    ca_cert: Optional[Path] = None
    client_cert: Optional[Path] = None
    client_key: Optional[Path] = None


class LLMSettings(BaseSettings):
    """LLM API configuration"""
    
    model_config = SettingsConfigDict(
        env_prefix="LLM_",
        env_file=".env",
        extra="ignore"
    )
    
    provider: Literal["anthropic", "openai", "local"] = "anthropic"
    api_key: SecretStr = Field(default=SecretStr(""))
    model: str = "claude-sonnet-4-5-20250929"
    max_tokens: int = Field(default=4096, ge=100, le=100000)
    temperature: float = Field(default=0.1, ge=0, le=2)
    
    # Rate limiting
    requests_per_minute: int = Field(default=60, ge=1)
    tokens_per_minute: int = Field(default=100000, ge=1000)
    
    # Retry settings
    max_retries: int = Field(default=3, ge=0, le=10)
    retry_delay: float = Field(default=1.0, ge=0.1)
    
    @property
    def api_base_url(self) -> str:
        urls = {
            "anthropic": "https://api.anthropic.com/v1",
            "openai": "https://api.openai.com/v1",
            "local": "http://localhost:8080/v1"
        }
        return urls.get(self.provider, urls["anthropic"])


class CacheSettings(BaseSettings):
    """Cache configuration"""
    
    model_config = SettingsConfigDict(
        env_prefix="CACHE_",
        env_file=".env",
        extra="ignore"
    )
    
    backend: Literal["memory", "redis", "file"] = "memory"
    redis_url: Optional[str] = "redis://localhost:6379/0"
    file_path: Path = Path("/tmp/intentforge_cache")
    
    # TTL settings
    default_ttl: int = Field(default=3600, ge=60)  # 1 hour
    max_entries: int = Field(default=10000, ge=100)
    
    # Fingerprint settings
    include_context_in_fingerprint: bool = True
    include_constraints_in_fingerprint: bool = True


class ValidationSettings(BaseSettings):
    """Code validation configuration"""
    
    model_config = SettingsConfigDict(
        env_prefix="VALIDATION_",
        env_file=".env",
        extra="ignore"
    )
    
    # Validation levels
    enable_syntax_check: bool = True
    enable_security_check: bool = True
    enable_semantic_check: bool = True
    
    # Security settings
    max_code_length: int = Field(default=50000, ge=1000)
    forbidden_imports: List[str] = Field(default_factory=lambda: [
        "os.system", "subprocess.call", "eval", "exec",
        "__import__", "compile", "open"  # open allowed only in sandbox
    ])
    forbidden_patterns: List[str] = Field(default_factory=lambda: [
        r"rm\s+-rf",
        r"DROP\s+TABLE",
        r"DELETE\s+FROM.*WHERE\s+1\s*=\s*1",
        r";\s*--",  # SQL comment injection
    ])
    
    # Sandbox settings
    sandbox_timeout: int = Field(default=30, ge=5, le=300)
    sandbox_memory_limit: str = "256m"


class RedisSettings(BaseSettings):
    """Redis configuration"""
    
    model_config = SettingsConfigDict(
        env_prefix="REDIS_",
        env_file=".env",
        extra="ignore"
    )
    
    host: str = "localhost"
    port: int = 6379
    port_external: int = 6379
    url: str = "redis://localhost:6379/0"


class Settings(BaseSettings):
    """Main application settings"""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )
    
    # Application
    app_name: str = "IntentForge"
    app_version: str = "0.1.0"
    environment: Literal["development", "staging", "production"] = "development"
    debug: bool = False
    
    # Server (using APP_ prefix for clarity)
    app_host: str = Field(default="0.0.0.0", alias="APP_HOST")
    app_port: int = Field(default=8000, alias="APP_PORT")
    app_workers: int = Field(default=4, ge=1, le=32, alias="APP_WORKERS")
    
    # Web port
    web_port: int = Field(default=80, alias="WEB_PORT")
    
    # Logging
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    log_format: str = "json"
    
    # Component settings
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    mqtt: MQTTSettings = Field(default_factory=MQTTSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    cache: CacheSettings = Field(default_factory=CacheSettings)
    redis: RedisSettings = Field(default_factory=RedisSettings)
    validation: ValidationSettings = Field(default_factory=ValidationSettings)
    
    # Backwards compatibility properties
    @property
    def host(self) -> str:
        return self.app_host
    
    @property
    def port(self) -> int:
        return self.app_port
    
    @property
    def workers(self) -> int:
        return self.app_workers
    
    @field_validator("environment", mode="before")
    @classmethod
    def validate_environment(cls, v: str) -> str:
        env_aliases = {
            "dev": "development",
            "prod": "production",
            "stage": "staging"
        }
        return env_aliases.get(v.lower(), v.lower())
    
    @property
    def is_production(self) -> bool:
        return self.environment == "production"
    
    @property
    def is_development(self) -> bool:
        return self.environment == "development"


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()


def get_database_settings() -> DatabaseSettings:
    """Get database settings"""
    return get_settings().database


def get_mqtt_settings() -> MQTTSettings:
    """Get MQTT settings"""
    return get_settings().mqtt


def get_llm_settings() -> LLMSettings:
    """Get LLM settings"""
    return get_settings().llm


def get_redis_settings() -> RedisSettings:
    """Get Redis settings"""
    return get_settings().redis


def get_cache_settings() -> CacheSettings:
    """Get cache settings"""
    return get_settings().cache


def get_validation_settings() -> ValidationSettings:
    """Get validation settings"""
    return get_settings().validation


# Environment-specific configuration loader
class ConfigLoader:
    """Load configuration from multiple sources"""
    
    def __init__(self, env_files: Optional[List[str]] = None):
        self.env_files = env_files or [
            ".env",
            ".env.local",
            f".env.{os.getenv('ENVIRONMENT', 'development')}"
        ]
    
    def load(self) -> Settings:
        """Load settings with environment overrides"""
        # Load base .env files
        for env_file in self.env_files:
            if Path(env_file).exists():
                self._load_env_file(env_file)
        
        # Clear cache and reload
        get_settings.cache_clear()
        return get_settings()
    
    def _load_env_file(self, path: str) -> None:
        """Load a single .env file"""
        from dotenv import load_dotenv
        load_dotenv(path, override=True)
    
    def to_dict(self, mask_secrets: bool = True) -> Dict[str, Any]:
        """Export settings as dictionary"""
        settings = get_settings()
        data = settings.model_dump()
        
        if mask_secrets:
            self._mask_secrets(data)
        
        return data
    
    def _mask_secrets(self, data: Dict[str, Any], depth: int = 0) -> None:
        """Recursively mask secret values"""
        if depth > 10:  # Prevent infinite recursion
            return
            
        for key, value in data.items():
            if isinstance(value, dict):
                self._mask_secrets(value, depth + 1)
            elif "password" in key.lower() or "secret" in key.lower() or "key" in key.lower():
                if isinstance(value, str) and value:
                    data[key] = "****"
