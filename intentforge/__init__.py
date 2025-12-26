"""
IntentForge - NLP-driven Code Generation Framework
Generates backend and firmware code from natural language intents via MQTT
"""

__version__ = "0.1.0"
__author__ = "Softreck"

from .core import IntentForge, Intent, IntentResult, IntentType, TargetPlatform
from .broker import MQTTIntentBroker
from .generator import CodeGenerator, DSLType, GenerationContext
from .validator import CodeValidator, ValidationLevel, ValidationResult
from .executor import DynamicExecutor
from .cache import IntentCache
from .config import get_settings, Settings
from .schema_registry import SchemaRegistry, get_registry, SchemaType
from .env_handler import EnvHandler, get_env, configure_env, EnvConfig
from .patterns import FullstackPatterns, PatternConfig, PatternType
from .simple import generate, crud, form, query, validate, save, Forge

__all__ = [
    # Core
    "IntentForge",
    "Intent", 
    "IntentResult",
    "IntentType",
    "TargetPlatform",
    
    # Broker
    "MQTTIntentBroker",
    
    # Generator
    "CodeGenerator",
    "DSLType",
    "GenerationContext",
    
    # Validator
    "CodeValidator",
    "ValidationLevel",
    "ValidationResult",
    
    # Executor
    "DynamicExecutor",
    
    # Cache
    "IntentCache",
    
    # Config
    "get_settings",
    "Settings",
    
    # Schema Registry
    "SchemaRegistry",
    "get_registry",
    "SchemaType",
    
    # Environment Handler
    "EnvHandler",
    "get_env",
    "configure_env",
    "EnvConfig",
    
    # Patterns
    "FullstackPatterns",
    "PatternConfig",
    "PatternType",
    
    # Simple API (one-liners)
    "generate",
    "crud",
    "form",
    "query",
    "validate",
    "save",
    "Forge",
]
