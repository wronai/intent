"""
Core module - Intent definitions and main IntentForge orchestrator
"""

import hashlib
import json
import uuid
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class IntentType(Enum):
    """Types of intents the system can process"""

    API_ENDPOINT = "api_endpoint"
    DATABASE_SCHEMA = "database_schema"
    FIRMWARE_FUNCTION = "firmware_function"
    EVENT_HANDLER = "event_handler"
    WORKFLOW = "workflow"
    VALIDATION_RULE = "validation_rule"
    UI_COMPONENT = "ui_component"
    MQTT_HANDLER = "mqtt_handler"


class TargetPlatform(Enum):
    """Target platforms for code generation"""

    PYTHON_FASTAPI = "python_fastapi"
    PYTHON_FLASK = "python_flask"
    NODEJS_EXPRESS = "nodejs_express"
    ARDUINO_CPP = "arduino_cpp"
    ESP32_MICROPYTHON = "esp32_micropython"
    JETSON_PYTHON = "jetson_python"
    RASPBERRY_PI = "raspberry_pi"
    GENERIC_PYTHON = "generic_python"


@dataclass
class Intent:
    """
    Represents a natural language intent that should be converted to code
    """

    description: str
    intent_type: IntentType = IntentType.API_ENDPOINT
    target_platform: TargetPlatform = TargetPlatform.PYTHON_FASTAPI
    context: dict[str, Any] = field(default_factory=dict)
    constraints: list[str] = field(default_factory=list)
    examples: list[dict[str, Any]] = field(default_factory=list)
    priority: int = 5

    # Auto-generated fields
    intent_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict[str, Any]:
        return {
            "intent_id": self.intent_id,
            "description": self.description,
            "intent_type": self.intent_type.value,
            "target_platform": self.target_platform.value,
            "context": self.context,
            "constraints": self.constraints,
            "examples": self.examples,
            "priority": self.priority,
            "created_at": self.created_at.isoformat(),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Intent":
        return cls(
            intent_id=data.get("intent_id", str(uuid.uuid4())),
            description=data["description"],
            intent_type=IntentType(data.get("intent_type", "api_endpoint")),
            target_platform=TargetPlatform(data.get("target_platform", "python_fastapi")),
            context=data.get("context", {}),
            constraints=data.get("constraints", []),
            examples=data.get("examples", []),
            priority=data.get("priority", 5),
        )

    def get_fingerprint(self) -> str:
        """Generate unique fingerprint for caching"""
        content = f"{self.description}|{self.intent_type.value}|{self.target_platform.value}"
        content += f"|{json.dumps(self.context, sort_keys=True)}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]


@dataclass
class IntentResult:
    """
    Result of intent processing - generated code and metadata
    """

    intent_id: str
    success: bool
    generated_code: str | None = None
    language: str | None = None
    validation_passed: bool = False
    validation_errors: list[str] = field(default_factory=list)
    execution_result: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    processing_time_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict())


class IntentForge:
    """
    Main orchestrator for intent-to-code generation
    Coordinates between MQTT broker, code generator, validator, and executor
    """

    def __init__(
        self,
        mqtt_broker: str = "localhost",
        mqtt_port: int = 1883,
        api_key: str | None = None,
        model: str = "claude-sonnet-4-5-20250929",
        enable_auto_deploy: bool = False,
        sandbox_mode: bool = True,
    ):
        self.mqtt_broker = mqtt_broker
        self.mqtt_port = mqtt_port
        self.api_key = api_key
        self.model = model
        self.enable_auto_deploy = enable_auto_deploy
        self.sandbox_mode = sandbox_mode

        # Component instances (lazy loaded)
        self._broker = None
        self._generator = None
        self._validator = None
        self._executor = None

        # Intent registry
        self._intents: dict[str, Intent] = {}
        self._results: dict[str, IntentResult] = {}

        # Observers/Hooks
        self._observers: dict[str, list[Callable]] = {
            "on_intent_received": [],
            "on_code_generated": [],
            "on_validation_complete": [],
            "on_deployed": [],
            "on_error": [],
        }

        # Code cache
        self._cache: dict[str, IntentResult] = {}

    @property
    def broker(self):
        if self._broker is None:
            from .broker import MQTTIntentBroker

            self._broker = MQTTIntentBroker(host=self.mqtt_broker, port=self.mqtt_port, forge=self)
        return self._broker

    @property
    def generator(self):
        if self._generator is None:
            from .generator import CodeGenerator

            self._generator = CodeGenerator(api_key=self.api_key, model=self.model)
        return self._generator

    @property
    def validator(self):
        if self._validator is None:
            from .validator import CodeValidator

            self._validator = CodeValidator(sandbox_mode=self.sandbox_mode)
        return self._validator

    @property
    def executor(self):
        if self._executor is None:
            from .executor import DynamicExecutor

            self._executor = DynamicExecutor(sandbox_mode=self.sandbox_mode)
        return self._executor

    def register_observer(self, event: str, callback: Callable) -> None:
        """Register callback for specific events"""
        if event in self._observers:
            self._observers[event].append(callback)

    def _emit(self, event: str, data: Any) -> None:
        """Emit event to all registered observers"""
        for callback in self._observers.get(event, []):
            try:
                callback(data)
            except Exception as e:
                print(f"Observer error for {event}: {e}")

    async def process_intent(self, intent: Intent) -> IntentResult:
        """
        Main processing pipeline for an intent
        """
        import time

        start_time = time.time()

        # Check cache
        fingerprint = intent.get_fingerprint()
        if fingerprint in self._cache:
            cached = self._cache[fingerprint]
            cached.metadata["from_cache"] = True
            return cached

        self._intents[intent.intent_id] = intent
        self._emit("on_intent_received", intent)

        try:
            # Generate code
            generated_code, language = await self.generator.generate(intent)

            self._emit(
                "on_code_generated",
                {"intent_id": intent.intent_id, "code": generated_code, "language": language},
            )

            # Validate
            is_valid, errors = await self.validator.validate(
                generated_code, language, intent.target_platform
            )

            self._emit(
                "on_validation_complete",
                {"intent_id": intent.intent_id, "valid": is_valid, "errors": errors},
            )

            # Execute/Deploy if enabled and valid
            execution_result = None
            if is_valid and self.enable_auto_deploy:
                execution_result = await self.executor.execute(
                    generated_code, language, intent.target_platform
                )
                self._emit(
                    "on_deployed", {"intent_id": intent.intent_id, "result": execution_result}
                )

            result = IntentResult(
                intent_id=intent.intent_id,
                success=True,
                generated_code=generated_code,
                language=language,
                validation_passed=is_valid,
                validation_errors=errors,
                execution_result=execution_result,
                processing_time_ms=(time.time() - start_time) * 1000,
            )

        except Exception as e:
            self._emit("on_error", {"intent_id": intent.intent_id, "error": str(e)})
            result = IntentResult(
                intent_id=intent.intent_id,
                success=False,
                validation_errors=[str(e)],
                processing_time_ms=(time.time() - start_time) * 1000,
            )

        self._results[intent.intent_id] = result
        self._cache[fingerprint] = result

        return result

    def start(self) -> None:
        """Start the MQTT broker listener"""
        self.broker.connect()
        self.broker.start_listening()

    def stop(self) -> None:
        """Stop all components"""
        if self._broker:
            self._broker.disconnect()

    # Convenience methods for quick intent creation
    def create_api_endpoint(
        self, description: str, method: str = "GET", auth_required: bool = False, **kwargs
    ) -> Intent:
        """Quick helper to create API endpoint intent"""
        return Intent(
            description=description,
            intent_type=IntentType.API_ENDPOINT,
            context={"http_method": method, "auth_required": auth_required, **kwargs},
        )

    def create_firmware_function(
        self,
        description: str,
        target: TargetPlatform = TargetPlatform.ESP32_MICROPYTHON,
        pins: list[int] | None = None,
        **kwargs,
    ) -> Intent:
        """Quick helper to create firmware function intent"""
        return Intent(
            description=description,
            intent_type=IntentType.FIRMWARE_FUNCTION,
            target_platform=target,
            context={"gpio_pins": pins or [], **kwargs},
        )

    def create_event_handler(self, description: str, event_source: str, **kwargs) -> Intent:
        """Quick helper to create event handler intent"""
        return Intent(
            description=description,
            intent_type=IntentType.EVENT_HANDLER,
            context={"event_source": event_source, **kwargs},
        )
