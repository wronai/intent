"""
IntentForge Plugin & Middleware System
======================================

Enables code reuse across all service layers through:
1. Middleware - preprocessing/postprocessing for all requests
2. Plugins - extend functionality without modifying core
3. Hooks - lifecycle events for customization
4. Decorators - reusable behaviors

Usage:
    from intentforge.plugins import PluginManager, middleware, hook

    # Register middleware
    @middleware('before')
    def log_request(request):
        print(f"Request: {request}")
        return request

    # Register plugin
    class MyPlugin(BasePlugin):
        def on_form_submit(self, form_id, data):
            # Custom logic for all forms
            pass

    plugins.register(MyPlugin())
"""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps
from typing import Any, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


# =============================================================================
# Types
# =============================================================================


class MiddlewarePhase(str, Enum):
    BEFORE = "before"
    AFTER = "after"
    ERROR = "error"


class HookEvent(str, Enum):
    # Request lifecycle
    REQUEST_START = "request:start"
    REQUEST_END = "request:end"
    REQUEST_ERROR = "request:error"

    # Form events
    FORM_VALIDATE = "form:validate"
    FORM_SUBMIT = "form:submit"
    FORM_SUCCESS = "form:success"

    # Payment events
    PAYMENT_INIT = "payment:init"
    PAYMENT_SUCCESS = "payment:success"
    PAYMENT_FAILED = "payment:failed"

    # Email events
    EMAIL_SEND = "email:send"
    EMAIL_SENT = "email:sent"

    # Data events
    DATA_QUERY = "data:query"
    DATA_CREATE = "data:create"
    DATA_UPDATE = "data:update"
    DATA_DELETE = "data:delete"

    # Camera events
    CAMERA_SNAPSHOT = "camera:snapshot"
    CAMERA_DETECT = "camera:detect"
    CAMERA_ALERT = "camera:alert"

    # LLM events
    LLM_GENERATE = "llm:generate"
    LLM_RESPONSE = "llm:response"

    # Cache events
    CACHE_HIT = "cache:hit"
    CACHE_MISS = "cache:miss"


@dataclass
class MiddlewareResult:
    """Result from middleware execution"""

    data: Any
    modified: bool = False
    skip_next: bool = False
    error: Exception | None = None


@dataclass
class HookContext:
    """Context passed to hooks"""

    event: HookEvent
    data: dict[str, Any]
    timestamp: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)


# =============================================================================
# Middleware System
# =============================================================================


class MiddlewareChain:
    """
    Chain of middleware functions executed in order.

    Usage:
        chain = MiddlewareChain()

        @chain.add(MiddlewarePhase.BEFORE)
        def validate(data):
            if not data.get('email'):
                raise ValueError("Email required")
            return data

        @chain.add(MiddlewarePhase.AFTER)
        def log_result(data):
            print(f"Result: {data}")
            return data

        result = await chain.execute(my_data)
    """

    def __init__(self):
        self._middleware: dict[MiddlewarePhase, list[Callable]] = {
            phase: [] for phase in MiddlewarePhase
        }

    def add(self, phase: MiddlewarePhase, priority: int = 0):
        """Decorator to add middleware"""

        def decorator(func: Callable) -> Callable:
            self._middleware[phase].append((priority, func))
            self._middleware[phase].sort(key=lambda x: x[0])
            return func

        return decorator

    def use(self, func: Callable, phase: MiddlewarePhase = MiddlewarePhase.BEFORE):
        """Add middleware function directly"""
        self._middleware[phase].append((0, func))

    async def execute(self, data: Any, handler: Callable | None = None) -> MiddlewareResult:
        """Execute middleware chain"""
        result = MiddlewareResult(data=data)

        try:
            # BEFORE phase
            for _, mw in self._middleware[MiddlewarePhase.BEFORE]:
                output = await self._call(mw, result.data)
                if output is not None:
                    result.data = output
                    result.modified = True

            # Main handler
            if handler:
                result.data = await self._call(handler, result.data)
                result.modified = True

            # AFTER phase
            for _, mw in self._middleware[MiddlewarePhase.AFTER]:
                output = await self._call(mw, result.data)
                if output is not None:
                    result.data = output
                    result.modified = True

        except Exception as e:
            result.error = e

            # ERROR phase
            for _, mw in self._middleware[MiddlewarePhase.ERROR]:
                try:
                    output = await self._call(mw, e, result.data)
                    if output is not None:
                        result.data = output
                        result.error = None
                        break
                except Exception:
                    pass

            if result.error:
                raise result.error

        return result

    async def _call(self, func: Callable, *args) -> Any:
        """Call sync or async function"""
        result = func(*args)
        if asyncio.iscoroutine(result):
            return await result
        return result


# =============================================================================
# Global Middleware Registry
# =============================================================================

_global_chain = MiddlewareChain()


def middleware(phase: MiddlewarePhase | str = MiddlewarePhase.BEFORE, priority: int = 0):
    """
    Global middleware decorator

    Usage:
        @middleware('before')
        def validate_request(data):
            return data

        @middleware('after', priority=10)
        def log_response(data):
            logger.info(f"Response: {data}")
            return data
    """
    if isinstance(phase, str):
        phase = MiddlewarePhase(phase)
    return _global_chain.add(phase, priority)


def use_middleware(func: Callable, phase: MiddlewarePhase = MiddlewarePhase.BEFORE):
    """Add middleware function globally"""
    _global_chain.use(func, phase)


# =============================================================================
# Hook System
# =============================================================================


class HookRegistry:
    """
    Event hook registry for lifecycle events.

    Usage:
        hooks = HookRegistry()

        @hooks.on(HookEvent.FORM_SUBMIT)
        def on_form_submit(ctx: HookContext):
            print(f"Form submitted: {ctx.data}")

        await hooks.emit(HookEvent.FORM_SUBMIT, {'form_id': 'contact'})
    """

    def __init__(self):
        self._handlers: dict[HookEvent, list[Callable]] = defaultdict(list)

    def on(self, event: HookEvent | str):
        """Decorator to register hook handler"""
        if isinstance(event, str):
            event = HookEvent(event)

        def decorator(func: Callable) -> Callable:
            self._handlers[event].append(func)
            return func

        return decorator

    def register(self, event: HookEvent, handler: Callable):
        """Register handler directly"""
        self._handlers[event].append(handler)

    def unregister(self, event: HookEvent, handler: Callable):
        """Unregister handler"""
        if handler in self._handlers[event]:
            self._handlers[event].remove(handler)

    async def emit(
        self,
        event: HookEvent,
        data: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> list[Any]:
        """Emit event to all registered handlers"""
        ctx = HookContext(event=event, data=data or {}, metadata=metadata or {})

        results = []
        for handler in self._handlers[event]:
            try:
                result = handler(ctx)
                if asyncio.iscoroutine(result):
                    result = await result
                results.append(result)
            except Exception as e:
                logger.error(f"Hook error for {event}: {e}")

        return results


# Global hook registry
hooks = HookRegistry()


def hook(event: HookEvent | str):
    """Global hook decorator"""
    return hooks.on(event)


# =============================================================================
# Plugin System
# =============================================================================


class BasePlugin(ABC):
    """
    Base class for plugins.

    Plugins can implement hooks and provide services.

    Usage:
        class AnalyticsPlugin(BasePlugin):
            name = "analytics"

            def setup(self, config):
                self.tracker = AnalyticsTracker(config)

            @hook(HookEvent.FORM_SUBMIT)
            def track_form(self, ctx):
                self.tracker.track('form_submit', ctx.data)

            @hook(HookEvent.PAYMENT_SUCCESS)
            def track_payment(self, ctx):
                self.tracker.track('payment', ctx.data)
    """

    name: str = "base"
    version: str = "1.0.0"

    def __init__(self):
        self._enabled = True

    @abstractmethod
    def setup(self, config: dict[str, Any]) -> None:
        """Initialize plugin with configuration"""
        pass

    def teardown(self) -> None:
        """Cleanup when plugin is disabled"""
        pass

    def enable(self) -> None:
        self._enabled = True

    def disable(self) -> None:
        self._enabled = False

    @property
    def enabled(self) -> bool:
        return self._enabled


class PluginManager:
    """
    Manages plugin lifecycle and registration.

    Usage:
        plugins = PluginManager()

        # Register plugins
        plugins.register(AnalyticsPlugin())
        plugins.register(NotificationPlugin())

        # Configure
        plugins.setup({
            'analytics': {'api_key': 'xxx'},
            'notification': {'webhook': 'https://...'}
        })

        # Use in services
        plugins.emit(HookEvent.FORM_SUBMIT, data)
    """

    def __init__(self):
        self._plugins: dict[str, BasePlugin] = {}
        self._hooks = HookRegistry()

    def register(self, plugin: BasePlugin) -> None:
        """Register a plugin"""
        self._plugins[plugin.name] = plugin

        # Auto-register hook methods
        for attr_name in dir(plugin):
            attr = getattr(plugin, attr_name)
            if hasattr(attr, "_hook_event"):
                self._hooks.register(attr._hook_event, attr)

        logger.info(f"Plugin registered: {plugin.name} v{plugin.version}")

    def unregister(self, name: str) -> None:
        """Unregister a plugin"""
        if name in self._plugins:
            self._plugins[name].teardown()
            del self._plugins[name]

    def get(self, name: str) -> BasePlugin | None:
        """Get plugin by name"""
        return self._plugins.get(name)

    def setup(self, config: dict[str, dict[str, Any]]) -> None:
        """Setup all plugins with configuration"""
        for name, plugin in self._plugins.items():
            plugin_config = config.get(name, {})
            plugin.setup(plugin_config)

    async def emit(self, event: HookEvent, data: dict[str, Any] | None = None) -> list[Any]:
        """Emit event to all plugin hooks"""
        return await self._hooks.emit(event, data)

    def list(self) -> list[str]:
        """List registered plugins"""
        return list(self._plugins.keys())


# Global plugin manager
plugins = PluginManager()


# =============================================================================
# Reusable Decorators
# =============================================================================


def cached(ttl: int = 3600, key_func: Callable | None = None):
    """
    Caching decorator for service methods.

    Usage:
        @cached(ttl=300)
        async def get_products():
            return await db.query("SELECT * FROM products")
    """

    def decorator(func: Callable) -> Callable:
        cache = {}

        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Generate cache key
            if key_func:
                key = key_func(*args, **kwargs)
            else:
                key = str(args) + str(sorted(kwargs.items()))

            # Check cache
            if key in cache:
                entry = cache[key]
                if time.time() - entry["time"] < ttl:
                    await hooks.emit(HookEvent.CACHE_HIT, {"key": key})
                    return entry["value"]

            # Call function
            await hooks.emit(HookEvent.CACHE_MISS, {"key": key})
            result = await func(*args, **kwargs)

            # Store in cache
            cache[key] = {"value": result, "time": time.time()}
            return result

        wrapper.cache_clear = lambda: cache.clear()
        return wrapper

    return decorator


def retry(max_attempts: int = 3, delay: float = 1.0, backoff: float = 2.0):
    """
    Retry decorator with exponential backoff.

    Usage:
        @retry(max_attempts=5, delay=1.0)
        async def call_external_api():
            response = await http.get('https://api.example.com')
            return response
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            current_delay = delay
            last_error = None

            for attempt in range(max_attempts):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_error = e
                    if attempt < max_attempts - 1:
                        await asyncio.sleep(current_delay)
                        current_delay *= backoff

            raise last_error

        return wrapper

    return decorator


def rate_limit(calls: int, period: float):
    """
    Rate limiting decorator.

    Usage:
        @rate_limit(calls=100, period=60)  # 100 calls per minute
        async def api_endpoint(request):
            return process(request)
    """

    def decorator(func: Callable) -> Callable:
        timestamps = []

        @wraps(func)
        async def wrapper(*args, **kwargs):
            now = time.time()

            # Remove old timestamps
            timestamps[:] = [t for t in timestamps if now - t < period]

            if len(timestamps) >= calls:
                wait_time = period - (now - timestamps[0])
                raise Exception(f"Rate limit exceeded. Try again in {wait_time:.1f}s")

            timestamps.append(now)
            return await func(*args, **kwargs)

        return wrapper

    return decorator


def validate_input(schema: dict[str, Any]):
    """
    Input validation decorator using JSON Schema.

    Usage:
        @validate_input({
            'type': 'object',
            'required': ['email', 'message'],
            'properties': {
                'email': {'type': 'string', 'format': 'email'},
                'message': {'type': 'string', 'minLength': 10}
            }
        })
        async def submit_form(data):
            return save_form(data)
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(data: dict[str, Any], *args, **kwargs):
            # Simple validation (use jsonschema for full validation)
            errors = []

            for field_name in schema.get("required", []):
                if field_name not in data:
                    errors.append(f"Missing required field: {field_name}")

            for prop_name, rules in schema.get("properties", {}).items():
                if prop_name in data:
                    value = data[prop_name]

                    if rules.get("type") == "string" and not isinstance(value, str):
                        errors.append(f"{prop_name} must be a string")

                    if rules.get("minLength") and len(str(value)) < rules["minLength"]:
                        errors.append(
                            f"{prop_name} must be at least {rules['minLength']} characters"
                        )

            if errors:
                raise ValueError(f"Validation failed: {', '.join(errors)}")

            return await func(data, *args, **kwargs)

        return wrapper

    return decorator


def audit_log(action: str):
    """
    Audit logging decorator.

    Usage:
        @audit_log("user:delete")
        async def delete_user(user_id):
            return await db.delete('users', user_id)
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start = time.time()
            result = None
            error = None

            try:
                result = await func(*args, **kwargs)
                return result
            except Exception as e:
                error = e
                raise
            finally:
                # Log audit entry
                logger.info(
                    f"AUDIT: {action} | args={args} | kwargs={kwargs} | "
                    f"duration={time.time() - start:.3f}s | "
                    f"error={error}"
                )

        return wrapper

    return decorator


# =============================================================================
# Built-in Plugins
# =============================================================================


class LoggingPlugin(BasePlugin):
    """Logs all events for debugging"""

    name = "logging"

    def setup(self, config: dict[str, Any]) -> None:
        self.log_level = config.get("level", "INFO")

    def _log(self, ctx: HookContext):
        logger.log(getattr(logging, self.log_level), f"[{ctx.event.value}] {ctx.data}")


class MetricsPlugin(BasePlugin):
    """Collects metrics for monitoring"""

    name = "metrics"

    def setup(self, config: dict[str, Any]) -> None:
        self.metrics = defaultdict(list)

    def track(self, event: str, value: float = 1.0):
        self.metrics[event].append({"value": value, "timestamp": time.time()})

    def get_stats(self, event: str) -> dict[str, float]:
        values = [m["value"] for m in self.metrics[event]]
        if not values:
            return {}
        return {
            "count": len(values),
            "sum": sum(values),
            "avg": sum(values) / len(values),
            "min": min(values),
            "max": max(values),
        }


class NotificationPlugin(BasePlugin):
    """Sends notifications on events"""

    name = "notification"

    def setup(self, config: dict[str, Any]) -> None:
        self.webhook_url = config.get("webhook_url")
        self.email_to = config.get("email_to")

    async def notify(self, title: str, message: str):
        if self.webhook_url:
            import httpx

            async with httpx.AsyncClient() as client:
                await client.post(self.webhook_url, json={"title": title, "message": message})


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "BasePlugin",
    "HookContext",
    "HookEvent",
    "HookRegistry",
    # Built-in plugins
    "LoggingPlugin",
    "MetricsPlugin",
    # Core
    "MiddlewareChain",
    # Types
    "MiddlewarePhase",
    "MiddlewareResult",
    "NotificationPlugin",
    "PluginManager",
    "audit_log",
    "cached",
    "hook",
    # Global instances
    "hooks",
    # Decorators
    "middleware",
    "plugins",
    "rate_limit",
    "retry",
    "use_middleware",
    "validate_input",
]
