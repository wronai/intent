"""
IntentForge LLM Provider Abstraction
Supports: Anthropic, OpenAI, Ollama, LiteLLM (universal proxy)

Usage:
    # Direct provider
    llm = get_llm_provider("ollama", model="llama3")
    response = await llm.generate("Create REST API for users")
    
    # Via LiteLLM (recommended - unified interface)
    llm = get_llm_provider("litellm", model="ollama/llama3")
    response = await llm.generate("Create REST API for users")
"""

import os
import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, AsyncIterator, Literal
from enum import Enum

logger = logging.getLogger(__name__)


# =============================================================================
# Types and Configuration
# =============================================================================

class LLMProvider(str, Enum):
    """Supported LLM providers"""
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    OLLAMA = "ollama"
    LITELLM = "litellm"
    LOCAL = "local"  # Generic OpenAI-compatible endpoint


@dataclass
class LLMConfig:
    """LLM configuration"""
    provider: LLMProvider = LLMProvider.ANTHROPIC
    model: str = "claude-sonnet-4-5-20250929"
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    
    # Generation parameters
    max_tokens: int = 4096
    temperature: float = 0.1
    top_p: float = 1.0
    
    # Timeouts and retries
    timeout: float = 60.0
    max_retries: int = 3
    retry_delay: float = 1.0
    
    # Ollama specific
    ollama_host: str = "http://localhost:11434"
    
    # LiteLLM specific
    litellm_api_base: str = "http://localhost:4000"  # LiteLLM proxy server
    
    @classmethod
    def from_env(cls) -> "LLMConfig":
        """Create config from environment variables"""
        provider_str = os.getenv("LLM_PROVIDER", "anthropic").lower()
        
        try:
            provider = LLMProvider(provider_str)
        except ValueError:
            provider = LLMProvider.ANTHROPIC
        
        return cls(
            provider=provider,
            model=os.getenv("LLM_MODEL", cls._default_model(provider)),
            api_key=os.getenv("LLM_API_KEY") or os.getenv("ANTHROPIC_API_KEY") or os.getenv("OPENAI_API_KEY"),
            api_base=os.getenv("LLM_API_BASE"),
            max_tokens=int(os.getenv("LLM_MAX_TOKENS", "4096")),
            temperature=float(os.getenv("LLM_TEMPERATURE", "0.1")),
            timeout=float(os.getenv("LLM_TIMEOUT", "60")),
            ollama_host=os.getenv("OLLAMA_HOST", "http://localhost:11434"),
            litellm_api_base=os.getenv("LITELLM_API_BASE", "http://localhost:4000")
        )
    
    @staticmethod
    def _default_model(provider: LLMProvider) -> str:
        defaults = {
            LLMProvider.ANTHROPIC: "claude-sonnet-4-5-20250929",
            LLMProvider.OPENAI: "gpt-4o",
            LLMProvider.OLLAMA: "llama3",
            LLMProvider.LITELLM: "gpt-4o",
            LLMProvider.LOCAL: "local-model"
        }
        return defaults.get(provider, "claude-sonnet-4-5-20250929")


@dataclass
class LLMResponse:
    """Unified LLM response"""
    content: str
    model: str
    provider: str
    
    # Usage stats
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    
    # Timing
    latency_ms: float = 0
    
    # Metadata
    finish_reason: str = "stop"
    raw_response: Optional[Dict[str, Any]] = None


# =============================================================================
# Base Provider Interface
# =============================================================================

class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers"""
    
    def __init__(self, config: LLMConfig):
        self.config = config
    
    @abstractmethod
    async def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        **kwargs
    ) -> LLMResponse:
        """Generate completion from prompt"""
        pass
    
    @abstractmethod
    async def generate_stream(
        self,
        prompt: str,
        system: Optional[str] = None,
        **kwargs
    ) -> AsyncIterator[str]:
        """Stream completion from prompt"""
        pass
    
    async def generate_code(
        self,
        description: str,
        language: str = "python",
        context: Optional[Dict[str, Any]] = None
    ) -> LLMResponse:
        """Generate code from natural language description"""
        
        system = f"""You are an expert {language} developer. Generate clean, production-ready code.
Rules:
- Return ONLY code, no explanations
- Include proper error handling
- Use type hints (Python) or TypeScript (JavaScript)
- Follow best practices and security guidelines
- Use parameterized queries for SQL
- Never hardcode credentials"""
        
        prompt = f"Generate {language} code for: {description}"
        
        if context:
            prompt += f"\n\nContext:\n{context}"
        
        return await self.generate(prompt, system=system)
    
    def _measure_latency(self, start_time: float) -> float:
        """Calculate latency in milliseconds"""
        import time
        return (time.time() - start_time) * 1000


# =============================================================================
# Anthropic Provider
# =============================================================================

class AnthropicProvider(BaseLLMProvider):
    """Anthropic Claude provider"""
    
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self.api_key = config.api_key or os.getenv("ANTHROPIC_API_KEY")
        
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY required")
    
    async def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        **kwargs
    ) -> LLMResponse:
        import httpx
        import time
        
        start_time = time.time()
        
        messages = [{"role": "user", "content": prompt}]
        
        payload = {
            "model": self.config.model,
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
            "temperature": kwargs.get("temperature", self.config.temperature),
            "messages": messages
        }
        
        if system:
            payload["system"] = system
        
        async with httpx.AsyncClient(timeout=self.config.timeout) as client:
            response = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": self.api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json"
                },
                json=payload
            )
            response.raise_for_status()
            data = response.json()
        
        return LLMResponse(
            content=data["content"][0]["text"],
            model=data["model"],
            provider="anthropic",
            input_tokens=data["usage"]["input_tokens"],
            output_tokens=data["usage"]["output_tokens"],
            total_tokens=data["usage"]["input_tokens"] + data["usage"]["output_tokens"],
            latency_ms=self._measure_latency(start_time),
            finish_reason=data.get("stop_reason", "stop"),
            raw_response=data
        )
    
    async def generate_stream(
        self,
        prompt: str,
        system: Optional[str] = None,
        **kwargs
    ) -> AsyncIterator[str]:
        import httpx
        
        messages = [{"role": "user", "content": prompt}]
        
        payload = {
            "model": self.config.model,
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
            "temperature": kwargs.get("temperature", self.config.temperature),
            "messages": messages,
            "stream": True
        }
        
        if system:
            payload["system"] = system
        
        async with httpx.AsyncClient(timeout=self.config.timeout) as client:
            async with client.stream(
                "POST",
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": self.api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json"
                },
                json=payload
            ) as response:
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        import json
                        data = json.loads(line[6:])
                        if data["type"] == "content_block_delta":
                            yield data["delta"]["text"]


# =============================================================================
# OpenAI Provider
# =============================================================================

class OpenAIProvider(BaseLLMProvider):
    """OpenAI GPT provider"""
    
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self.api_key = config.api_key or os.getenv("OPENAI_API_KEY")
        self.api_base = config.api_base or "https://api.openai.com/v1"
        
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY required")
    
    async def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        **kwargs
    ) -> LLMResponse:
        import httpx
        import time
        
        start_time = time.time()
        
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        
        payload = {
            "model": self.config.model,
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
            "temperature": kwargs.get("temperature", self.config.temperature),
            "messages": messages
        }
        
        async with httpx.AsyncClient(timeout=self.config.timeout) as client:
            response = await client.post(
                f"{self.api_base}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json=payload
            )
            response.raise_for_status()
            data = response.json()
        
        return LLMResponse(
            content=data["choices"][0]["message"]["content"],
            model=data["model"],
            provider="openai",
            input_tokens=data["usage"]["prompt_tokens"],
            output_tokens=data["usage"]["completion_tokens"],
            total_tokens=data["usage"]["total_tokens"],
            latency_ms=self._measure_latency(start_time),
            finish_reason=data["choices"][0].get("finish_reason", "stop"),
            raw_response=data
        )
    
    async def generate_stream(
        self,
        prompt: str,
        system: Optional[str] = None,
        **kwargs
    ) -> AsyncIterator[str]:
        import httpx
        
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        
        payload = {
            "model": self.config.model,
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
            "temperature": kwargs.get("temperature", self.config.temperature),
            "messages": messages,
            "stream": True
        }
        
        async with httpx.AsyncClient(timeout=self.config.timeout) as client:
            async with client.stream(
                "POST",
                f"{self.api_base}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json=payload
            ) as response:
                async for line in response.aiter_lines():
                    if line.startswith("data: ") and line != "data: [DONE]":
                        import json
                        data = json.loads(line[6:])
                        delta = data["choices"][0].get("delta", {})
                        if "content" in delta:
                            yield delta["content"]


# =============================================================================
# Ollama Provider (Native)
# =============================================================================

class OllamaProvider(BaseLLMProvider):
    """
    Ollama native provider - connects directly to Ollama API
    
    Setup:
        1. Install Ollama: curl -fsSL https://ollama.com/install.sh | sh
        2. Pull model: ollama pull llama3
        3. Set OLLAMA_HOST=http://localhost:11434 (default)
    
    Usage:
        llm = OllamaProvider(LLMConfig(model="llama3"))
        response = await llm.generate("Create REST API")
    """
    
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self.host = config.ollama_host or os.getenv("OLLAMA_HOST", "http://localhost:11434")
    
    async def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        **kwargs
    ) -> LLMResponse:
        import httpx
        import time
        
        start_time = time.time()
        
        payload = {
            "model": self.config.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": kwargs.get("temperature", self.config.temperature),
                "num_predict": kwargs.get("max_tokens", self.config.max_tokens)
            }
        }
        
        if system:
            payload["system"] = system
        
        async with httpx.AsyncClient(timeout=self.config.timeout) as client:
            response = await client.post(
                f"{self.host}/api/generate",
                json=payload
            )
            response.raise_for_status()
            data = response.json()
        
        return LLMResponse(
            content=data["response"],
            model=data.get("model", self.config.model),
            provider="ollama",
            input_tokens=data.get("prompt_eval_count", 0),
            output_tokens=data.get("eval_count", 0),
            total_tokens=data.get("prompt_eval_count", 0) + data.get("eval_count", 0),
            latency_ms=self._measure_latency(start_time),
            finish_reason="stop",
            raw_response=data
        )
    
    async def generate_stream(
        self,
        prompt: str,
        system: Optional[str] = None,
        **kwargs
    ) -> AsyncIterator[str]:
        import httpx
        
        payload = {
            "model": self.config.model,
            "prompt": prompt,
            "stream": True,
            "options": {
                "temperature": kwargs.get("temperature", self.config.temperature),
                "num_predict": kwargs.get("max_tokens", self.config.max_tokens)
            }
        }
        
        if system:
            payload["system"] = system
        
        async with httpx.AsyncClient(timeout=self.config.timeout) as client:
            async with client.stream(
                "POST",
                f"{self.host}/api/generate",
                json=payload
            ) as response:
                async for line in response.aiter_lines():
                    if line:
                        import json
                        data = json.loads(line)
                        if "response" in data:
                            yield data["response"]
    
    async def list_models(self) -> List[str]:
        """List available Ollama models"""
        import httpx
        
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{self.host}/api/tags")
            response.raise_for_status()
            data = response.json()
        
        return [model["name"] for model in data.get("models", [])]
    
    async def pull_model(self, model: str) -> bool:
        """Pull a model from Ollama registry"""
        import httpx
        
        async with httpx.AsyncClient(timeout=600.0) as client:  # Long timeout for download
            response = await client.post(
                f"{self.host}/api/pull",
                json={"name": model}
            )
            return response.status_code == 200


# =============================================================================
# LiteLLM Provider (Universal Proxy)
# =============================================================================

class LiteLLMProvider(BaseLLMProvider):
    """
    LiteLLM - Universal LLM API
    
    Supports 100+ models through unified interface:
    - OpenAI: gpt-4, gpt-3.5-turbo
    - Anthropic: claude-3-opus, claude-3-sonnet
    - Ollama: ollama/llama3, ollama/mistral
    - Azure: azure/gpt-4
    - AWS Bedrock: bedrock/anthropic.claude-v2
    - Google: gemini/gemini-pro
    - Cohere: command-nightly
    - Many more...
    
    Setup Option 1 - Direct (no proxy server):
        pip install litellm
        llm = LiteLLMProvider(LLMConfig(model="ollama/llama3"))
    
    Setup Option 2 - Via LiteLLM Proxy (recommended for production):
        litellm --model ollama/llama3 --port 4000
        llm = LiteLLMProvider(LLMConfig(litellm_api_base="http://localhost:4000"))
    """
    
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self.api_base = config.litellm_api_base or os.getenv("LITELLM_API_BASE")
        self._use_proxy = bool(self.api_base)
        
        # Check if litellm is installed for direct usage
        if not self._use_proxy:
            try:
                import litellm
                self._litellm = litellm
            except ImportError:
                raise ImportError(
                    "LiteLLM not installed. Install with: pip install litellm\n"
                    "Or set LITELLM_API_BASE to use proxy mode."
                )
    
    async def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        **kwargs
    ) -> LLMResponse:
        import time
        start_time = time.time()
        
        if self._use_proxy:
            return await self._generate_via_proxy(prompt, system, start_time, **kwargs)
        else:
            return await self._generate_direct(prompt, system, start_time, **kwargs)
    
    async def _generate_direct(
        self,
        prompt: str,
        system: Optional[str],
        start_time: float,
        **kwargs
    ) -> LLMResponse:
        """Direct LiteLLM call (no proxy)"""
        import asyncio
        
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        
        # Run sync litellm.completion in thread pool
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: self._litellm.completion(
                model=self.config.model,
                messages=messages,
                max_tokens=kwargs.get("max_tokens", self.config.max_tokens),
                temperature=kwargs.get("temperature", self.config.temperature)
            )
        )
        
        return LLMResponse(
            content=response.choices[0].message.content,
            model=response.model,
            provider="litellm",
            input_tokens=response.usage.prompt_tokens,
            output_tokens=response.usage.completion_tokens,
            total_tokens=response.usage.total_tokens,
            latency_ms=self._measure_latency(start_time),
            finish_reason=response.choices[0].finish_reason,
            raw_response=response.model_dump()
        )
    
    async def _generate_via_proxy(
        self,
        prompt: str,
        system: Optional[str],
        start_time: float,
        **kwargs
    ) -> LLMResponse:
        """Call via LiteLLM proxy server"""
        import httpx
        
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        
        payload = {
            "model": self.config.model,
            "messages": messages,
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
            "temperature": kwargs.get("temperature", self.config.temperature)
        }
        
        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        
        async with httpx.AsyncClient(timeout=self.config.timeout) as client:
            response = await client.post(
                f"{self.api_base}/chat/completions",
                headers=headers,
                json=payload
            )
            response.raise_for_status()
            data = response.json()
        
        return LLMResponse(
            content=data["choices"][0]["message"]["content"],
            model=data.get("model", self.config.model),
            provider="litellm",
            input_tokens=data.get("usage", {}).get("prompt_tokens", 0),
            output_tokens=data.get("usage", {}).get("completion_tokens", 0),
            total_tokens=data.get("usage", {}).get("total_tokens", 0),
            latency_ms=self._measure_latency(start_time),
            finish_reason=data["choices"][0].get("finish_reason", "stop"),
            raw_response=data
        )
    
    async def generate_stream(
        self,
        prompt: str,
        system: Optional[str] = None,
        **kwargs
    ) -> AsyncIterator[str]:
        """Stream via LiteLLM"""
        if self._use_proxy:
            async for chunk in self._stream_via_proxy(prompt, system, **kwargs):
                yield chunk
        else:
            async for chunk in self._stream_direct(prompt, system, **kwargs):
                yield chunk
    
    async def _stream_direct(
        self,
        prompt: str,
        system: Optional[str],
        **kwargs
    ) -> AsyncIterator[str]:
        """Direct streaming"""
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        
        response = self._litellm.completion(
            model=self.config.model,
            messages=messages,
            max_tokens=kwargs.get("max_tokens", self.config.max_tokens),
            temperature=kwargs.get("temperature", self.config.temperature),
            stream=True
        )
        
        for chunk in response:
            delta = chunk.choices[0].delta
            if hasattr(delta, "content") and delta.content:
                yield delta.content
    
    async def _stream_via_proxy(
        self,
        prompt: str,
        system: Optional[str],
        **kwargs
    ) -> AsyncIterator[str]:
        """Streaming via proxy"""
        import httpx
        import json
        
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        
        payload = {
            "model": self.config.model,
            "messages": messages,
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
            "temperature": kwargs.get("temperature", self.config.temperature),
            "stream": True
        }
        
        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        
        async with httpx.AsyncClient(timeout=self.config.timeout) as client:
            async with client.stream(
                "POST",
                f"{self.api_base}/chat/completions",
                headers=headers,
                json=payload
            ) as response:
                async for line in response.aiter_lines():
                    if line.startswith("data: ") and line != "data: [DONE]":
                        data = json.loads(line[6:])
                        delta = data["choices"][0].get("delta", {})
                        if "content" in delta:
                            yield delta["content"]


# =============================================================================
# Provider Factory
# =============================================================================

def get_llm_provider(
    provider: str = None,
    model: str = None,
    **kwargs
) -> BaseLLMProvider:
    """
    Factory function to get appropriate LLM provider
    
    Args:
        provider: Provider name (anthropic, openai, ollama, litellm)
                  If None, auto-detect from environment
        model: Model name (provider-specific)
        **kwargs: Additional config options
    
    Returns:
        BaseLLMProvider instance
    
    Examples:
        # Auto-detect from environment
        llm = get_llm_provider()
        
        # Specific provider
        llm = get_llm_provider("ollama", model="llama3")
        
        # Via LiteLLM (supports any backend)
        llm = get_llm_provider("litellm", model="ollama/llama3")
        llm = get_llm_provider("litellm", model="anthropic/claude-3-sonnet")
    """
    
    # Auto-detect from environment
    if provider is None:
        provider = os.getenv("LLM_PROVIDER", "anthropic")
    
    # Parse provider
    try:
        provider_enum = LLMProvider(provider.lower())
    except ValueError:
        # Default to LiteLLM for unknown providers
        provider_enum = LLMProvider.LITELLM
        if model is None:
            model = provider  # Use provider string as model
    
    # Build config
    config = LLMConfig.from_env()
    config.provider = provider_enum
    
    if model:
        config.model = model
    
    # Apply kwargs
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
    
    # Create provider instance
    providers = {
        LLMProvider.ANTHROPIC: AnthropicProvider,
        LLMProvider.OPENAI: OpenAIProvider,
        LLMProvider.OLLAMA: OllamaProvider,
        LLMProvider.LITELLM: LiteLLMProvider,
        LLMProvider.LOCAL: OpenAIProvider  # Local uses OpenAI-compatible API
    }
    
    provider_class = providers.get(provider_enum, LiteLLMProvider)
    return provider_class(config)


# =============================================================================
# Convenience Functions
# =============================================================================

async def generate(
    prompt: str,
    model: str = None,
    provider: str = None,
    system: str = None,
    **kwargs
) -> str:
    """
    Quick generation helper
    
    Usage:
        code = await generate("Create REST API for users")
        code = await generate("Create API", model="ollama/llama3")
    """
    llm = get_llm_provider(provider, model, **kwargs)
    response = await llm.generate(prompt, system=system, **kwargs)
    return response.content


async def generate_code(
    description: str,
    language: str = "python",
    model: str = None,
    provider: str = None,
    **kwargs
) -> str:
    """
    Quick code generation helper
    
    Usage:
        code = await generate_code("REST API for products")
        code = await generate_code("MQTT handler", model="ollama/codellama")
    """
    llm = get_llm_provider(provider, model, **kwargs)
    response = await llm.generate_code(description, language)
    return response.content


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    # Types
    "LLMProvider",
    "LLMConfig", 
    "LLMResponse",
    
    # Providers
    "BaseLLMProvider",
    "AnthropicProvider",
    "OpenAIProvider",
    "OllamaProvider",
    "LiteLLMProvider",
    
    # Factory
    "get_llm_provider",
    
    # Helpers
    "generate",
    "generate_code"
]
