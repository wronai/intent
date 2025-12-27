"""
IntentForge LLM Module
Unified interface for multiple LLM providers

Supported Providers:
- Anthropic Claude
- OpenAI GPT
- Ollama (local)
- LiteLLM (universal proxy for 100+ models)

Quick Start:
    from intentforge.llm import get_llm_provider, generate_code

    # Using Ollama
    llm = get_llm_provider("ollama", model="llama3")
    code = await llm.generate("Create REST API")

    # Using LiteLLM (any backend)
    code = await generate_code("Create API", model="ollama/codellama")
"""

from .providers import (
    AnthropicProvider,
    # Providers
    BaseLLMProvider,
    LiteLLMProvider,
    LLMConfig,
    # Types
    LLMProvider,
    LLMResponse,
    OllamaProvider,
    OpenAIProvider,
    # Helpers
    generate,
    generate_code,
    # Factory
    get_llm_provider,
)

__all__ = [
    "LLMProvider",
    "LLMConfig",
    "LLMResponse",
    "BaseLLMProvider",
    "AnthropicProvider",
    "OpenAIProvider",
    "OllamaProvider",
    "LiteLLMProvider",
    "get_llm_provider",
    "generate",
    "generate_code",
]
