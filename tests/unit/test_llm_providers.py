"""
Unit tests for LLM providers
Tests Ollama, Anthropic, OpenAI, LiteLLM integration
"""

import asyncio

# Import from your module
import sys
from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, ".")

from intentforge.llm.providers import (
    AnthropicProvider,
    BaseLLMProvider,
    LiteLLMProvider,
    LLMConfig,
    LLMProvider,
    LLMResponse,
    OllamaProvider,
    OpenAIProvider,
    generate,
    generate_code,
    get_llm_provider,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def ollama_config():
    """Ollama configuration"""
    return LLMConfig(
        provider=LLMProvider.OLLAMA,
        model="llama3",
        ollama_host="http://localhost:11434",
        max_tokens=1000,
        temperature=0.1,
    )


@pytest.fixture
def anthropic_config():
    """Anthropic configuration"""
    return LLMConfig(
        provider=LLMProvider.ANTHROPIC,
        model="claude-3-sonnet",
        api_key="test-api-key",
        max_tokens=1000,
    )


@pytest.fixture
def openai_config():
    """OpenAI configuration"""
    return LLMConfig(
        provider=LLMProvider.OPENAI, model="gpt-4o", api_key="test-api-key", max_tokens=1000
    )


# =============================================================================
# LLMConfig Tests
# =============================================================================


class TestLLMConfig:
    """Tests for LLMConfig dataclass"""

    def test_default_values(self):
        """Test default configuration values"""
        config = LLMConfig()

        assert config.provider == LLMProvider.ANTHROPIC
        assert config.max_tokens == 4096
        assert config.temperature == 0.1
        assert config.timeout == 60.0
        assert config.max_retries == 3

    def test_from_env_ollama(self, monkeypatch):
        """Test loading config from environment (Ollama)"""
        monkeypatch.setenv("LLM_PROVIDER", "ollama")
        monkeypatch.setenv("LLM_MODEL", "codellama")
        monkeypatch.setenv("OLLAMA_HOST", "http://custom:11434")

        config = LLMConfig.from_env()

        assert config.provider == LLMProvider.OLLAMA
        assert config.model == "codellama"
        assert config.ollama_host == "http://custom:11434"

    def test_from_env_anthropic(self, monkeypatch):
        """Test loading config from environment (Anthropic)"""
        monkeypatch.setenv("LLM_PROVIDER", "anthropic")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")

        config = LLMConfig.from_env()

        assert config.provider == LLMProvider.ANTHROPIC
        assert config.api_key == "sk-ant-test"

    def test_default_model_per_provider(self):
        """Test default model selection per provider"""
        assert LLMConfig._default_model(LLMProvider.OLLAMA) == "llama3"
        assert LLMConfig._default_model(LLMProvider.OPENAI) == "gpt-4o"
        assert "claude" in LLMConfig._default_model(LLMProvider.ANTHROPIC)


# =============================================================================
# LLMResponse Tests
# =============================================================================


class TestLLMResponse:
    """Tests for LLMResponse dataclass"""

    def test_response_creation(self):
        """Test creating response object"""
        response = LLMResponse(
            content="Hello world",
            model="llama3",
            provider="ollama",
            input_tokens=10,
            output_tokens=5,
            total_tokens=15,
            latency_ms=100.5,
        )

        assert response.content == "Hello world"
        assert response.total_tokens == 15
        assert response.latency_ms == 100.5

    def test_default_values(self):
        """Test default values"""
        response = LLMResponse(content="Test", model="test", provider="test")

        assert response.input_tokens == 0
        assert response.finish_reason == "stop"
        assert response.raw_response is None


# =============================================================================
# OllamaProvider Tests
# =============================================================================


class TestOllamaProvider:
    """Tests for Ollama provider"""

    @pytest.mark.asyncio
    async def test_generate_success(self, ollama_config):
        """Test successful generation"""
        provider = OllamaProvider(ollama_config)

        # Mock httpx response
        mock_response = {
            "response": "Generated code here",
            "model": "llama3",
            "prompt_eval_count": 10,
            "eval_count": 20,
        }

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_instance.post.return_value = MagicMock(
                json=lambda: mock_response, raise_for_status=lambda: None
            )
            mock_client.return_value.__aenter__.return_value = mock_instance

            response = await provider.generate("Create REST API")

            assert response.content == "Generated code here"
            assert response.model == "llama3"
            assert response.provider == "ollama"

    @pytest.mark.asyncio
    async def test_list_models(self, ollama_config):
        """Test listing available models"""
        provider = OllamaProvider(ollama_config)

        mock_response = {"models": [{"name": "llama3"}, {"name": "codellama"}, {"name": "mistral"}]}

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_instance.get.return_value = MagicMock(
                json=lambda: mock_response, raise_for_status=lambda: None
            )
            mock_client.return_value.__aenter__.return_value = mock_instance

            models = await provider.list_models()

            assert "llama3" in models
            assert "codellama" in models
            assert len(models) == 3

    def test_host_configuration(self, ollama_config):
        """Test host is configured correctly"""
        provider = OllamaProvider(ollama_config)
        assert provider.host == "http://localhost:11434"


# =============================================================================
# AnthropicProvider Tests
# =============================================================================


class TestAnthropicProvider:
    """Tests for Anthropic provider"""

    def test_requires_api_key(self, anthropic_config):
        """Test that API key is required"""
        config = LLMConfig(provider=LLMProvider.ANTHROPIC, api_key=None)

        with pytest.raises(ValueError, match="ANTHROPIC_API_KEY required"):
            AnthropicProvider(config)

    @pytest.mark.asyncio
    async def test_generate_with_system(self, anthropic_config):
        """Test generation with system prompt"""
        provider = AnthropicProvider(anthropic_config)

        mock_response = {
            "content": [{"text": "Response text"}],
            "model": "claude-3-sonnet",
            "usage": {"input_tokens": 50, "output_tokens": 100},
            "stop_reason": "end_turn",
        }

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_instance.post.return_value = MagicMock(
                json=lambda: mock_response, raise_for_status=lambda: None
            )
            mock_client.return_value.__aenter__.return_value = mock_instance

            response = await provider.generate("Create API", system="You are a Python expert")

            assert response.content == "Response text"
            assert response.provider == "anthropic"


# =============================================================================
# OpenAIProvider Tests
# =============================================================================


class TestOpenAIProvider:
    """Tests for OpenAI provider"""

    def test_requires_api_key(self):
        """Test that API key is required"""
        config = LLMConfig(provider=LLMProvider.OPENAI, api_key=None)

        with pytest.raises(ValueError, match="OPENAI_API_KEY required"):
            OpenAIProvider(config)

    def test_custom_api_base(self, openai_config):
        """Test custom API base URL"""
        openai_config.api_base = "https://custom.openai.com/v1"
        provider = OpenAIProvider(openai_config)

        assert provider.api_base == "https://custom.openai.com/v1"


# =============================================================================
# Factory Function Tests
# =============================================================================


class TestGetLLMProvider:
    """Tests for get_llm_provider factory"""

    def test_get_ollama_provider(self, monkeypatch):
        """Test getting Ollama provider"""
        monkeypatch.setenv("OLLAMA_HOST", "http://localhost:11434")

        provider = get_llm_provider("ollama", model="llama3")

        assert isinstance(provider, OllamaProvider)
        assert provider.config.model == "llama3"

    def test_get_anthropic_provider(self, monkeypatch):
        """Test getting Anthropic provider"""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

        provider = get_llm_provider("anthropic")

        assert isinstance(provider, AnthropicProvider)

    def test_auto_detect_from_env(self, monkeypatch):
        """Test auto-detection from environment"""
        monkeypatch.setenv("LLM_PROVIDER", "ollama")
        monkeypatch.setenv("OLLAMA_HOST", "http://localhost:11434")

        provider = get_llm_provider()

        assert isinstance(provider, OllamaProvider)

    def test_unknown_provider_defaults_to_litellm(self):
        """Test unknown provider falls back to LiteLLM"""
        # This should not raise, but use LiteLLM
        try:
            provider = get_llm_provider("unknown_provider")
            assert isinstance(provider, LiteLLMProvider)
        except ImportError:
            # LiteLLM not installed, which is fine
            pass


# =============================================================================
# Helper Function Tests
# =============================================================================


class TestHelperFunctions:
    """Tests for generate() and generate_code() helpers"""

    @pytest.mark.asyncio
    async def test_generate_helper(self, monkeypatch):
        """Test generate() helper function"""
        monkeypatch.setenv("LLM_PROVIDER", "ollama")
        monkeypatch.setenv("OLLAMA_HOST", "http://localhost:11434")

        with patch.object(OllamaProvider, "generate") as mock_generate:
            mock_generate.return_value = LLMResponse(
                content="Test response", model="llama3", provider="ollama"
            )

            result = await generate("Test prompt")

            assert result == "Test response"

    @pytest.mark.asyncio
    async def test_generate_code_helper(self, monkeypatch):
        """Test generate_code() helper function"""
        monkeypatch.setenv("LLM_PROVIDER", "ollama")
        monkeypatch.setenv("OLLAMA_HOST", "http://localhost:11434")

        with patch.object(OllamaProvider, "generate_code") as mock_generate:
            mock_generate.return_value = LLMResponse(
                content="def hello(): pass", model="codellama", provider="ollama"
            )

            result = await generate_code("Create hello function")

            assert "def hello" in result


# =============================================================================
# Integration Tests (require running Ollama)
# =============================================================================


@pytest.mark.integration
@pytest.mark.skipif(
    not pytest.importorskip("httpx", reason="httpx not installed"),
    reason="Integration tests require running Ollama",
)
class TestOllamaIntegration:
    """Integration tests with real Ollama instance"""

    @pytest.mark.asyncio
    async def test_real_generation(self):
        """Test real generation (requires Ollama running)"""
        try:
            provider = get_llm_provider("ollama", model="llama3")
            response = await provider.generate("Say 'test' and nothing else")
            assert len(response.content) > 0
        except Exception:
            pytest.skip("Ollama not available")


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-x"])
