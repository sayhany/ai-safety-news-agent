"""Tests for configuration module."""

import pytest
from pathlib import Path

from aisafety_news.config import Settings, ModelConfig, validate_config


def test_settings_creation(mock_env):
    """Test settings creation with environment variables."""
    settings = Settings()
    
    assert settings.openrouter_api_key == "test-key"
    assert settings.max_per_domain == 10
    assert settings.global_parallel == 30
    assert settings.w_source_priority == 0.4
    assert settings.w_recency == 0.25
    assert settings.w_gov == 0.2
    assert settings.w_llm_importance == 0.15


def test_settings_weight_validation():
    """Test weight validation."""
    with pytest.raises(ValueError, match="Weight must be between 0 and 1"):
        Settings(
            openrouter_api_key="test",
            w_source_priority=1.5  # Invalid weight
        )


def test_settings_threshold_validation():
    """Test deduplication threshold validation."""
    with pytest.raises(ValueError, match="Deduplication threshold must be between 0 and 1"):
        Settings(
            openrouter_api_key="test",
            dedupe_threshold=1.5  # Invalid threshold
        )


def test_directory_creation(temp_dir, mock_env):
    """Test that directories are created automatically."""
    settings = Settings()
    
    assert settings.data_dir.exists()
    assert settings.cache_dir.exists()
    assert settings.data_dir.is_dir()
    assert settings.cache_dir.is_dir()


def test_model_config_loading():
    """Test model configuration loading."""
    # This will use the models.yaml file we created
    try:
        config = ModelConfig()
        
        # Test LLM route loading
        summarizer = config.get_llm_route("summarizer")
        assert summarizer.primary == "anthropic/claude-3-haiku"
        assert "openai/gpt-4o-mini" in summarizer.fallback
        
        relevance = config.get_llm_route("relevance")
        assert relevance.primary == "openai/gpt-4o-mini"
        
        # Test model settings
        model_settings = config.get_model_settings()
        assert model_settings.temperature == 0.2
        assert model_settings.max_tokens == 2048
        
        # Test source priorities
        priorities = config.get_source_priorities()
        assert priorities.get("nist.gov") == 1.0
        assert priorities.get("default") == 0.5
        
    except FileNotFoundError:
        pytest.skip("models.yaml not found - expected in some test environments")


def test_config_validation(mock_env):
    """Test configuration validation."""
    # Should pass with valid config
    assert validate_config() == True


def test_config_validation_missing_api_key(monkeypatch):
    """Test validation fails without API key."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "")
    
    # Should fail without API key
    assert validate_config() == False


def test_config_validation_invalid_weights(mock_env, monkeypatch):
    """Test validation fails with invalid weights."""
    # Set weights that don't sum to 1.0
    monkeypatch.setenv("W_SOURCE_PRIORITY", "0.5")
    monkeypatch.setenv("W_RECENCY", "0.5")
    monkeypatch.setenv("W_GOV", "0.5")
    monkeypatch.setenv("W_LLM_IMPORTANCE", "0.5")
    
    # Should fail with weights that don't sum to 1.0
    assert validate_config() == False
