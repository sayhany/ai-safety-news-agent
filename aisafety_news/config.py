"""Configuration management for AI Safety Newsletter Agent."""

import os
import yaml
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, HttpUrl, field_validator, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMRoute(BaseModel):
    """LLM model routing configuration."""
    primary: str
    fallback: List[str] = Field(default_factory=list)


class SourceConfig(BaseModel):
    """News source configuration."""
    url: HttpUrl
    domain: str
    category: str
    priority: float = 0.5


class ModelSettings(BaseModel):
    """LLM model-specific settings."""
    temperature: float = 0.2
    max_tokens: int = 2048
    timeout_seconds: int = 40
    retry_attempts: int = 3
    backoff_factor: float = 2.0


class Settings(BaseSettings):
    """Main application settings."""
    
    # ── LLM Configuration ──────────────────────────────────────────────────
    openrouter_api_key: str = Field(..., description="OpenRouter API key")
    
    # ── Ingestion Settings ─────────────────────────────────────────────────
    max_per_domain: int = Field(10, description="Max concurrent requests per domain")
    global_parallel: int = Field(30, description="Global semaphore limit")
    user_agent: str = Field(
        "AISafetyNewsletterBot/0.1 (+https://github.com/your/repo)",
        description="User agent for web requests"
    )
    
    # ── Scoring Weights ────────────────────────────────────────────────────
    w_source_priority: float = Field(0.4, description="Source priority weight")
    w_recency: float = Field(0.25, description="Recency weight")
    w_gov: float = Field(0.2, description="Government source weight")
    w_llm_importance: float = Field(0.15, description="LLM importance weight")
    
    # ── Storage & Caching ──────────────────────────────────────────────────
    data_dir: Path = Field(Path("./data"), description="Data directory")
    cache_dir: Path = Field(Path("./cache/http"), description="HTTP cache directory")
    
    # ── Processing Settings ────────────────────────────────────────────────
    dedupe_threshold: float = Field(0.85, description="Vector similarity threshold")
    max_articles: int = Field(20, description="Maximum articles in newsletter")
    
    # ── API Settings ───────────────────────────────────────────────────────
    api_port: int = Field(8000, description="HTTP API port")
    api_host: str = Field("127.0.0.1", description="API host binding")
    
    # ── Logging ────────────────────────────────────────────────────────────
    log_level: str = Field("INFO", description="Log level")
    json_logging: bool = Field(True, description="Enable JSON logging")
    
    # ── Security ───────────────────────────────────────────────────────────
    data_retention_days: int = Field(30, description="Data retention period")
    https_only: bool = Field(True, description="HTTPS-only mode")
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False
    )
    
    @field_validator("data_dir", "cache_dir")
    @classmethod
    def ensure_directories(cls, v: Path) -> Path:
        """Ensure directories exist."""
        v.mkdir(parents=True, exist_ok=True)
        return v.resolve()
    
    @field_validator("w_source_priority", "w_recency", "w_gov", "w_llm_importance")
    @classmethod
    def validate_weights(cls, v: float) -> float:
        """Validate weight values are between 0 and 1."""
        if not 0 <= v <= 1:
            raise ValueError("Weight must be between 0 and 1")
        return v
    
    @field_validator("dedupe_threshold")
    @classmethod
    def validate_threshold(cls, v: float) -> float:
        """Validate deduplication threshold."""
        if not 0 <= v <= 1:
            raise ValueError("Deduplication threshold must be between 0 and 1")
        return v


class ModelConfig:
    """Model configuration loader."""
    
    def __init__(self, config_path: Union[str, Path] = "models.yaml"):
        self.config_path = Path(config_path)
        self._config: Dict[str, Any] = {}
        self.load_config()
    
    def load_config(self) -> None:
        """Load model configuration from YAML file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Model config file not found: {self.config_path}")
        
        with open(self.config_path, "r", encoding="utf-8") as f:
            self._config = yaml.safe_load(f)
    
    def get_llm_route(self, route_name: str) -> LLMRoute:
        """Get LLM route configuration."""
        if route_name not in self._config:
            raise ValueError(f"LLM route '{route_name}' not found in config")
        
        route_data = self._config[route_name]
        return LLMRoute(**route_data)
    
    def get_model_settings(self) -> ModelSettings:
        """Get model settings."""
        settings_data = self._config.get("model_settings", {})
        return ModelSettings(**settings_data)
    
    def get_source_priorities(self) -> Dict[str, float]:
        """Get source priority mapping."""
        return self._config.get("source_priorities", {})
    
    def get_approved_sources(self) -> List[SourceConfig]:
        """Get approved news sources."""
        sources_data = self._config.get("approved_sources", [])
        return [SourceConfig(**source) for source in sources_data]
    
    def get_ai_safety_keywords(self) -> Dict[str, List[str]]:
        """Get AI safety keywords for filtering."""
        return self._config.get("ai_safety_keywords", {"primary": [], "secondary": []})


# Global instances
settings = Settings()
model_config = ModelConfig()


def get_settings() -> Settings:
    """Get application settings."""
    return settings


def get_model_config() -> ModelConfig:
    """Get model configuration."""
    return model_config


def validate_config() -> bool:
    """Validate configuration completeness."""
    try:
        # Check required settings
        if not settings.openrouter_api_key:
            raise ValueError("OPENROUTER_API_KEY is required")
        
        # Validate weight sum (should be close to 1.0)
        weight_sum = (
            settings.w_source_priority +
            settings.w_recency +
            settings.w_gov +
            settings.w_llm_importance
        )
        if abs(weight_sum - 1.0) > 0.01:
            raise ValueError(f"Scoring weights sum to {weight_sum}, should be 1.0")
        
        # Check model config
        model_config.get_llm_route("summarizer")
        model_config.get_llm_route("relevance")
        
        return True
    
    except Exception as e:
        print(f"Configuration validation failed: {e}")
        return False


if __name__ == "__main__":
    # Configuration validation
    if validate_config():
        print("✅ Configuration is valid")
    else:
        print("❌ Configuration validation failed")
        exit(1)
