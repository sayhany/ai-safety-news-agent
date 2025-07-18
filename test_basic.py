#!/usr/bin/env python3
"""Basic test to validate project structure without external dependencies."""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Set required environment variables
os.environ["OPENROUTER_API_KEY"] = "test-key"
os.environ["LOG_LEVEL"] = "INFO"
os.environ["JSON_LOGGING"] = "false"

def test_imports():
    """Test that core modules can be imported."""
    try:
        # Test basic imports that don't require external dependencies
        from aisafety_news import __version__
        print(f"✅ Package version: {__version__}")
        
        # Test utility functions
        from aisafety_news.utils import normalize_url, extract_domain, clean_text
        print("✅ Utils module imported")
        
        # Test basic URL functions
        test_url = "https://example.com/path/"
        normalized = normalize_url(test_url)
        domain = extract_domain(test_url)
        print(f"✅ URL normalization: {normalized}")
        print(f"✅ Domain extraction: {domain}")
        
        # Test text cleaning
        dirty_text = "  This is   a test   with   extra   spaces  "
        clean = clean_text(dirty_text)
        print(f"✅ Text cleaning: '{clean}'")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def test_config():
    """Test configuration without external dependencies."""
    try:
        # Mock the models.yaml to avoid file dependency
        import tempfile
        import yaml
        
        mock_config = {
            "summarizer": {
                "primary": "test-model",
                "fallback": ["fallback-model"]
            },
            "relevance": {
                "primary": "test-model",
                "fallback": []
            },
            "model_settings": {
                "temperature": 0.2,
                "max_tokens": 1000
            },
            "source_priorities": {
                "default": 0.5
            },
            "approved_sources": [],
            "ai_safety_keywords": {
                "primary": ["ai safety"],
                "secondary": ["artificial intelligence"]
            }
        }
        
        # Create temporary config file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(mock_config, f)
            temp_config_path = f.name
        
        try:
            from aisafety_news.config import ModelConfig
            config = ModelConfig(temp_config_path)
            
            # Test route loading
            summarizer = config.get_llm_route("summarizer")
            print(f"✅ Summarizer route: {summarizer.primary}")
            
            # Test model settings
            settings = config.get_model_settings()
            print(f"✅ Model settings: temp={settings.temperature}")
            
            return True
            
        finally:
            # Clean up temp file
            os.unlink(temp_config_path)
            
    except Exception as e:
        print(f"❌ Config test failed: {e}")
        return False

def test_text_utils():
    """Test text processing utilities."""
    try:
        from aisafety_news.processing.text_utils import (
            canonical_title, extract_keywords, clean_html_text
        )
        
        # Test title canonicalization
        title = "OpenAI Debuts GPT-4o?!!!"
        canonical = canonical_title(title)
        print(f"✅ Canonical title: '{canonical}'")
        
        # Test keyword extraction
        text = "This is a test article about AI safety and machine learning."
        keywords = extract_keywords(text)
        print(f"✅ Keywords: {keywords}")
        
        # Test HTML cleaning
        html = "<p>This is <b>bold</b> text with &amp; entities.</p>"
        clean = clean_html_text(html)
        print(f"✅ HTML cleaned: '{clean}'")
        
        return True
        
    except Exception as e:
        print(f"❌ Text utils test failed: {e}")
        return False

def test_project_structure():
    """Test that all required files and directories exist."""
    required_files = [
        "pyproject.toml",
        "README.md",
        ".env.example",
        "models.yaml",
        ".gitignore",
        "Dockerfile",
        ".github/workflows/ci.yml",
        "aisafety_news/__init__.py",
        "aisafety_news/config.py",
        "aisafety_news/utils.py",
        "aisafety_news/logging.py",
        "aisafety_news/orchestrator.py",
        "aisafety_news/models/__init__.py",
        "aisafety_news/models/llm_client.py",
        "aisafety_news/ingest/__init__.py",
        "aisafety_news/ingest/sources.py",
        "aisafety_news/processing/__init__.py",
        "aisafety_news/processing/text_utils.py",
        "templates/newsletter.j2",
        "templates/article_summary.j2",
        "templates/relevance_filter.j2",
        "prompts.yaml",
        "tests/__init__.py",
        "tests/conftest.py",
        "tests/test_config.py"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not (project_root / file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        print(f"❌ Missing files: {missing_files}")
        return False
    else:
        print(f"✅ All {len(required_files)} required files exist")
        return True

def main():
    """Run all tests."""
    print("🚀 Running AI Safety Newsletter Agent basic tests...\n")
    
    tests = [
        ("Project Structure", test_project_structure),
        ("Core Imports", test_imports),
        ("Configuration", test_config),
        ("Text Utils", test_text_utils),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n📋 Testing {test_name}:")
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} PASSED")
            else:
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            print(f"❌ {test_name} ERROR: {e}")
    
    print(f"\n📊 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Project structure is valid.")
        return 0
    else:
        print("⚠️  Some tests failed. Check the output above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
