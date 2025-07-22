# AI Safety Newsletter Agent 🤖📰

An intelligent Python agent that automatically discovers, analyzes, and curates AI safety news into well-formatted newsletters. The agent searches multiple sources, filters for relevance, removes duplicates, and generates professional newsletters using advanced LLMs.

## 🌟 Features

- **🔍 Smart Content Discovery**: Uses Exa, Brave, and Bing search APIs to find the latest AI safety content
- **🧠 AI-Powered Filtering**: Leverages Google Gemini and OpenAI models to assess relevance and importance
- **🔄 Intelligent Deduplication**: Uses FAISS embeddings to identify and remove duplicate articles
- **📊 Weighted Scoring System**: Configurable scoring that prioritizes the most important content
- **⚡ Async Processing**: Multi-stage pipeline for fast, efficient processing
- **🎨 Beautiful Output**: Generates clean, professional Markdown newsletters
- **🔧 Flexible Configuration**: Easily customizable via YAML configuration files
- **📝 Comprehensive Logging**: Detailed logging with performance metrics
- **🤝 Respectful Crawling**: Follows robots.txt and implements rate limiting

## 🚀 Quick Start

### Prerequisites

- **Python 3.11+** 
- **API Keys** for at least one LLM provider and one search provider

### Easy Installation (Recommended)

Install directly from the repository:

```bash
pip install git+https://github.com/yourusername/ai-safety-news-agent.git
```

Or install locally:

```bash
git clone <repository-url>
cd ai-safety-news-agent
pip install .
```

### Development Installation

For development or customization:

1. **Clone and enter the project:**
   ```bash
   git clone <repository-url>
   cd ai-safety-news-agent
   ```

2. **Install with Poetry:**
   ```bash
   pip install poetry
   poetry install
   ```

### Setup API Keys

Create a `.env` file with your API keys:
```bash
cp .env.example .env
```

Edit `.env` with your API keys:
```bash
# Required: OpenAI for LLM
OPENAI_API_KEY="your-openai-key"

# Required: Exa for search
EXA_API_KEY="your-exa-key"

# Optional: Google AI for embeddings (fallback to OpenAI if not provided)
GOOGLE_AI_API_KEY="your-google-ai-key"
```

### Your First Newsletter

**Activate the poetry environment:**
```bash
poetry shell
```

**Generate a newsletter for the last 7 days:**
```bash
aistr-news-agent 2024-01-15
```

**Or run directly with poetry:**
```bash
poetry run aistr-news-agent 2024-01-15
```

That's it! The agent will automatically:
1. Search for AI safety content from the specified date
2. Filter articles for relevance and quality
3. Remove duplicates using AI embeddings
4. Score and rank the content
5. Generate a beautiful newsletter in Markdown

## 📖 Usage Guide

### Basic Commands

**Generate newsletter for specific date (after `poetry shell`):**
```bash
aistr-news-agent 2024-01-15
```

**Or run directly with poetry:**
```bash
poetry run aistr-news-agent 2024-01-15
```

**Save newsletter to file:**
```bash
poetry run aistr-news-agent 2024-01-15 --output my_newsletter.md
```

**Test with mock data (no API calls):**
```bash
poetry run aistr-news-agent 2024-01-15 --mock
```

**Check if your sources are working:**
```bash
poetry run check-sources
```

### Advanced Options

**Customize the number of articles:**
```bash
poetry run aistr-news-agent 2024-01-15 --max-articles 10
```

**Use specific LLM model:**
```bash
poetry run aistr-news-agent 2024-01-15 --model "gpt-4o"
```

**Enable debug logging:**
```bash
poetry run aistr-news-agent 2024-01-15 --log-level DEBUG
```

**Validate your configuration:**
```bash
poetry run aistr-news-agent --validate-config
```

### Complete CLI Reference

| Option | Description | Example |
|--------|-------------|---------|
| `start_date` | **Required.** Start date for article search (YYYY-MM-DD) | `2024-01-15` |
| `--mock` | Use mock data for testing (no real API calls) | `--mock` |
| `--max-articles` | Maximum articles in newsletter (default: 15) | `--max-articles 10` |
| `--output` | Save newsletter to specific file | `--output newsletter.md` |
| `--log-level` | Logging verbosity (DEBUG, INFO, WARNING, ERROR) | `--log-level DEBUG` |
| `--model` | Override default LLM model | `--model "openai/gpt-4o"` |
| `--validate-config` | Validate configuration and exit | `--validate-config` |

## ⚙️ Configuration

The agent is highly configurable through YAML files:

### models.yaml
Configure LLM models, search sources, and processing parameters:
```yaml
llm_routes:
  relevance:
    primary: "google/gemini-1.5-flash"
    fallback: "openai/gpt-4o-mini"
  
search_sources:
  - name: "exa"
    priority: 1
    max_results: 50
```

### prompts.yaml  
Customize AI prompts for different processing stages:
```yaml
relevance_filter: |
  Analyze this article for AI safety relevance...
  
scoring: |
  Rate this article's importance for AI safety...
```

## 🔧 Troubleshooting

### Common Issues

**"No API key found" error:**
- Make sure you've created `.env` from `.env.example`
- Verify your API keys are correctly set in `.env`
- Check that you have at least one LLM and one search API key

**"No articles found" error:**  
- Try a different date range
- Run `poetry run check-sources` to verify sources are accessible
- Check your internet connection

**Performance issues:**
- Reduce `--max-articles` to process fewer articles
- Use `--mock` flag to test without API calls
- Check the `MAX_PER_DOMAIN` setting in your environment

### Getting Help

- Check [TROUBLESHOOTING.md](TROUBLESHOOTING.md) for detailed solutions
- Review logs with `--log-level DEBUG` for detailed diagnostics
- Validate configuration with `--validate-config`

## 🏗️ Development

### Running Tests

```bash
# Full test suite
poetry run pytest

# With coverage report  
poetry run pytest --cov=aisafety_news --cov-report=html

# Test specific functionality
poetry run pytest tests/test_search_api.py -v
```

### Code Quality

```bash
# Format code
poetry run ruff format

# Lint code
poetry run ruff check

# Type checking
poetry run mypy aisafety_news/
```

### Project Architecture

```
aisafety_news/
├── ingest/           # Article discovery and parsing
│   ├── sources.py    # Search API integrations
│   └── adapters/     # Source-specific parsers
├── processing/       # Content processing pipeline
│   ├── relevance.py  # AI relevance filtering
│   ├── dedupe.py     # Duplicate removal
│   └── scoring.py    # Article importance scoring
├── models/           # AI model clients
│   ├── llm_client.py # LLM integrations
│   └── embedding_client.py # Embedding models
├── orchestrator.py   # Main pipeline coordination
├── render.py         # Newsletter generation
└── config.py         # Configuration management
```

## 🤝 Contributing

We welcome contributions! Here's how to get started:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests (`poetry run pytest`)
5. Submit a pull request

### Development Setup

```bash
# Install with development dependencies
poetry install --with dev

# Run pre-commit hooks
poetry run ruff check
poetry run mypy aisafety_news/
poetry run pytest
```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Built with modern Python async/await patterns
- Uses Google Gemini and OpenAI for AI processing
- Powered by Exa, Brave, and Bing search APIs
- FAISS for efficient similarity search
- Rich for beautiful CLI output