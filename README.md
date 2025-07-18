# AI Safety Newsletter Agent

An autonomous Python 3.11+ agent that crawls approved news sources, filters for AI safety relevance, deduplicates and ranks articles, then generates an English-only Markdown newsletter using LLMs through OpenRouter.

## Features

- **Async orchestrator** with multi-stage pipeline
- **Pluggable LLM backends** with fallback and retry logic
- **Smart deduplication** using embeddings and FAISS (≥0.85 similarity threshold)
- **Weighted scoring** system with YAML-tunable parameters
- **Rate limiting** and robots.txt compliance
- **Structured JSON logging** with structlog
- **Comprehensive testing** with ≥90% coverage

## Architecture

### Pipeline Stages
1. **Source ingestion** - Crawl/query approved news sources
2. **Relevance filtering** - Keyword + LLM-based AI safety filtering
3. **Deduplication** - Hash + embedding-based duplicate removal
4. **Scoring & ranking** - Weighted salience scoring
5. **Summarization** - LLM-generated article summaries
6. **Template rendering** - Markdown newsletter generation

### Module Structure
```
aisafety_news/
├── config.py              # Environment & YAML configuration
├── models/llm_client.py   # Async OpenRouter wrapper with fallback
├── ingest/sources.py      # Source registry + per-domain adapters
├── processing/
│   ├── relevance.py       # Keyword + LLM filtering
│   ├── dedupe.py          # Hash + embedding deduplication
│   └── scoring.py         # Weighted ranking system
├── summarize.py           # Article summarization via LLM
├── render.py              # Markdown template assembly
└── orchestrator.py        # Async pipeline & CLI
```

## Installation

### Prerequisites
- Python 3.11+
- Poetry (recommended) or pip

### Setup
```bash
# Clone the repository
git clone <repository-url>
cd ai-safety-news-agent

# Install dependencies
poetry install --with dev

# Or with pip
pip install -e .
```

### Configuration

1. Copy the example environment file:
```bash
cp .env.example .env
```

2. Configure your environment variables:
```bash
# Required
OPENROUTER_API_KEY=your_openrouter_api_key_here

# Optional (defaults provided)
MAX_PER_DOMAIN=10
GLOBAL_PARALLEL=30
W_SOURCE_PRIORITY=0.4
W_RECENCY=0.25
W_GOV=0.2
W_LLM_IMPORTANCE=0.15
```

3. Customize model configuration in `models.yaml`:
```yaml
summarizer:
  primary: "anthropic/claude-3-haiku"
  fallback:
    - "openai/gpt-4o-mini"
    - "meta-llama/llama-3.1-8b-instruct"

relevance:
  primary: "openai/gpt-4o-mini"
  fallback:
    - "anthropic/claude-3-haiku"
```

## Usage

### Command Line Interface
```bash
# Generate newsletter for a specific date
aisafety-news 2025-07-18

# Run with mock data (for testing)
aisafety-news 2025-07-18 --mock

# Output to file
aisafety-news 2025-07-18 > newsletter.md
```

### HTTP API
```bash
# Start the HTTP server
python -m aisafety_news.api

# Generate newsletter via API
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"start_date": "2025-07-18"}'
```

## Configuration Reference

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENROUTER_API_KEY` | *required* | OpenRouter API key for LLM access |
| `MAX_PER_DOMAIN` | `10` | Maximum concurrent requests per domain |
| `GLOBAL_PARALLEL` | `30` | Global semaphore limit for concurrent requests |
| `W_SOURCE_PRIORITY` | `0.4` | Weight for source priority in scoring |
| `W_RECENCY` | `0.25` | Weight for article recency in scoring |
| `W_GOV` | `0.2` | Weight for government/official sources |
| `W_LLM_IMPORTANCE` | `0.15` | Weight for LLM-assessed importance |
| `DATA_DIR` | `./data` | Directory for intermediate artifacts |
| `CACHE_DIR` | `./cache/http` | Directory for HTTP cache |

### Scoring Weights

The scoring system uses weighted factors (configurable in YAML):
- **Source priority** (0.4): Trusted sources score higher
- **Recency** (0.25): More recent articles score higher  
- **Government/official** (0.2): Official sources get priority
- **LLM importance** (0.15): AI-assessed relevance and impact

### Output Format

The generated newsletter includes:
- **Bold headlines** (≤100 characters)
- **2-3 bullet points** per article (≤35 words each)
- **Category** classification
- **Importance** rating
- **Metric units** and **24-hour time** format

## Development

### Testing
```bash
# Run all tests
poetry run pytest

# Run with coverage
poetry run pytest --cov=aisafety_news --cov-report=html

# Run specific test file
poetry run pytest tests/test_orchestrator.py -v
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

### Docker
```bash
# Build image
docker build -t aisafety-newsletter .

# Run container
docker run -e OPENROUTER_API_KEY=your_key aisafety-newsletter 2025-07-18
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes with tests
4. Ensure tests pass and coverage ≥90%
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Security

- All secrets managed via environment variables or AWS Secrets Manager
- HTTPS-only external requests
- Raw scraped data retained for ≤30 days
- Rate limiting and robots.txt compliance enforced
