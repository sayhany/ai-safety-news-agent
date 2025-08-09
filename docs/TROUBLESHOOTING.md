# AI Safety News Agent - Troubleshooting Guide

This guide helps you resolve common issues with the AI Safety News Agent.

## Table of Contents
- [Source Connectivity Issues](#source-connectivity-issues)
- [GPU/FAISS Warnings](#gpufaiss-warnings)
- [External Dependencies](#external-dependencies)
- [Performance Issues](#performance-issues)
- [Debugging Tips](#debugging-tips)

## Source Connectivity Issues

### Problem: Search API requests failing
The agent may encounter connectivity issues with the configured search APIs (e.g., Exa, Bing).

#### Solutions:

1. **Check Source Health**
   ```bash
   poetry run check-sources
   ```
   This command shows the health status of all configured sources.

2. **Check API Keys**
   Ensure your search API keys are set correctly in your `.env` file:
   ```
   EXA_API_KEY="your_exa_api_key"
   BING_API_KEY="your_bing_api_key"
   ```

3. **Manual Source Configuration**
   Edit `models.yaml` to add or modify search queries:
   ```yaml
   search_sources:
     - name: "Reuters AI Safety News"
       api: "exa"
       query: '(AI safety OR AI governance OR AI regulation) site:reuters.com'
       date_range: 7
       priority: 0.9
   ```

4. **Temporary Network Issues**
   - The agent will automatically retry failed requests with exponential backoff.
   - Use the `--mock` flag for testing without network access.

## GPU/FAISS Warnings

### Problem: "Faiss not compiled with GPU support" warning
This is a non-critical warning that appears when using the CPU-only version of FAISS.

#### Solutions:

1. **Disable the Warning** (Already implemented)
   The agent now suppresses this warning automatically.

2. **Disable Embedding Deduplication**
   Set in your `.env` file:
   ```bash
   USE_EMBEDDING_DEDUPLICATION=false
   ```

3. **Install GPU Version** (Optional)
   If you have a CUDA-capable GPU:
   ```bash
   pip uninstall faiss-cpu
   pip install faiss-gpu
   ```

## External Dependencies

### Problem: OpenRouter API errors

#### Solutions:

1. **Check API Key**
   Ensure your API key is set correctly:
   ```bash
   export OPENROUTER_API_KEY="your-api-key"
   # or in .env file
   OPENROUTER_API_KEY=your-api-key
   ```

2. **API Rate Limits**
   The agent includes:
   - Automatic retry with backoff
   - Fallback models for each operation
   - Configurable timeouts

3. **Model Availability**
   Check `models.yaml` for fallback models:
   ```yaml
   summarizer:
     primary: "anthropic/claude-3-haiku"
     fallback:
       - "openai/gpt-4o-mini"
       - "meta-llama/llama-3.1-8b-instruct"
   ```

### Problem: Missing Python dependencies

#### Solutions:
```bash
# Reinstall all dependencies
poetry install

# Update dependencies
poetry update

# Clear cache and reinstall
poetry cache clear pypi --all
poetry install
```

## Performance Issues

### Problem: Slow processing

#### Solutions:

1. **Adjust Concurrency Settings**
   In `.env`:
   ```bash
   MAX_PER_DOMAIN=5  # Reduce concurrent requests per domain
   GLOBAL_PARALLEL=20  # Reduce total concurrent requests
   ```

2. **Limit Articles Per Source**
   ```bash
   poetry run aisafety-news 2025-07-18 --limit-per-source 5
   ```

3. **Disable Embedding Deduplication**
   This can significantly speed up processing:
   ```bash
   USE_EMBEDDING_DEDUPLICATION=false
   ```

### Problem: High memory usage

#### Solutions:
- Reduce batch sizes in processing
- Clear caches regularly: `rm -rf cache/`
- Use `--mock` flag for testing with minimal resources

## Debugging Tips

### Enable Verbose Logging
```bash
# Set log level
export LOG_LEVEL=DEBUG

# Or run with debug flag
poetry run aisafety-news 2025-07-18 --debug
```

### Check Logs
The agent uses structured JSON logging. To view logs:
```bash
# Pretty print logs
poetry run aisafety-news 2025-07-18 | jq '.'

# Filter by level
poetry run aisafety-news 2025-07-18 | jq 'select(.level=="ERROR")'

# Search for specific issues
poetry run aisafety-news 2025-07-18 | jq 'select(.msg | contains("search"))'
```

### Test Individual Components

1. **Test Sources Only**
   ```bash
   poetry run check-sources
   ```

2. **Test with Mock Data**
   ```bash
   poetry run aisafety-news 2025-07-18 --mock
   ```

3. **Test Specific Pipeline Stage**
   ```python
   # In Python
   from aisafety_news.ingest.sources import gather_articles
   articles = await gather_articles("2025-07-18", mock=True)
   ```

### Common Error Messages

| Error | Cause | Solution |
|-------|-------|----------|
| `aiohttp.ClientError` | Network connectivity | Check internet connection, retry |
| `ValueError: Exa Search API key is not set.` | Missing API key | Add `EXA_API_KEY` to your `.env` file. |
| `FAISS not available` | Missing dependency | `poetry add faiss-cpu` |
| `LLMError: timeout` | API timeout | Increase timeout, check API status |

## Getting Help

1. **Check Implementation Status**
   Review `IMPLEMENTATION_STATUS.md` for feature completion status.

2. **Review Logs**
   Most issues are logged with context and suggestions.

3. **Run Tests**
   ```bash
   poetry run pytest -v
   ```

4. **File an Issue**
   Include:
   - Error messages
   - Log output (with `--debug`)
   - Source check results
   - Environment details

## Environment Variables Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENROUTER_API_KEY` | Required | API key for LLM access |
| `LOG_LEVEL` | INFO | Logging verbosity |
| `USE_EMBEDDING_DEDUPLICATION` | true | Enable semantic deduplication |
| `MAX_PER_DOMAIN` | 10 | Max concurrent requests per domain |
| `GLOBAL_PARALLEL` | 30 | Total concurrent requests |
| `DATA_RETENTION_DAYS` | 30 | How long to keep cached data |

## Quick Fixes

### Reset Everything
```bash
# Clear all caches and data
rm -rf cache/ data/

# Reinstall dependencies
poetry install

# Test with mock data
poetry run aisafety-news 2025-07-18 --mock
```

### Minimal Configuration
For testing with minimal external dependencies:
```bash
# Disable embedding deduplication
export USE_EMBEDDING_DEDUPLICATION=false

# Use mock data
poetry run aisafety-news 2025-07-18 --mock

# Or limit scope
poetry run aisafety-news 2025-07-18 --limit-per-source 2