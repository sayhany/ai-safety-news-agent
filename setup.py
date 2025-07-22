#!/usr/bin/env python3
"""Setup script for AI Safety Newsletter Agent."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="aisafety-news-agent",
    version="0.1.0",
    author="AI Safety Team",
    author_email="team@example.com",
    description="Autonomous agent that compiles an AI-safety newsletter",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/ai-safety-news-agent",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Information Technology",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Internet :: WWW/HTTP :: Indexing/Search",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Linguistic",
    ],
    python_requires=">=3.11",
    install_requires=[
        "pydantic-settings>=2.2",
        "httpx>=0.27",
        "aiohttp>=3.9",
        "selectolax>=0.3",
        "beautifulsoup4>=4.12",
        "sentence-transformers>=2.4",
        "faiss-cpu>=1.8",
        "structlog>=24.1",
        "orjson>=3.10",
        # "tiktoken>=0.6",  # Optional, can be removed if causing build issues
        "aiosqlite>=0.20",
        "click>=8.1",
        "jinja2>=3.1",
        "pyyaml>=6.0",
        "rich>=13.7.1",
        "exa-py>=1.1.2",
        "openai>=1.97.0",
        "google-genai>=0.5.0",  # Optional for embeddings
        "numpy",
    ],
    extras_require={
        "dev": [
            "pytest>=8.2",
            "pytest-asyncio>=0.23",
            "ruff>=0.4",
            "mypy>=1.10",
            "coverage>=7.5",
            "pytest-cov>=4.1",
            "aresponses>=3.0.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "aisafety-news=aisafety_news.orchestrator:cli",
            "aistr-news-agent=aisafety_news.orchestrator:cli",
            "check-sources=aisafety_news.check_sources:main",
        ],
    },
    include_package_data=True,
    package_data={
        "aisafety_news": ["*.yaml", "templates/*.md"],
    },
)