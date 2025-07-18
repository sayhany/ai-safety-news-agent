"""Async OpenRouter LLM client with fallback and retry logic."""

import asyncio
import json
import time
from typing import Any, Dict, Iterable, List, Optional, Union

import httpx
from pydantic import BaseModel

from ..config import get_model_config, get_settings, LLMRoute, ModelSettings
from ..logging import get_logger, log_api_request, log_error
from ..utils import retry_async

logger = get_logger(__name__)

OPENROUTER_ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"


class ChatMessage(BaseModel):
    """Chat message for LLM interaction."""
    role: str
    content: str


class LLMResponse(BaseModel):
    """LLM response wrapper."""
    content: str
    model: str
    usage: Optional[Dict[str, Any]] = None
    response_time: Optional[float] = None


class LLMError(Exception):
    """LLM-specific error."""
    pass


class LLMClient:
    """Async OpenRouter client with fallback and retry capabilities."""
    
    def __init__(self, route_name: str):
        """Initialize LLM client.
        
        Args:
            route_name: Name of the LLM route configuration
        """
        self.route_name = route_name
        self.settings = get_settings()
        self.model_config = get_model_config()
        
        try:
            self.route: LLMRoute = self.model_config.get_llm_route(route_name)
            self.model_settings: ModelSettings = self.model_config.get_model_settings()
        except Exception as e:
            logger.error(f"Failed to load LLM route '{route_name}': {e}")
            raise LLMError(f"Invalid LLM route: {route_name}") from e
        
        self.api_key = self.settings.openrouter_api_key
        if not self.api_key:
            raise LLMError("OPENROUTER_API_KEY is required")
    
    async def _make_request(
        self,
        model: str,
        messages: List[ChatMessage],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        timeout: Optional[int] = None
    ) -> LLMResponse:
        """Make a single request to OpenRouter API.
        
        Args:
            model: Model name
            messages: Chat messages
            temperature: Sampling temperature
            max_tokens: Maximum tokens
            timeout: Request timeout
            
        Returns:
            LLM response
            
        Raises:
            LLMError: If request fails
        """
        start_time = time.time()
        
        # Prepare request payload
        payload = {
            "model": model,
            "messages": [msg.model_dump() for msg in messages],
            "temperature": temperature or self.model_settings.temperature,
            "max_tokens": max_tokens or self.model_settings.max_tokens,
        }
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/your/repo",
            "User-Agent": self.settings.user_agent,
        }
        
        timeout_seconds = timeout or self.model_settings.timeout_seconds
        
        try:
            async with httpx.AsyncClient(timeout=timeout_seconds) as client:
                logger.debug(
                    "Making LLM request",
                    model=model,
                    messages_count=len(messages),
                    temperature=payload["temperature"],
                    max_tokens=payload["max_tokens"]
                )
                
                response = await client.post(
                    OPENROUTER_ENDPOINT,
                    json=payload,
                    headers=headers
                )
                
                response_time = time.time() - start_time
                
                # Log API request
                logger.info(
                    **log_api_request(
                        method="POST",
                        url=OPENROUTER_ENDPOINT,
                        status_code=response.status_code,
                        response_time=response_time,
                        model=model
                    )
                )
                
                if response.status_code != 200:
                    error_msg = f"OpenRouter API error {response.status_code}: {response.text}"
                    logger.error(error_msg, model=model, status_code=response.status_code)
                    raise LLMError(error_msg)
                
                response_data = response.json()
                
                # Extract response content
                if "choices" not in response_data or not response_data["choices"]:
                    raise LLMError("No choices in OpenRouter response")
                
                choice = response_data["choices"][0]
                if "message" not in choice or "content" not in choice["message"]:
                    raise LLMError("Invalid response format from OpenRouter")
                
                content = choice["message"]["content"]
                usage = response_data.get("usage", {})
                
                logger.debug(
                    "LLM request successful",
                    model=model,
                    response_time=response_time,
                    content_length=len(content),
                    usage=usage
                )
                
                return LLMResponse(
                    content=content,
                    model=model,
                    usage=usage,
                    response_time=response_time
                )
        
        except httpx.TimeoutException as e:
            error_msg = f"Request timeout for model {model}"
            logger.warning(error_msg, timeout=timeout_seconds)
            raise LLMError(error_msg) from e
        
        except httpx.RequestError as e:
            error_msg = f"Request error for model {model}: {str(e)}"
            logger.warning(error_msg)
            raise LLMError(error_msg) from e
        
        except json.JSONDecodeError as e:
            error_msg = f"Invalid JSON response from model {model}"
            logger.warning(error_msg)
            raise LLMError(error_msg) from e
        
        except Exception as e:
            error_msg = f"Unexpected error for model {model}: {str(e)}"
            logger.error(**log_error(e, context=f"LLM request to {model}"))
            raise LLMError(error_msg) from e
    
    async def chat(
        self,
        messages: Union[List[ChatMessage], List[Dict[str, str]], str],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        timeout: Optional[int] = None
    ) -> LLMResponse:
        """Send chat messages to LLM with fallback and retry.
        
        Args:
            messages: Chat messages (various formats accepted)
            temperature: Sampling temperature
            max_tokens: Maximum tokens
            timeout: Request timeout
            
        Returns:
            LLM response
            
        Raises:
            LLMError: If all models and retries fail
        """
        # Normalize messages to ChatMessage format
        normalized_messages = self._normalize_messages(messages)
        
        if not normalized_messages:
            raise LLMError("No messages provided")
        
        # Try all models in the route
        models_to_try = [self.route.primary] + self.route.fallback
        last_error = None
        
        for model in models_to_try:
            logger.debug(f"Trying model: {model}")
            
            # Retry logic for each model
            async def make_request():
                return await self._make_request(
                    model=model,
                    messages=normalized_messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    timeout=timeout
                )
            
            try:
                response = await retry_async(
                    make_request,
                    max_retries=self.model_settings.retry_attempts,
                    backoff_factor=self.model_settings.backoff_factor,
                    exceptions=(LLMError, httpx.RequestError, httpx.TimeoutException)
                )
                
                logger.info(
                    "LLM request successful",
                    route=self.route_name,
                    model=model,
                    response_time=response.response_time
                )
                
                return response
            
            except Exception as e:
                last_error = e
                logger.warning(
                    f"Model {model} failed after retries",
                    route=self.route_name,
                    error=str(e)
                )
                continue
        
        # All models failed
        error_msg = f"All models failed for route '{self.route_name}'"
        logger.error(error_msg, last_error=str(last_error) if last_error else None)
        raise LLMError(error_msg) from last_error
    
    def _normalize_messages(
        self, 
        messages: Union[List[ChatMessage], List[Dict[str, str]], str]
    ) -> List[ChatMessage]:
        """Normalize messages to ChatMessage format.
        
        Args:
            messages: Messages in various formats
            
        Returns:
            List of ChatMessage objects
        """
        if isinstance(messages, str):
            return [ChatMessage(role="user", content=messages)]
        
        elif isinstance(messages, list):
            normalized = []
            for msg in messages:
                if isinstance(msg, ChatMessage):
                    normalized.append(msg)
                elif isinstance(msg, dict):
                    if "role" in msg and "content" in msg:
                        normalized.append(ChatMessage(role=msg["role"], content=msg["content"]))
                    else:
                        raise LLMError(f"Invalid message format: {msg}")
                else:
                    raise LLMError(f"Unsupported message type: {type(msg)}")
            return normalized
        
        else:
            raise LLMError(f"Unsupported messages type: {type(messages)}")
    
    async def summarize(
        self,
        text: str,
        max_length: Optional[int] = None,
        style: str = "concise"
    ) -> str:
        """Summarize text using the LLM.
        
        Args:
            text: Text to summarize
            max_length: Maximum summary length
            style: Summary style (concise, detailed, bullets)
            
        Returns:
            Summary text
        """
        if not text.strip():
            return ""
        
        # Create summarization prompt
        prompt = self._create_summarization_prompt(text, max_length, style)
        
        response = await self.chat(prompt)
        return response.content.strip()
    
    def _create_summarization_prompt(
        self,
        text: str,
        max_length: Optional[int],
        style: str
    ) -> str:
        """Create summarization prompt.
        
        Args:
            text: Text to summarize
            max_length: Maximum length
            style: Summary style
            
        Returns:
            Prompt string
        """
        length_instruction = ""
        if max_length:
            length_instruction = f" Keep it under {max_length} characters."
        
        style_instructions = {
            "concise": "Provide a concise summary focusing on key points.",
            "detailed": "Provide a detailed summary covering main topics.",
            "bullets": "Provide a bullet-point summary with 2-3 key points."
        }
        
        style_instruction = style_instructions.get(style, style_instructions["concise"])
        
        return f"""Please summarize the following text. {style_instruction}{length_instruction}

Text to summarize:
{text}

Summary:"""


class MockLLMClient(LLMClient):
    """Mock LLM client for testing."""
    
    def __init__(self, route_name: str):
        """Initialize mock client."""
        self.route_name = route_name
        # Don't call parent __init__ to avoid API key requirements
    
    async def chat(
        self,
        messages: Union[List[ChatMessage], List[Dict[str, str]], str],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        timeout: Optional[int] = None
    ) -> LLMResponse:
        """Mock chat response."""
        # Simulate processing time
        await asyncio.sleep(0.1)
        
        # Generate mock response based on input
        if isinstance(messages, str):
            content = f"Mock response to: {messages[:50]}..."
        else:
            last_message = messages[-1] if messages else ""
            if isinstance(last_message, ChatMessage):
                content = f"Mock response to: {last_message.content[:50]}..."
            elif isinstance(last_message, dict):
                content = f"Mock response to: {last_message.get('content', '')[:50]}..."
            else:
                content = "Mock response"
        
        return LLMResponse(
            content=content,
            model="mock-model",
            usage={"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
            response_time=0.1
        )


def create_llm_client(route_name: str, mock: bool = False) -> LLMClient:
    """Factory function to create LLM client.
    
    Args:
        route_name: LLM route name
        mock: Whether to use mock client
        
    Returns:
        LLM client instance
    """
    if mock:
        return MockLLMClient(route_name)
    else:
        return LLMClient(route_name)


# Convenience functions for common operations
async def summarize_article(
    text: str,
    mock: bool = False,
    max_length: int = 500
) -> str:
    """Summarize article text.
    
    Args:
        text: Article text
        mock: Use mock client
        max_length: Maximum summary length
        
    Returns:
        Article summary
    """
    client = create_llm_client("summarizer", mock=mock)
    return await client.summarize(text, max_length=max_length, style="bullets")


async def assess_relevance(
    title: str,
    content: str,
    mock: bool = False
) -> float:
    """Assess AI safety relevance of article.
    
    Args:
        title: Article title
        content: Article content
        mock: Use mock client
        
    Returns:
        Relevance score (0.0 to 1.0)
    """
    if mock:
        # Mock relevance based on keywords
        text = f"{title} {content}".lower()
        ai_keywords = ["ai", "artificial intelligence", "machine learning", "safety", "ethics"]
        score = sum(1 for keyword in ai_keywords if keyword in text) / len(ai_keywords)
        return min(score, 1.0)
    
    client = create_llm_client("relevance", mock=mock)
    
    prompt = f"""Rate the relevance of this article to AI safety on a scale of 0.0 to 1.0.

AI safety includes topics like: AI alignment, AI governance, AI regulation, AI ethics, AI risk assessment, algorithmic bias, AI transparency, AI accountability, responsible AI development.

Article Title: {title}

Article Content: {content[:1000]}...

Provide only a numeric score between 0.0 and 1.0:"""
    
    try:
        response = await client.chat(prompt)
        # Extract numeric score from response
        import re
        score_match = re.search(r'(\d+\.?\d*)', response.content)
        if score_match:
            score = float(score_match.group(1))
            return max(0.0, min(1.0, score))
        else:
            logger.warning("Could not parse relevance score", response=response.content)
            return 0.0
    except Exception as e:
        logger.error("Failed to assess relevance", error=str(e))
        return 0.0


if __name__ == "__main__":
    # Test the LLM client
    async def test_client():
        client = create_llm_client("summarizer", mock=True)
        
        test_messages = [
            ChatMessage(role="user", content="What is AI safety?")
        ]
        
        response = await client.chat(test_messages)
        print(f"Response: {response.content}")
        print(f"Model: {response.model}")
        print(f"Response time: {response.response_time}")
    
    asyncio.run(test_client())
