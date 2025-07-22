"""Async OpenAI LLM client with fallback and retry logic."""

import asyncio
import json
import time
from typing import Any, Dict, Iterable, List, Optional, Union

import httpx
from pydantic import BaseModel

try:
    from openai import AsyncOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    AsyncOpenAI = None

try:
    from google import genai
    GOOGLE_AI_AVAILABLE = True
except ImportError:
    GOOGLE_AI_AVAILABLE = False
    genai = None

from ..config import get_model_config, get_settings, LLMRoute, ModelSettings
from ..logging import get_logger, log_api_request, log_error
from ..utils import retry_async

logger = get_logger(__name__)

# OpenAI API model mapping
OPENAI_MODELS = {
    "gpt-4o": "gpt-4o",
    "gpt-4o-mini": "gpt-4o-mini",
    "gpt-4": "gpt-4",
    "gpt-3.5-turbo": "gpt-3.5-turbo"
}

# Gemini API model mapping (fallback support)
GEMINI_MODELS = {
    "gemini-2.5-flash": "gemini-2.5-flash",
    "gemini-2.5-flash-lite-preview-06-17": "gemini-2.5-flash-lite-preview-06-17",
}


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
    """Async Google Gemini client with fallback and retry capabilities."""
    
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
        
        # Initialize OpenAI client by default
        self.openai_api_key = getattr(self.settings, 'openai_api_key', None)
        if self.openai_api_key and OPENAI_AVAILABLE:
            self._openai_client = AsyncOpenAI(api_key=self.openai_api_key)
            logger.info("OpenAI client initialized")
        else:
            self._openai_client = None
            logger.warning("OpenAI API key not found or OpenAI SDK not available")
        
        # Initialize Gemini client as fallback
        self.google_api_key = getattr(self.settings, 'google_ai_api_key', None)
        if self.google_api_key and GOOGLE_AI_AVAILABLE:
            self._gemini_client = genai.Client(api_key=self.google_api_key)
            logger.info("Gemini client initialized as fallback")
        else:
            self._gemini_client = None
            logger.warning("Google AI API key not found or Google AI SDK not available")
        
        if not self._openai_client and not self._gemini_client:
            raise LLMError("Neither OpenAI nor Google AI API keys are available")
    
    def _is_openai_model(self, model: str) -> bool:
        """Check if model is an OpenAI model."""
        return model in OPENAI_MODELS
    
    def _is_gemini_model(self, model: str) -> bool:
        """Check if model is a Gemini model."""
        return model in GEMINI_MODELS
    
    async def _make_openai_request(
        self,
        model: str,
        messages: List[ChatMessage],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        timeout: Optional[int] = None
    ) -> LLMResponse:
        """Make request to OpenAI API."""
        if not self._openai_client:
            raise LLMError("OpenAI client not initialized")
        
        start_time = time.time()
        openai_model = OPENAI_MODELS.get(model, model)
        
        # Convert messages to OpenAI format
        openai_messages = []
        for msg in messages:
            openai_messages.append({
                "role": msg.role,
                "content": msg.content
            })
        
        try:
            response = await self._openai_client.chat.completions.create(
                model=openai_model,
                messages=openai_messages,
                temperature=temperature or self.model_settings.temperature,
                max_tokens=max_tokens or self.model_settings.max_tokens,
                timeout=timeout or self.model_settings.timeout_seconds
            )
            
            response_time = time.time() - start_time
            content = response.choices[0].message.content
            
            if not content:
                raise LLMError("Empty response content from OpenAI")
            
            logger.info(
                "OpenAI API request successful",
                model=openai_model,
                response_time=response_time
            )
            
            return LLMResponse(
                content=content,
                model=openai_model,
                usage={
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                } if response.usage else None,
                response_time=response_time
            )
            
        except Exception as e:
            error_msg = f"OpenAI API error for model {openai_model}: {str(e)}"
            logger.error(error_msg)
            raise LLMError(error_msg) from e
    
    async def _make_gemini_request(
        self,
        model: str,
        messages: List[ChatMessage],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        timeout: Optional[int] = None
    ) -> LLMResponse:
        """Make request to Gemini API."""
        if not self._gemini_client:
            raise LLMError("Gemini client not initialized")
        
        start_time = time.time()
        gemini_model = GEMINI_MODELS.get(model, model)
        
        # Convert messages to Gemini format
        gemini_messages = self._convert_to_gemini_format(messages)
        
        # Prepare generation config
        generation_config = {
            "temperature": temperature or self.model_settings.temperature,
            "max_output_tokens": max_tokens or self.model_settings.max_tokens,
        }
        
        try:
            # Run in thread pool to avoid blocking async event loop
            loop = asyncio.get_event_loop()
            
            response = await loop.run_in_executor(
                None,
                lambda: self._gemini_client.models.generate_content(
                    model=f"models/{gemini_model}",
                    contents=gemini_messages,
                    config=generation_config
                )
            )
            
            response_time = time.time() - start_time
            
            # Extract response content (simplified from previous complex logic)
            if hasattr(response, 'text') and response.text:
                content = response.text
            else:
                raise LLMError("Could not extract content from Gemini response")
            
            if not content or content.strip() == "":
                raise LLMError("Empty response content from Gemini")
            
            logger.info(
                "Gemini API request successful",
                model=gemini_model,
                response_time=response_time
            )
            
            # Estimate usage (Gemini doesn't provide exact token counts)
            usage = {
                "prompt_tokens": sum(len(msg.content.split()) * 1.3 for msg in messages),
                "completion_tokens": len(content.split()) * 1.3,
                "total_tokens": 0
            }
            usage["total_tokens"] = usage["prompt_tokens"] + usage["completion_tokens"]
            
            return LLMResponse(
                content=content,
                model=gemini_model,
                usage=usage,
                response_time=response_time
            )
        
        except Exception as e:
            error_msg = f"Gemini API error for model {gemini_model}: {str(e)}"
            logger.error(error_msg)
            raise LLMError(error_msg) from e
    
    async def _make_request(
        self,
        model: str,
        messages: List[ChatMessage],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        timeout: Optional[int] = None
    ) -> LLMResponse:
        """Route request to appropriate API based on model type."""
        if self._is_openai_model(model):
            if not self._openai_client:
                raise LLMError(f"OpenAI model {model} requested but OpenAI client not available")
            return await self._make_openai_request(model, messages, temperature, max_tokens, timeout)
        elif self._is_gemini_model(model):
            if not self._gemini_client:
                raise LLMError(f"Gemini model {model} requested but Gemini client not available")
            return await self._make_gemini_request(model, messages, temperature, max_tokens, timeout)
        else:
            # Default to OpenAI if available, otherwise Gemini
            if self._openai_client:
                logger.warning(f"Unknown model {model}, defaulting to OpenAI")
                return await self._make_openai_request(model, messages, temperature, max_tokens, timeout)
            elif self._gemini_client:
                logger.warning(f"Unknown model {model}, defaulting to Gemini")
                return await self._make_gemini_request(model, messages, temperature, max_tokens, timeout)
            else:
                raise LLMError(f"No available clients for model {model}")
    
    def _convert_to_gemini_format(self, messages: List[ChatMessage]) -> List[Dict[str, Any]]:
        """Convert chat messages to Gemini format.
        
        Args:
            messages: List of chat messages
            
        Returns:
            List of content objects for Gemini API
        """
        # Gemini expects a list of content objects
        # System messages are combined with the first user message
        # Only user and model roles are supported
        gemini_contents = []
        system_prompt = ""
        
        # Extract system messages first
        for msg in messages:
            if msg.role == "system":
                system_prompt += msg.content + "\n\n"
        
        # Convert remaining messages to Gemini format
        for msg in messages:
            if msg.role == "user":
                content = msg.content
                # Prepend system prompt to first user message
                if system_prompt and not gemini_contents:
                    content = system_prompt + content
                
                gemini_contents.append({
                    "role": "user",
                    "parts": [{"text": content}]
                })
            elif msg.role == "assistant":
                gemini_contents.append({
                    "role": "model", 
                    "parts": [{"text": msg.content}]
                })
            # Skip system messages as they're already handled
        
        # If no user message exists but we have system prompt, create one
        if not gemini_contents and system_prompt:
            gemini_contents.append({
                "role": "user",
                "parts": [{"text": system_prompt.strip()}]
            })
        
        return gemini_contents
    
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
        
        # Check for model override from CLI/settings
        if self.settings.llm_model_override:
            models_to_try = [self.settings.llm_model_override]
            logger.info(f"Using model override: {self.settings.llm_model_override}")
        else:
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
