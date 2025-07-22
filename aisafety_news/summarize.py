"""
Article summarization using LLM-based approaches.

This module provides intelligent summarization of AI safety news articles
with different summary types and lengths.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from .models.llm_client import LLMClient
from .config import Settings
from .processing.text_utils import clean_html_text, extract_keywords

logger = logging.getLogger(__name__)


class SummaryType(Enum):
    """Types of summaries to generate."""
    BRIEF = "brief"          # 1-2 sentences
    STANDARD = "standard"    # 3-5 sentences
    DETAILED = "detailed"    # 1-2 paragraphs
    TECHNICAL = "technical"  # Focus on technical details
    IMPACT = "impact"        # Focus on implications and impact


@dataclass
class SummaryConfig:
    """Configuration for summary generation."""
    summary_type: SummaryType = SummaryType.STANDARD
    max_length: int = 200
    focus_keywords: List[str] = None
    include_implications: bool = True
    include_technical_details: bool = False


@dataclass
class ArticleSummary:
    """Generated summary with metadata."""
    summary: str
    summary_type: SummaryType
    word_count: int
    key_points: List[str]
    implications: Optional[str] = None
    technical_details: Optional[str] = None
    confidence_score: float = 0.0


class ArticleSummarizer:
    """LLM-based article summarization system."""
    
    def __init__(self, settings: Settings, llm_client: LLMClient):
        """Initialize summarizer."""
        self.settings = settings
        self.llm_client = llm_client
        
        # Default models for different summary types
        self.default_models = {
            SummaryType.BRIEF: "anthropic/claude-3-haiku",
            SummaryType.STANDARD: "anthropic/claude-3-haiku",
            SummaryType.DETAILED: "anthropic/claude-3-sonnet",
            SummaryType.TECHNICAL: "anthropic/claude-3-sonnet",
            SummaryType.IMPACT: "anthropic/claude-3-sonnet",
        }
        
        # Summary length guidelines
        self.length_guidelines = {
            SummaryType.BRIEF: {"sentences": "1-2", "words": "30-50"},
            SummaryType.STANDARD: {"sentences": "3-5", "words": "80-150"},
            SummaryType.DETAILED: {"sentences": "6-10", "words": "200-400"},
            SummaryType.TECHNICAL: {"sentences": "5-8", "words": "150-300"},
            SummaryType.IMPACT: {"sentences": "4-7", "words": "120-250"},
        }
    
    def _build_summary_prompt(self, article: Dict, config: SummaryConfig) -> str:
        """Build prompt for summary generation."""
        title = article.get('title', '')
        content = clean_html_text(article.get('content', ''))
        url = article.get('url', '')
        
        # Truncate content if too long
        max_content_length = 3000
        if len(content) > max_content_length:
            content = content[:max_content_length] + "..."
        
        # Get length guidelines
        guidelines = self.length_guidelines[config.summary_type]
        
        # Build base prompt
        base_prompt = f"""
Summarize this AI safety news article. Focus on the key information relevant to AI safety, alignment, and governance.

Title: {title}
URL: {url}
Content: {content}

Requirements:
- Length: {guidelines['sentences']} sentences ({guidelines['words']} words)
- Focus on AI safety implications and significance
- Be concise but informative
- Use clear, accessible language
"""
        
        # Add specific instructions based on summary type
        if config.summary_type == SummaryType.BRIEF:
            base_prompt += "\n- Provide only the most essential information in 1-2 sentences"
            
        elif config.summary_type == SummaryType.TECHNICAL:
            base_prompt += "\n- Include technical details and methodological information"
            base_prompt += "\n- Explain any AI safety techniques or approaches mentioned"
            
        elif config.summary_type == SummaryType.IMPACT:
            base_prompt += "\n- Emphasize potential implications for AI safety and society"
            base_prompt += "\n- Discuss relevance to AI governance and policy"
            
        elif config.summary_type == SummaryType.DETAILED:
            base_prompt += "\n- Provide comprehensive coverage of all key points"
            base_prompt += "\n- Include context and background information"
        
        # Add focus keywords if provided
        if config.focus_keywords:
            keywords_str = ", ".join(config.focus_keywords)
            base_prompt += f"\n- Pay special attention to: {keywords_str}"
        
        base_prompt += "\n\nProvide just the summary text, no additional formatting or labels."
        
        return base_prompt
    
    def _extract_key_points_prompt(self, article: Dict, summary: str) -> str:
        """Build prompt for extracting key points."""
        return f"""
Based on this article and its summary, extract 3-5 key points as bullet points.

Article Title: {article.get('title', '')}
Summary: {summary}

Extract the most important points that an AI safety researcher should know. Format as:
• Point 1
• Point 2
• Point 3
etc.

Provide only the bullet points, no additional text.
"""
    
    def _extract_implications_prompt(self, article: Dict, summary: str) -> str:
        """Build prompt for extracting implications."""
        return f"""
Based on this AI safety article, what are the key implications for:
1. AI safety research
2. AI governance and policy
3. AI development practices

Article Title: {article.get('title', '')}
Summary: {summary}

Provide a brief analysis (2-3 sentences) focusing on practical implications.
"""
    
    async def _generate_summary(self, article: Dict, config: SummaryConfig) -> str:
        """Generate the main summary."""
        prompt = self._build_summary_prompt(article, config)
        model = self.default_models[config.summary_type]
        
        try:
            response = await self.llm_client.chat(
                messages=prompt,
                max_tokens=min(500, config.max_length * 2),  # Rough token estimate
                temperature=0.3
            )
            
            return response.content.strip()
            
        except Exception as e:
            logger.error(f"Failed to generate summary: {e}")
            # Fallback to simple truncation
            content = article.get('content', '')
            return content[:config.max_length] + "..." if len(content) > config.max_length else content
    
    async def _extract_key_points(self, article: Dict, summary: str) -> List[str]:
        """Extract key points from the article."""
        prompt = self._extract_key_points_prompt(article, summary)
        
        try:
            response_obj = await self.llm_client.chat(
                messages=prompt,
                max_tokens=200,
                temperature=0.2
            )
            response = response_obj.content
            
            # Parse bullet points
            lines = response.strip().split('\n')
            key_points = []
            
            for line in lines:
                line = line.strip()
                if line.startswith('•') or line.startswith('-') or line.startswith('*'):
                    point = line[1:].strip()
                    if point:
                        key_points.append(point)
            
            return key_points[:5]  # Limit to 5 points
            
        except Exception as e:
            logger.warning(f"Failed to extract key points: {e}")
            return []
    
    async def _extract_implications(self, article: Dict, summary: str) -> Optional[str]:
        """Extract implications and impact analysis."""
        prompt = self._extract_implications_prompt(article, summary)
        
        try:
            response = await self.llm_client.chat(
                messages=prompt,
                max_tokens=150,
                temperature=0.3
            )
            
            return response.content.strip()
            
        except Exception as e:
            logger.warning(f"Failed to extract implications: {e}")
            return None
    
    def _calculate_confidence_score(self, article: Dict, summary: str) -> float:
        """Calculate confidence score for the summary."""
        # Simple heuristic-based confidence scoring
        score = 0.5  # Base score
        
        # Content quality indicators
        content = article.get('content', '')
        if len(content) > 500:
            score += 0.1
        if len(content) > 1000:
            score += 0.1
        
        # Summary quality indicators
        if len(summary) > 50:
            score += 0.1
        if len(summary.split('.')) >= 2:  # Multiple sentences
            score += 0.1
        
        # Source credibility
        source_tier = article.get('source_tier', 'unknown')
        if source_tier == 'tier_1':
            score += 0.2
        elif source_tier == 'tier_2':
            score += 0.1
        
        return min(1.0, score)
    
    async def summarize_article(self, article: Dict, 
                              config: Optional[SummaryConfig] = None) -> ArticleSummary:
        """Generate comprehensive summary for an article."""
        if config is None:
            config = SummaryConfig()
        
        logger.debug(f"Summarizing article: {article.get('title', 'Unknown')}")
        
        # Generate main summary
        summary = await self._generate_summary(article, config)
        
        # Generate additional components concurrently
        tasks = []
        
        # Key points
        tasks.append(self._extract_key_points(article, summary))
        
        # Implications (if requested)
        if config.include_implications:
            tasks.append(self._extract_implications(article, summary))
        else:
            tasks.append(asyncio.create_task(asyncio.sleep(0)))  # Dummy task
        
        # Execute tasks
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        key_points = results[0] if not isinstance(results[0], Exception) else []
        implications = results[1] if len(results) > 1 and not isinstance(results[1], Exception) else None
        
        # Calculate confidence score
        confidence_score = self._calculate_confidence_score(article, summary)
        
        # Count words
        word_count = len(summary.split())
        
        return ArticleSummary(
            summary=summary,
            summary_type=config.summary_type,
            word_count=word_count,
            key_points=key_points,
            implications=implications,
            confidence_score=confidence_score
        )
    
    async def summarize_articles(self, articles: List[Dict], 
                               config: Optional[SummaryConfig] = None) -> List[Dict]:
        """Summarize multiple articles with concurrency control."""
        if not articles:
            return []
        
        logger.info(f"Summarizing {len(articles)} articles")
        
        # Limit concurrency to avoid overwhelming the LLM API
        semaphore = asyncio.Semaphore(5)
        
        async def summarize_with_limit(article):
            async with semaphore:
                try:
                    summary_obj = await self.summarize_article(article, config)
                    
                    # Add summary information to article
                    article['summary'] = summary_obj.summary
                    article['summary_type'] = summary_obj.summary_type.value
                    article['summary_word_count'] = summary_obj.word_count
                    article['key_points'] = summary_obj.key_points
                    article['implications'] = summary_obj.implications
                    article['summary_confidence'] = summary_obj.confidence_score
                    
                    return article
                    
                except Exception as e:
                    logger.error(f"Failed to summarize article {article.get('title', 'Unknown')}: {e}")
                    # Add fallback summary
                    article['summary'] = article.get('content', '')[:200] + "..."
                    article['summary_type'] = 'fallback'
                    article['summary_confidence'] = 0.1
                    return article
        
        # Execute summarization tasks
        summarized_articles = await asyncio.gather(
            *[summarize_with_limit(article) for article in articles]
        )
        
        logger.info(f"Summarization complete for {len(summarized_articles)} articles")
        return summarized_articles


async def summarize_articles(articles: List[Dict], settings: Settings, 
                           llm_client: LLMClient, 
                           config: Optional[SummaryConfig] = None) -> List[Dict]:
    """Convenience function for article summarization."""
    summarizer = ArticleSummarizer(settings, llm_client)
    return await summarizer.summarize_articles(articles, config)
