"""
Deduplication system for news articles using hash-based and embedding-based methods.

This module provides multiple deduplication strategies:
1. URL-based deduplication (exact matches)
2. Hash-based deduplication (content similarity)
3. Embedding-based deduplication using FAISS (semantic similarity)
"""

import asyncio
import hashlib
import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta

try:
    import numpy as np
    # Suppress FAISS GPU warnings by setting environment variable
    import os
    os.environ['FAISS_NO_AVX2'] = '1'  # Disable AVX2 warnings
    
    import faiss
    FAISS_AVAILABLE = True
    
    # Suppress GPU warnings if no GPU FAISS available
    import logging
    faiss_logger = logging.getLogger('faiss')
    faiss_logger.setLevel(logging.ERROR)
    
    # Log FAISS availability
    logger = logging.getLogger(__name__)
    logger.debug("FAISS loaded successfully (CPU mode)")
    
except ImportError:
    FAISS_AVAILABLE = False
    faiss = None
    np = None
    logger = logging.getLogger(__name__)
    logger.info("FAISS not available - embedding deduplication disabled")

from ..models.llm_client import LLMClient
from ..models.embedding_client import get_embedding_client, is_embedding_available
from ..config import Settings
from .text_utils import clean_html_text, canonical_title

logger = logging.getLogger(__name__)


@dataclass
class DuplicateGroup:
    """Group of duplicate articles."""
    canonical_article: Dict
    duplicates: List[Dict]
    similarity_scores: List[float]
    method: str  # 'url', 'hash', 'embedding'


class ArticleDeduplicator:
    """Multi-strategy article deduplication system."""
    
    def __init__(self, settings: Settings, llm_client: Optional[LLMClient] = None):
        """Initialize deduplicator."""
        self.settings = settings
        self.llm_client = llm_client
        
        # Thresholds
        self.hash_similarity_threshold = getattr(settings, 'hash_similarity_threshold', 0.85)
        self.embedding_similarity_threshold = getattr(settings, 'embedding_similarity_threshold', 0.85)
        
        # Check if embedding deduplication is enabled
        self.use_embeddings = getattr(settings, 'use_embedding_deduplication', True) and FAISS_AVAILABLE
        
        # Cache settings
        self.cache_dir = Path(getattr(settings, 'cache_dir', 'cache'))
        self.cache_dir.mkdir(exist_ok=True)
        self.embedding_cache_file = self.cache_dir / 'embeddings.pkl'
        self.faiss_index_file = self.cache_dir / 'faiss_index.bin'
        
        # In-memory caches
        self.url_cache: Set[str] = set()
        self.hash_cache: Dict[str, str] = {}  # hash -> article_id
        self.embedding_cache: Dict[str, np.ndarray] = {}
        self.faiss_index: Optional[faiss.Index] = None
        self.index_to_id: Dict[int, str] = {}  # FAISS index -> article_id
        
        # Load existing caches
        self._load_caches()
    
    def _load_caches(self) -> None:
        """Load existing caches from disk."""
        try:
            if self.embedding_cache_file.exists():
                with open(self.embedding_cache_file, 'rb') as f:
                    cache_data = pickle.load(f)
                    self.embedding_cache = cache_data.get('embeddings', {})
                    self.index_to_id = cache_data.get('index_to_id', {})
                    logger.info(f"Loaded {len(self.embedding_cache)} cached embeddings")
            
            if FAISS_AVAILABLE and self.faiss_index_file.exists():
                self.faiss_index = faiss.read_index(str(self.faiss_index_file))
                logger.info(f"Loaded FAISS index with {self.faiss_index.ntotal} vectors")
                
        except Exception as e:
            logger.warning(f"Failed to load caches: {e}")
            self.embedding_cache = {}
            self.index_to_id = {}
            self.faiss_index = None
    
    def _save_caches(self) -> None:
        """Save caches to disk."""
        try:
            # Save embedding cache
            cache_data = {
                'embeddings': self.embedding_cache,
                'index_to_id': self.index_to_id,
                'timestamp': datetime.now().isoformat()
            }
            with open(self.embedding_cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
            
            # Save FAISS index
            if FAISS_AVAILABLE and self.faiss_index is not None:
                faiss.write_index(self.faiss_index, str(self.faiss_index_file))
                
        except Exception as e:
            logger.warning(f"Failed to save caches: {e}")
    
    def _generate_content_hash(self, article: Dict) -> str:
        """Generate content hash for article."""
        # Combine title and content for hashing
        title = clean_html_text(article.get('title', ''))
        content = clean_html_text(article.get('content', ''))
        
        # Normalize text for better matching
        normalized_title = canonical_title(title)
        normalized_content = canonical_title(content)
        
        # Create hash from normalized content
        combined = f"{normalized_title}\n{normalized_content}"
        return hashlib.sha256(combined.encode('utf-8')).hexdigest()
    
    def _calculate_hash_similarity(self, hash1: str, hash2: str) -> float:
        """Calculate similarity between two hashes (simplified)."""
        # For exact hash matching, similarity is binary
        return 1.0 if hash1 == hash2 else 0.0
    
    async def _get_embedding(self, text: str, article_id: str) -> Optional[np.ndarray]:
        """Get embedding for text, using cache if available."""
        if not FAISS_AVAILABLE:
            return None
        
        # Check cache first
        if article_id in self.embedding_cache:
            return self.embedding_cache[article_id]
        
        try:
            # Get embedding client
            embedding_client = get_embedding_client()
            if not embedding_client.is_available():
                logger.debug("Embedding client not available")
                return None
            
            # Generate embedding using Gemini
            # Use first 1000 chars to avoid token limits
            truncated_text = text[:1000]
            embedding = await embedding_client.generate_embedding(
                text=truncated_text,
                task_type="SEMANTIC_SIMILARITY"
            )
            
            if embedding is not None:
                # Cache the result
                self.embedding_cache[article_id] = embedding
                logger.debug(f"Generated Gemini embedding for article {article_id}")
                return embedding
            else:
                logger.warning(f"Failed to generate embedding for article {article_id}")
                return None
            
        except Exception as e:
            logger.warning(f"Failed to generate embedding for {article_id}: {e}")
            return None
    
    def _add_to_faiss_index(self, embedding: np.ndarray, article_id: str) -> int:
        """Add embedding to FAISS index."""
        if not FAISS_AVAILABLE:
            return -1
        
        # Initialize index if needed or if dimension mismatch
        if self.faiss_index is None or self.faiss_index.d != embedding.shape[0]:
            dimension = embedding.shape[0]
            logger.info(f"Creating new FAISS index with dimension {dimension}")
            self.faiss_index = faiss.IndexFlatIP(dimension)  # Inner product (cosine similarity)
            # Clear the index mapping since we're starting fresh
            self.index_to_id.clear()
        
        # Add to index
        embedding_2d = embedding.reshape(1, -1).astype(np.float32)
        index_id = self.faiss_index.ntotal
        self.faiss_index.add(embedding_2d)
        self.index_to_id[index_id] = article_id
        
        return index_id
    
    def _search_similar_embeddings(self, embedding: np.ndarray, 
                                 threshold: float = 0.85) -> List[Tuple[str, float]]:
        """Search for similar embeddings in FAISS index."""
        if not FAISS_AVAILABLE or self.faiss_index is None or self.faiss_index.ntotal == 0:
            return []
        
        try:
            # Search for similar vectors
            embedding_2d = embedding.reshape(1, -1).astype(np.float32)
            scores, indices = self.faiss_index.search(embedding_2d, min(10, self.faiss_index.ntotal))
            
            # Filter by threshold and convert to article IDs
            similar_articles = []
            for score, idx in zip(scores[0], indices[0]):
                if score >= threshold and idx in self.index_to_id:
                    article_id = self.index_to_id[idx]
                    similar_articles.append((article_id, float(score)))
            
            return similar_articles
            
        except Exception as e:
            logger.warning(f"FAISS search failed: {e}")
            return []
    
    def deduplicate_by_url(self, articles: List[Dict]) -> Tuple[List[Dict], List[DuplicateGroup]]:
        """Remove exact URL duplicates."""
        seen_urls = set()
        unique_articles = []
        duplicate_groups = []
        
        for article in articles:
            url = article.get('url', '')
            if url and url in seen_urls:
                # Find the canonical article
                canonical = next(a for a in unique_articles if a.get('url') == url)
                duplicate_groups.append(DuplicateGroup(
                    canonical_article=canonical,
                    duplicates=[article],
                    similarity_scores=[1.0],
                    method='url'
                ))
            else:
                if url:
                    seen_urls.add(url)
                unique_articles.append(article)
        
        logger.info(f"URL deduplication: {len(articles)} -> {len(unique_articles)} articles")
        return unique_articles, duplicate_groups
    
    def deduplicate_by_hash(self, articles: List[Dict]) -> Tuple[List[Dict], List[DuplicateGroup]]:
        """Remove content hash duplicates."""
        hash_to_article = {}
        unique_articles = []
        duplicate_groups = []
        
        for article in articles:
            content_hash = self._generate_content_hash(article)
            article['content_hash'] = content_hash
            
            if content_hash in hash_to_article:
                # Found duplicate
                canonical = hash_to_article[content_hash]
                duplicate_groups.append(DuplicateGroup(
                    canonical_article=canonical,
                    duplicates=[article],
                    similarity_scores=[1.0],
                    method='hash'
                ))
            else:
                hash_to_article[content_hash] = article
                unique_articles.append(article)
        
        logger.info(f"Hash deduplication: {len(articles)} -> {len(unique_articles)} articles")
        return unique_articles, duplicate_groups
    
    async def deduplicate_by_embedding(self, articles: List[Dict]) -> Tuple[List[Dict], List[DuplicateGroup]]:
        """Remove semantic duplicates using embeddings."""
        if not self.use_embeddings:
            logger.debug("Embedding deduplication disabled")
            return articles, []
            
        if not FAISS_AVAILABLE or not is_embedding_available():
            logger.debug("Embedding deduplication not available (missing FAISS or Gemini embedding client)")
            return articles, []
        
        unique_articles = []
        duplicate_groups = []
        processed_ids = set()
        
        # Generate all embeddings in parallel first for speed
        embedding_client = get_embedding_client()
        
        # Prepare texts for batch processing
        article_texts = []
        article_ids = []
        for i, article in enumerate(articles):
            article_id = article.get('id', f"article_{i}")
            title = article.get('title', '')
            content = article.get('content', '')
            text = f"{title}\n{content[:1000]}"  # Limit content for speed
            article_texts.append(text)
            article_ids.append(article_id)
        
        logger.info(f"Generating embeddings for {len(articles)} articles in parallel...")
        
        # Generate embeddings in parallel batches
        embeddings = await embedding_client.generate_embeddings_batch(
            article_texts, 
            task_type="SEMANTIC_SIMILARITY"
        )
        
        # Now process articles for deduplication
        for i, (article, embedding, article_id) in enumerate(zip(articles, embeddings, article_ids)):
            if article_id in processed_ids:
                continue
                
            if embedding is None:
                unique_articles.append(article)
                continue
            
            # Search for similar articles
            similar_articles = self._search_similar_embeddings(
                embedding, self.embedding_similarity_threshold
            )
            
            if similar_articles:
                # Found duplicates - group them
                duplicates = []
                similarity_scores = []
                
                for similar_id, score in similar_articles:
                    if similar_id != article_id and similar_id not in processed_ids:
                        # Find the similar article by ID
                        similar_idx = next(
                            (j for j, aid in enumerate(article_ids) if aid == similar_id),
                            None
                        )
                        if similar_idx is not None:
                            duplicates.append(articles[similar_idx])
                            similarity_scores.append(score)
                            processed_ids.add(similar_id)
                
                if duplicates:
                    duplicate_groups.append(DuplicateGroup(
                        canonical_article=article,
                        duplicates=duplicates,
                        similarity_scores=similarity_scores,
                        method='embedding'
                    ))
            
            # Add to index and unique articles
            self._add_to_faiss_index(embedding, article_id)
            unique_articles.append(article)
            processed_ids.add(article_id)
        
        logger.info(f"Embedding deduplication: {len(articles)} -> {len(unique_articles)} articles")
        return unique_articles, duplicate_groups
    
    async def deduplicate_articles(self, articles: List[Dict]) -> Tuple[List[Dict], List[DuplicateGroup]]:
        """Full deduplication pipeline using all methods."""
        logger.info(f"Starting deduplication of {len(articles)} articles")
        
        all_duplicate_groups = []
        
        # Add IDs to articles if missing
        for i, article in enumerate(articles):
            if 'id' not in article:
                article['id'] = f"article_{i}_{hash(article.get('url', ''))}"
        
        # Step 1: URL deduplication
        articles, url_dupes = self.deduplicate_by_url(articles)
        all_duplicate_groups.extend(url_dupes)
        
        # Step 2: Hash deduplication
        articles, hash_dupes = self.deduplicate_by_hash(articles)
        all_duplicate_groups.extend(hash_dupes)
        
        # Step 3: Embedding deduplication (if enabled and available)
        if self.use_embeddings:
            articles, embedding_dupes = await self.deduplicate_by_embedding(articles)
            all_duplicate_groups.extend(embedding_dupes)
        
        # Save caches
        self._save_caches()
        
        logger.info(f"Deduplication complete: {len(articles)} unique articles, "
                   f"{len(all_duplicate_groups)} duplicate groups found")
        
        return articles, all_duplicate_groups


async def deduplicate_articles(articles: List[Dict], settings: Settings,
                             llm_client: Optional[LLMClient] = None) -> Tuple[List[Dict], List[DuplicateGroup]]:
    """Convenience function for article deduplication."""
    deduplicator = ArticleDeduplicator(settings, llm_client)
    return await deduplicator.deduplicate_articles(articles)
