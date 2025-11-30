"""
Embedding and similarity tools using OpenAI embeddings and FAISS.
"""

import numpy as np
from typing import List, Tuple, Optional
import time
from openai import OpenAI
import faiss
from config import Config


# Initialize OpenAI client
_openai_client: Optional[OpenAI] = None


def _get_openai_client() -> OpenAI:
    """Get or create OpenAI client (singleton pattern)."""
    global _openai_client
    if _openai_client is None:
        if not Config.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY not configured")
        _openai_client = OpenAI(api_key=Config.OPENAI_API_KEY)
    return _openai_client


def get_embeddings(
    texts: List[str],
    model: str = None,
    batch_size: int = 100,
    rate_limit_delay: float = None
) -> np.ndarray:
    """
    Get OpenAI embeddings for a list of texts.

    Args:
        texts: List of text strings to embed
        model: Embedding model to use (default from config)
        batch_size: Process texts in batches of this size
        rate_limit_delay: Delay between API calls (default from config)

    Returns:
        numpy array of shape (len(texts), embedding_dim)

    Raises:
        ValueError: If API key is not configured
        Exception: If API call fails
    """
    if not texts:
        return np.array([])

    if model is None:
        model = Config.EMBEDDING_MODEL

    if rate_limit_delay is None:
        rate_limit_delay = 1.0 / Config.OPENAI_RATE_LIMIT

    client = _get_openai_client()
    all_embeddings = []

    # Process in batches to handle rate limits and large lists
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]

        try:
            response = client.embeddings.create(
                model=model,
                input=batch
            )

            # Extract embeddings from response
            batch_embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embeddings)

            # Rate limiting
            if i + batch_size < len(texts):
                time.sleep(rate_limit_delay)

        except Exception as e:
            print(f"Error getting embeddings for batch {i//batch_size + 1}: {e}")
            raise

    return np.array(all_embeddings, dtype=np.float32)


def get_single_embedding(text: str, model: str = None) -> np.ndarray:
    """
    Get embedding for a single text.

    Args:
        text: Text to embed
        model: Embedding model to use

    Returns:
        1D numpy array of the embedding
    """
    embeddings = get_embeddings([text], model=model)
    return embeddings[0] if len(embeddings) > 0 else np.array([])


def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    """
    Create a FAISS index for similarity search using inner product (cosine similarity).

    Args:
        embeddings: numpy array of shape (n_vectors, embedding_dim)

    Returns:
        FAISS index ready for similarity search

    Note:
        For cosine similarity, embeddings should be L2-normalized before indexing.
    """
    if embeddings.size == 0:
        raise ValueError("Cannot build index from empty embeddings")

    # Normalize embeddings for cosine similarity
    normalized_embeddings = normalize_embeddings(embeddings)

    # Create FAISS index (Inner Product = cosine similarity for normalized vectors)
    dimension = normalized_embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)

    # Add vectors to index
    index.add(normalized_embeddings)

    return index


def normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    """
    L2-normalize embeddings for cosine similarity.

    Args:
        embeddings: numpy array of embeddings

    Returns:
        Normalized embeddings
    """
    if embeddings.ndim == 1:
        # Single embedding
        norm = np.linalg.norm(embeddings)
        return embeddings / norm if norm > 0 else embeddings

    # Multiple embeddings
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)  # Avoid division by zero
    return embeddings / norms


def compute_similarity(
    text1: str,
    text2: str,
    model: str = None
) -> float:
    """
    Compute cosine similarity between two texts.

    Args:
        text1: First text
        text2: Second text
        model: Embedding model to use

    Returns:
        Similarity score between 0.0 and 1.0
    """
    if not text1 or not text2:
        return 0.0

    # Get embeddings
    embeddings = get_embeddings([text1, text2], model=model)

    if len(embeddings) != 2:
        return 0.0

    # Normalize and compute cosine similarity
    emb1_norm = normalize_embeddings(embeddings[0])
    emb2_norm = normalize_embeddings(embeddings[1])

    similarity = np.dot(emb1_norm, emb2_norm)

    # Convert to range [0, 1] from [-1, 1]
    similarity = (similarity + 1.0) / 2.0

    return float(similarity)


def compute_similarity_batch(
    text: str,
    candidates: List[str],
    model: str = None
) -> List[float]:
    """
    Compute similarity between one text and multiple candidate texts.

    More efficient than computing pairwise similarities one by one.

    Args:
        text: Query text
        candidates: List of candidate texts
        model: Embedding model to use

    Returns:
        List of similarity scores (same length as candidates)
    """
    if not text or not candidates:
        return [0.0] * len(candidates)

    # Get all embeddings at once
    all_texts = [text] + candidates
    embeddings = get_embeddings(all_texts, model=model)

    query_embedding = normalize_embeddings(embeddings[0])
    candidate_embeddings = normalize_embeddings(embeddings[1:])

    # Compute similarities
    similarities = np.dot(candidate_embeddings, query_embedding)

    # Convert to range [0, 1]
    similarities = (similarities + 1.0) / 2.0

    return similarities.tolist()


def find_relevant_passage(
    claim: str,
    source_chunks: List[str],
    top_k: int = 3,
    model: str = None,
    min_similarity: float = None
) -> List[Tuple[str, float]]:
    """
    Find the most relevant passages in source that relate to the claim.

    Uses semantic similarity search with FAISS.

    Args:
        claim: The claim text to search for
        source_chunks: List of text chunks from the source document
        top_k: Number of top results to return
        model: Embedding model to use
        min_similarity: Minimum similarity threshold (default from config)

    Returns:
        List of (passage, similarity_score) tuples, sorted by similarity (highest first)
    """
    if not source_chunks:
        return []

    if min_similarity is None:
        min_similarity = Config.SIMILARITY_THRESHOLD

    # Get embeddings for claim and all source chunks
    all_texts = [claim] + source_chunks
    embeddings = get_embeddings(all_texts, model=model)

    query_embedding = embeddings[0:1]  # Keep 2D shape
    source_embeddings = embeddings[1:]

    # Build FAISS index
    normalized_sources = normalize_embeddings(source_embeddings)
    normalized_query = normalize_embeddings(query_embedding)

    dimension = normalized_sources.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(normalized_sources)

    # Search for top k similar passages
    k = min(top_k, len(source_chunks))
    similarities, indices = index.search(normalized_query, k)

    # Convert similarities from [-1, 1] to [0, 1]
    similarities = (similarities[0] + 1.0) / 2.0

    # Build results
    results = []
    for idx, sim in zip(indices[0], similarities):
        if sim >= min_similarity:
            results.append((source_chunks[idx], float(sim)))

    return results


def chunk_text(
    text: str,
    chunk_size: int = 500,
    overlap: int = 50
) -> List[str]:
    """
    Split text into overlapping chunks for better semantic search.

    Args:
        text: Text to split
        chunk_size: Target size of each chunk (in characters)
        overlap: Number of characters to overlap between chunks

    Returns:
        List of text chunks
    """
    if not text:
        return []

    # Clean text
    text = text.strip()

    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size

        # Try to break at sentence boundary
        if end < len(text):
            # Look for sentence end near the boundary
            chunk_text = text[start:end]
            sentence_end = max(
                chunk_text.rfind('. '),
                chunk_text.rfind('! '),
                chunk_text.rfind('? ')
            )

            if sentence_end > chunk_size // 2:  # Only break if we're at least halfway
                end = start + sentence_end + 1

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        # Move start position with overlap
        start = end - overlap if end < len(text) else end

    return chunks


def get_embedding_dimension(model: str = None) -> int:
    """
    Get the dimension of embeddings for a given model.

    Args:
        model: Embedding model name

    Returns:
        Embedding dimension
    """
    if model is None:
        model = Config.EMBEDDING_MODEL

    # Common OpenAI embedding dimensions
    dimensions = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
    }

    return dimensions.get(model, 1536)  # Default to 1536


# Cache for embeddings to avoid recomputation
_embedding_cache = {}


def get_embeddings_cached(
    texts: List[str],
    model: str = None,
    cache_enabled: bool = True
) -> np.ndarray:
    """
    Get embeddings with optional caching.

    Args:
        texts: Texts to embed
        model: Embedding model
        cache_enabled: Whether to use cache

    Returns:
        Embeddings array
    """
    if not cache_enabled:
        return get_embeddings(texts, model=model)

    if model is None:
        model = Config.EMBEDDING_MODEL

    # Check cache
    embeddings_list = []
    texts_to_embed = []
    text_indices = []

    for i, text in enumerate(texts):
        cache_key = (model, text[:200])  # Use first 200 chars as key
        if cache_key in _embedding_cache:
            embeddings_list.append((i, _embedding_cache[cache_key]))
        else:
            texts_to_embed.append(text)
            text_indices.append(i)

    # Get embeddings for uncached texts
    if texts_to_embed:
        new_embeddings = get_embeddings(texts_to_embed, model=model)

        for i, text, embedding in zip(text_indices, texts_to_embed, new_embeddings):
            cache_key = (model, text[:200])
            _embedding_cache[cache_key] = embedding
            embeddings_list.append((i, embedding))

    # Sort by original index and extract embeddings
    embeddings_list.sort(key=lambda x: x[0])
    return np.array([emb for _, emb in embeddings_list], dtype=np.float32)


def clear_embedding_cache():
    """Clear the embedding cache."""
    global _embedding_cache
    _embedding_cache = {}
