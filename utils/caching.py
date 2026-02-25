"""
Caching layer for AutoDataLab.
Implements efficient caching for expensive operations using Streamlit and memory cache.
"""

import hashlib
import pickle
import json
from functools import wraps
from typing import Any, Callable, Dict, Optional, Tuple
from pathlib import Path
import pandas as pd
import numpy as np
import streamlit as st

from utils.logger import get_logger
from utils.exceptions import CachingError
from config import settings

logger = get_logger(__name__)

# Cache directory
CACHE_DIR = Path(__file__).parent.parent / ".cache"
CACHE_DIR.mkdir(exist_ok=True, parents=True)


def get_cache_key(*args, **kwargs) -> str:
    """
    Generate cache key from function arguments.
    
    Args:
        *args: Positional arguments
        **kwargs: Keyword arguments
        
    Returns:
        str: Hash-based cache key
    """
    try:
        key_data = str(args) + str(sorted(kwargs.items()))
        return hashlib.md5(key_data.encode()).hexdigest()
    except Exception as e:
        logger.error(f"Error generating cache key: {e}")
        raise CachingError(f"Failed to generate cache key: {e}")


def get_dataframe_hash(df: pd.DataFrame) -> str:
    """
    Generate hash of DataFrame content.
    
    Args:
        df: pandas DataFrame
        
    Returns:
        str: DataFrame content hash
    """
    try:
        df_bytes = pickle.dumps(df)
        return hashlib.md5(df_bytes).hexdigest()
    except Exception as e:
        logger.error(f"Error hashing DataFrame: {e}")
        raise CachingError(f"Failed to hash DataFrame: {e}")


def cache_to_disk(func: Callable) -> Callable:
    """
    Decorator for caching function results to disk.
    Useful for expensive I/O operations and transformations.
    
    Args:
        func: Function to cache
        
    Returns:
        Callable: Wrapped function with disk caching
        
    Example:
        >>> @cache_to_disk
        ... def expensive_computation(df):
        ...     return df.groupby('column').sum()
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        if not settings.CACHE_ENABLED:
            return func(*args, **kwargs)
        
        try:
            cache_key = get_cache_key(func.__name__, *args, **kwargs)
            cache_file = CACHE_DIR / f"{cache_key}.pkl"
            
            # Check if cache exists
            if cache_file.exists():
                logger.debug(f"Cache hit for {func.__name__}")
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            
            # Compute result
            logger.debug(f"Computing {func.__name__}, will cache result")
            result = func(*args, **kwargs)
            
            # Save to cache
            with open(cache_file, 'wb') as f:
                pickle.dump(result, f)
            
            logger.info(f"Cached result for {func.__name__}")
            return result
            
        except Exception as e:
            logger.warning(f"Caching failed for {func.__name__}: {e}, computing without cache")
            return func(*args, **kwargs)
    
    return wrapper


def streamlit_cache(
    ttl: int = 3600,
    show_spinner: bool = True
) -> Callable:
    """
    Decorator for caching using Streamlit's @st.cache_data.
    Optimal for DataFrames and computed results within a session.
    
    Args:
        ttl: Time to live in seconds (default: 1 hour)
        show_spinner: Show spinner while computing
        
    Returns:
        Callable: Wrapped function with Streamlit caching
        
    Example:
        >>> @streamlit_cache(ttl=3600)
        ... def load_preprocessed_data(df):
        ...     return df.dropna()
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not settings.CACHE_ENABLED:
                return func(*args, **kwargs)
            
            # Use Streamlit's cache decorator
            cached_func = st.cache_data(
                func=func,
                ttl=ttl,
                show_spinner=show_spinner
            )
            return cached_func(*args, **kwargs)
        
        return wrapper
    
    return decorator


class DataFrameCache:
    """
    Cache manager for DataFrame operations.
    Tracks DataFrame modifications and invalidates relevant caches.
    """
    
    def __init__(self):
        """Initialize DataFrame cache manager."""
        self.df_hashes: Dict[str, str] = {}
        self.operation_cache: Dict[str, Any] = {}
        logger.info("DataFrameCache initialized")
    
    def update_dataframe_hash(self, df_name: str, df: pd.DataFrame) -> None:
        """
        Update hash for a DataFrame.
        
        Args:
            df_name: Name/identifier for DataFrame
            df: pandas DataFrame
        """
        try:
            hash_value = get_dataframe_hash(df)
            self.df_hashes[df_name] = hash_value
            logger.debug(f"Updated hash for {df_name}: {hash_value}")
        except Exception as e:
            logger.error(f"Failed to update DataFrame hash: {e}")
    
    def is_dataframe_changed(self, df_name: str, df: pd.DataFrame) -> bool:
        """
        Check if DataFrame has changed since last hash.
        
        Args:
            df_name: Name/identifier for DataFrame
            df: pandas DataFrame
            
        Returns:
            bool: True if DataFrame changed, False otherwise
        """
        try:
            if df_name not in self.df_hashes:
                return True
            
            current_hash = get_dataframe_hash(df)
            has_changed = current_hash != self.df_hashes[df_name]
            
            if has_changed:
                logger.debug(f"DataFrame {df_name} has changed")
            
            return has_changed
            
        except Exception as e:
            logger.error(f"Failed to check DataFrame change: {e}")
            return True  # Assume changed on error
    
    def invalidate_related_cache(self, df_name: str) -> None:
        """
        Invalidate cached operations related to a DataFrame.
        
        Args:
            df_name: Name/identifier for DataFrame
        """
        try:
            keys_to_remove = [
                key for key in self.operation_cache.keys()
                if df_name in key
            ]
            
            for key in keys_to_remove:
                del self.operation_cache[key]
                logger.debug(f"Invalidated cache: {key}")
            
            if keys_to_remove:
                logger.info(f"Invalidated {len(keys_to_remove)} cache entries for {df_name}")
        
        except Exception as e:
            logger.error(f"Error invalidating cache: {e}")
    
    def cache_operation(self, operation_key: str, result: Any) -> None:
        """
        Cache result of an operation.
        
        Args:
            operation_key: Unique key for operation
            result: Operation result to cache
        """
        try:
            self.operation_cache[operation_key] = result
            logger.debug(f"Cached operation: {operation_key}")
        except Exception as e:
            logger.error(f"Failed to cache operation: {e}")
    
    def get_cached_operation(
        self,
        operation_key: str,
        default: Optional[Any] = None
    ) -> Optional[Any]:
        """
        Retrieve cached operation result.
        
        Args:
            operation_key: Unique key for operation
            default: Default value if not found
            
        Returns:
            Cached result or default value
        """
        return self.operation_cache.get(operation_key, default)
    
    def clear_cache(self) -> None:
        """Clear all caches."""
        try:
            self.df_hashes.clear()
            self.operation_cache.clear()
            logger.info("All caches cleared")
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")


# Global cache instance
df_cache = DataFrameCache()


def clear_disk_cache() -> None:
    """Clear all disk-based caches."""
    try:
        for cache_file in CACHE_DIR.glob("*.pkl"):
            cache_file.unlink()
        logger.info("Disk cache cleared")
    except Exception as e:
        logger.error(f"Error clearing disk cache: {e}")
        raise CachingError(f"Failed to clear disk cache: {e}")


def get_cache_stats() -> Dict[str, Any]:
    """
    Get cache statistics.
    
    Returns:
        dict: Cache statistics
    """
    try:
        cache_files = list(CACHE_DIR.glob("*.pkl"))
        total_size = sum(f.stat().st_size for f in cache_files)
        
        return {
            "cached_items": len(cache_files),
            "total_size_bytes": total_size,
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "cache_dir": str(CACHE_DIR)
        }
    except Exception as e:
        logger.error(f"Error getting cache stats: {e}")
        return {}


if __name__ == "__main__":
    # Test caching
    logger.info("Testing caching system")
    
    @cache_to_disk
    def expensive_operation(x):
        import time
        time.sleep(1)
        return x ** 2
    
    print("First call (will compute)...")
    result1 = expensive_operation(5)
    print(f"Result: {result1}")
    
    print("Second call (will use cache)...")
    result2 = expensive_operation(5)
    print(f"Result: {result2}")
    
    print("\nCache statistics:")
    stats = get_cache_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
