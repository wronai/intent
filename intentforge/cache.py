"""
Cache System - Smart caching with fingerprinting for code reuse
Supports memory, Redis, and file-based backends
"""

import hashlib
import json
import logging
import pickle
import threading
from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Generic, TypeVar

from .config import CacheSettings, get_settings
from .core import Intent, IntentResult

logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass
class CacheEntry[T]:
    """Cached item with metadata"""

    key: str
    value: T
    created_at: datetime = field(default_factory=datetime.utcnow)
    expires_at: datetime | None = None
    hit_count: int = 0
    fingerprint: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_expired(self) -> bool:
        if self.expires_at is None:
            return False
        return datetime.utcnow() > self.expires_at

    def touch(self) -> None:
        """Update hit count"""
        self.hit_count += 1


class FingerprintGenerator:
    """
    Generate unique fingerprints for intents
    Determines when cached results can be reused
    """

    def __init__(self, include_context: bool = True, include_constraints: bool = True):
        self.include_context = include_context
        self.include_constraints = include_constraints

    def generate(self, intent: Intent) -> str:
        """Generate fingerprint for an intent"""
        components = [
            intent.description.lower().strip(),
            intent.intent_type.value,
            intent.target_platform.value,
        ]

        if self.include_context and intent.context:
            # Sort context for consistent hashing
            context_str = json.dumps(intent.context, sort_keys=True)
            components.append(context_str)

        if self.include_constraints and intent.constraints:
            constraints_str = "|".join(sorted(intent.constraints))
            components.append(constraints_str)

        combined = "::".join(str(c) for c in components)
        return hashlib.sha256(combined.encode()).hexdigest()[:32]

    def generate_partial(self, description: str, intent_type: str) -> str:
        """Generate partial fingerprint for fuzzy matching"""
        # Normalize description
        normalized = self._normalize_description(description)
        combined = f"{normalized}::{intent_type}"
        return hashlib.md5(combined.encode()).hexdigest()[:16]

    def _normalize_description(self, description: str) -> str:
        """Normalize description for better matching"""
        import re

        # Convert to lowercase
        text = description.lower()
        # Remove extra whitespace
        text = re.sub(r"\s+", " ", text).strip()
        # Remove common stop words
        stop_words = {"a", "an", "the", "to", "for", "with", "that", "this"}
        words = [w for w in text.split() if w not in stop_words]
        return " ".join(words)


class CacheBackend(ABC):
    """Abstract cache backend"""

    @abstractmethod
    def get(self, key: str) -> CacheEntry | None:
        pass

    @abstractmethod
    def set(self, entry: CacheEntry) -> None:
        pass

    @abstractmethod
    def delete(self, key: str) -> bool:
        pass

    @abstractmethod
    def clear(self) -> None:
        pass

    @abstractmethod
    def keys(self) -> list:
        pass

    @abstractmethod
    def stats(self) -> dict[str, Any]:
        pass


class MemoryCache(CacheBackend):
    """
    In-memory LRU cache with TTL support
    Thread-safe implementation
    """

    def __init__(self, max_entries: int = 10000, default_ttl: int = 3600):
        self.max_entries = max_entries
        self.default_ttl = default_ttl
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = threading.RLock()
        self._hits = 0
        self._misses = 0

    def get(self, key: str) -> CacheEntry | None:
        with self._lock:
            entry = self._cache.get(key)

            if entry is None:
                self._misses += 1
                return None

            if entry.is_expired:
                self._cache.pop(key, None)
                self._misses += 1
                return None

            # Move to end (LRU)
            self._cache.move_to_end(key)
            entry.touch()
            self._hits += 1

            return entry

    def set(self, entry: CacheEntry) -> None:
        with self._lock:
            # Set expiration if not set
            if entry.expires_at is None:
                entry.expires_at = datetime.utcnow() + timedelta(seconds=self.default_ttl)

            # Evict if necessary
            while len(self._cache) >= self.max_entries:
                self._cache.popitem(last=False)

            self._cache[entry.key] = entry

    def delete(self, key: str) -> bool:
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0

    def keys(self) -> list:
        with self._lock:
            return list(self._cache.keys())

    def stats(self) -> dict[str, Any]:
        with self._lock:
            total = self._hits + self._misses
            hit_rate = (self._hits / total * 100) if total > 0 else 0

            return {
                "backend": "memory",
                "entries": len(self._cache),
                "max_entries": self.max_entries,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": f"{hit_rate:.2f}%",
            }


class RedisCache(CacheBackend):
    """Redis-based cache backend"""

    def __init__(self, redis_url: str = "redis://localhost:6379/0", prefix: str = "intentforge:"):
        self.redis_url = redis_url
        self.prefix = prefix
        self._client = None

    @property
    def client(self):
        if self._client is None:
            import redis

            self._client = redis.from_url(self.redis_url)
        return self._client

    def _make_key(self, key: str) -> str:
        return f"{self.prefix}{key}"

    def get(self, key: str) -> CacheEntry | None:
        try:
            data = self.client.get(self._make_key(key))
            if data:
                entry_data = json.loads(data)
                entry = self._deserialize_entry(entry_data)
                if not entry.is_expired:
                    entry.touch()
                    # Update hit count in Redis
                    self.client.hset(f"{self._make_key(key)}:meta", "hit_count", entry.hit_count)
                    return entry
            return None
        except Exception as e:
            logger.error(f"Redis get error: {e}")
            return None

    def set(self, entry: CacheEntry) -> None:
        try:
            key = self._make_key(entry.key)
            data = self._serialize_entry(entry)

            if entry.expires_at:
                ttl = int((entry.expires_at - datetime.utcnow()).total_seconds())
                if ttl > 0:
                    self.client.setex(key, ttl, json.dumps(data))
            else:
                self.client.set(key, json.dumps(data))
        except Exception as e:
            logger.error(f"Redis set error: {e}")

    def delete(self, key: str) -> bool:
        try:
            return self.client.delete(self._make_key(key)) > 0
        except Exception as e:
            logger.error(f"Redis delete error: {e}")
            return False

    def clear(self) -> None:
        try:
            keys = self.client.keys(f"{self.prefix}*")
            if keys:
                self.client.delete(*keys)
        except Exception as e:
            logger.error(f"Redis clear error: {e}")

    def keys(self) -> list:
        try:
            keys = self.client.keys(f"{self.prefix}*")
            return [k.decode().replace(self.prefix, "") for k in keys]
        except Exception as e:
            logger.error(f"Redis keys error: {e}")
            return []

    def stats(self) -> dict[str, Any]:
        try:
            info = self.client.info()
            return {
                "backend": "redis",
                "entries": len(self.keys()),
                "used_memory": info.get("used_memory_human", "unknown"),
                "connected_clients": info.get("connected_clients", 0),
            }
        except Exception as e:
            return {"backend": "redis", "error": str(e)}

    def _serialize_entry(self, entry: CacheEntry) -> dict[str, Any]:
        return {
            "key": entry.key,
            "value": entry.value.to_dict() if hasattr(entry.value, "to_dict") else entry.value,
            "created_at": entry.created_at.isoformat(),
            "expires_at": entry.expires_at.isoformat() if entry.expires_at else None,
            "hit_count": entry.hit_count,
            "fingerprint": entry.fingerprint,
            "metadata": entry.metadata,
        }

    def _deserialize_entry(self, data: dict[str, Any]) -> CacheEntry:
        return CacheEntry(
            key=data["key"],
            value=data["value"],
            created_at=datetime.fromisoformat(data["created_at"]),
            expires_at=datetime.fromisoformat(data["expires_at"]) if data["expires_at"] else None,
            hit_count=data["hit_count"],
            fingerprint=data["fingerprint"],
            metadata=data["metadata"],
        )


class FileCache(CacheBackend):
    """File-based cache backend"""

    def __init__(self, cache_dir: Path = Path("/tmp/intentforge_cache")):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._index_file = self.cache_dir / "_index.json"
        self._lock = threading.RLock()

    def _get_path(self, key: str) -> Path:
        # Use hash to avoid filesystem issues with special characters
        safe_key = hashlib.md5(key.encode()).hexdigest()
        return self.cache_dir / f"{safe_key}.cache"

    def get(self, key: str) -> CacheEntry | None:
        with self._lock:
            path = self._get_path(key)
            if not path.exists():
                return None

            try:
                with open(path, "rb") as f:
                    entry = pickle.load(f)

                if entry.is_expired:
                    path.unlink()
                    return None

                entry.touch()
                # Update file with new hit count
                with open(path, "wb") as f:
                    pickle.dump(entry, f)

                return entry
            except Exception as e:
                logger.error(f"File cache get error: {e}")
                return None

    def set(self, entry: CacheEntry) -> None:
        with self._lock:
            path = self._get_path(entry.key)
            try:
                with open(path, "wb") as f:
                    pickle.dump(entry, f)
            except Exception as e:
                logger.error(f"File cache set error: {e}")

    def delete(self, key: str) -> bool:
        with self._lock:
            path = self._get_path(key)
            if path.exists():
                path.unlink()
                return True
            return False

    def clear(self) -> None:
        with self._lock:
            for path in self.cache_dir.glob("*.cache"):
                path.unlink()

    def keys(self) -> list:
        # Note: We can't recover original keys from hashes without an index
        return [p.stem for p in self.cache_dir.glob("*.cache")]

    def stats(self) -> dict[str, Any]:
        cache_files = list(self.cache_dir.glob("*.cache"))
        total_size = sum(f.stat().st_size for f in cache_files)

        return {
            "backend": "file",
            "entries": len(cache_files),
            "cache_dir": str(self.cache_dir),
            "total_size_bytes": total_size,
            "total_size_mb": f"{total_size / 1024 / 1024:.2f}",
        }


class IntentCache:
    """
    High-level cache manager for IntentForge
    Handles fingerprinting and automatic backend selection
    """

    def __init__(self, settings: CacheSettings | None = None):
        self.settings = settings or get_settings().cache
        self.fingerprint_gen = FingerprintGenerator(
            include_context=self.settings.include_context_in_fingerprint,
            include_constraints=self.settings.include_constraints_in_fingerprint,
        )

        # Initialize backend
        self._backend = self._create_backend()

    def _create_backend(self) -> CacheBackend:
        """Create appropriate cache backend"""
        if self.settings.backend == "redis":
            return RedisCache(redis_url=self.settings.redis_url or "redis://localhost:6379/0")
        elif self.settings.backend == "file":
            return FileCache(cache_dir=self.settings.file_path)
        else:
            return MemoryCache(
                max_entries=self.settings.max_entries, default_ttl=self.settings.default_ttl
            )

    def get_result(self, intent: Intent) -> IntentResult | None:
        """Get cached result for an intent"""
        fingerprint = self.fingerprint_gen.generate(intent)
        entry = self._backend.get(fingerprint)

        if entry:
            logger.debug(f"Cache hit for intent {intent.intent_id} (fingerprint: {fingerprint})")
            result = entry.value
            if isinstance(result, dict):
                # Reconstruct IntentResult from dict
                return IntentResult(**result)
            return result

        logger.debug(f"Cache miss for intent {intent.intent_id}")
        return None

    def set_result(self, intent: Intent, result: IntentResult, ttl: int | None = None) -> None:
        """Cache a result for an intent"""
        fingerprint = self.fingerprint_gen.generate(intent)

        expires_at = None
        if ttl or self.settings.default_ttl:
            expires_at = datetime.utcnow() + timedelta(seconds=ttl or self.settings.default_ttl)

        entry = CacheEntry(
            key=fingerprint,
            value=result.to_dict(),
            fingerprint=fingerprint,
            expires_at=expires_at,
            metadata={
                "intent_id": intent.intent_id,
                "intent_type": intent.intent_type.value,
                "description_preview": intent.description[:100],
            },
        )

        self._backend.set(entry)
        logger.debug(f"Cached result for intent {intent.intent_id}")

    def invalidate(self, intent: Intent) -> bool:
        """Invalidate cache for a specific intent"""
        fingerprint = self.fingerprint_gen.generate(intent)
        return self._backend.delete(fingerprint)

    def invalidate_by_type(self, intent_type: str) -> int:
        """Invalidate all cached entries of a specific type"""
        count = 0
        for key in self._backend:
            entry = self._backend.get(key)
            if entry and entry.metadata.get("intent_type") == intent_type:
                self._backend.delete(key)
                count += 1
        return count

    def clear(self) -> None:
        """Clear entire cache"""
        self._backend.clear()

    def stats(self) -> dict[str, Any]:
        """Get cache statistics"""
        return self._backend.stats()

    def find_similar(
        self, description: str, intent_type: str, threshold: float = 0.8
    ) -> IntentResult | None:
        """
        Find similar cached result using fuzzy matching
        Useful when exact match isn't found
        """
        _ = self.fingerprint_gen.generate_partial(description, intent_type)

        # Search through cache for similar entries
        for key in self._backend:
            entry = self._backend.get(key)
            if entry and entry.metadata.get("intent_type") == intent_type:
                cached_desc = entry.metadata.get("description_preview", "")
                similarity = self._calculate_similarity(description, cached_desc)
                if similarity >= threshold:
                    return entry.value

        return None

    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Simple similarity calculation based on common words"""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = words1 & words2
        union = words1 | words2

        return len(intersection) / len(union)
