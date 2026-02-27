"""Agent memory management adapter — short-term (Redis) and long-term (PostgreSQL) memory.

Implements AgentMemoryManagerProtocol with dual-layer storage:
- Short-term memory: Redis with TTL, holds current conversation and working context.
- Long-term memory: PostgreSQL with embedding similarity search for relevant retrieval.
- Memory consolidation promotes important short-term entries to long-term storage.
- Per-agent namespace isolation enforces cross-tenant data boundaries.
"""

import json
import uuid
from datetime import UTC, datetime
from typing import Any

from redis.asyncio import Redis
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from aumos_common.observability import get_logger

logger = get_logger(__name__)

_DEFAULT_SHORT_TERM_TTL_SECONDS = 3600  # 1 hour
_DEFAULT_LONG_TERM_RETENTION_DAYS = 90
_DEFAULT_RELEVANCE_DECAY_FACTOR = 0.95
_DEFAULT_MAX_SHORT_TERM_ENTRIES = 200
_DEFAULT_CONSOLIDATION_THRESHOLD = 0.7  # importance score above this → promote to long-term
_EMBEDDING_DIMENSION = 1536  # OpenAI ada-002 / text-embedding-3-small


class AgentMemoryManager:
    """Dual-layer agent memory with Redis short-term and PostgreSQL long-term storage.

    Short-term memory stores the current working context (conversation turns,
    intermediate reasoning steps, tool outputs) with TTL-based expiry.
    Long-term memory persists facts, learned patterns, and embeddings across sessions.
    Retrieval uses cosine similarity on embeddings for relevance ranking.
    """

    def __init__(
        self,
        redis_client: Redis,  # type: ignore[type-arg]
        db_session: AsyncSession,
        short_term_ttl_seconds: int = _DEFAULT_SHORT_TERM_TTL_SECONDS,
        max_short_term_entries: int = _DEFAULT_MAX_SHORT_TERM_ENTRIES,
        consolidation_threshold: float = _DEFAULT_CONSOLIDATION_THRESHOLD,
    ) -> None:
        """Initialize with Redis and database session.

        Args:
            redis_client: Async Redis client for short-term memory.
            db_session: Async SQLAlchemy session for long-term memory.
            short_term_ttl_seconds: TTL in seconds for Redis short-term entries.
            max_short_term_entries: Maximum entries in short-term memory before pruning.
            consolidation_threshold: Importance score threshold for long-term promotion.
        """
        self._redis = redis_client
        self._db = db_session
        self._short_term_ttl = short_term_ttl_seconds
        self._max_short_term_entries = max_short_term_entries
        self._consolidation_threshold = consolidation_threshold

    # ─── Namespace helpers ────────────────────────────────────────────────────

    def _short_term_key(self, agent_id: uuid.UUID, tenant_id: str) -> str:
        """Build Redis namespace key for agent short-term memory list."""
        return f"mem:st:{tenant_id}:{agent_id}"

    def _working_context_key(self, agent_id: uuid.UUID, tenant_id: str) -> str:
        """Build Redis namespace key for agent working context hash."""
        return f"mem:ctx:{tenant_id}:{agent_id}"

    # ─── Short-term memory (Redis) ────────────────────────────────────────────

    async def add_short_term(
        self,
        agent_id: uuid.UUID,
        tenant_id: str,
        content: str,
        memory_type: str,
        importance: float = 0.5,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Add an entry to the agent's short-term memory.

        Args:
            agent_id: Agent whose memory this belongs to.
            tenant_id: Tenant context for namespace isolation.
            content: The memory content string.
            memory_type: Category label (e.g., 'conversation', 'tool_output', 'reasoning').
            importance: Importance score 0.0-1.0, used for consolidation gating.
            metadata: Optional additional metadata dict.

        Returns:
            Generated memory entry ID string.
        """
        entry_id = str(uuid.uuid4())
        entry = {
            "id": entry_id,
            "content": content,
            "memory_type": memory_type,
            "importance": importance,
            "metadata": metadata or {},
            "created_at": datetime.now(UTC).isoformat(),
        }

        list_key = self._short_term_key(agent_id, tenant_id)
        serialized = json.dumps(entry)

        # Append to sorted list, then enforce max size with LTRIM
        await self._redis.lpush(list_key, serialized)
        await self._redis.ltrim(list_key, 0, self._max_short_term_entries - 1)
        await self._redis.expire(list_key, self._short_term_ttl)

        logger.debug(
            "Short-term memory entry added",
            agent_id=str(agent_id),
            entry_id=entry_id,
            memory_type=memory_type,
            importance=importance,
            tenant_id=tenant_id,
        )
        return entry_id

    async def get_short_term(
        self,
        agent_id: uuid.UUID,
        tenant_id: str,
        limit: int = 50,
        memory_type: str | None = None,
    ) -> list[dict[str, Any]]:
        """Retrieve short-term memory entries, newest first.

        Args:
            agent_id: Agent whose memory to retrieve.
            tenant_id: Tenant context.
            limit: Maximum number of entries to return.
            memory_type: Optional filter by memory type category.

        Returns:
            List of memory entry dicts, ordered newest-first.
        """
        list_key = self._short_term_key(agent_id, tenant_id)
        raw_entries = await self._redis.lrange(list_key, 0, limit * 2 - 1)

        entries: list[dict[str, Any]] = []
        for raw in raw_entries:
            entry = json.loads(raw)
            if memory_type is None or entry.get("memory_type") == memory_type:
                entries.append(entry)
            if len(entries) >= limit:
                break

        return entries

    async def update_working_context(
        self,
        agent_id: uuid.UUID,
        tenant_id: str,
        context_updates: dict[str, Any],
    ) -> None:
        """Update the agent's working context hash in Redis.

        Working context holds the current task state, active goal, and
        intermediate variables the agent needs during its current run.

        Args:
            agent_id: Agent whose context to update.
            tenant_id: Tenant context.
            context_updates: Key-value pairs to merge into the context.
        """
        ctx_key = self._working_context_key(agent_id, tenant_id)
        serialized_updates = {k: json.dumps(v) for k, v in context_updates.items()}
        await self._redis.hset(ctx_key, mapping=serialized_updates)  # type: ignore[arg-type]
        await self._redis.expire(ctx_key, self._short_term_ttl)

        logger.debug(
            "Working context updated",
            agent_id=str(agent_id),
            keys=list(context_updates.keys()),
            tenant_id=tenant_id,
        )

    async def get_working_context(
        self,
        agent_id: uuid.UUID,
        tenant_id: str,
    ) -> dict[str, Any]:
        """Retrieve the agent's full working context.

        Args:
            agent_id: Agent whose context to retrieve.
            tenant_id: Tenant context.

        Returns:
            Working context dict, empty if no context exists.
        """
        ctx_key = self._working_context_key(agent_id, tenant_id)
        raw_context = await self._redis.hgetall(ctx_key)

        if not raw_context:
            return {}

        return {k.decode(): json.loads(v) for k, v in raw_context.items()}

    # ─── Long-term memory (PostgreSQL) ────────────────────────────────────────

    async def store_long_term(
        self,
        agent_id: uuid.UUID,
        tenant_id: str,
        content: str,
        memory_type: str,
        topic: str,
        importance: float,
        embedding: list[float] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Store a memory entry in long-term PostgreSQL storage.

        Args:
            agent_id: Agent whose memory this belongs to.
            tenant_id: Tenant context.
            content: The memory content string.
            memory_type: Category label for filtering.
            topic: High-level topic tag for grouping.
            importance: Importance score 0.0-1.0.
            embedding: Optional vector embedding for similarity search.
            metadata: Optional additional metadata.

        Returns:
            UUID string of the created memory record.
        """
        entry_id = uuid.uuid4()
        embedding_str = f"[{','.join(str(v) for v in embedding)}]" if embedding else None

        stmt = text("""
            INSERT INTO agf_agent_memories (
                id, agent_id, tenant_id, content, memory_type, topic,
                importance, embedding, metadata, created_at, last_accessed_at
            ) VALUES (
                :id, :agent_id, :tenant_id, :content, :memory_type, :topic,
                :importance,
                CASE WHEN :embedding IS NOT NULL
                     THEN :embedding::vector
                     ELSE NULL END,
                :metadata::jsonb, :created_at, :last_accessed_at
            )
        """)

        await self._db.execute(
            stmt,
            {
                "id": entry_id,
                "agent_id": agent_id,
                "tenant_id": uuid.UUID(tenant_id),
                "content": content,
                "memory_type": memory_type,
                "topic": topic,
                "importance": importance,
                "embedding": embedding_str,
                "metadata": json.dumps(metadata or {}),
                "created_at": datetime.now(UTC),
                "last_accessed_at": datetime.now(UTC),
            },
        )
        await self._db.flush()

        logger.info(
            "Long-term memory entry stored",
            agent_id=str(agent_id),
            entry_id=str(entry_id),
            memory_type=memory_type,
            topic=topic,
            importance=importance,
            has_embedding=embedding is not None,
            tenant_id=tenant_id,
        )
        return str(entry_id)

    async def search_long_term(
        self,
        agent_id: uuid.UUID,
        tenant_id: str,
        query_embedding: list[float],
        limit: int = 10,
        topic: str | None = None,
        memory_type: str | None = None,
        min_importance: float = 0.0,
        since_hours: int | None = None,
    ) -> list[dict[str, Any]]:
        """Retrieve long-term memories by embedding similarity.

        Uses cosine similarity via pgvector for relevance-ranked retrieval.

        Args:
            agent_id: Agent whose memories to search.
            tenant_id: Tenant context.
            query_embedding: Embedding vector for similarity comparison.
            limit: Maximum entries to return.
            topic: Optional topic filter.
            memory_type: Optional memory type filter.
            min_importance: Minimum importance score filter.
            since_hours: Optional time filter, only entries within N hours.

        Returns:
            List of memory dicts ordered by cosine similarity (most relevant first).
        """
        embedding_str = f"[{','.join(str(v) for v in query_embedding)}]"

        conditions = [
            "agent_id = :agent_id",
            "tenant_id = :tenant_id",
            "importance >= :min_importance",
        ]
        params: dict[str, Any] = {
            "agent_id": agent_id,
            "tenant_id": uuid.UUID(tenant_id),
            "min_importance": min_importance,
            "limit": limit,
            "embedding": embedding_str,
        }

        if topic:
            conditions.append("topic = :topic")
            params["topic"] = topic
        if memory_type:
            conditions.append("memory_type = :memory_type")
            params["memory_type"] = memory_type
        if since_hours:
            conditions.append(
                "created_at >= NOW() - INTERVAL ':since_hours hours'"
            )
            params["since_hours"] = since_hours

        where_clause = " AND ".join(conditions)
        stmt = text(f"""
            SELECT id, content, memory_type, topic, importance, metadata,
                   created_at, last_accessed_at,
                   1 - (embedding <=> :embedding::vector) AS similarity
            FROM agf_agent_memories
            WHERE {where_clause}
            ORDER BY embedding <=> :embedding::vector
            LIMIT :limit
        """)

        result = await self._db.execute(stmt, params)
        rows = result.fetchall()

        # Update last_accessed_at for retrieved entries
        if rows:
            accessed_ids = [str(row.id) for row in rows]
            await self._db.execute(
                text("""
                    UPDATE agf_agent_memories
                    SET last_accessed_at = :now
                    WHERE id = ANY(:ids::uuid[])
                """),
                {"now": datetime.now(UTC), "ids": "{" + ",".join(accessed_ids) + "}"},
            )

        return [
            {
                "id": str(row.id),
                "content": row.content,
                "memory_type": row.memory_type,
                "topic": row.topic,
                "importance": row.importance,
                "metadata": row.metadata,
                "created_at": row.created_at.isoformat(),
                "last_accessed_at": row.last_accessed_at.isoformat(),
                "similarity": float(row.similarity),
            }
            for row in rows
        ]

    # ─── Consolidation and pruning ────────────────────────────────────────────

    async def consolidate_memory(
        self,
        agent_id: uuid.UUID,
        tenant_id: str,
        get_embedding_fn: Any,  # Callable[[str], Awaitable[list[float]]]
    ) -> int:
        """Promote high-importance short-term entries to long-term memory.

        Iterates short-term entries, generates embeddings for those above the
        importance threshold, and writes them to PostgreSQL long-term storage.

        Args:
            agent_id: Agent whose memory to consolidate.
            tenant_id: Tenant context.
            get_embedding_fn: Async callable that takes a string and returns an embedding.

        Returns:
            Number of entries promoted to long-term storage.
        """
        entries = await self.get_short_term(agent_id, tenant_id, limit=self._max_short_term_entries)
        promoted_count = 0

        for entry in entries:
            if entry.get("importance", 0.0) >= self._consolidation_threshold:
                try:
                    embedding = await get_embedding_fn(entry["content"])
                    await self.store_long_term(
                        agent_id=agent_id,
                        tenant_id=tenant_id,
                        content=entry["content"],
                        memory_type=entry.get("memory_type", "consolidated"),
                        topic=entry.get("metadata", {}).get("topic", "general"),
                        importance=entry["importance"],
                        embedding=embedding,
                        metadata=entry.get("metadata"),
                    )
                    promoted_count += 1
                except Exception as exc:
                    logger.warning(
                        "Memory consolidation failed for entry",
                        agent_id=str(agent_id),
                        entry_id=entry["id"],
                        error=str(exc),
                        tenant_id=tenant_id,
                    )

        logger.info(
            "Memory consolidation complete",
            agent_id=str(agent_id),
            promoted=promoted_count,
            total_evaluated=len(entries),
            tenant_id=tenant_id,
        )
        return promoted_count

    async def prune_long_term(
        self,
        agent_id: uuid.UUID,
        tenant_id: str,
        retention_days: int = _DEFAULT_LONG_TERM_RETENTION_DAYS,
        min_importance_to_retain: float = 0.3,
    ) -> int:
        """Prune expired and low-relevance long-term memory entries.

        Args:
            agent_id: Agent whose memories to prune.
            tenant_id: Tenant context.
            retention_days: Delete entries not accessed within this many days.
            min_importance_to_retain: Entries below this importance score are candidates.

        Returns:
            Number of entries deleted.
        """
        result = await self._db.execute(
            text("""
                WITH deleted AS (
                    DELETE FROM agf_agent_memories
                    WHERE agent_id = :agent_id
                      AND tenant_id = :tenant_id
                      AND (
                          last_accessed_at < NOW() - INTERVAL ':retention_days days'
                          OR importance < :min_importance
                      )
                    RETURNING id
                )
                SELECT COUNT(*) FROM deleted
            """),
            {
                "agent_id": agent_id,
                "tenant_id": uuid.UUID(tenant_id),
                "retention_days": retention_days,
                "min_importance": min_importance_to_retain,
            },
        )
        deleted_count = result.scalar() or 0

        logger.info(
            "Long-term memory pruned",
            agent_id=str(agent_id),
            deleted=deleted_count,
            retention_days=retention_days,
            tenant_id=tenant_id,
        )
        return int(deleted_count)

    async def clear_short_term(self, agent_id: uuid.UUID, tenant_id: str) -> None:
        """Clear all short-term memory and working context for an agent.

        Called at the end of an agent session or on explicit reset.

        Args:
            agent_id: Agent whose short-term memory to clear.
            tenant_id: Tenant context.
        """
        list_key = self._short_term_key(agent_id, tenant_id)
        ctx_key = self._working_context_key(agent_id, tenant_id)
        await self._redis.delete(list_key, ctx_key)

        logger.info(
            "Short-term memory cleared",
            agent_id=str(agent_id),
            tenant_id=tenant_id,
        )
