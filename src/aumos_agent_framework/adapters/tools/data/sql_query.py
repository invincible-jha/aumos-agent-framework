"""Read-only SQL query tool via aumos-data-layer connection.

Privilege level 2 — requires EXECUTE_SAFE; SELECT only, no mutations.
"""

from typing import Any

from pydantic import Field

from aumos_agent_framework.core.interfaces import ToolInputSchema, ToolOutputSchema
from aumos_common.observability import get_logger

logger = get_logger(__name__)


class SQLQueryInput(ToolInputSchema):
    """Input schema for the read-only SQL query tool."""

    query: str = Field(..., description="SQL SELECT query to execute")
    max_rows: int = Field(default=100, ge=1, le=5000, description="Maximum rows to return")
    parameters: list[Any] = Field(default_factory=list, description="Positional query parameters")


class SQLQueryOutput(ToolOutputSchema):
    """Output schema for the SQL query tool."""

    result: list[dict[str, Any]] = Field(description="Query result rows as list of dicts")
    metadata: dict[str, Any] = Field(default_factory=dict)


class SQLQueryTool:
    """Execute a read-only SQL SELECT query via the aumos-data-layer connection pool.

    Only SELECT statements are permitted. Any query containing DML keywords
    (INSERT, UPDATE, DELETE, DROP, TRUNCATE, ALTER, CREATE) will be rejected.
    """

    tool_id: str = "sql_query"
    display_name: str = "SQL Query"
    category: str = "data"
    description: str = "Execute a read-only SQL SELECT query against the tenant database and return rows."
    privilege_level: int = 2
    input_schema: type[SQLQueryInput] = SQLQueryInput
    output_schema: type[SQLQueryOutput] = SQLQueryOutput

    _FORBIDDEN_KEYWORDS = frozenset(
        {"insert", "update", "delete", "drop", "truncate", "alter", "create", "replace", "merge"}
    )

    def _validate_read_only(self, query: str) -> None:
        """Raise ValueError if query contains write keywords."""
        normalized = query.lower()
        for keyword in self._FORBIDDEN_KEYWORDS:
            if keyword in normalized:
                raise ValueError(
                    f"SQL query tool only allows SELECT queries. Forbidden keyword detected: {keyword!r}"
                )

    async def execute(
        self,
        input_data: SQLQueryInput,
        tenant_id: str,
        config: dict[str, str],
    ) -> SQLQueryOutput:
        """Execute a read-only SQL query.

        Args:
            input_data: SQL query, row limit, and parameters.
            tenant_id: Tenant context for RLS enforcement.
            config: Must contain 'DATABASE_URL' for the connection string.

        Returns:
            SQLQueryOutput with result rows.
        """
        import asyncpg

        self._validate_read_only(input_data.query)

        database_url = config.get("DATABASE_URL", "")
        if not database_url:
            return SQLQueryOutput(result=[], metadata={"error": "DATABASE_URL not configured"})

        conn = await asyncpg.connect(database_url)
        try:
            await conn.execute(f"SET app.current_tenant = '{tenant_id}'")
            limited_query = f"SELECT * FROM ({input_data.query}) _q LIMIT {input_data.max_rows}"
            rows = await conn.fetch(limited_query, *input_data.parameters)
            result = [dict(row) for row in rows]
        finally:
            await conn.close()

        logger.info(
            "SQL query executed",
            rows_returned=len(result),
            tenant_id=tenant_id,
        )

        return SQLQueryOutput(
            result=result,
            metadata={"rows_returned": len(result), "max_rows": input_data.max_rows},
        )


TOOL = SQLQueryTool()
