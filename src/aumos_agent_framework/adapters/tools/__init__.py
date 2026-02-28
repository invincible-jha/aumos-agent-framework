"""Pre-built tool implementations for the AumOS agent framework.

Tools are organized into categories:
- web: Web search, scraping, RSS feeds
- data: SQL queries, CSV loading, generic JSON APIs
- communication: Email, Slack, webhooks
- document: PDF extraction, Word reader, HTML-to-Markdown
- code: Python REPL, shell commands
- ai: LLM summarization, image description, embedding search
- aumos: AumOS platform integrations (workflow trigger, data pipeline)

Each tool implements AumOSToolProtocol from core/interfaces.py.
"""

from aumos_agent_framework.adapters.tools.registry import BuiltinToolRegistry

__all__ = ["BuiltinToolRegistry"]
