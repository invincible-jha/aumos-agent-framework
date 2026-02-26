# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| 0.1.x   | Yes       |

## Reporting a Vulnerability

**Do NOT open a public GitHub issue for security vulnerabilities.**

Report security issues to: security@aumos.ai

Include:
- Description of the vulnerability
- Steps to reproduce
- Potential impact
- Suggested fix (if any)

We will acknowledge receipt within 48 hours and provide a fix timeline within 7 days.

## Security Considerations

This service handles:
- Multi-tenant agent execution — tenant isolation via RLS is critical
- 5-level privilege system — privilege escalation is a high-severity concern
- Human-in-the-loop gates — unauthorized approvals are critical
- Agent tool access — tool misuse by unprivileged agents is high-severity
- Circuit breaker state — manipulation could expose system to cascading failures

All security reports for these areas will be treated as critical priority.
