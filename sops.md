# Operating Procedures

## Tech Stack
- Language: Python 3.12+
- Data models: pydantic >=2.0
- HTTP client: httpx (async)
- Testing: pytest + pytest-asyncio
- Config: YAML (pyyaml)
- No ML framework dependencies in core — fine-tuning backends are plugin interfaces

## Project Structure
- Source: `src/apprentice/`
- Tests: `tests/`
- Entry point: CLI via `apprentice = "apprentice.cli:main"` or library import via `from apprentice import Apprentice`

## Standards
- Type annotations on all public functions
- Prefer composition over inheritance
- All external boundaries (APIs, local model servers, file I/O) behind abstract interfaces
- Config parsing is strict — fail fast on invalid config, not at runtime
- Keep files under 300 lines
- snake_case functions, PascalCase classes

## Verification
- All functions must have at least one test
- Tests must run without external services — mock all API calls and model servers
- No test should depend on GPU, API keys, or network access
- Use dependency injection for testability (pass clients, not construct them internally)
- Run: `python3 -m pytest tests/ -v`

## Design Principles
- The unified interface is the product — everything else serves it
- Config is the only user-facing surface — if it's not in config, the user can't control it
- Phases are emergent from confidence scores, not manually triggered states
- Sampling frequency is continuous, not discrete tiers
- Budget enforcement is hard — the system stops spending, never exceeds limits
- Local model failures are graceful — fall back to remote, don't crash
- Every decision the system makes should be logged and auditable

## Error Handling
- Network errors to remote APIs: retry with exponential backoff, then degrade
- Local model unavailable: fall back to remote (counts against budget)
- Budget exhausted: serve from local only (accept quality risk), log warning
- Fine-tuning failure: keep current model version, retry next batch
- Config errors: fail at startup with clear message, never silently default

## Preferences
- Prefer stdlib over third-party libraries
- httpx over requests (async support)
- Pydantic models for all data flowing between components
- YAML for config, JSON for internal state persistence
- Structured logging (JSON) for machine-readable audit trail
- No global state — all state flows through explicit parameters or injected dependencies
