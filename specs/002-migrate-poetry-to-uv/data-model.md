# Data Model: Migrate from Poetry to uv

**Feature**: 002-migrate-poetry-to-uv

## Scope

This feature is a toolchain migration. No domain entities or data models are introduced or modified. The following configuration artifacts are affected:

| Artifact | Change |
|----------|--------|
| `pyproject.toml` | Poetry sections replaced by PEP 621 + uv_build + dependency-groups |
| `uv.lock` | New lock file (replaces poetry.lock) |
| `poetry.lock` | Removed |
| `poetry.toml` | Removed |
| `[tool.pyscaf.documentation]` | Add `package_paths` for parse_doc.py |

No database, API, or domain model changes.
