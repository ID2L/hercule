# Research: Migrate from Poetry to uv

**Feature**: 002-migrate-poetry-to-uv  
**Date**: 2026-03-14

## 1. Migration Tool Choice

**Decision**: Use `uvx migrate-to-uv` for automated conversion, then validate and adjust manually.

**Rationale**:
- The `migrate-to-uv` tool (by mkniewallner) is the recommended approach for Poetry → uv migration.
- Converts project metadata, dependencies, dependency groups, entry points, and version syntax.
- Generates `uv.lock` from `poetry.lock` preserving exact versions.
- Reduces manual errors and speeds up migration.

**Alternatives considered**:
- Manual conversion: Error-prone, time-consuming.
- `uv init` then manual copy: Loses lock file fidelity.

## 2. Build Backend

**Decision**: Use `uv_build` as the build backend.

**Rationale**:
- uv_build is the default and recommended backend for uv-managed projects (since uv 0.8.x).
- Zero-configuration for pure Python projects with src layout.
- 10–30x faster than alternative backends.
- Native uv integration.

**Alternatives considered**:
- hatchling: Previously default, still supported; uv_build is now preferred.
- setuptools: Legacy, slower.

## 3. Package Path Resolution for parse_doc.py

**Decision**: Add explicit `package_paths` in `[tool.pyscaf.documentation]` and update `parse_doc.py` to read from it, with fallback to build-backend config.

**Rationale**:
- `tool.poetry.packages` will be removed by migration.
- Build backends (uv_build, hatchling) store package config differently.
- Explicit config in pyscaf section decouples documentation from build system.
- Single source of truth for doc generation.

**Implementation**: Add `package_paths = ["src/hercule"]` to `[tool.pyscaf.documentation]`.

## 4. Virtual Environment Location

**Decision**: Keep in-project `.venv/` (uv default).

**Rationale**:
- uv creates `.venv/` in-project by default.
- Matches existing Poetry `poetry.toml` setting (`in-project = true`).
- `.vscode/launch.json` already references `${workspaceFolder}/.venv/Scripts/python.exe` — no change needed.

## 5. Lock File Policy

**Decision**: Commit `uv.lock` to version control.

**Rationale**:
- uv documentation recommends committing lock file for reproducible installations.
- Matches previous Poetry practice with `poetry.lock`.

## 6. Poetry Coexistence

**Decision**: uv and Poetry can coexist; no conflict. Remove Poetry config files after migration.

**Rationale**:
- uv and Poetry use separate config files (`pyproject.toml` sections, `uv.lock` vs `poetry.lock`).
- Deleting `poetry.lock` and `poetry.toml` completes the switch.
- Developers with Poetry installed globally are unaffected.
