# Quickstart: Migrate from Poetry to uv

**Feature**: 002-migrate-poetry-to-uv

## Post-Migration Verification

After migration, run these commands to verify the setup:

### 1. Install dependencies

```bash
uv sync
```

Expected: Virtual environment created in `.venv/`, all dependencies installed.

### 2. Run tests

```bash
uv run pytest
```

Expected: All tests pass.

### 3. Run CLI entry points

```bash
uv run hercule --help
uv run gen-doc
uv run serve-doc
```

Expected: Each command executes without error.

### 4. Build package

```bash
uv build
```

Expected: `dist/` contains wheel and/or sdist.

### 5. Add/remove dependency (smoke test)

```bash
uv add --dev black
uv remove black
```

Expected: pyproject.toml and uv.lock updated correctly.
