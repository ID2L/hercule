# Implementation Plan: Migrate from Poetry to uv

**Branch**: `002-migrate-poetry-to-uv` | **Date**: 2026-03-14 | **Spec**: [spec.md](./spec.md)  
**Input**: Feature specification from `/specs/002-migrate-poetry-to-uv/spec.md`

## Summary

Replace Poetry with uv as the sole dependency management and packaging tool. Use `uvx migrate-to-uv` for automated conversion, then update README, parse_doc.py, and remove Poetry artifacts. All CLI entry points and tests must continue to work.

## Technical Context

**Language/Version**: Python 3.10+  
**Primary Dependencies**: gymnasium, click, pyyaml, numpy, pydantic, jinja2, matplotlib, pandas, torch  
**Storage**: N/A  
**Testing**: pytest  
**Target Platform**: Cross-platform (Windows, Linux, macOS)  
**Project Type**: CLI / library (RL framework)  
**Performance Goals**: N/A  
**Constraints**: Existing tests must pass; no changes to test code  
**Scale/Scope**: Single Python package with src layout

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

No constitution file present. Proceeding with standard practices.

## Project Structure

### Documentation (this feature)

```text
specs/002-migrate-poetry-to-uv/
├── plan.md              # This file
├── research.md          # Phase 0 output
├── quickstart.md        # Phase 1 output
├── checklists/
│   └── requirements.md
└── tasks.md             # Phase 2 output (/speckit.tasks)
```

### Source Code (repository root)

```text
hercule/
├── pyproject.toml       # Migrated to uv format
├── uv.lock              # Replaces poetry.lock
├── src/
│   └── hercule/         # Main package
│       ├── cli/
│       ├── config/
│       ├── environnements/
│       ├── models/
│       ├── reports/
│       ├── run/
│       └── ...
├── pyscaf/
│   └── documentation/
│       └── scripts/
│           └── parse_doc.py   # Update package path resolution
├── tests/
├── experiments/
├── README.md            # Update Poetry → uv instructions
└── .venv/               # In-project (unchanged)
```

**Structure Decision**: Single Python package with src layout. No structural changes; only configuration and toolchain migration.

## Complexity Tracking

N/A — No constitution violations.
