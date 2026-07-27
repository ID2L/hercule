# Feature Specification: Migrate from Poetry to uv

**Feature Branch**: `002-migrate-poetry-to-uv`  
**Created**: 2026-03-14  
**Status**: Draft  
**Input**: User description: "Replace Poetry with uv for dependency management. Everything must be operational with uv, and README / agent files must be updated."

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Developer installs project dependencies with uv (Priority: P1)

A developer clones the repository and uses uv to install all project dependencies (main and dev) in a single command. The virtual environment is created automatically and all packages are available.

**Why this priority**: Without working dependency installation, no other development activity is possible. This is the foundational capability.

**Independent Test**: Can be fully tested by running the install command in a clean clone and verifying all imports resolve. Delivers: a working development environment.

**Acceptance Scenarios**:

1. **Given** a fresh clone of the repository, **When** the developer runs the uv install command, **Then** all main and dev dependencies are installed and the virtual environment is created in-project.
2. **Given** the dependencies are installed, **When** the developer imports any project module (e.g., `import hercule`), **Then** the import succeeds without error.
3. **Given** the dependencies are installed, **When** the developer runs the test suite, **Then** all tests pass as they did under Poetry.

---

### User Story 2 - Developer runs CLI entry points with uv (Priority: P1)

A developer uses uv to run the project's CLI tools (`hercule`, `gen-doc`, `serve-doc`) without needing to manually activate a virtual environment.

**Why this priority**: The CLI entry points are the primary interface for the project. They must work seamlessly with uv.

**Independent Test**: Can be tested by running each CLI entry point via uv and confirming expected output. Delivers: functional CLI tools under the new toolchain.

**Acceptance Scenarios**:

1. **Given** dependencies are installed with uv, **When** the developer runs the `hercule` CLI, **Then** the CLI starts and displays help or executes as expected.
2. **Given** dependencies are installed with uv, **When** the developer runs `gen-doc` and `serve-doc` commands, **Then** documentation generation and serving work correctly.

---

### User Story 3 - Developer reads updated documentation (Priority: P2)

A developer opens the README and finds accurate, up-to-date instructions for setting up and working with the project using uv. No references to Poetry remain in user-facing documentation.

**Why this priority**: Outdated documentation causes confusion and onboarding friction. Important but not blocking actual development.

**Independent Test**: Can be tested by reviewing all documentation files for Poetry references and verifying uv instructions are present and correct. Delivers: clear onboarding path for new contributors.

**Acceptance Scenarios**:

1. **Given** the migration is complete, **When** a developer reads the README, **Then** all setup instructions reference uv instead of Poetry.
2. **Given** the migration is complete, **When** a developer searches all project files for "poetry" references, **Then** no user-facing documentation mentions Poetry (internal build-system metadata excluded if technically necessary).

---

### User Story 4 - Developer adds or removes a dependency (Priority: P2)

A developer needs to add a new package or remove an existing one using uv's dependency management commands.

**Why this priority**: Dependency management is a recurring developer task. Must be straightforward with uv.

**Independent Test**: Can be tested by adding and removing a test package and verifying the lock file updates correctly. Delivers: ongoing dependency management workflow.

**Acceptance Scenarios**:

1. **Given** the project is set up with uv, **When** the developer adds a new dependency, **Then** the dependency is added to `pyproject.toml` and the lock file is updated.
2. **Given** the project has dependencies, **When** the developer removes a dependency, **Then** it is removed from `pyproject.toml` and the lock file is updated.

---

### User Story 5 - Build system produces an installable package (Priority: P3)

The project can be built into a distributable Python package (wheel/sdist) using uv's build capabilities.

**Why this priority**: Package distribution is important but less frequently needed than day-to-day development tasks.

**Independent Test**: Can be tested by building the package and verifying the resulting artifact can be installed in a fresh environment. Delivers: distributable package.

**Acceptance Scenarios**:

1. **Given** the project is configured for uv, **When** a developer builds the project, **Then** a valid wheel and/or sdist is produced.
2. **Given** the built package, **When** installed in a clean environment, **Then** the `hercule` CLI entry point is available and functional.

---

### Edge Cases

- What happens when a developer has Poetry installed globally — does uv coexist without conflict?
- How does the migration handle the existing `poetry.lock` — is it replaced by `uv.lock`?
- What happens if the developer has an existing `.venv` created by Poetry — does uv reuse or recreate it?
- How does the `parse_doc.py` script behave after migration, given it reads `tool.poetry.packages` from `pyproject.toml`?

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The project MUST use uv as the sole dependency management and packaging tool, replacing Poetry entirely.
- **FR-002**: The `pyproject.toml` MUST be updated to use a uv-compatible build system (e.g., `hatchling`) instead of `poetry-core`.
- **FR-003**: All project dependencies (main and dev) MUST be declared in PEP 621-compliant `[project]` and `[project.optional-dependencies]` or `[dependency-groups]` sections.
- **FR-004**: CLI entry points (`hercule`, `gen-doc`, `serve-doc`) MUST be declared using PEP 621 `[project.scripts]` syntax.
- **FR-005**: A `uv.lock` file MUST replace `poetry.lock` for reproducible installations.
- **FR-006**: The `poetry.toml` configuration file MUST be removed.
- **FR-007**: The `poetry.lock` file MUST be removed from the repository.
- **FR-008**: The README MUST be updated to replace all Poetry instructions and references with uv equivalents.
- **FR-009**: The `pyscaf/documentation/scripts/parse_doc.py` script MUST be updated to read package paths from PEP 621-compatible configuration instead of `tool.poetry.packages`.
- **FR-010**: The `.gitignore` MUST include uv-specific entries if any are needed (e.g., `uv.lock` policy: tracked or ignored).
- **FR-011**: The virtual environment MUST continue to be created in-project (`.venv/` directory).
- **FR-012**: The existing test suite MUST continue to pass without modification to test code.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: A developer can set up the complete project environment from a fresh clone in under 2 minutes using uv.
- **SC-002**: All existing tests pass with zero failures after the migration.
- **SC-003**: All three CLI entry points (`hercule`, `gen-doc`, `serve-doc`) execute successfully under uv.
- **SC-004**: Zero references to Poetry remain in user-facing documentation (README, inline help).
- **SC-005**: The project can be built into a distributable package using uv.
- **SC-006**: Dependency addition and removal workflows function correctly with uv.

## Assumptions

- uv is already installed or can be installed by the developer (standard toolchain prerequisite).
- The project will use `hatchling` as the build backend, which is the recommended backend for uv-managed projects and supports `src/` layout natively.
- The `uv.lock` file will be committed to the repository for reproducible builds.
- The existing `.venv/` directory convention is preserved (in-project virtual environments).
- Dev dependencies will use uv's `[dependency-groups]` feature (PEP 735) for grouping development dependencies.
- The `.vscode/launch.json` path to `.venv/Scripts/python.exe` remains valid since uv also creates `.venv/` in-project.
