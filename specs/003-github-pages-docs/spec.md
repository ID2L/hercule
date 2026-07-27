# Feature Specification: Publish API Documentation on GitHub Pages

**Feature Branch**: `003-github-pages-docs`  
**Created**: 2026-07-28  
**Status**: Draft  
**Input**: User description: "Publish the pdoc-generated API documentation of the hercule repository on GitHub Pages, via an automated CI workflow (GitHub Actions) so that docs/ no longer needs to be committed to the repository."

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Reader consults the API documentation online (Priority: P1)

Anyone interested in Hercule — a contributor, a researcher evaluating the framework, or a future maintainer —
opens a public URL in a browser and reads the API documentation of the `hercule` package (modules, classes,
methods, docstrings) without cloning the repository, installing Python, or running any command.

**Why this priority**: This is the point of the feature. Without a reachable published site none of the other
stories deliver value. It is also the smallest useful slice: a single publication already satisfies it.

**Independent Test**: Open the published URL in a private browsing window (not signed in, no repository access)
and confirm the index lists the package modules and that a class page renders its docstrings. Delivers: a
publicly readable API reference.

**Acceptance Scenarios**:

1. **Given** the documentation has been published at least once, **When** a reader opens the published URL,
   **Then** the index page loads and lists every public module of the `hercule` package.
2. **Given** the reader is on the index page, **When** they follow the link to a module such as the models
   package, **Then** the page displays that module's classes, methods, and their docstrings.
3. **Given** a reader is not authenticated on the hosting platform, **When** they open the published URL,
   **Then** the content is served without requiring any sign-in.

---

### User Story 2 - Documentation stays current without manual action (Priority: P1)

A maintainer merges a change into the default branch — a new algorithm, a renamed method, an improved docstring.
The published documentation reflects that change shortly afterwards, with nobody running a command, generating
files locally, or committing anything.

**Why this priority**: Documentation that needs a manual step drifts out of date, which is the failure mode this
feature exists to remove. Together with Story 1 it forms the MVP.

**Independent Test**: Change a docstring on the default branch, wait for the automated publication to finish,
reload the corresponding page and confirm the new text appears. Delivers: self-maintaining documentation.

**Acceptance Scenarios**:

1. **Given** a commit that modifies a docstring is merged into the default branch, **When** the automated
   publication completes, **Then** the published page for that symbol shows the updated text.
2. **Given** a commit adds a new algorithm sub-package, **When** the automated publication completes, **Then**
   the new module appears in the published documentation index.
3. **Given** a maintainer wants to republish without any code change, **When** they trigger the publication
   manually, **Then** the documentation is regenerated and republished.
4. **Given** two commits are merged in quick succession, **When** both publications are triggered, **Then** the
   final published content corresponds to the most recent commit, never to the older one.

---

### User Story 3 - Generated documentation disappears from the repository (Priority: P2)

A contributor generates the documentation locally to preview it. The generated output does not show up as
untracked changes, cannot be committed by accident, and never has to be reviewed in a pull request.

**Why this priority**: This is the "so that" half of the request and the reason the working tree currently
carries 19 untracked generated HTML files. It depends on Stories 1–2: the output can only stop being tracked
once it is published elsewhere.

**Independent Test**: Generate the documentation locally, then inspect the repository status and confirm no
generated file is reported. Delivers: a clean working tree and review-noise-free pull requests.

**Acceptance Scenarios**:

1. **Given** a clean checkout, **When** the contributor generates the documentation locally, **Then** the
   repository reports no new untracked or modified files.
2. **Given** the contributor has generated the documentation locally, **When** they stage all changes, **Then**
   no generated documentation file is staged.
3. **Given** the local preview command is used, **When** it runs, **Then** it still behaves exactly as before
   this feature (local preview capability is not lost).

---

### User Story 4 - Broken documentation is caught before merge (Priority: P2)

A contributor opens a pull request containing a change that breaks documentation generation (an import error, a
malformed docstring, a renamed package path). The pull request reports the failure, and the published
documentation is left untouched.

**Why this priority**: Protects Story 2's promise. Without it a broken change reaches the default branch and
either publishes a degraded site or silently stops updating it. Valuable, but not required for the MVP.

**Independent Test**: Open a pull request that deliberately breaks documentation generation and confirm the pull
request reports a failed check while the published site still serves the previous content. Delivers: protection
against publishing broken documentation.

**Acceptance Scenarios**:

1. **Given** a pull request whose changes break documentation generation, **When** the automated check runs,
   **Then** the check fails and the failure is visible on the pull request.
2. **Given** that same pull request, **When** the check fails, **Then** nothing is published and the live
   documentation still serves the previously published content.
3. **Given** a pull request whose changes do not break documentation generation, **When** the automated check
   runs, **Then** the check passes and still nothing is published (publication happens only from the default
   branch).

---

### User Story 5 - Reader finds the documentation from the repository (Priority: P3)

Someone landing on the repository home page discovers the link to the published documentation without searching
for it.

**Why this priority**: Pure discoverability. The documentation is usable without it, but an unlinked site is
rarely found.

**Independent Test**: Open the repository home page and confirm a visible link leads to the published
documentation. Delivers: discoverability.

**Acceptance Scenarios**:

1. **Given** a reader opens the repository home page, **When** they read the introduction, **Then** a clearly
   labelled link to the published documentation is present and resolves to the live site.

---

### Edge Cases

- **Generation fails on the default branch**: publication must not overwrite the live site with partial or empty
  content; the previously published version keeps being served and the failure is reported.
- **Two publications run concurrently**: they must not interleave; the outcome must be deterministic and
  correspond to the newest commit.
- **Pull request from a fork**: the validation check must run without granting the fork any ability to publish.
- **Hosting not yet activated**: the very first publication depends on a one-time repository setting; until it is
  done the automated run must fail with an actionable message rather than appear to succeed.
- **Package layout changes**: if the documented package paths change in the project configuration, the published
  output must follow that configuration rather than a value duplicated inside the automation.
- **Reader opens a deep link to a page that no longer exists** (a module removed since their last visit): the
  hosting platform's standard not-found response is acceptable; no custom handling is required.
- **Repository visibility becomes private**: publication may stop being publicly reachable; this is a hosting
  plan constraint and is out of scope.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The system MUST publish the API documentation of the `hercule` package at a stable, publicly
  reachable URL that does not change between publications.
- **FR-002**: The system MUST regenerate and republish the documentation automatically for every change merged
  into the default branch, with no manual step.
- **FR-003**: The published content MUST be generated from the repository source at the published commit, using
  the project's existing documentation generation command and its configured package paths, so that published
  output and locally generated output cannot diverge.
- **FR-004**: A maintainer MUST be able to trigger a republication manually, without pushing a commit.
- **FR-005**: The repository MUST NOT track generated documentation output; generating it locally MUST leave the
  working tree clean.
- **FR-006**: Local documentation generation and local preview MUST continue to work unchanged.
- **FR-007**: The system MUST validate that documentation generation succeeds for every pull request targeting
  the default branch, without publishing anything from that pull request.
- **FR-008**: When documentation generation fails, the system MUST report the failure visibly and MUST leave the
  previously published content intact.
- **FR-009**: Concurrent publications MUST be serialised or superseded so that the final published state always
  corresponds to the most recent qualifying commit.
- **FR-010**: The automated publication MUST install dependencies from the project's committed lock file so that
  a documentation build is reproducible.
- **FR-011**: The automation MUST hold only the permissions required to read the repository and publish the
  documentation, and MUST NOT be able to publish from pull requests originating in forks.
- **FR-012**: The repository home page MUST link to the published documentation.
- **FR-013**: The one-time manual setup required to activate hosting MUST be documented in the repository so any
  maintainer can reproduce it.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Within 10 minutes of a change being merged into the default branch, the published documentation
  reflects that change, measured from merge time to content availability.
- **SC-002**: Publishing requires **0** manual actions per update after the one-time activation.
- **SC-003**: After generating the documentation locally, the repository reports **0** untracked or modified
  generated files.
- **SC-004**: **0** generated documentation files are added to the repository history from this feature onward.
- **SC-005**: **100%** of the public modules of the `hercule` package are reachable from the published index.
- **SC-006**: **100%** of pull requests that break documentation generation are flagged before merge, and **0%**
  of them alter the live published content.
- **SC-007**: A reader starting from the repository home page reaches the published documentation in **1** click.
- **SC-008**: A maintainer can determine, from the repository alone, how to activate hosting on a fresh clone or
  fork in under **5** minutes.

## Assumptions

- **Hosting**: the repository is public and hosted on GitHub, so GitHub Pages is available at no cost; the
  published URL is the default `github.io` address for the repository. No custom domain is required.
- **One-time activation**: enabling Pages and selecting the automated-workflow source is a repository setting
  performed once by someone with admin rights. The current maintainer has admin rights on the repository, so no
  permission escalation is needed.
- **Scope of publication**: only the latest state of the default branch is published. Versioned documentation
  (one site per release or tag) is out of scope.
- **Documented packages**: the site documents the package paths already declared in the project configuration
  (`src/hercule`). The vendored scaffolding package is not published.
- **Generation tooling**: the existing project documentation command is the single source of generation; this
  feature changes neither the generator, its options, nor its output format.
- **Trigger scope**: publication triggers on the default branch only; other branches get validation only.
- **Existing untracked output**: the 19 generated HTML files currently in the working tree are build artefacts to
  be discarded locally, not content to migrate.
- **Independence from tests**: publication is independent of the test suite; a failing test does not block
  publication, and this feature adds no documentation checks to the existing test suite.

## Dependencies

- Continuous-integration capability must be enabled on the repository. No workflow exists today; this feature
  introduces the first one.
- The committed dependency lock file must stay valid, since the automated build installs from it.
- The documentation generation command must stay functional. It was repaired in feature 002 after the package
  manager migration and now reads its package paths from the project configuration.

## Out of Scope

- Versioned or multi-version documentation sites.
- Custom domain, analytics, or a search backend beyond what the generator already produces.
- Publishing narrative or tutorial documentation beyond the generated API reference.
- Changing the documentation generator or its theme.
- Publishing documentation for the vendored scaffolding package.
