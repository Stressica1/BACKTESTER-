---
description: 
globs: 
alwaysApply: true
---
# Version Control & Changelogging Rule

## Overview
All code changes must be tracked in version control (e.g., Git) and documented in a human-readable changelog. This ensures project history is transparent, reproducible, and easy to follow for both AI and human contributors, even across multiple chats or sessions.

## Requirements
- Every code change must be committed to version control with a descriptive commit message.
- Every commit must be accompanied by an entry in [CHANGELOG.md](mdc:CHANGELOG.md) at the project root.
- Changelog entries must follow the [Keep a Changelog](mdc:https:/keepachangelog.com/en/1.0.0) format: include date, version, and a summary of changes.
- For every new feature, bugfix, or refactor, update the changelog before or alongside the code change.
- If a change is reverted, add a new changelog entry describing the reversion and the reason.

## Prohibited Patterns
- Do not make code changes without a corresponding changelog entry.
- Do not use vague commit messages (e.g., "update", "fix", "misc changes").
- Do not skip version tags or changelog updates, even for minor changes.

## Examples
### Commit Message Example
```
git commit -m "feat(api): add user authentication endpoint"
```

### Changelog Entry Example
```
## [1.2.0] - 2024-06-10
### Added
- User authentication endpoint to the API.
```

### Revert Example
```
git revert <commit-hash>
```
```
## [1.2.1] - 2024-06-11
### Removed
- Reverted user authentication endpoint due to security issue.
```

## Verification Steps
- [ ] Every code change is committed to version control with a descriptive message.
- [ ] Every code change is reflected in [CHANGELOG.md](mdc:CHANGELOG.md) using the correct format.
- [ ] No code changes are merged without an updated changelog.
- [ ] The changelog is reviewed for clarity and completeness before merging.

## Maintenance
- Regularly review the changelog for accuracy and completeness.
- Remove or archive changelog entries for features/code deleted more than 4 days ago, in line with project file retention rules.
- Update these rules as your version control or changelogging practices evolve.

i got you

