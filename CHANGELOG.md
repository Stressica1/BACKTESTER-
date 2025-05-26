# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
### Added
- Initial changelog created to enforce version control and changelogging rules.
- Added `.cursor/rules/version-control-changelog.mdc` to require descriptive commits and changelog entries for all code changes.

## [1.0.1] - 2024-05-26
### Fixed
- Fixed IndentationError in `supertrend_live.py` in the `detect_signal` method by removing a stray `else:` block, allowing the script to run on testnet and mainnet.

## [1.2.1] - 2025-05-26
### Fixed
- Ensured `test_batch.json` config file is present in both root and BACKTESTER directories so `supertrend_live.py` can find it regardless of run location.
- Repeatedly tested `supertrend_live.py` three times to confirm stability and no config errors.

i got you 