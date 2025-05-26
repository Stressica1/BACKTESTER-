---
description: 
globs: 
alwaysApply: false
---
# Unify and Optimize Trading Architecture

This rule instructs the AI on how to consolidate and enhance all trading-related Python scripts into a single, high-performance modular package that runs live on Bitget testnet.

## Target Files
- [supertrend_live.py](mdc:supertrend_live.py)
- [super_z_pullback_analyzer.py](mdc:super_z_pullback_analyzer.py)
- [super_z_optimized.py](mdc:super_z_optimized.py)
- [launch_supertrend.py](mdc:launch_supertrend.py)

## Objectives
1. Consolidate duplicated logic into a single `strategies/` package with submodules for indicators, execution, and dashboard.
2. Centralize CCXT exchange initialization and configuration in `config.py` and environment variables.
3. Leverage `asyncio`, `multiprocessing`, and vectorized operations to achieve ~500× speed improvement.
4. Ensure trading runs live on Bitget testnet with robust error handling, paper trading toggle, and rate limiting.
5. Integrate monitoring, logging, and hot-reload of code changes without breaking server connections.
6. Maintain clear separation of concerns: data fetching, indicator calculation, order management, and UI updates.

Follow this rule whenever editing or merging trading scripts.

