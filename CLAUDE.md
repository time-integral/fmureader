# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
uv sync && uvx pre-commit install  # Install dependencies (make install)
make test                          # Run all tests with coverage
uv run pytest tests/test_fmi2.py  # Run a single test file
uv run pytest -k "test_name"      # Run a single test by name
make check                        # Lint (pre-commit) + type check (pyright)
make format                       # Auto-format with pre-commit hooks
```

## Architecture

`fmureader` is a read-only parser for FMI model description XML files into Pydantic models. Source is under `src/fmureader/`.

- **`fmi2.py`** — FMI 2.0 Pydantic models + `read_model_description()`. Handles `.xml`, `.fmu` (ZIP), and unzipped FMU directories.
- **`fmi3.py`** — Same for FMI 3.0. Has richer variable types (Float32/64, Int8–Int64, UInt8–UInt64, Clock, Binary).
- **`fmi.py`** — Version-agnostic entry point: `get_fmi_version()` and `read_model_description()` that auto-detects FMI 2.0 vs 3.0 and delegates to the appropriate module.
- **`__init__.py`** — Package init.

Each `fmi2`/`fmi3` module is self-contained: it defines all Pydantic models for that version's schema, plus a standalone `read_model_description()` function. Tests mirror this split: `test_fmi2.py`, `test_fmi3.py`, and `test_fmi.py` (version-agnostic).

Tests use `--doctest-modules`, so doctests in source files are also executed.
