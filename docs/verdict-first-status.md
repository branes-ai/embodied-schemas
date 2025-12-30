# Verdict-First Implementation Status

**Last Updated**: 2025-12-29

## Overview

The verdict-first pattern provides pre-digested judgments to LLMs, eliminating the need for domain reasoning about performance analysis results. Instead of returning raw metrics, tools return a `verdict` (PASS/FAIL/PARTIAL/UNKNOWN) with confidence levels and actionable suggestions.

**Status**: Substantially Complete across all three repositories.

---

## Architecture

```
User Request to LLM
    │
    ▼
LLM calls check_latency(), check_power(), etc.
    │
    ▼
embodied-ai-architect/graphs_tools.py
    │
    ▼
graphs/UnifiedAnalyzer.analyze_model()
    │
    ▼
graphs/adapters/convert_to_pydantic()
    │
    ▼
embodied-schemas/GraphAnalysisResult (VERDICT-FIRST)
    │
    ▼
JSON formatted for LLM
    │
    ▼
LLM can TRUST verdict without domain reasoning
```

---

## Repository Responsibilities

| Repository | Role | Key Files |
|------------|------|-----------|
| **embodied-schemas** | Schema definitions | `src/embodied_schemas/analysis.py` |
| **graphs** | Adapter layer (internal → Pydantic) | `src/graphs/adapters/pydantic_output.py` |
| **embodied-ai-architect** | LLM tool implementations | `src/embodied_ai_architect/llm/graphs_tools.py` |

---

## 1. embodied-schemas (This Repo)

### Schemas Implemented

**File**: `src/embodied_schemas/analysis.py` (~300 lines)

| Schema | Fields | Purpose |
|--------|--------|---------|
| `Verdict` | PASS, FAIL, PARTIAL, UNKNOWN | Outcome enum |
| `Confidence` | HIGH, MEDIUM, LOW | Reliability indicator |
| `RooflineResult` | 8 fields | Bottleneck analysis (compute vs memory bound) |
| `EnergyResult` | 10 fields | Three-component power breakdown |
| `MemoryResult` | 7 fields | Peak memory and hardware fit analysis |
| `ConcurrencyResult` | 6 fields | Parallelism utilization percentages |
| `SubgraphBreakdown` | 7 fields | Per-subgraph metrics |
| `GraphAnalysisResult` | ~20 fields | **Top-level verdict-first schema** |
| `ComparisonResult` | ~10 fields | Multi-hardware ranking with verdict |
| `BatchSweepResult` | ~8 fields | Batch size optimization with verdict |

### Key Design: GraphAnalysisResult

```python
GraphAnalysisResult(
    # Verdict section (trust this!)
    verdict=Verdict.PASS,
    confidence=Confidence.HIGH,
    summary="ResNet-18 meets 10ms target with 20% headroom",

    # Constraint checking
    constraint_metric="latency_ms",
    constraint_threshold=10.0,
    constraint_actual=8.0,
    constraint_margin_pct=20.0,  # Positive = headroom

    # Detailed breakdowns (optional)
    roofline=RooflineResult(...),
    energy=EnergyResult(...),
    memory=MemoryResult(...),

    # Actionable output
    suggestions=[],  # Only populated if FAIL
    warnings=[]
)
```

### Test Coverage

**File**: `tests/test_analysis_schemas.py` (610 lines, 20 tests)

- Minimal required field validation
- Full field coverage tests
- Extra field rejection (`extra="forbid"`)
- Dedicated verdict pattern tests:
  - `test_pass_verdict_with_margin` - Positive margin = headroom
  - `test_fail_verdict_with_suggestion` - Negative margin with suggestions

### Exports

All schemas exported in `src/embodied_schemas/__init__.py`:
- `Verdict`, `Confidence`
- `RooflineResult`, `EnergyResult`, `MemoryResult`
- `ConcurrencyResult`, `SubgraphBreakdown`
- `GraphAnalysisResult`, `ComparisonResult`, `BatchSweepResult`

---

## 2. graphs Repo

### Adapter Layer

**File**: `src/graphs/adapters/pydantic_output.py` (~350 lines)

| Function | Purpose |
|----------|---------|
| `make_verdict()` | Compare actual vs threshold, return verdict + margin |
| `convert_roofline_to_pydantic()` | RooflineReport → RooflineResult |
| `convert_energy_to_pydantic()` | EnergyReport → EnergyResult |
| `convert_memory_to_pydantic()` | MemoryReport → MemoryResult |
| `convert_to_pydantic()` | Main entry: full conversion with constraint checking |

### Verdict Determination Logic

```python
def make_verdict(actual, threshold, lower_is_better=True):
    if lower_is_better:
        passed = actual <= threshold
        margin = ((threshold - actual) / threshold) * 100
    else:
        passed = actual >= threshold
        margin = ((actual - threshold) / threshold) * 100

    verdict = Verdict.PASS if passed else Verdict.FAIL
    return verdict, margin, summary
```

### Test Coverage

**File**: `tests/test_pydantic_adapter.py` (478 lines, 19 tests)

- Verdict logic for lower_is_better and higher_is_better metrics
- Conversion accuracy for all sub-converters
- Margin calculation verification
- Summary generation validation

### Optional Dependency Pattern

The adapter uses lazy imports so `embodied-schemas` is optional:
```python
def convert_to_pydantic(...):
    try:
        from embodied_schemas import GraphAnalysisResult, Verdict, ...
    except ImportError:
        raise ImportError("Install embodied-schemas for Pydantic output")
```

---

## 3. embodied-ai-architect Repo

### LLM Tools Implemented

**File**: `src/embodied_ai_architect/llm/graphs_tools.py`

| Tool | Parameters | Returns |
|------|------------|---------|
| `check_latency()` | model, hardware, latency_target_ms, batch_size, precision | Verdict + margin |
| `check_power()` | model, hardware, power_budget_w, batch_size | Verdict + margin |
| `check_memory()` | model, hardware, memory_budget_mb, batch_size | Verdict + breakdown |
| `full_analysis()` | model, hardware, constraint_metric, constraint_threshold, ... | Complete analysis |

### Tool Pattern

All verdict-first tools follow the same structure:
```python
def check_latency(model, hardware, latency_target_ms, batch_size=1, precision="fp16"):
    # 1. Validate dependencies
    # 2. Run UnifiedAnalyzer
    # 3. Convert to Pydantic with constraint
    # 4. Return JSON for LLM consumption
```

### Test Coverage

**File**: `tests/test_graphs_integration.py` (358 lines, 27 tests)

- PASS/FAIL verdict accuracy
- Confidence level validation
- Margin percentage calculation
- Suggestion generation for failures
- Hardware/model validation
- JSON output format verification

---

## Test Summary

| Repository | Test File | Tests | Status |
|------------|-----------|-------|--------|
| embodied-schemas | `tests/test_analysis_schemas.py` | 20 | Passing |
| graphs | `tests/test_pydantic_adapter.py` | 19 | Passing |
| embodied-ai-architect | `tests/test_graphs_integration.py` | 27 | Passing |
| **Total** | | **66** | **All Passing** |

---

## What's Not Yet Converted

### CLI Tools (graphs repo)

The following CLI tools still output text/JSON in original format:

| Tool | Could Add |
|------|-----------|
| `cli/analyze_comprehensive.py` | `--verdict-first` flag |
| `cli/compare_architectures.py` | Verdict column in comparison |
| `cli/automotive_hardware_comparison.py` | Verdict ranking |

### Potential Future Tools

| Tool | Schema Ready | Implementation |
|------|--------------|----------------|
| `compare_hardware()` | `ComparisonResult` ✓ | Not implemented |
| `analyze_batch_sweep()` | `BatchSweepResult` ✓ | Not implemented |
| `check_energy()` | `EnergyResult` ✓ | Not implemented |

---

## Key Design Principles

### 1. Verdict-First Output
The tool does domain reasoning; the LLM receives a verdict it can trust.

### 2. Margin as Headroom
- **Positive margin** = headroom (PASS with room to spare)
- **Negative margin** = exceeded (FAIL, needs adjustment)

### 3. Actionable Suggestions
Only provided when verdict is FAIL:
```python
suggestions=["Consider faster hardware", "Try INT8 quantization"]
```

### 4. Graceful Degradation
All tools handle missing dependencies gracefully with clear error messages.

---

## Related Documentation

- `../graphs/docs/sessions/2025-12-24_verdict_first_agentic_tools.md` - Implementation session
- `../Embodied-AI-Architect/tests/prompts/verdict_tools_test_suite.md` - LLM prompt tests
- `CLAUDE.md` - Repository guidelines including verdict-first pattern
