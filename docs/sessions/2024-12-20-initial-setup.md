# Session Log: Initial Repository Setup

**Date**: December 20, 2024
**Focus**: Creating the shared schema repository for embodied AI codesign

---

## Context

This session established `branes-ai/embodied-schemas` as a shared package between:
- `branes-ai/graphs` - Analysis tools, roofline models, calibration
- `branes-ai/Embodied-AI-Architect` - LLM orchestration, knowledge base, CLI

The goal is to provide a single source of truth for hardware, model, sensor, and use case specifications that both repositories can import.

---

## Discussion Topics

### 1. Tool Registration and Orchestration

Analyzed current tool registration patterns in Embodied-AI-Architect:
- Orchestrator uses manual dictionary-based registration with hard-coded execution order
- LLM tools have dual registration (definitions + executors) that must stay in sync
- No validation of tool completeness before execution

### 2. Subtask Decomposition Design

Before designing tools, we need to understand how tasks decompose:
- Who decomposes: LLM-driven vs template workflows vs goal-oriented
- Granularity: What constitutes an atomic subtask
- Replanning: Can decomposition change mid-execution

### 3. Tool Granularity Decision

Evaluated four approaches:
- Generic tool + arguments → Hard parameterization
- Task-specific by name → Easy parameterization, parallel execution
- Context-adaptive → Unpredictable, hard to debug
- Do-everything → Slow, output overload

**Decision**: Dimension-specific tools (analyze_latency, analyze_power, analyze_memory, etc.)
- LLMs excel at semantic matching (goal → tool name)
- Parallel execution when dimensions are independent
- Clear verdict per dimension

### 4. Tool Output Schema Design

Designed verdict-first output format:
```json
{
  "verdict": "PASS | FAIL | PARTIAL | UNKNOWN",
  "confidence": "high | medium | low",
  "summary": "One sentence description",
  "metric": {"name": "...", "measured": N, "required": N, "unit": "..."},
  "evidence": "How determined",
  "suggestion": "Actionable next step if not PASS"
}
```

**Principle**: Tools do domain reasoning, LLM receives verdict it can trust.

### 5. Domain Knowledge Architecture

Evaluated storage options:
- File tree (YAML/JSON) - Simple, version-controlled
- RDF/Knowledge Graph - Rich relationships, complex
- Relational DB - Mature, good query
- System prompt - Token expensive, doesn't scale
- Vector embeddings - Fuzzy matching, imprecise for facts

**Decision**: Hybrid architecture
- YAML files as source of truth (version-controlled)
- Optional graph loading for relationships
- Vector index only for fuzzy semantic matching

Research on current agentic systems (2025):
- A-MEM: Zettelkasten-style linked notes
- Zep/Graphiti: Temporal knowledge graphs showing 18.5% accuracy improvement
- Mem0: Hybrid vector + graph + key-value

### 6. Shared Schema Repository Decision

**Decision**: Create `branes-ai/embodied-schemas` as separate package

Rationale:
- Schemas are the contract between repos
- Clean dependency management
- Natural home for factual hardware/chip data
- Independent versioning

### 7. Data Split with graphs/hardware_registry

Analyzed existing `graphs/hardware_registry/` structure:
- 43+ hardware profiles across cpu/, gpu/, accelerator/, dsp/, boards/
- Each device has spec.json + calibrations/*.json

**Split Decision**:

| embodied-schemas | graphs |
|------------------|--------|
| Datasheet specs | ops_per_clock |
| Power profiles | theoretical_peaks |
| Physical specs | Calibration data |
| Environmental specs | Operation profiles |
| Interface specs | Measured efficiency |

---

## Implementation

### Created Repository Structure

```
embodied-schemas/
├── src/embodied_schemas/
│   ├── __init__.py      # Package exports
│   ├── hardware.py      # 20+ models/enums
│   ├── models.py        # ML model schemas
│   ├── sensors.py       # Camera, depth, LiDAR
│   ├── usecases.py      # Constraint templates
│   ├── benchmarks.py    # Result schemas
│   ├── constraints.py   # Tier definitions
│   ├── loaders.py       # YAML loading
│   ├── registry.py      # Unified access
│   └── data/            # YAML data directories
├── tests/
│   └── test_schemas.py  # 15 passing tests
├── pyproject.toml
├── README.md
└── LICENSE
```

### Key Schema Highlights

**HardwareEntry** - Extended for embodied AI:
- PhysicalSpec: weight, dimensions, form factor
- EnvironmentalSpec: temp range, IP rating, vibration/shock
- InterfaceSpec: CSI, USB, PCIe, CAN bus counts
- PowerSpec: Multiple power modes with frequencies

**ModelEntry** - Perception-focused:
- Architecture: backbone, neck, head, params, FLOPs
- Variants: fp32, fp16, int8 with accuracy delta
- Accuracy benchmarks per dataset

**UseCaseEntry** - Constraint templates:
- Hard/soft constraints with criticality levels
- Success criteria for validation
- Recommended hardware/model/sensor configs

**BenchmarkResult** - Verdict-first output:
- Latency, power, memory, thermal metrics
- Verdict + confidence + suggestion format

### Test Results

```
15 passed in 0.12s
```

All schema models validated successfully.

---

## Planning Documents Created

1. **docs/plans/agentic-tool-architecture.md** (Embodied-AI-Architect)
   - Tool registration patterns
   - Granularity decision
   - Output schema design
   - Knowledge architecture

2. **docs/plans/embodied-ai-codesign-subtasks.md** (Embodied-AI-Architect)
   - 70+ enumerated subtasks
   - Knowledge base, analysis, recommendation, synthesis tools
   - Phased implementation plan

3. **docs/plans/knowledge-base-schema.md** (Embodied-AI-Architect)
   - Complete schema design
   - Directory structure
   - Query interface design

4. **docs/plans/shared-schema-repo-architecture.md** (Embodied-AI-Architect)
   - Repository decision and rationale
   - Dependency flow
   - Data split with graphs
   - Migration strategy

---

## Next Steps

1. **Seed initial data** - Add YAML files for key hardware platforms and models
2. **Integrate with graphs** - Add embodied-schemas as dependency, update hardware registry
3. **Integrate with Embodied-AI-Architect** - Migrate knowledge_base.py to use shared schemas
4. **Implement analysis tools** - Build dimension-specific tools using schemas

---

## Key Decisions Summary

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Tool granularity | Dimension-specific by name | LLMs match semantically, parallel execution |
| Output format | Verdict-first structured | LLM reads verdict, doesn't interpret |
| Knowledge storage | YAML + optional graph | Version-controlled, queryable |
| Schema location | Separate shared repo | Clean dependencies, single source |
| Data split | Datasheet → schemas, calibration → graphs | Universal vs tool-specific |

---

*Session duration: ~2 hours*
*Lines of code: ~1,500 (schemas + tests)*
