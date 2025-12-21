# CLAUDE.md

This file provides guidance to Claude Code when working with code in this repository.

## Repository Role: Shared Schema Source

**This is a shared dependency package** imported by two downstream repositories:

```
embodied-schemas (this repo)
       ↑              ↑
       │              │
   graphs      Embodied-AI-Architect
```

- **graphs** (`../graphs`) - Uses schemas for roofline analysis, hardware simulation, benchmark validation
- **Embodied-AI-Architect** (`../Embodied-AI-Architect`) - Uses schemas for LLM tool outputs, knowledge base, CLI

**Any schema change here affects both downstream repos.** Consider compatibility carefully.

## What Belongs Here vs Downstream Repos

### This Repo Owns (Datasheet Facts)
- Vendor-published hardware specs (memory, TDP, compute units)
- Physical specs (weight, dimensions, form factor)
- Environmental specs (temp range, IP rating, vibration/shock)
- Interface specs (CSI lanes, USB ports, PCIe lanes)
- Power profiles and modes (TDP at different frequencies)
- Model architectures and published accuracy benchmarks
- Sensor specifications (resolution, FPS, FoV)
- Use case constraint templates

### graphs Repo Owns (Analysis-Specific)
- `ops_per_clock` - Roofline model parameters
- `theoretical_peaks` - Computed performance ceilings
- Calibration data - Measured real-world performance
- Operation profiles - GEMM, CONV, attention micro-benchmarks
- Efficiency curves - Power vs performance tradeoffs

### Embodied-AI-Architect Owns (Orchestration)
- LLM tool definitions and execution logic
- Agentic workflows and decomposition
- CLI and user interface
- Report generation templates

## Build & Development Commands

```bash
# Install in development mode
pip install -e ".[dev]"

# Run tests
pytest tests/

# Run single test
pytest tests/test_schemas.py::test_hardware_entry_minimal -v

# Validate all YAML data files
pytest tests/test_data_validity.py

# Linting and formatting
black src/ tests/ --line-length 100
ruff check src/ tests/

# Type checking
mypy src/
```

## Architecture

### Schema Modules (`src/embodied_schemas/`)

| Module | Purpose |
|--------|---------|
| `hardware.py` | Hardware platforms, capabilities, physical/environmental/power/interface specs |
| `models.py` | ML model specs, architectures, variants (fp32/fp16/int8), accuracy |
| `sensors.py` | Camera, depth, LiDAR sensor specifications |
| `usecases.py` | Use case templates with constraints and success criteria |
| `benchmarks.py` | Benchmark results with verdict-first output schema |
| `constraints.py` | Tier definitions (latency, power, memory classes) |
| `loaders.py` | YAML loading with Pydantic validation |
| `registry.py` | Unified query API for the data catalog |

### Data Catalog (`src/embodied_schemas/data/`)

```
data/
├── hardware/{vendor}/     # Dev kits and platforms
├── chips/{vendor}/        # Raw SoC specifications
├── models/{type}/         # ML models by category
├── sensors/{category}/    # Sensors by type
├── usecases/{platform}/   # Use case templates
└── constraints/           # Tier definitions
```

## Key Design Patterns

### Verdict-First Output Schema

Tools return pre-digested judgments, not raw data:

```python
BenchmarkResult(
    verdict="PASS",           # PASS | FAIL | PARTIAL | UNKNOWN
    confidence="high",        # high | medium | low
    summary="Meets 30fps requirement with margin",
    metric={"measured": 28.5, "required": 33.3, "unit": "ms"},
    evidence="Measured over 100 inference runs",
    suggestion=None,          # Only if not PASS
)
```

**Principle**: The tool does domain reasoning; the LLM receives a verdict it can trust.

### Pydantic Models with Optional Fields

Use `| None = None` for optional fields to allow minimal entries:

```python
class HardwareEntry(BaseModel):
    id: str                           # Required
    name: str                         # Required
    capabilities: HardwareCapability  # Required
    physical: PhysicalSpec | None = None       # Optional
    environmental: EnvironmentalSpec | None = None  # Optional
```

### ID Conventions

- Hardware: `{vendor}_{product}_{variant}` (e.g., `nvidia_jetson_orin_nano_8gb`)
- Models: `{family}{size}` (e.g., `yolov8s`, `yolov8n`)
- Sensors: `{vendor}_{product}` (e.g., `intel_realsense_d435i`)
- Use cases: `{platform}_{application}` (e.g., `drone_inspection`)

## Adding New Data

### Adding Hardware

1. Create `src/embodied_schemas/data/hardware/{vendor}/{device}.yaml`
2. Follow `HardwareEntry` schema from `hardware.py`
3. Include `datasheet_url` for traceability
4. Run `pytest tests/test_data_validity.py` to validate

### Adding Models

1. Create `src/embodied_schemas/data/models/{type}/{model}.yaml`
2. Follow `ModelEntry` schema from `models.py`
3. Include accuracy benchmarks from published papers
4. Add variants if quantized versions exist

### Schema Changes

- **Non-breaking** (new optional field): Bump minor version
- **Breaking** (required field, type change, removal): Bump major version
- Add deprecation warnings before removing fields
- Update both downstream repos after publishing

## Code Style

- Line length: 100 characters
- Python target: 3.10+
- Use type hints everywhere
- Format with Black, lint with Ruff
- All schemas must have docstrings

## Testing

- `test_schemas.py` - Schema validation with various inputs
- `test_data_validity.py` - Validate all YAML files load correctly
- All tests must pass before merging

## Versioning

This package uses semantic versioning. The downstream repos pin to compatible versions:

```toml
# In graphs/pyproject.toml
dependencies = ["embodied-schemas>=0.2.0,<1.0.0"]

# In Embodied-AI-Architect/pyproject.toml
dependencies = ["embodied-schemas>=0.2.0,<1.0.0"]
```

## Related Documentation

- `../Embodied-AI-Architect/docs/plans/shared-schema-repo-architecture.md` - Original architecture decision
- `../Embodied-AI-Architect/docs/plans/knowledge-base-schema.md` - Schema design rationale
- `../graphs/hardware_registry/` - Analysis-specific data that stays in graphs
