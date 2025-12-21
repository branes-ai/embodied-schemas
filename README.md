# Embodied Schemas

Shared Pydantic schemas and factual hardware/model data for embodied AI hardware/software codesign.

## Overview

This package provides:

1. **Pydantic Schema Models** - Type-safe data structures for hardware platforms, ML models, sensors, use cases, and benchmarks
2. **Factual Data Catalog** - YAML files with validated specifications for chips, dev kits, perception models, and sensors
3. **Registry API** - Unified access to query and filter the data catalog

## Installation

```bash
pip install embodied-schemas

# For development
pip install -e ".[dev]"
```

## Usage

### Import Schema Models

```python
from embodied_schemas import (
    HardwareEntry,
    ModelEntry,
    SensorEntry,
    UseCaseEntry,
    BenchmarkResult,
)
from embodied_schemas.hardware import HardwareType, PowerMode
from embodied_schemas.constraints import LatencyTier, PowerClass
```

### Access Data Catalog

```python
from embodied_schemas import Registry

# Load the catalog
registry = Registry.load()

# Look up hardware by ID
jetson_nano = registry.hardware.get("nvidia_jetson_nano_4gb")

# Find hardware matching criteria
edge_devices = registry.hardware.find(
    suitable_for="edge",
    power_watts_max=15
)

# Look up models
yolov8s = registry.models.get("yolov8s")
detection_models = registry.models.find(type="object_detection")

# Check compatibility
is_compatible = registry.is_compatible(
    model_id="yolov8s",
    hardware_id="nvidia_jetson_nano_4gb",
    variant="fp16"
)
```

### Use in Tool Outputs

```python
from embodied_schemas import BenchmarkResult
from embodied_schemas.benchmarks import LatencyMetrics

result = BenchmarkResult(
    model_id="yolov8s",
    hardware_id="nvidia_jetson_nano_4gb",
    variant="fp16",
    verdict="PASS",
    confidence="high",
    latency=LatencyMetrics(
        mean_ms=28.5,
        std_ms=2.1,
        p95_ms=32.0,
    ),
)
```

## Data Catalog Structure

```
data/
├── hardware/           # Complete dev kits and platforms
│   ├── nvidia/
│   ├── qualcomm/
│   ├── hailo/
│   ├── google/
│   └── ...
├── chips/              # Raw SoC/chip specifications
│   ├── nvidia/
│   ├── qualcomm/
│   └── ...
├── models/             # ML model specifications
│   ├── detection/
│   ├── segmentation/
│   └── ...
├── sensors/            # Cameras, depth sensors, LiDAR
│   ├── cameras/
│   ├── depth/
│   └── lidar/
├── usecases/           # Application constraint templates
│   ├── drone/
│   ├── quadruped/
│   ├── amr/
│   └── edge/
└── constraints/        # Tier definitions
    ├── latency_tiers.yaml
    └── power_classes.yaml
```

## Related Projects

- [graphs](https://github.com/branes-ai/graphs) - Analysis tools, roofline models, simulators
- [Embodied-AI-Architect](https://github.com/branes-ai/Embodied-AI-Architect) - LLM orchestration, CLI, knowledge base

## Contributing

### Adding Hardware Data

1. Create YAML file: `src/embodied_schemas/data/hardware/{vendor}/{name}.yaml`
2. Follow the schema in `src/embodied_schemas/hardware.py`
3. Run validation: `pytest tests/test_data_validity.py`
4. Include datasheet URL in the file

### Schema Changes

- Schema changes require version bump
- Breaking changes require major version bump
- Add deprecation warnings before removing fields

## License

MIT License - see LICENSE file for details.
