"""ML model schemas for perception and AI workloads.

Defines data structures for neural network models used in embodied AI systems,
including architecture specs, accuracy benchmarks, and deployment variants.
"""

from enum import Enum
from pydantic import BaseModel, Field


class ModelType(str, Enum):
    """Types of perception/AI models."""

    OBJECT_DETECTION = "object_detection"
    INSTANCE_SEGMENTATION = "instance_segmentation"
    SEMANTIC_SEGMENTATION = "semantic_segmentation"
    PANOPTIC_SEGMENTATION = "panoptic_segmentation"
    POSE_ESTIMATION = "pose_estimation"
    DEPTH_ESTIMATION = "depth_estimation"
    OPTICAL_FLOW = "optical_flow"
    CLASSIFICATION = "classification"
    TRACKING = "tracking"
    SLAM = "slam"
    SCENE_UNDERSTANDING = "scene_understanding"
    LANE_DETECTION = "lane_detection"
    FACE_DETECTION = "face_detection"
    OCR = "ocr"


class ModelFormat(str, Enum):
    """Model serialization formats."""

    PYTORCH = "pytorch"
    ONNX = "onnx"
    TENSORRT = "tensorrt"
    TFLITE = "tflite"
    OPENVINO = "openvino"
    COREML = "coreml"
    HAILO = "hailo"
    EDGE_TPU = "edge_tpu"
    NCNN = "ncnn"
    MNN = "mnn"


class DataType(str, Enum):
    """Model weight data types."""

    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"
    INT8 = "int8"
    INT4 = "int4"
    MIXED = "mixed"


class ModelVariant(BaseModel):
    """A specific variant/quantization of a model."""

    name: str = Field(..., description="Variant name, e.g., 'fp16', 'int8-tensorrt'")
    dtype: DataType = Field(..., description="Primary data type")
    format: ModelFormat = Field(..., description="Model format")
    file_size_mb: float | None = Field(None, description="Model file size in MB")
    accuracy_delta: float = Field(
        0.0, description="Accuracy change relative to FP32 baseline (negative = worse)"
    )
    calibration_dataset: str | None = Field(
        None, description="Dataset used for quantization calibration"
    )
    calibration_samples: int | None = Field(
        None, description="Number of samples used for calibration"
    )
    optimizations: list[str] = Field(
        default_factory=list,
        description="Applied optimizations: layer_fusion, channel_pruning, etc.",
    )
    notes: str | None = Field(None, description="Variant-specific notes")


class InputSpec(BaseModel):
    """Model input specification."""

    shape: list[int] = Field(..., description="Input shape [N, C, H, W] or similar")
    dtype: str = Field("float32", description="Input data type")
    normalization: list[float] | None = Field(
        None, description="Normalization range, e.g., [0, 1] or [-1, 1]"
    )
    mean: list[float] | None = Field(
        None, description="Per-channel mean for normalization"
    )
    std: list[float] | None = Field(
        None, description="Per-channel std for normalization"
    )
    channel_order: str = Field("rgb", description="Channel order: rgb, bgr")
    dynamic_axes: dict[str, list[int]] | None = Field(
        None, description="Dynamic axes for variable input sizes"
    )


class OutputSpec(BaseModel):
    """Model output specification."""

    format: str = Field(..., description="Output format description")
    num_classes: int | None = Field(None, description="Number of output classes")
    max_detections: int | None = Field(None, description="Max detections for detection models")
    output_names: list[str] = Field(
        default_factory=list, description="Named output tensor names"
    )


class ArchitectureSpec(BaseModel):
    """Model architecture specification."""

    type: ModelType = Field(..., description="Model task type")
    family: str | None = Field(None, description="Model family: yolov8, rt-detr, etc.")
    backbone: str | None = Field(None, description="Backbone architecture")
    neck: str | None = Field(None, description="Neck architecture: fpn, pan, etc.")
    head: str | None = Field(None, description="Head architecture")
    params_millions: float = Field(..., description="Total parameters in millions")
    flops_billions: float | None = Field(None, description="FLOPs in billions (at reference input)")
    macs_billions: float | None = Field(None, description="MACs in billions")
    layers: int | None = Field(None, description="Total number of layers")
    attention_layers: int | None = Field(None, description="Number of attention layers")


class AccuracyBenchmark(BaseModel):
    """Accuracy benchmark on a specific dataset."""

    dataset: str = Field(..., description="Dataset name: coco_val2017, imagenet, etc.")
    map_50: float | None = Field(None, description="mAP@0.5 for detection")
    map_50_95: float | None = Field(None, description="mAP@0.5:0.95 for detection")
    map_small: float | None = Field(None, description="mAP for small objects")
    map_medium: float | None = Field(None, description="mAP for medium objects")
    map_large: float | None = Field(None, description="mAP for large objects")
    miou: float | None = Field(None, description="Mean IoU for segmentation")
    accuracy_top1: float | None = Field(None, description="Top-1 accuracy for classification")
    accuracy_top5: float | None = Field(None, description="Top-5 accuracy for classification")
    pck: float | None = Field(None, description="PCK for pose estimation")
    abs_rel: float | None = Field(None, description="Absolute relative error for depth")
    rmse: float | None = Field(None, description="RMSE for depth estimation")
    methodology: str | None = Field(
        None, description="Evaluation methodology: official, custom, etc."
    )
    notes: str | None = Field(None, description="Additional notes about the benchmark")


class MemoryRequirements(BaseModel):
    """Runtime memory requirements."""

    weights_mb: float = Field(..., description="Model weights size in MB")
    peak_activation_mb: float | None = Field(
        None, description="Peak activation memory at batch=1"
    )
    workspace_mb: float = Field(0, description="Additional workspace memory (TensorRT, etc.)")
    total_inference_mb: float | None = Field(
        None, description="Total memory during inference"
    )


class ModelEntry(BaseModel):
    """Complete model catalog entry."""

    # Identity
    id: str = Field(..., description="Unique identifier, e.g., yolov8s")
    name: str = Field(..., description="Human-readable name")
    version: str = Field(..., description="Model version")
    vendor: str = Field(..., description="Model creator/vendor")

    # Architecture
    architecture: ArchitectureSpec = Field(..., description="Architecture specifications")

    # Input/Output
    input: InputSpec = Field(..., description="Input specification")
    output: OutputSpec = Field(..., description="Output specification")

    # Accuracy benchmarks
    accuracy: dict[str, AccuracyBenchmark] = Field(
        default_factory=dict,
        description="Accuracy benchmarks keyed by dataset name",
    )

    # Variants
    variants: list[ModelVariant] = Field(
        default_factory=list, description="Available model variants"
    )

    # Memory
    memory: MemoryRequirements = Field(..., description="Memory requirements")

    # Compatibility
    compatible_hardware: list[str] = Field(
        default_factory=list, description="List of compatible hardware IDs"
    )
    suitable_for: list[str] = Field(
        default_factory=list, description="List of suitable use case IDs"
    )

    # Optimization notes
    optimization_notes: dict[str, str] = Field(
        default_factory=dict,
        description="Optimization notes keyed by runtime/hardware",
    )

    # Metadata
    license: str = Field(..., description="License: MIT, Apache-2.0, AGPL-3.0, etc.")
    source_url: str | None = Field(None, description="Source repository URL")
    paper_url: str | None = Field(None, description="Research paper URL")
    weights_url: str | None = Field(None, description="Pre-trained weights URL")
    last_updated: str = Field(..., description="Last update date (YYYY-MM-DD)")

    model_config = {"extra": "forbid"}
