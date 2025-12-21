"""Sensor schemas for perception systems.

Defines data structures for cameras, depth sensors, LiDAR, and other
sensors used in embodied AI systems.
"""

from enum import Enum
from pydantic import BaseModel, Field


class SensorCategory(str, Enum):
    """Categories of sensors."""

    CAMERA = "camera"
    DEPTH = "depth"
    LIDAR = "lidar"
    RADAR = "radar"
    IMU = "imu"
    GPS = "gps"
    ULTRASONIC = "ultrasonic"
    THERMAL = "thermal"
    EVENT = "event"  # Event cameras


class InterfaceType(str, Enum):
    """Sensor interface types."""

    CSI = "csi"
    USB = "usb"
    ETHERNET = "ethernet"
    GMSL = "gmsl"
    MIPI = "mipi"
    I2C = "i2c"
    SPI = "spi"
    UART = "uart"
    PCIE = "pcie"


class DepthType(str, Enum):
    """Types of depth sensing technology."""

    STEREO = "stereo"
    TOF = "tof"  # Time of Flight
    STRUCTURED_LIGHT = "structured_light"
    LIDAR = "lidar"
    ACTIVE_STEREO = "active_stereo"


class CameraSpec(BaseModel):
    """Camera sensor specifications."""

    sensor_type: str = Field(..., description="Sensor type: CMOS, CCD")
    resolution: list[int] = Field(..., description="Resolution [width, height]")
    pixel_size_um: float | None = Field(None, description="Pixel size in micrometers")
    sensor_size_mm: list[float] | None = Field(
        None, description="Sensor size [width, height] in mm"
    )
    max_fps: int = Field(..., description="Maximum frame rate")
    dynamic_range_db: float | None = Field(None, description="Dynamic range in dB")
    sensitivity_iso: list[int] | None = Field(
        None, description="ISO sensitivity range [min, max]"
    )
    shutter_type: str | None = Field(
        None, description="Shutter type: rolling, global"
    )
    hdr_support: bool = Field(False, description="HDR capture support")
    color_filter: str = Field("bayer", description="Color filter: bayer, mono, rgb")


class DepthSpec(BaseModel):
    """Depth sensor specifications."""

    depth_type: DepthType = Field(..., description="Depth sensing technology")
    depth_resolution: list[int] = Field(..., description="Depth resolution [width, height]")
    rgb_resolution: list[int] | None = Field(
        None, description="RGB resolution if available [width, height]"
    )
    depth_fps: int = Field(..., description="Depth frame rate")
    rgb_fps: int | None = Field(None, description="RGB frame rate if available")
    range_m: list[float] = Field(..., description="Depth range [min, max] in meters")
    accuracy_percent: float | None = Field(
        None, description="Depth accuracy as percentage at reference distance"
    )
    accuracy_reference_m: float | None = Field(
        None, description="Reference distance for accuracy spec"
    )
    baseline_mm: float | None = Field(
        None, description="Stereo baseline in mm (for stereo sensors)"
    )
    indoor_outdoor: str = Field(
        "both", description="Suitability: indoor, outdoor, both"
    )


class LidarSpec(BaseModel):
    """LiDAR sensor specifications."""

    lidar_type: str = Field(..., description="LiDAR type: mechanical, solid_state, mems")
    channels: int = Field(..., description="Number of channels/lines")
    points_per_second: int = Field(..., description="Points per second")
    range_m: list[float] = Field(..., description="Range [min, max] in meters")
    accuracy_cm: float | None = Field(None, description="Range accuracy in cm")
    horizontal_fov_deg: float = Field(..., description="Horizontal field of view")
    vertical_fov_deg: float = Field(..., description="Vertical field of view")
    angular_resolution_deg: float | None = Field(
        None, description="Angular resolution in degrees"
    )
    scan_rate_hz: float | None = Field(None, description="Scan rate in Hz")
    wavelength_nm: int | None = Field(None, description="Laser wavelength in nm")
    eye_safe: bool = Field(True, description="Eye-safe classification")


class ImuSpec(BaseModel):
    """IMU sensor specifications."""

    axes: int = Field(6, description="Number of axes (6 or 9)")
    gyro_range_dps: float | None = Field(None, description="Gyroscope range in degrees/sec")
    accel_range_g: float | None = Field(None, description="Accelerometer range in g")
    gyro_noise_dps: float | None = Field(None, description="Gyroscope noise density")
    accel_noise_mg: float | None = Field(None, description="Accelerometer noise density")
    sample_rate_hz: int | None = Field(None, description="Maximum sample rate")
    has_magnetometer: bool = Field(False, description="Includes magnetometer")


class SensorInterface(BaseModel):
    """Sensor interface specification."""

    type: InterfaceType = Field(..., description="Interface type")
    version: str | None = Field(None, description="Interface version: 3.0, 2.1, etc.")
    lanes: int | None = Field(None, description="Number of lanes (for CSI/MIPI)")
    data_rate_gbps: float | None = Field(None, description="Data rate in Gbps")


class SensorPower(BaseModel):
    """Sensor power specifications."""

    voltage_v: float = Field(..., description="Operating voltage")
    current_ma: float | None = Field(None, description="Typical current in mA")
    power_watts: float = Field(..., description="Power consumption in watts")
    standby_watts: float | None = Field(None, description="Standby power consumption")


class SensorPhysical(BaseModel):
    """Sensor physical specifications."""

    weight_grams: float | None = Field(None, description="Weight in grams")
    dimensions_mm: list[float] | None = Field(
        None, description="Dimensions [L, W, H] in mm"
    )
    connector: str | None = Field(None, description="Connector type")
    mounting: str | None = Field(None, description="Mounting type")


class SensorEnvironmental(BaseModel):
    """Sensor environmental specifications."""

    operating_temp_c: list[float] = Field(
        ..., description="Operating temperature [min, max] in Celsius"
    )
    storage_temp_c: list[float] | None = Field(None, description="Storage temperature range")
    ip_rating: str | None = Field(None, description="IP rating")
    outdoor_capable: bool = Field(False, description="Suitable for outdoor use")


class SensorEntry(BaseModel):
    """Complete sensor catalog entry."""

    # Identity
    id: str = Field(..., description="Unique identifier")
    name: str = Field(..., description="Human-readable name")
    vendor: str = Field(..., description="Manufacturer")
    model: str = Field(..., description="Model name/number")
    category: SensorCategory = Field(..., description="Sensor category")

    # Type-specific specs (one should be populated based on category)
    camera_spec: CameraSpec | None = Field(None, description="Camera specifications")
    depth_spec: DepthSpec | None = Field(None, description="Depth sensor specifications")
    lidar_spec: LidarSpec | None = Field(None, description="LiDAR specifications")
    imu_spec: ImuSpec | None = Field(None, description="IMU specifications")

    # Common specs
    interface: SensorInterface = Field(..., description="Interface specification")
    power: SensorPower = Field(..., description="Power specification")
    physical: SensorPhysical | None = Field(None, description="Physical specifications")
    environmental: SensorEnvironmental | None = Field(
        None, description="Environmental specifications"
    )

    # Compatibility
    compatible_interfaces: list[str] = Field(
        default_factory=list,
        description="Compatible interface types on hardware",
    )
    suitable_for: list[str] = Field(
        default_factory=list, description="Suitable use case IDs"
    )

    # Cost and availability
    cost_usd: float | None = Field(None, description="Approximate cost in USD")
    availability: str = Field("widely_available", description="Product availability")

    # Metadata
    datasheet_url: str | None = Field(None, description="Datasheet link")
    product_url: str | None = Field(None, description="Product page link")
    last_updated: str = Field(..., description="Last update date (YYYY-MM-DD)")

    model_config = {"extra": "forbid"}
