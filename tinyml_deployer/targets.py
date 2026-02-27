"""MCU target definitions for supported microcontrollers."""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class MCUTarget:
    """Hardware specification for a microcontroller target."""

    name: str
    flash_kb: int
    ram_kb: int
    clock_mhz: int
    fpu: bool
    framework: str
    supported_ops: list[str] = field(default_factory=list)


ESP32 = MCUTarget(
    name="esp32",
    flash_kb=4096,
    ram_kb=520,
    clock_mhz=240,
    fpu=False,
    framework="ESP-IDF",
    supported_ops=[
        "CONV_2D",
        "DEPTHWISE_CONV_2D",
        "FULLY_CONNECTED",
        "SOFTMAX",
        "RESHAPE",
        "AVERAGE_POOL_2D",
        "MAX_POOL_2D",
        "ADD",
        "MUL",
        "QUANTIZE",
        "DEQUANTIZE",
    ],
)

ESP32S3 = MCUTarget(
    name="esp32s3",
    flash_kb=8192,
    ram_kb=512,
    clock_mhz=240,
    fpu=False,
    framework="ESP-IDF",
    supported_ops=[
        "CONV_2D",
        "DEPTHWISE_CONV_2D",
        "FULLY_CONNECTED",
        "SOFTMAX",
        "RESHAPE",
        "AVERAGE_POOL_2D",
        "MAX_POOL_2D",
        "ADD",
        "MUL",
        "QUANTIZE",
        "DEQUANTIZE",
        "LOGISTIC",
        "MEAN",
        "PAD",
    ],
)

STM32F4 = MCUTarget(
    name="stm32f4",
    flash_kb=1024,
    ram_kb=192,
    clock_mhz=168,
    fpu=True,
    framework="STM32CubeAI",
    supported_ops=[
        "CONV_2D",
        "DEPTHWISE_CONV_2D",
        "FULLY_CONNECTED",
        "SOFTMAX",
        "RESHAPE",
        "AVERAGE_POOL_2D",
        "MAX_POOL_2D",
        "ADD",
        "QUANTIZE",
        "DEQUANTIZE",
    ],
)

STM32H7 = MCUTarget(
    name="stm32h7",
    flash_kb=2048,
    ram_kb=1024,
    clock_mhz=480,
    fpu=True,
    framework="STM32CubeAI",
    supported_ops=[
        "CONV_2D",
        "DEPTHWISE_CONV_2D",
        "FULLY_CONNECTED",
        "SOFTMAX",
        "RESHAPE",
        "AVERAGE_POOL_2D",
        "MAX_POOL_2D",
        "ADD",
        "MUL",
        "QUANTIZE",
        "DEQUANTIZE",
        "LOGISTIC",
        "MEAN",
        "PAD",
        "CONCATENATION",
        "SPLIT",
    ],
)

TARGETS: dict[str, MCUTarget] = {
    "esp32": ESP32,
    "esp32s3": ESP32S3,
    "stm32f4": STM32F4,
    "stm32h7": STM32H7,
}


def get_target(name: str) -> MCUTarget:
    """Look up an MCU target by name.

    Raises:
        ValueError: If the target name is not recognized.
    """
    target = TARGETS.get(name.lower())
    if target is None:
        available = ", ".join(sorted(TARGETS.keys()))
        raise ValueError(f"Unknown target '{name}'. Available targets: {available}")
    return target
