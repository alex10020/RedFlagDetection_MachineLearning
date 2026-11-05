"""Red Flag Detector: score dating-app messages for toxic language and gaslighting."""

from .data import (
    clean_text,
    deduplicate,
    load_gaslighting,
    load_jigsaw,
    load_stress_test,
    load_toxic_conversations,
    make_splits,
)

__all__ = [
    "clean_text",
    "deduplicate",
    "load_gaslighting",
    "load_jigsaw",
    "load_stress_test",
    "load_toxic_conversations",
    "make_splits",
]
__version__ = "0.1.0"
