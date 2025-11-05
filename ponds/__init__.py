from ponds.core import PONDS
from ponds.shapegen import ShapeMethod, SINGLE, CELL, ELLIPSIS, CUSTOM, BLOB, CLUSTER
from ponds.shiftgen import ShiftMethod, SIGMOID, BIMODAL
from ponds.backgroundgen import BackgroundMethod, WhiteNoise, Trend
from ponds.correlationgen import (
    CorrelationMethod,
    NoCorr,
    Noise,
    Gauss,
    Custom,
    Propagating,
)

from typing import Any, Dict

__all__ = [
    "PONDS",
    "ShapeMethod",
    "SINGLE",
    "CELL",
    "ELLIPSIS",
    "CUSTOM",
    "BLOB",
    "CLUSTER",
    "ShiftMethod",
    "SIGMOID",
    "BIMODAL",
    "BackgroundMethod",
    "WhiteNoise",
    "Trend",
    "CorrelationMethod",
    "NoCorr",
    "Noise",
    "Gauss",
    "Custom",
    "Propagating",
]
type_dict: Dict[str, Any] = {
    "SINGLE": SINGLE,
    "CELL": CELL,
    "ELLIPSIS": ELLIPSIS,
    "CUSTOM": CUSTOM,
    "BLOB": BLOB,
    "CLUSTER": CLUSTER,
    "SIGMOID": SIGMOID,
    "BIMODAL": BIMODAL,
    "WhiteNoise": WhiteNoise,
    "Trend": Trend,
    "NoCorr": NoCorr,
    "Noise": Noise,
    "Gauss": Gauss,
    "Custom": Custom,
    "Propagating": Propagating,
}
