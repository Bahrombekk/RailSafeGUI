"""
RailSafe AI - Detectors Module

Detektorlar:
- CarDetector: Oddiy YOLO detector
- BatchCarDetector: Batch inference detector
- RealtimeMultiCameraDetector: Real-time multi-camera detector (TensorRT native + fallback)
"""

from .car_detector import CarDetector
from .batch_detector import BatchCarDetector
from .realtime_detector import RealtimeMultiCameraDetector

__all__ = [
    'CarDetector',
    'BatchCarDetector',
    'RealtimeMultiCameraDetector',
]
