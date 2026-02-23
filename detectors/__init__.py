"""
RailSafe AI - Detectors Module

Detektorlar:
- CarDetector: Oddiy YOLO detector
- RealtimeMultiCameraDetector: Real-time multi-camera detector (TensorRT native + fallback)
"""

from .car_detector import CarDetector
from .realtime_detector import RealtimeMultiCameraDetector

__all__ = [
    'CarDetector',
    'RealtimeMultiCameraDetector',
]
