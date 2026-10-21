"""Explicit spatial-mapping adapters; importing this package starts no SDK resources."""
from .config import SpatialMappingConfig
from .factory import MappingSession, create_spatial_mapping
from .frames import CameraSnapshots, SharedFrames

__all__ = ['SpatialMappingConfig', 'MappingSession', 'create_spatial_mapping',
           'CameraSnapshots', 'SharedFrames']
