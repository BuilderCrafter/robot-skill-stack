"""Construct semantic perception from already initialized camera resources."""
from robot_skill_stack.world.model.provider import WorldObservationProvider

from .config import MultiViewConfig


def build_perception_provider(kind: str, source, *, scene,
                              settings: MultiViewConfig | None = None) -> WorldObservationProvider:
    """Select an independent V1 or multiview observation pipeline."""
    if kind == 'perception':
        from .v1.factory import build_v1_provider
        return build_v1_provider(source, scene)
    if kind == 'multi_view':
        from .multiview.provider import MultiViewProvider
        settings = MultiViewConfig() if settings is None else settings
        return MultiViewProvider(source, scene, voxel_size=settings.voxel_size, stride=settings.stride)
    raise ValueError(f'Unsupported semantic perception pipeline: {kind!r}')
