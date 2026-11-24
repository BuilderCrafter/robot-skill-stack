"""Shared depth-edge sampling and row-major pixel-ribbon deduplication."""
import numpy as np


DEPTH_TOLERANCE = .002
PIXEL_RADIUS = 3
EDGE_SAMPLES = 48
RIBBON_OFFSETS = np.array([(x, y) for x in range(-PIXEL_RADIUS, PIXEL_RADIUS+1)
                                for y in range(-PIXEL_RADIUS, PIXEL_RADIUS+1)])
PIXEL_NEIGHBORS = np.array([(x, y) for x in (-1, 0, 1) for y in (-1, 0, 1) if x or y])
EDGE_POSITIONS = np.linspace(-.4, .4, EDGE_SAMPLES)


def unique_pixels(raw: np.ndarray) -> np.ndarray:
    """Deduplicate bounded image-coordinate ribbons in row-major order."""
    low = raw.min(0)
    width = int(raw[:, 0].max()-low[0])+1
    codes = np.unique((raw[:, 1]-low[1])*width+raw[:, 0]-low[0])
    return np.column_stack((codes % width+low[0], codes // width+low[1]))
