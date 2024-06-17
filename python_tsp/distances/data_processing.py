"""Common data processing tasks between all distances"""

from typing import Optional, Tuple

import numpy as np


def process_input(
    sources: np.ndarray, destinations: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Pre-process input
    This function ensures ``sources`` and ``destinations`` have at least two
    dimensions, and if ``destinations`` is `None`, set it equal to ``sources``.
    """
    if destinations is None:
        destinations = sources

    sources = np.atleast_2d(sources)
    destinations = np.atleast_2d(destinations)

    return sources, destinations  # type: ignore
