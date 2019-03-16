import numpy as np

DEFAULT_PIXEL_SIZE = 4e-6
SAGGITAL = 1
MERIDIONAL = 2
BOTH_AXES = 3
SFR_HEADER = [
    'blockid',
    'edgex',
    'edgey',
    'edgeangle',
    'radialangle'
]


def diffraction_mtf(freq):
    return 2.0 / np.pi * (np.arccos(freq) - freq * (1 - freq ** 2) ** 0.5)


RAW_SFR_FREQUENCIES = [x / 64 for x in range(64)]  # List of sfr frequencies in cycles/pixel