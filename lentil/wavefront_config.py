import numpy as np
from collections import OrderedDict

# modelwavelengths = np.linspace(0.47, 0.62, 3)
# modelwavelengths = modelwavelengths[1:-1]
modelwavelengths = np.array([0.575])


freqs = np.arange(2, 30, 3) / 64

DIVIDE_BY_APERTURE = lambda f: 1.0 / f
MULTIPLY_BY_APERTURE = lambda f: f
NOP = lambda f: 1.0

params_options = OrderedDict(
    # param_name=(Min, Initial, Max, normalise_with_aperture, per_focusset)
    df_offset=(None, None, None, NOP, True),
    df_step=(0.015, 0.2, 1, DIVIDE_BY_APERTURE, True),
    z7=(-np.inf, 0, np.inf, DIVIDE_BY_APERTURE, False),
    z8=(-np.inf, 0, np.inf, DIVIDE_BY_APERTURE, False),
    z9=(-np.inf, 0, np.inf, DIVIDE_BY_APERTURE, False),
    z16=(-np.inf, 0, np.inf, DIVIDE_BY_APERTURE, False),
    z25=(-np.inf, 0, np.inf, DIVIDE_BY_APERTURE, False),
    z36=(-np.inf, 0, np.inf, DIVIDE_BY_APERTURE, False),
    fstop=(1/1.07, 1.0, 1.07, MULTIPLY_BY_APERTURE, False),
    loca=(-np.inf, 0, np.inf, DIVIDE_BY_APERTURE, False),
    spca=(-np.inf, 0.01, np.inf, DIVIDE_BY_APERTURE, False),
    locaref=(0.500, 0.60, 0.680, NOP, False),
    zero=(0, 0.005, 0.05, NOP, False),
)

optimise_params = [
    'df_offset',
    'df_step',
    'z7',
    # 'z8',
    'z9',
    'z16',
    'z25',
    'z36',
    'fstop',
    # 'loca',
    # 'spca',
    # 'locaref',
    # 'zero',
]

fixed_params = []
for key in params_options.keys():
    if key not in optimise_params:
        fixed_params.append(key)
