import cProfile
import lentil
import prysm
import prysm as prysmc
import numpy as np
import pstats
from lentil.constants_utils import *
from lentil.wavefront_test__old import try_wavefront
from lentil.wavefront_config__old import SPACIAL_FREQS
from lentil import wavefront_config__old

BASE_PATH = "/home/sam/nashome/MTFMapper Stuff/"

PATHS = [
    "Bernard/",

    "56mm/f1.2/",
    "56mm/f2.8/",
    "56mm/f5.6/",
    "56mm/f8/",]

# focusset = lentil.FocusSet(fallback_results_path(os.path.join(BASE_PATH, PATHS[2]), 3), include_all=1,
#                            use_calibration=1)
e = np
import cupy as cp

# phase = cp.array(pupil.change_phase_unit(to='waves', inplace=False))
# prysmc.config.backend = 'cuda'


def fcn():
    """Complex wavefunction associated with the pupil."""

    pupil = prysmc.NollZernike(z4=0.1, z5=0.1, z6=0.1, z7=0.1, mask_target='none', samples=256)
    psf = prysmc.PSF.from_pupil(pupil, efl=28, Q=2)

    # fcn = e.exp(1j * 2 * e.pi * phase)  # phase implicitly in units of waves, no 2pi/l
    # guard against nans in phase
    # fcn[e.isnan(phase)] = 0

    # if pupil.mask_target in ('fcn', 'both'):
    #     fcn *= pupil._mask

    # return fcn

# m= prysm.MTF.from_psf(psf)

# sag_x, sag_mod = m.sag
# zkwargs = {'z9': 0.057499999999999996, 'z10': 0.057499999999999996, 'z11': 0.057499999999999996, 'z12': 0.057499999999999996}

# prysm.config.backend = "cuda"
# prysm.config.precision = 32

def runme():
    for _ in range(50):
        # fcn()
        # prysm.config.backend = 'numpy'
        # prysm.config.precision = wavefront_config.PRECISION
        # pupil = prysm.NollZernike(z4=0.1, z5=0.1, z6=0.1, z7=0.1, mask_target='none', samples=256)
        # for _ in range(7):
        #     pupil = prysm.FringeZernike(Z4=0.1,
        #                                 dia=10, norm=True,
        #                                 wavelength=0.5,
        #                                 opd_unit="um",
        #                                 samples=256,
        #                                 **{'z9': 0.057499999999999996, 'z10': 0.057499999999999996,
        #                                    'z11': 0.057499999999999996,
        #                                    'z12': 0.057499999999999996})
        # print(pupil.phase)
        try_wavefront(10, dict(df_offset=10, df_step=0.1, z9=0.1, z10=0.1, z11=0.1, z12=0.1, base_fstop=2.0, fstop=2.0), samples=256, mono=False)
        # sag_x, sag_mod = m.sag
        # sag_x = cp.asnumpy(sag_x)
        # sag_mod = cp.asnumpy(sag_mod)
        # interpfn = interpolate.InterpolatedUnivariateSpline(cp.asnumpy(sag_x), cp.asnumpy(sag_mod), k=1)
        # interpfn = interpolate.InterpolatedUnivariateSpline(sag_x, sag_mod, k=1)
        # a = interpfn(SPACIAL_FREQS / DEFAULT_PIXEL_SIZE * 1e-3)

for _ in range(1):
    runme()

cProfile.run('runme()', 'profilestats')
p = pstats.Stats('profilestats')
p.strip_dirs().sort_stats('cumulative').print_stats()
p.strip_dirs().sort_stats('time').print_stats()