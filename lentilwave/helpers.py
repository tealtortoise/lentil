import numpy as np
from scipy import ndimage
try:
    import cupy as cp
except ImportError:
    cp = None

import prysm

from lentilwave import config, processing_details
from lentil import constants_utils as lentilconf


class TestSettings:
    def __init__(self, p, x_loc=None, y_loc=None, defocus=0.0):
        self.defocus = defocus
        self.p = p
        self.mono = False
        self.plot = False
        self.dummy = False
        self.allow_cuda = config.USE_CUDA
        self.id_or_hash = 0
        self.strehl_estimate = 1.0
        self.fftsize = None
        self.phasesamples = None
        self.return_otf = True
        self.return_otf_mtf = False
        self.return_psf = False
        self.return_prysm_mtf = False
        self.return_mask = False
        self.mask = None
        self.prysm_mtf = None
        self.cpu_gpu_arraysize_boundary = config.CPU_GPU_ARRAYSIZE_BOUNDARY
        self.effective_q = None
        self.guide_mtf = None
        self.q_autosize_scalar = config.Q_AUTOSIZE_SCALAR
        self.phase_autosize_scalar = config.PHASE_AUTOSIZE_SCALAR
        self.cache_sizes = True
        if x_loc is None:
            x_loc = lentilconf.IMAGE_WIDTH / 2
        if y_loc is None:
            y_loc = lentilconf.IMAGE_HEIGHT / 2
        self.x_loc = x_loc
        self.y_loc = y_loc
        self.pixel_vignetting = True
        self.lens_vignetting = True
        self.default_exit_pupil_position_mm = 100
        self.exif = None
        self.fix_pupil_rotation = True
        self.zernike_flags = None
        self.used_zernikes = None
        self.max_zernike = None
        self.zernike_array = None
        self.zernike_index = None
        self.zernike_array_indexed = None

    @property
    def fftshape(self):
        if self.fftsize is None:
            return None
        return self.fftsize, self.fftsize

    @property
    def phaseshape(self):
        if self.phasesamples is None:
            return None
        return self.phasesamples, self.phasesamples

    @property
    def is_valid(self):
        if self.fftsize is None:
            return False
        if self.phasesamples is None:
            return False
        if self.p is None:
            return False
        if self.defocus is None:
            return False
        return True

    def get_processing_details(self, cache_=None):
        processing_details.get_processing_details(self, cache_=cache_)
        return self

    def get_used_zernikes(self):
        me = np
        flags = me.zeros(48, dtype="int")
        idx = me.zeros(48, dtype="int") - 1
        all = me.zeros(48, dtype="float64")
        used = [4, 9]
        count = 0
        for key, val in self.p.items():
            if key[0].lower() == "z" and key[1].isdigit():
                z_number = int(key[1:])
                if z_number == 4:
                    raise Exception("Z4 cannot be used directly, use defocus attribute")
                if z_number == 9:
                    # We've already added this
                    continue
                flags[z_number - 1] = 1  # Zero indexed
                all[z_number - 1] = val
                used.append(z_number)
        flags[3] = 1  # We always need defocus
        flags[8] = 1  # We always need first sphere

        used_indexed = me.zeros(len(used))
        used.sort()
        for z in used:
            idx[z - 1] = count
            used_indexed[count] = all[z - 1]
            count += 1

        self.used_zernikes = tuple(used)
        self.max_zernike = max(used)
        self.zernike_array_indexed = used_indexed
        self.zernike_index = idx
        self.zernike_flags = flags
        self.zernike_array = all
        return self


class TestResults:
    def __init__(self):
        self.lsf = None
        self.psf = None
        self.otf = None
        self.timings = None
        self.strehl = None
        self.fftsize = None
        self.samples = None
        self.id_or_hash = None
        self.used_cuda = None

        self.prysm_mtf: prysm.MTF = None
        self.psf: prysm.PSF = None
        self.otf = None
        self.mtf = None

    def copy_important_settings(self, s: TestSettings):
        self.fftsize = s.fftsize
        self.samples = s.phasesamples
        self.id_or_hash = s.id_or_hash

    def get_mtf(self):
        return abs(self.otf[0]), abs(self.otf[1])


def zoom2d(inarr, xfactor=1.0, yfactor=1.0, xoffset=0.0, yoffset=0.0, affine_transform=ndimage.affine_transform, me=np):
    shape = inarr.shape
    transform = me.array(((1.0 / yfactor, 0), (0, 1.0 / xfactor)))
    offset_x = (shape[0]) / 2 * (1.0 - 1.0 / xfactor) - xoffset / xfactor
    offset_y = (shape[1]) / 2 * (1.0 - 1.0 / yfactor) - yoffset / yfactor
    if me is cp:
        return affine_transform(inarr, transform, (offset_y, offset_x),
                                           order=1)
    else:
        if inarr.dtype == "complex64" or inarr.dtype == "complex128":
            real = affine_transform(inarr.real, transform, (offset_y, offset_x),
                                               order=config.PSF_SPLINE_ORDER)
            imag = affine_transform(inarr.imag, transform, (offset_y, offset_x),
                                               order=config.PSF_SPLINE_ORDER)
            return real + 1j * imag
        else:
            return affine_transform(inarr, transform, (offset_y, offset_x),
                             order=config.PSF_SPLINE_ORDER)


def zoom1d(inarr, factor=1.0, offset=0.0, affine_transform=ndimage.affine_transform, me=np):
    shape = inarr.shape
    transform = me.array((1.0 / factor,))
    offset_ = (shape[0]) / 2 * (1.0 - 1.0 / factor) - offset / factor
    if me is cp:
        return affine_transform(inarr, transform, (offset_,), order=1)
    else:
        if inarr.dtype == "complex64" or inarr.dtype == "complex128":
            real = affine_transform(inarr.real, transform, (offset,),
                                               order=config.PSF_SPLINE_ORDER)
            imag = affine_transform(inarr.imag, transform, (offset_,),
                                               order=config.PSF_SPLINE_ORDER)
            return real + 1j * imag
        else:
            return affine_transform(inarr, transform, (offset_,),
                             order=config.PSF_SPLINE_ORDER)


def get_z9(p, modelwavelength):
    rel_wv = modelwavelength / config.BASE_WAVELENGTH
    spca = p.get('spca', 0.0) * 30
    spca2 = p.get('spca2', 0.0) * 30

    spcaz9 = (modelwavelength / config.BASE_WAVELENGTH - 1.0) * spca + spca * 0.028
    spca2z9 = (rel_wv - 1.0) ** 2 * spca2 * 10 - spca2 * 0.06
    return (p.get('z9', 0.0) + spcaz9 + spca2z9)


def get_z4(defocus, p, modelwavelength):
    fstop_base_ratio = p['fstop'] / p['base_fstop']
    rel_wv = modelwavelength / config.BASE_WAVELENGTH
    loca = p.get('loca', 0.0) * 30
    loca1 = p.get('loca1', 0.0) * 30

    locadefocus = (rel_wv - 1.0) ** 2 * 10 * loca - loca * 0.06
    loca1defocus = (rel_wv - 1.0) * 1 * loca1 + loca1 * 0.027
    base_z4 = ((defocus - p.get('df_offset', 0)) * p.get('df_step', 1)) * fstop_base_ratio ** 2
    # print(p)
    return -(base_z4 - locadefocus - loca1defocus)


def get_lca_shifts(s: TestSettings, modelwavelength, samplespacing, old=False):
    rel_wv = modelwavelength / 0.54

    img_height = lentilconf.calc_image_height(s.x_loc, s.y_loc)

    px = s.p.get('tca_slr', 0.0) * 1e3 * img_height
    py = 0

    shiftx = (rel_wv - 1.0) ** 2 * px * 10 - px / 14
    shifty = (rel_wv - 1.0) ** 2 * py * 10 - py / 14
    if old:
        return shiftx / samplespacing / s.fftsize, shifty / samplespacing / s.fftsize
    return shiftx / samplespacing / 1.5, shifty / samplespacing / 1.5


# def get_polywaves(n):


strehl_estimate_cache = None
mtf_cache = None
return_cache = None

