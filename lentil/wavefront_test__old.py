import time
from collections import deque
import cupy as cp
import copy
import multiprocessing
import cupyx.scipy.ndimage
import cupyx.scipy.fftpack as scipyfftpack
import numpy as np
import prysm
from scipy import interpolate, ndimage, fftpack, optimize, signal
import matplotlib.pyplot as plt

from lentil import wavefront_config__old as conf
from lentil.wavefront_config__old import SPACIAL_FREQS, BASE_WAVELENGTH, MODEL_WVLS, DEFAULT_SAMPLES
from lentil.constants_utils import *


zcache = prysm.zernike.zcache.regular
cupyzcache = {}
numpyzcache_arrays = {}
cupyzcache_arrays = {}
cache_idx = {}
cache_idx["cp"] = None
cache_idx["np"] = None
window_cache = {}

all_but_z4_and_z9_phasecache = dict(np={}, cp={})

RETURN_MTF = 1
RETURN_OTF = 2
RETURN_LSF = 3
RETURN_PSF = 4
RETURN_WITH_PROCESSING_DETAILS = 5

settings_cache = {}

mask_cache = deque(maxlen=conf.MASK_CACHE_SIZE)

deltas = np.linspace(-1,1, 19) * 1e-12
a = np.random.random((256, 256))
psf_size = 256
mses = []
for d in deltas:
    normshift_x = d
    normshift_y = 0
    zoom_factor = 1.0000
    transform = np.array(((1.0 / zoom_factor, 0), (0, 1.0 / zoom_factor)))
    offset_x = (psf_size - 1 + 1) / 2 * (1.0 - 1.0 / zoom_factor - normshift_x)
    offset_y = (psf_size - 1 + 1) / 2 * (1.0 - 1.0 / zoom_factor - normshift_y)
    order = conf.PSF_SPLINE_ORDER
    zoomed_mono_psf = ndimage.affine_transform(a, transform, (offset_y, offset_x),
                                       order=order)
    mses.append(((zoomed_mono_psf - a)**2).sum())
    print(d, mses[-1])

# plt.plot(deltas, mses)
# plt.show()
# exit()
class TestSettings:
    def __init__(self, defocus, p):
        self.defocus = defocus
        self.p = p
        self.mono = False
        self.plot = False
        self.dummy = False
        self.allow_cuda = conf.USE_CUDA
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
        self.cpu_gpu_arraysize_boundary = conf.CPU_GPU_ARRAYSIZE_BOUNDARY
        self.effective_q = None
        self.guide_mtf = None
        self.q_autosize_scalar = conf.Q_AUTOSIZE_SCALAR
        self.phase_autosize_scalar = conf.PHASE_AUTOSIZE_SCALAR
        self.cache_sizes = True
        self.x_loc = IMAGE_WIDTH / 2
        self.y_loc = IMAGE_HEIGHT / 2
        self.pixel_vignetting = True
        self.lens_vignetting = True
        self.default_exit_pupil_position_mm = 100
        self.exif = None
        self.fix_pupil_rotation = True

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

    def get_processing_details(self):
        get_processing_details(self)
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

    def copy_important_settings(self, s: TestSettings):
        self.fftsize = s.fftsize
        self.samples = s.phasesamples
        self.id_or_hash = s.id_or_hash

    def get_mtf(self):
        return abs(self.otf[0]), abs(self.otf[1])


def zoom(inarr, xfactor=1.0, yfactor=1.0, xoffset=0.0, yoffset=0.0, affine_transform=ndimage.affine_transform, me=np):
    shape = inarr.shape
    transform = me.array(((1.0 / yfactor, 0), (0, 1.0 / xfactor)))
    offset_x = (shape[0]) / 2 * (1.0 - 1.0 / xfactor) - xoffset / xfactor
    offset_y = (shape[1]) / 2 * (1.0 - 1.0 / yfactor) - yoffset / yfactor
    if me is cp:
        return affine_transform(inarr, transform, (offset_y, offset_x),
                                           order=1)
    else:
        real = affine_transform(inarr.real, transform, (offset_y, offset_x),
                                           order=conf.PSF_SPLINE_ORDER)
        imag = affine_transform(inarr.imag, transform, (offset_y, offset_x),
                                           order=conf.PSF_SPLINE_ORDER)
        return real + 1j * imag


def get_z9(p, modelwavelength):
    rel_wv = modelwavelength / BASE_WAVELENGTH
    spca = p.get('spca', 0.0) * 30
    spca2 = p.get('spca2', 0.0) * 30

    spcaz9 = (modelwavelength / BASE_WAVELENGTH - 1.0) * spca + spca * 0.028
    spca2z9 = (rel_wv - 1.0) ** 2 * spca2 * 10 - spca2 * 0.06
    return (p.get('z9', 0.0) + spcaz9 + spca2z9) * conf.BASE_WAVELENGTH


def get_z4(defocus, p, modelwavelength):
    fstop_base_ratio = p['fstop'] / p['base_fstop']
    rel_wv = modelwavelength / BASE_WAVELENGTH
    loca = p.get('loca', 0.0) * 30
    loca1 = p.get('loca1', 0.0) * 30

    locadefocus = (rel_wv - 1.0) ** 2 * 10 * loca - loca * 0.06
    loca1defocus = (rel_wv - 1.0) * 1 * loca1 + loca1 * 0.027
    base_z4 = ((defocus - p.get('df_offset', 0)) * p.get('df_step', 1)) * fstop_base_ratio ** 2
    # print(p)
    return -(base_z4 - locadefocus - loca1defocus) * conf.BASE_WAVELENGTH


def get_lca_shifts(s: TestSettings, modelwavelength, samplespacing):
    rel_wv = modelwavelength / 0.54

    img_height = calc_image_height(s.x_loc, s.y_loc)

    px = s.p.get('tca_slr', 0.0) * 1e2 * img_height
    py = 0

    shiftx = (rel_wv - 1.0) ** 2 * px * 10 - px / 14
    shifty = (rel_wv - 1.0) ** 2 * py * 10 - py / 14
    return shiftx / samplespacing / s.fftsize, shifty / samplespacing / s.fftsize


strehl_estimate_cache = None
mtf_cache = None
return_cache = None


def get_used_zernikes(iterable, cache=True, me=np):
    max = 0
    arr = me.zeros(48, dtype="int")
    idx = me.zeros(48, dtype="int") - 1
    used = []
    count = 0
    for item in iterable:
        if item[0].lower() == "z" and item[1].isdigit():
            print(item)
            zn = int(item[1:])
            arr[zn - 1] = 1
            if (zn) > max:
                max = zn
            idx[zn - 1] = count
            count += 1
            used.append(zn)
    arr[3] = 1
    print(arr, max, idx, 99)
    return arr, max, idx, used


def get_processing_details(s: TestSettings):
    if s.id_or_hash is not None and s.id_or_hash in settings_cache:
        stup = settings_cache[s.id_or_hash]
        if s.fftsize is None:
            s.fftsize = stup[0]
        if s.phasesamples is None:
            s.phasesamples = stup[1]
        if s.effective_q is None:
            s.effective_q = stup[2]
        s.allow_cuda = s.allow_cuda and s.fftsize > s.cpu_gpu_arraysize_boundary
        return s

    # if s.return_type == RETURN_LSF:
    #     s.fftsize = 128
    #     s.phasesamples = 64
    #     s.effective_q = 2
    #     return s

    minimum_q = np.clip((s.strehl_estimate * 4) * s.q_autosize_scalar, 2, 3)
    # minimum_q = np.clip((0.5 + s.strehl_estimate * 3.5) * s.q_autosize_scalar, 1.0, 5)
    min_samples = -np.inf
    f_stopped_down = s.p['fstop'] / s.p['base_fstop']
    if s.guide_mtf is None:
        min_samples = 384
    else:
        for otf in s.guide_mtf:
            freqs = np.arange(0, 65) / 64
            zero_plus_spacial_freqs = np.concatenate(([0], SPACIAL_FREQS, [1.0, 2.0]))
            interpotf_real = interpolate.InterpolatedUnivariateSpline(zero_plus_spacial_freqs, np.concatenate(([1.0], otf.real, [0,0])), k=2)(freqs)
            interpotf_imag = interpolate.InterpolatedUnivariateSpline(zero_plus_spacial_freqs, np.concatenate(([1.0], otf.imag, [0,0])), k=2)(freqs)
            interpotf = interpotf_real + 1j * interpotf_imag
            fftin = np.concatenate((interpotf[:-1], np.flip(interpotf[1:])))
            lsfshifted = np.abs(fftpack.ifft(fftin))
            lsfmax = np.maximum(lsfshifted[:64], np.flip(lsfshifted[64:]))
            lsfmax /= lsfmax.max()

            fitweights = np.clip((0.1 - lsfmax)*25, 0, 0.999) ** 4
            fitweights = (0.01 < lsfmax) * (lsfmax < 0.12)
            x_arr = np.arange(len(lsfmax))

            def cost(params, return_curve=False):
                a, b = params
                c = 0
                expcurve = b * np.exp(-0.1 * a * x_arr) + c
                if return_curve:
                    return expcurve
                return ((lsfmax - expcurve) ** 2 * fitweights).mean()

            a, b = optimize.minimize(cost, (1.0, 1.0,), bounds=((0.01, 70), (0.1, 30),)).x
            c = 0
            cutoff = 0.03

            needed_width = -10 / a * np.log((cutoff - c) / b)

            # plt.plot(cost((a, b), return_curve=True), label="fit")
            # plt.plot(lsfshifted / lsfshifted.max(), label="lsfshifted")
            # plt.plot(fftin, label="fftin")
            # plt.plot(lsfmax, label="lsfmax")
            # plt.ylim(0, 1)
            # plt.hlines([cutoff], 0, 64)
            # plt.legend()
            # plt.show()

            min_samples_this_axis = needed_width * s.phase_autosize_scalar * 9
            if min_samples_this_axis > min_samples:
                min_samples = min_samples_this_axis
    # min_samples = 64 + (1.0 - minmtf[3])**2 * 400

    # samples = int(min_samples / 2 + 1) * 2

    # for samples in CUDA_GOOD_FFT_SIZES: # limit sizes for cache reasons
    #     if samples >= min_samples:
    #         break
    for power in range(4, 10):
        samples = 2 ** power
        if samples > min_samples:
            break
        samples = int((2 ** power * 1.5) / 2 + 0.5) * 2
        if samples > min_samples:
            break

    effective_q_without_padding = f_stopped_down

    min_fftsize = minimum_q * samples / effective_q_without_padding


    for fftsize in (CUDA_GOOD_FFT_SIZES if s.allow_cuda else CPU_GOOD_FFT_SIZES):
        if fftsize >= min_fftsize:
            break

    # s.allow_cuda = s.allow_cuda and (samples + fftsize) >= (s.cpu_gpu_arraysize_boundary / 2)
    s.allow_cuda = s.allow_cuda and fftsize >= s.cpu_gpu_arraysize_boundary

    effective_q = fftsize / samples * effective_q_without_padding

    assert (fftsize - samples) % 2 == 0
    assert fftsize % 2 == 0
    assert samples % 2 == 0

    if s.fftsize is None:
        s.fftsize = fftsize
    if s.phasesamples is None:
        s.phasesamples = samples

    s.effective_q = effective_q

    if s.id_or_hash is not None:
        settings_cache[s.id_or_hash] = fftsize, samples, effective_q

    # s.fftsize = 1024
    # s.phasesamples = 512

    return s

# s = TestSettings(0, dict(fstop=5.6, base_fstop=1.4))
# s.phasesamples = 256
# get_processing_details(s)
# exit()


def try_wavefront(s: TestSettings):
    global window_cache

    orig_s = copy.copy(s)
    orig_s.p = s.p.copy()
    t = time.time()

    tr = TestResults()
    tr.copy_important_settings(s)
    if s.id_or_hash == 1:
        print("step", s.p['df_step'])

    if s.dummy:
        return tr

    if s.p['fstop'] < s.p['base_fstop']:
        raise ValueError("Base_fstop must be wider (lower) than fstop {} < {}".format(s.p['fstop'], s.p['base_fstop']))
    mul = 1
    mtfs = []
    pupilslices = []
    bestpupil = (np.inf, None, None)

    if not s.is_valid:
        s = get_processing_details(s)

    use_cuda = s.allow_cuda
    # prysm_path_q = 4
    # s.fftsize = prysm_path_q * s.phasesamples
    prysm_path_q = max(1, s.fftsize / s.phasesamples)


    if use_cuda:
        fft2 = cupyx.scipy.fftpack.fft2
        fftpack = cupyx.scipy.fftpack
        affine_transform = cupyx.scipy.ndimage.affine_transform
        # affine_transform = ndimage.affine_transform
    else:
        fft2 = scipyfftpack.fft2
        fftpack = scipyfftpack
        affine_transform = ndimage.affine_transform

    tr.used_cuda = use_cuda

    SAM_RADIOMETRIC_MODEL = True

    USE_PSF = False

    # cudadevice = cp.cuda.Device(0)

    def sync():
        pass
        # if use_cuda:
        #     cudadevice.synchronize()

    realdtype = "float32" if conf.PRECISION == 32 else "float64"
    complexdtype = "complex64" if conf.PRECISION == 32 else "complex128"

    me, engine_string = (cp, 'np') if use_cuda else (np, 'np')

    eval_wavelengths = [BASE_WAVELENGTH] if s.mono else MODEL_WVLS
    # eval_wavelengths = [BASE_WAVELENGTH] if s.mono else [0.5, 0.9]

    # s.p['loca'] *= 0.1
    # s.p['loca1'] *= 0.1
    # s.p['spca2'] *= 0.1
    # s.p['spca'] *= 0.1
    # s.p['tca_slr'] = 0
    if s.return_psf or USE_PSF or s.return_prysm_mtf:
        mono_psf_stack = me.zeros(s.fftshape, dtype="float64")

    psf_sag = np.zeros((s.fftsize, ), dtype="float64")
    psf_tan = np.zeros((s.fftsize, ), dtype="float64")
    psf_lst = []
    samplelst = []

    polychromatic_weights = np.array([float(photopic_fn(wv * 1e3) * d50_interpolator(wv)) for wv in eval_wavelengths])
    # polychromatic_weights **= 2
    # polychromatic_weights = me.array([0.5, 0.5])

    t_misc = 0
    t_pupils = 0
    t_get_phases = 0
    t_get_fcns = 0
    t_fcntransforms = 0
    t_pads = 0
    t_ffts = 0
    t_cudasyncs = 0
    t_affines = 0
    t_mtfs = 0
    t_init = time.time() - t
    t = time.time()

    mask = mask_pupil(s, engine=me, dtype=realdtype)

    if s.return_mask:
        tr.mask = mask

    t_maskmaking = time.time() - t

    used_zernikes_flags, max_zernike, cache_idx_, used_zernikes = get_used_zernikes(s.p.keys())

    if cache_idx[engine_string] is not None:
        if not me.all(cache_idx[engine_string] == cache_idx_):
            raise Exception("Zernicke cache does not match P dict")
    else:
        cache_idx[engine_string] = cache_idx_

    zkwargs = {}
    for key, value in s.p.items():

        # if key.upper() == 'Z9':
        #     zkwargs[key] = z9
        #     continue
        # if key.upper() == 'Z4':
        #     raise ValueError("No Z4 separately!")
        if key.lower().startswith('z') and key[1].isdigit():
            # if key not in ['z9', 'z16', 'z25', 'z36']:
            #     continue
            zkwargs[key] = value * mul * conf.BASE_WAVELENGTH

    z_arr_no_z4_z9 = np.zeros(used_zernikes_flags.sum(), dtype=realdtype)
    for key, value in zkwargs.items():
        idx = cache_idx['np'][int(key[1:]) - 1]
        z_arr_no_z4_z9[idx] = value
    z_arr_no_z4_z9[cache_idx[engine_string][3]] = 0
    z_arr_no_z4_z9[cache_idx[engine_string][8]] = 0
    print(zkwargs)
    print(zkwargs)
    print(zkwargs)
    print(zkwargs)
    print(z_arr_no_z4_z9)
    print(z_arr_no_z4_z9)
    print(z_arr_no_z4_z9)
    print(z_arr_no_z4_z9)
    print(z_arr_no_z4_z9)
    print(z_arr_no_z4_z9)
    print(z_arr_no_z4_z9)
    zhash = hash(tuple(z_arr_no_z4_z9))

    if me is not np:
        z_arr_no_z4_z9 = me.array(z_arr_no_z4_z9, dtype=realdtype)

    if me is cp:
        # cache = cupyzcache
        cache_array = cupyzcache_arrays
    else:
        # cache = zcache
        cache_array = numpyzcache_arrays

    for wvl_num, (model_wvl, polych_weight) in enumerate(zip(eval_wavelengths, polychromatic_weights)):
        t = time.time()
        rel_wv = model_wvl / BASE_WAVELENGTH

        z4 = get_z4(s.defocus, s.p, model_wvl)
        z9 = get_z9(s.p, model_wvl)

        samplelst.append(s.phasesamples)
        t_misc += time.time() - t
        if s.phasesamples not in cache_array:
            t = time.time()
            if s.phasesamples not in zcache:
                pupil = prysm.FringeZernike(used_zernikes_flags,
                                            dia=10, norm=False,
                                            wavelength=model_wvl,
                                            opd_unit="um",
                                            mask_target='none',
                                            samples=s.phasesamples, )  # This is just here to fill the coeff cache
            t_pupils += time.time() - t
            t = time.time()

            # if me is cp:
            #     cache[s.phasesamples] = {}
            cache_array[s.phasesamples] = me.empty((s.phasesamples, s.phasesamples, used_zernikes_flags.sum()), dtype=realdtype)
            # cache_array[s.phasesamples] = 1

            # for key, val in cache[s.phasesamples].items():
            if 1:
                idx = cache_idx['np'][key]
                # if me is cp:
                #     cache[s.phasesamples][key] = me.array(val)
                cache_array[s.phasesamples][:, :, idx] = me.array(val, dtype=realdtype)

                cache_idx[key] = idx
            print("graaagl")

            sync()
            t_get_phases += time.time() - t
        t = time.time()
        pupil = prysm.FringeZernike(dia=10, wavelength=model_wvl, norm=False,
                                    opd_unit="um",
                                    mask_target='none',
                                    samples=s.phasesamples, )
        t_pupils += time.time() - t
        t = time.time()

        if s.phasesamples not in all_but_z4_and_z9_phasecache[engine_string] or \
                all_but_z4_and_z9_phasecache[engine_string][s.phasesamples][0] != zhash:
            all_but_z4_and_z9_phasecache[engine_string][s.phasesamples] = zhash, cache_array[s.phasesamples] @ z_arr_no_z4_z9

        if me is cp:
            phase = cache_array[s.phasesamples][:, :, 3] * z4
            phase += all_but_z4_and_z9_phasecache[engine_string][s.phasesamples][1]
            phase += cache_array[s.phasesamples][:, :, 8] * z9
        else:
            # print(cache)
            phase = cache[s.phasesamples][3] * z4
            phase += all_but_z4_and_z9_phasecache[engine_string][s.phasesamples][1]
            phase += cache[s.phasesamples][8] * z9
        phase /= model_wvl
        sync()
        t_get_phases += time.time() - t

        t = time.time()
        # phase = me.array(pupil.change_phase_unit(to='waves', inplace=False),
        #                  dtype=realdtype)
        # phase = me.zeros((samples, samples), dtype=realdtype)
        wavefunction = me.exp(1j * 2 * me.pi * phase)
        wavefunction *= mask


        # if me is cp:
        #     plt.imshow(cp.asnumpy(me.angle(wavefunction)))
        #     plt.show()
        # else:
        #     plt.imshow(me.angle(wavefunction))
        #     plt.show()
        sync()
        t_get_fcns += time.time() - t

        if model_wvl == min(eval_wavelengths):
            if s.plot:
                mono_psf = prysm.PSF.from_pupil(pupil, efl=s.p['base_fstop'] * 10, Q=prysm_path_q)

            # psf_x_units, psf_y_units = prysm.propagation.prop_pupil_plane_to_psf_plane_units(wavefunction,
            #                                                              pupil.sample_spacing,
            #                                                              s.p['base_fstop'] * 10, model_wvl,
            #                                                              prysm_path_q)

            psf_sample_spacing = prysm.propagation.pupil_sample_to_psf_sample(pupil_sample=pupil.sample_spacing,
                                                          samples=s.fftsize,
                                                          wavelength=model_wvl,
                                                          efl=s.p['base_fstop'] * 10) * 1e-3
            psf_units = np.arange(-s.fftsize / 2, s.fftsize / 2) * psf_sample_spacing


        padpx = int((s.fftsize - s.phasesamples) / 2)

        ellip = s.p.get('ellip', 0)
        if ellip == 0:
            ellip = None
        else:
            xellip = np.clip(1 + ellip, 0.5, 1.0)
            yellip = np.clip(1.0 - ellip, 0.5, 1.0)

        if padpx > 0:
            pt = padpx, padpx
            if conf.ENABLE_PUPIL_DISTORTION and ellip is not None:
                t = time.time()
                wavefunction = zoom(wavefunction, xellip, yellip, affine_transform=affine_transform, me=me)
                t_fcntransforms += time.time() - t
            t = time.time()
            padded_cropped_pupil_fcn = me.pad(me.array(wavefunction, dtype=complexdtype), (pt, pt), mode="constant")
            t_pads += time.time() - t
        elif padpx < 0:
            t = time.time()
            padded_cropped_pupil_fcn = me.array(wavefunction[-padpx:-padpx+s.fftsize, -padpx:-padpx+s.fftsize], dtype=complexdtype)
            t_pads += time.time() - t
            if conf.ENABLE_PUPIL_DISTORTION and ellip is not None:
                t = time.time()
                padded_cropped_pupil_fcn = zoom(padded_cropped_pupil_fcn, xellip, yellip, affine_transform=affine_transform, me=me)
                t_fcntransforms += time.time() - t

        else:
            # padded_cropped_pupil_fcn = me.array(wavefunction, dtype=complexdtype)
            padded_cropped_pupil_fcn = wavefunction
            if conf.ENABLE_PUPIL_DISTORTION and ellip is not None:
                t = time.time()
                padded_cropped_pupil_fcn = zoom(padded_cropped_pupil_fcn, xellip, yellip, affine_transform=affine_transform, me=me)
                t_fcntransforms += time.time() - t
        t = time.time()
        pad = me.fft.fftshift(padded_cropped_pupil_fcn)
        sync()
        t_pads += time.time() - t
        t = time.time()

        # pad = me.fft.fft2(pad, norm='ortho')
        pad = fft2(pad, overwrite_x=True)
        shifted = me.fft.ifftshift(pad)
        impulse_response = me.absolute(shifted)
        impulse_response **= 2
        if not SAM_RADIOMETRIC_MODEL:
            impulse_response /= impulse_response.sum()
        impx = impulse_response.sum(axis=1)
        impy = impulse_response.sum(axis=0)

        sync()
        t_ffts += time.time() - t

        psf_size = impulse_response.shape[0]

        zoom_factor = float(np.clip(model_wvl / min(eval_wavelengths), 1.00001, np.inf))
        normshift_x, normshift_y = get_lca_shifts(s, model_wvl, psf_sample_spacing)

        if s.return_psf or USE_PSF:
            t = time.time()
            transform = me.array(((1.0 / zoom_factor, 0), (0, 1.0 / zoom_factor)))
            offset_x = (psf_size - 1 + 1) / 2 * (1.0 - 1.0 / zoom_factor - normshift_x)
            offset_y = (psf_size - 1 + 1) / 2 * (1.0 - 1.0 / zoom_factor - normshift_y)
            if use_cuda:
                order = 1
            else:
                order = conf.PSF_SPLINE_ORDER
            zoomed_mono_psf = affine_transform(impulse_response, transform, (offset_y, offset_x),
                                               order=order)
            if SAM_RADIOMETRIC_MODEL:
                zoomed_mono_psf *= polych_weight / zoomed_mono_psf.sum()
            else:
                zoomed_mono_psf *= polych_weight
            mono_psf_stack += zoomed_mono_psf
            sync()
            t_affines += time.time() - t
        # else:
        if 1:
            t = time.time()
            if me is cp:
                impx = cp.asnumpy(impx)
                impy = cp.asnumpy(impy)
            sync()
            t_cudasyncs += time.time() - t

            t = time.time()
            transform = np.array((1.0 / zoom_factor,))

            offset_x = (psf_size - 1 + 1) / 2 * (1.0 - 1.0 / zoom_factor - normshift_x)
            offset_y = (psf_size - 1 + 1) / 2 * (1.0 - 1.0 / zoom_factor - normshift_y)

            # impx /= impx.sum()
            # impy /= impy.sum()

            zoomx = ndimage.affine_transform(impx, transform, offset_x, order=conf.PSF_SPLINE_ORDER)
            zoomy = ndimage.affine_transform(impy, transform, offset_y, order=conf.PSF_SPLINE_ORDER)

            # zoomx = zoomed_mono_psf.mean(axis=0)
            # zoomy = zoomed_mono_psf.mean(axis=1)

            # plt.plot(zoomx / zoomx.max())
            # plt.plot(zoomx_ / zoomx_.max())
            # plt.show()

            # plt.plot(zoomy / zoomy.max())
            # plt.plot(zoomy_ / zoomy_.max())
            # plt.show()


            # print(model_wvl, s.id_or_hash)
            # plt.plot(psf_units, zoomx)
            # plt.plot(psf_units, zoomy)
            # plt.plot(psf_units, impx)
            # plt.plot(psf_units, impy)
            # plt.show()

            mul = 1.0 * zoom_factor
            # mul = 1

            if SAM_RADIOMETRIC_MODEL:
                # pass
                psf_sag += zoomx / zoomx.sum() * polych_weight
                psf_tan += zoomy / zoomy.sum() * polych_weight
            else:
                psf_sag += zoomx * polych_weight * mul
                psf_tan += zoomy * polych_weight * mul
            sync()
            t_affines += time.time() - t

        if s.plot:
            psf_lst.append(mono_psf)

        pupilslices.append(pupil.slice_x[1])

        metric = np.abs(rel_wv - 1)
        if metric < bestpupil[0]:
            bestpupil = metric, pupil, model_wvl, zkwargs

    t = time.time()

    if USE_PSF:
        psf_sag = mono_psf_stack.sum(axis=1)
        psf_tan = mono_psf_stack.sum(axis=0)
        if me is cp:
            psf_sag = cp.asnumpy(psf_sag)
            psf_tan = cp.asnumpy(psf_tan)
        # plt.plot(psf_sag / psf_sag.mean())
        # plt.plot(psf_sag_ / psf_sag_.mean())
        # plt.plot(((psf_sag / psf_sag.mean()) / (psf_sag_ / psf_sag_.mean()))[192:256+64])
        # plt.show()
    # ref_tr = _try_wavefront_prysmref(orig_s)

    if s.return_psf or s.return_prysm_mtf:
        npunits = cp.asnumpy(psf_units)
        prysm_psf = prysm.PSF(x=npunits, y=npunits, data=cp.asnumpy(mono_psf_stack))
        tr.psf = prysm_psf
        if 0:

            psf3 = prysm.PSF(x=npunits, y=npunits, data=cp.asnumpy(mono_psf_stack))
            psf2 = ref_tr.psf

            prysm_psf.data /= prysm_psf.data.sum()
            psf2.data /= psf2.data.sum()

            # assert np.allclose(prysm_psf.data, psf2.data)
            # assert np.allclose([prysm_psf.sample_spacing], [psf2.sample_spacing])
            psfnorm = prysm_psf.data / prysm_psf.data.sum()
            psf2norm = psf2.data / psf2.data.sum()
            psf3.data = np.clip(psfnorm / psf2norm, 0.3, 3)
            # print(psf3.data)
            # psf3.data[0,0] = max(psfnorm.max(), psf2norm.max())
            f, (a1, a2, a3) = plt.subplots(1, 3)

            tr.psf = prysm_psf
            prysm_psf.plot2d(axlim=18, ax=a1, fig=f)
            psf2.plot2d(axlim=18, ax=a2, fig=f)
            psf3.plot2d(axlim=18, ax=a3, fig=f)
            plt.show()
        # tr.otf = np.ones_like(SPACIAL_FREQS), np.ones_like(SPACIAL_FREQS)
        # return tr

    # plt.plot(psf_units, psf_sag / psf_sag.max()+ 0.1)
    # plt.plot(psf_units, mono_psf_stack.mean(axis=0) / mono_psf_stack.mean(axis=0).max())
    # plt.show()
    # plt.plot(psf_units, psf_tan / psf_tan.max() + 0.1)
    # plt.plot(psf_units, mono_psf_stack.mean(axis=1) / mono_psf_stack.mean(axis=1).max())
    # plt.show()

    # if s.return_type == RETURN_LSF:
    #     lsf_sag = mono_psf_stack.sum(axis=0)
    #     lsf_sag /= lsf_sag.max()
    #     lsf_tan = mono_psf_stack.sum(axis=1)
    #     lsf_tan /= lsf_tan.max()
    #     tr.lsf = lsf_sag, lsf_tan
    #     return tr

    centre = s.fftsize // 2
    mtf_x_units = prysm.fttools.forward_ft_unit(psf_sample_spacing * 1e-3, s.fftsize)
    sag_x = mtf_x_units[centre:]
    tan_x = mtf_x_units[centre:]

    if 0:
        if otf:
            mtf = me.fft.fftshift(me.fft.fft2(me.fft.ifftshift(mixedpsf)))
        # else:
    # mtf = me.absolute(me.fft.fft2(mono_psf_stack))
    # psf_y_units = prysm.fttools.forward_ft_unit((psf_x_units[1] - psf_x_units[0]) / 1e3, len(psf_y_units))
    # mtf = mtf / np.abs(mtf[0, 0])
    # sag_mod = mtf[0, :centre]
    # tan_mod = mtf[:centre, 0]

    # sag_mod = me.fft.fftshift(me.fft.fft(me.fft.ifftshift(psf_sag)))[centre:]
    # tan_mod = me.fft.fftshift(me.fft.fft(me.fft.ifftshift(psf_tan)))[centre:]

    # sag_mod /= np.abs(sag_mod)[0]
    # tan_mod /= np.abs(tan_mod)[0]

    mtf_mapper_fft_halfwindowsize_um = 16 * DEFAULT_PIXEL_SIZE * 1e6

    tukeykey = (s.fftsize, psf_sample_spacing)
    try:
        tukey_window = window_cache[tukeykey]
    except KeyError:
        if len(window_cache) > 300:
            window_cache = {}
        tukey_window = tukey(psf_units / mtf_mapper_fft_halfwindowsize_um, 0.6)
        window_cache[tukeykey] = tukey_window
        # print("PSF size {}um".format(psf_units[-1]*2), len(window_cache))

    # tukey_window = np.ones_like(psf_sag)

    # tukey_window = get_window(s.fftsize, psf_sample_spacing)
    # print("{:.3f} {:.3f}".format(np.sum(psf_sag * psf_units) / psf_sag.sum(), np.sum(psf_tan * psf_units) / psf_tan.sum()))
    sag_mod = normalised_centreing_fft(psf_sag * tukey_window, fftpack=scipyfftpack, engine=np)[:centre]
    tan_mod = normalised_centreing_fft(psf_tan * tukey_window, fftpack=scipyfftpack, engine=np)[:centre]

    # sag_mod = abs(fftpack.fft(np.fft.fftshift(psf_sag))[:centre])
    # tan_mod = abs(fftpack.fft(np.fft.fftshift(psf_tan))[:centre])
    # try:
    #     sag_mod /= sag_mod[0]
    # except FloatingPointError:
    #     sag_mod = np.zeros(centre)
    # try:
    #     tan_mod /= tan_mod[0]
    # except FloatingPointError:
    #     tan_mod = np.zeros(centre)
    # tan_mod /= tan_mod[0]

    # ref_mtf = ref_tr._prysm_mtf
    # (p_sag_x, p_sag_mod), (p_tan_x, p_tan_mod) = ref_mtf

    # assert np.allclose(p_sag_mod, abs(sag_mod))
    # assert np.allclose(p_tan_mod, abs(tan_mod))

    # plt.plot(p_sag_x, p_sag_mod)
    # plt.plot(sag_x, abs(sag_mod))
    # plt.show()
    # plt.plot(sag_x, abs(sag_mod) / p_sag_mod)
    # plt.show()
    # plt.plot(p_tan_x, p_tan_mod)
    # plt.plot(tan_x, abs(tan_mod))
    # plt.show()
    # plt.plot(tan_x, abs(tan_mod) / p_tan_mod)
    # plt.show()
    if s.return_prysm_mtf:
        prysm_mtf = prysm.MTF.from_psf(prysm_psf)
        tr.prysm_mtf = prysm_mtf

    if s.return_otf and s.return_otf_mtf:
        interpfn = interpolate.InterpolatedUnivariateSpline(sag_x, np.abs(sag_mod), k=1)
        sagmtf = interpfn(SPACIAL_FREQS / DEFAULT_PIXEL_SIZE * 1e-3)
        interpfn = interpolate.InterpolatedUnivariateSpline(tan_x, np.abs(tan_mod), k=1)
        tanmtf = interpfn(SPACIAL_FREQS / DEFAULT_PIXEL_SIZE * 1e-3)
        tr.otf = sagmtf, tanmtf

        # plt.plot(abs(tr.otf[0]))
        # plt.plot(ref_tr.otf[0])
        # plt.show()
        # plt.plot(abs(tr.otf[1]))
        # plt.plot(ref_tr.otf[1])
        # plt.show()

    if s.return_otf and not s.return_otf_mtf:
        interpfn = interpolate.InterpolatedUnivariateSpline(sag_x, np.real(sag_mod), k=1)
        sagmtf = interpfn(SPACIAL_FREQS / DEFAULT_PIXEL_SIZE * 1e-3)
        interpfn = interpolate.InterpolatedUnivariateSpline(tan_x, np.real(tan_mod), k=1)
        tanmtf = interpfn(SPACIAL_FREQS / DEFAULT_PIXEL_SIZE * 1e-3)
        interpfn = interpolate.InterpolatedUnivariateSpline(sag_x, np.imag(sag_mod), k=1)
        sagmtf_i = interpfn(SPACIAL_FREQS / DEFAULT_PIXEL_SIZE * 1e-3)
        interpfn = interpolate.InterpolatedUnivariateSpline(tan_x, np.imag(tan_mod), k=1)
        tanmtf_i = interpfn(SPACIAL_FREQS / DEFAULT_PIXEL_SIZE * 1e-3)
        # sagmtf = sagmtf + sagmtf_i * 1j
        # tanmtf = tanmtf + tanmtf_i * 1j

        # Normalise phase somehow

        # tr.otf = normalised_fft(sagmtf, sagmtf_i, SPACIAL_FREQS, inc_neg_freqs=False, return_type=COMPLEX_CARTESIAN), \
        #          normalised_fft(tanmtf, tanmtf_i, SPACIAL_FREQS, inc_neg_freqs=False, return_type=COMPLEX_CARTESIAN)

        # if s.id_or_hash == 0:
        #     normalised_fft(sagmtf, sagmtf_i, SPACIAL_FREQS, inc_neg_freqs=False,
        #                    return_type=COMPLEX_CARTESIAN, plot=True)
            # plt.plot(tuple(psf_sag))
            # plt.plot(tuple(psf_tan))
            # plt.show()
            # plt.plot(sagmtf, color='red')
            # plt.plot(tr.otf[0].real, '--', color='red')
            # plt.plot(tanmtf, color='green')
            # plt.plot(tr.otf[1].real, '--', color='green')
            # plt.plot(sagmtf_i, color='orange')
            # plt.plot(tr.otf[0].imag, '--', color='orange')
            # plt.plot(tanmtf_i, color='blue')
            # plt.plot(tr.otf[1].imag, '--', color='blue')
            # plt.show()

        tr.otf = sagmtf + 1j * sagmtf_i, tanmtf + 1j * tanmtf_i


    sync()
    t_mtfs += time.time() - t

    timings = dict(t_init=t_init,
                   t_maskmaking=t_maskmaking,
                   t_pupils=t_pupils,
                   t_get_phases=t_get_phases,
                   t_get_fcns=t_get_fcns,
                   t_fcntransforms=t_fcntransforms,
                   t_pads=t_pads,
                   t_ffts=t_ffts,
                   t_cudasyncs=t_cudasyncs,
                   t_affines=t_affines,
                   t_mtfs=t_mtfs,
                   t_misc=t_misc)

    tr.timings = timings

    sliceavg = me.average(me.array(pupilslices, dtype='float64'), axis=0, weights=polychromatic_weights)

    slice_ = bestpupil[1].slice_x[1]
    slicedv = me.abs(me.diff(sliceavg[me.isfinite(sliceavg)]))
    if slicedv.sum() != 0:
        peakiness = slicedv.max() / slicedv.mean()
    else:
        peakiness = 999.0
    strehl = float(bestpupil[1].strehl)

    if s.plot:
        # Plot each Z phase separately
        for key, value in bestpupil[3].items():
            if value != 0:
                pupil = prysm.FringeZernike(dia=10, norm=False,
                                            wavelength=bestpupil[2],
                                            opd_unit="um",
                                            samples=samples,
                                            **{key: value})
                pupil = mask_pupil(pupil, p['base_fstop'], p['fstop'], engine=me)
                slice = pupil.slice_x[1]
                rms = (slice[me.isfinite(slice)] ** 2).mean() ** 0.5
                plt.plot(cp.asnumpy(slice), label="{} : {:.3f} λRMS".format(key, rms / conf.BASE_WAVELENGTH))
        slice_ = bestpupil[1].slice_x[1]
        rms = (slice_[me.isfinite(slice_)] ** 2).mean() ** 0.5
        plt.plot(slice_, label="All : {:.3f} λRMS".format(rms / conf.BASE_WAVELENGTH))
        pupil = prysm.FringeZernike(z4=z4,
                                    dia=10, norm=False,
                                  wavelength=bestpupil[2],
                                  opd_unit="um",
                                  samples=samples,
                                  **bestpupil[3])

        pupil = mask_pupil(pupil, p['base_fstop'], p['fstop'])
        slice_ = pupil.slice_x[1]
        rms = (slice_[me.isfinite(slice_)] ** 2).mean() ** 0.5
        plt.plot(slice_, '--', label="ZeroZ4 : {:.3f} λRMS".format(rms / conf.BASE_WAVELENGTH), color='black')
        plt.legend()
        plt.show()
        mono_psf = prysm.PSF.from_pupil(pupil, efl=p['base_fstop']*10)
        mono_psf.plot2d(axlim=8)
        plt.show()

        slicedv = np.abs(np.diff(sliceavg[me.isfinite(slice_)]))
        peakiness = slicedv.max() / slicedv.mean()

        plt.plot(np.diff(slice_), label="Pupil slice derivative at best focus {:.3f}".format(peakiness))
        plt.legend()
        plt.show()

    return tr


def _try_wavefront_prysmref(s: TestSettings):
    t = time.time()

    tr = TestResults()
    tr.copy_important_settings(s)

    if s.dummy:
        return tr

    if s.p['fstop'] < s.p['base_fstop']:
        raise ValueError("Base_fstop must be wider (lower) than fstop {} < {}".format(s.p['fstop'], s.p['base_fstop']))
    pupilslices = []

    if not s.is_valid:
        s = get_processing_details(s)

    use_cuda = False

    if prysm.config.backend != np:
        prysm.config.backend = np
        prysm.config.precision = conf.PRECISION


    tr.used_cuda = use_cuda

    realdtype = "float32" if conf.PRECISION == 32 else "float64"
    complexdtype = "complex64" if conf.PRECISION == 32 else "complex128"

    me, mestr = (cp, 'np') if use_cuda else (np, 'np')

    eval_wavelengths = [BASE_WAVELENGTH] if s.mono else MODEL_WVLS

    # eval_wavelengths = [BASE_WAVELENGTH] if s.mono else [0.5, 0.9]

    # s.p['loca'] *= 0.1
    # s.p['loca1'] *= 0.1
    # s.p['spca2'] *= 0.1
    # s.p['spca'] *= 0.1

    psf_lst = []

    # s.fftsize = s.phasesamples * 4

    prysm_path_q = max(1, s.fftsize / s.phasesamples)

    samplelst = []
    print(s.id_or_hash)

    polychromatic_weights = me.array([float(photopic_fn(wv * 1e3) * d50_interpolator(wv)) for wv in eval_wavelengths])
    # polychromatic_weights = me.array([0.5, 0.5])
    t_misc = 0
    t_pupils = 0
    t_get_phases = 0
    t_get_fcns = 0
    t_pads = 0
    t_ffts = 0
    t_cudasyncs = 0
    t_affines = 0
    t_mtfs = 0
    t_init = time.time() - t
    t = time.time()

    mask = mask_pupil(s, engine=me, dtype=realdtype)

    t_maskmaking = time.time() - t

    bestpupil = None

    zkwargs = {}
    for key, value in s.p.items():
        if key.lower().startswith('z') and key[1].isdigit():
            if key.lower() != "z4" and key.lower() != "z9":
                zkwargs[key] = value * conf.BASE_WAVELENGTH

    for wvl_num, (model_wvl, polych_weight) in enumerate(zip(eval_wavelengths, polychromatic_weights)):
        t = time.time()
        rel_wv = model_wvl / BASE_WAVELENGTH

        z4 = get_z4(s.defocus, s.p, model_wvl)
        z9 = get_z9(s.p, model_wvl)

        samplelst.append(s.phasesamples)
        t_misc += time.time() - t
        t = time.time()
        pupil = prysm.FringeZernike(z4=z4, z9=z9, dia=10, norm=False,
                                    wavelength=model_wvl,
                                    opd_unit="um",
                                    mask_target='fcn',
                                    mask=mask,
                                    samples=s.phasesamples, **zkwargs)
        t_pupils += time.time() - t
        t = time.time()

        t_get_phases += time.time() - t
        t = time.time()

        t_get_phases += time.time() - t

        t = time.time()
        mono_psf = prysm.PSF.from_pupil(pupil, efl=s.p['base_fstop'] * 10, Q=prysm_path_q, norm='radiometric')
        # plt.plot(np.log10(mono_psf.data[int(mono_psf.data.shape[0]/2), :]))
        # mono_psf.plot2d()
        # plt.show()

        t_ffts += time.time() - t

        psf_lst.append(mono_psf)

        pupilslices.append(pupil.slice_x[1])

        metric = np.abs(rel_wv - 1)
        if bestpupil is None or metric < bestpupil[0]:
            bestpupil = metric, pupil, model_wvl, zkwargs

    t = time.time()

    mixedpsf = prysm.PSF.polychromatic(psf_lst, me.array(polychromatic_weights))
    tr.psf = mixedpsf
    # print(s.id_or_hash)
    # tr2 = _try_wavefront_prysmref(s)
    # mixedpsf.plot2d(axlim=18)
    # ax = plt.gca()
    # ax.set_title("A")
    # plt.show()
    # tr.otf = np.ones_like(SPACIAL_FREQS), np.ones_like(SPACIAL_FREQS)

    t_ffts += time.time() - t

    t = time.time()
    freqs = SPACIAL_FREQS / DEFAULT_PIXEL_SIZE * 1e-3
    mtf = prysm.MTF.from_psf(mixedpsf)
    tr._prysm_mtf = mtf.sag, mtf.tan
    tr.otf = mtf.exact_sag(freqs), mtf.exact_tan(freqs)
    # tr._prysm_otf = tr.otf
    if 0:
        centre = s.fftsize // 2

        otf = me.fft.fftshift(me.fft.fft2(me.fft.ifftshift(mixedpsf.data)))

        f_units = prysm.fttools.forward_ft_unit(mixedpsf.sample_spacing, len(mixedpsf.x))
        otf /= np.abs(otf[centre, centre])
        sag_mod = otf[centre, centre:]
        tan_mod = otf[centre:, centre]
        f_units = f_units[centre:]

        interpfn = interpolate.InterpolatedUnivariateSpline(f_units, np.real(sag_mod), k=1)
        sagmtf = interpfn(SPACIAL_FREQS / DEFAULT_PIXEL_SIZE * 1e-3)
        interpfn = interpolate.InterpolatedUnivariateSpline(f_units, np.real(tan_mod), k=1)
        tanmtf = interpfn(SPACIAL_FREQS / DEFAULT_PIXEL_SIZE * 1e-3)
        interpfn = interpolate.InterpolatedUnivariateSpline(f_units, np.imag(sag_mod), k=1)
        sagmtf_i = interpfn(SPACIAL_FREQS / DEFAULT_PIXEL_SIZE * 1e-3)
        interpfn = interpolate.InterpolatedUnivariateSpline(f_units, np.imag(tan_mod), k=1)
        tanmtf_i = interpfn(SPACIAL_FREQS / DEFAULT_PIXEL_SIZE * 1e-3)

        tr.otf = sagmtf + 1j * sagmtf_i, tanmtf + 1j * tanmtf_i

    t_mtfs += time.time() - t

    timings = dict(t_init=t_init,
                   t_maskmaking=t_maskmaking,
                   t_pupils=t_pupils,
                   t_get_phases=t_get_phases,
                   t_get_fcns=t_get_fcns,
                   t_pads=t_pads,
                   t_ffts=t_ffts,
                   t_cudasyncs=t_cudasyncs,
                   t_affines=t_affines,
                   t_mtfs=t_mtfs,
                   t_misc=t_misc)

    tr.timings = timings

    sliceavg = me.average(me.array(pupilslices, dtype='float64'), axis=0, weights=polychromatic_weights)

    slice_ = bestpupil[1].slice_x[1]
    slicedv = me.abs(me.diff(sliceavg[me.isfinite(sliceavg)]))
    if slicedv.sum() != 0:
        peakiness = slicedv.max() / slicedv.mean()
    else:
        peakiness = 999.0
    # strehl = float(bestpupil[1].strehl)
    strehl = 1

    if s.plot:
        # Plot each Z phase separately
        for key, value in bestpupil[3].items():
            if value != 0:
                pupil = prysm.FringeZernike(dia=10, norm=False,
                                            wavelength=bestpupil[2],
                                            opd_unit="um",
                                            samples=samples,
                                            **{key: value})
                pupil = mask_pupil(pupil, p['base_fstop'], p['fstop'], engine=me)
                slice = pupil.slice_x[1]
                rms = (slice[me.isfinite(slice)] ** 2).mean() ** 0.5
                plt.plot(cp.asnumpy(slice), label="{} : {:.3f} λRMS".format(key, rms / conf.BASE_WAVELENGTH))
        slice_ = bestpupil[1].slice_x[1]
        rms = (slice_[me.isfinite(slice_)] ** 2).mean() ** 0.5
        plt.plot(slice_, label="All : {:.3f} λRMS".format(rms / conf.BASE_WAVELENGTH))
        pupil = prysm.FringeZernike(z4=z4,
                                    dia=10, norm=False,
                                    wavelength=bestpupil[2],
                                    opd_unit="um",
                                    samples=samples,
                                    **bestpupil[3])

        pupil = mask_pupil(pupil, p['base_fstop'], p['fstop'])
        slice_ = pupil.slice_x[1]
        rms = (slice_[me.isfinite(slice_)] ** 2).mean() ** 0.5
        plt.plot(slice_, '--', label="ZeroZ4 : {:.3f} λRMS".format(rms / conf.BASE_WAVELENGTH), color='black')
        plt.legend()
        plt.show()
        mono_psf = prysm.PSF.from_pupil(pupil, efl=p['base_fstop']*10)
        mono_psf.plot2d(axlim=8)
        plt.show()

        slicedv = np.abs(np.diff(sliceavg[me.isfinite(slice_)]))
        peakiness = slicedv.max() / slicedv.mean()

        plt.plot(np.diff(slice_), label="Pupil slice derivative at best focus {:.3f}".format(peakiness))
        plt.legend()
        plt.show()

    return tr


tempcache = {}


def mask_pupil(s: TestSettings, engine=np, dtype="float64", plot=False):
    # s.p['v_rad'] = 1.0
    hashtuple = (s.p['base_fstop'],
                 s.p['fstop'],
                 s.x_loc,
                 s.y_loc,
                 s.phasesamples,
                 s.p.get('a', 0.0),
                 s.p.get('b', 0.0),
                 s.p.get('v_scr', 1.0),
                 s.p.get('v_rad', 1.0),
                 # s.p.get('v_x', 0.0),
                 # s.p.get('v_y', 0.0),
                 s.p.get('squariness', 0.5),
                 "np" if engine is np else "cp",
                 dtype)

    hash = hashtuple.__hash__()

    for cachehash, mask in mask_cache:
        if cachehash == hash:
            return mask

    smoothfactor = s.phasesamples / 1.5

    me = engine

    aperture_stop_norm_radius = s.p['base_fstop'] / s.p['fstop']

    na = 1 / (2.0 * s.p['base_fstop'])
    onaxis_peripheral_ray_angle = me.arcsin(na, dtype=dtype)

    pupil_radius_mm = me.tan(onaxis_peripheral_ray_angle, dtype=dtype) * s.default_exit_pupil_position_mm

    x_displacement_mm = (s.x_loc - IMAGE_WIDTH / 2) * DEFAULT_PIXEL_SIZE * 1e3
    y_displacement_mm = (s.y_loc - IMAGE_HEIGHT / 2) * DEFAULT_PIXEL_SIZE * 1e3

    # angle = s.p.get('v_angle', 0)
    magnitude = (x_displacement_mm ** 2 + y_displacement_mm ** 2) ** 0.5

    if s.fix_pupil_rotation:
        x_displacement_mm = -magnitude
        y_displacement_mm = 0

    x_displacement_mm_min = -x_displacement_mm - pupil_radius_mm
    x_displacement_mm_max = -x_displacement_mm + pupil_radius_mm
    y_displacement_mm_min = -y_displacement_mm - pupil_radius_mm
    y_displacement_mm_max = -y_displacement_mm + pupil_radius_mm

    x = me.linspace(x_displacement_mm_min, x_displacement_mm_max, s.phasesamples, dtype=dtype)
    y = me.linspace(y_displacement_mm_min, y_displacement_mm_max, s.phasesamples, dtype=dtype)
    gridx, gridy = me.meshgrid(x, y)
    displacement_grid = (gridx**2 + gridy**2) ** 0.5
    squariness = (2**0.5 - displacement_grid / me.maximum(abs(gridx), abs(gridy))) ** 2
    pixel_angle_grid = me.arctan(displacement_grid / s.default_exit_pupil_position_mm *
                                 (1.0 + squariness * s.p.get('squariness', 0.5)), dtype=dtype)

    normarr = me.linspace(-1, 1, s.phasesamples, dtype=dtype)
    gridx, gridy = me.meshgrid(normarr, normarr)
    pupil_norm_radius_grid = (gridx ** 2 + gridy ** 2) ** 0.5

    stopmask = np.clip( (aperture_stop_norm_radius - pupil_norm_radius_grid) * smoothfactor + 0.5, 0, 1)

    a = s.p.get('a', 1.0)
    b = s.p.get('b', 1.0)

    if not s.pixel_vignetting:
        mask = stopmask
    else:
        coeff_4 = -18.73 * a
        corff_6 = 485 * b
        square_grid = 1.0 / (1.0 + (pixel_angle_grid**4 * coeff_4 + pixel_angle_grid ** 6 * corff_6))
        mask = stopmask * square_grid

    if s.lens_vignetting:
        # Lens vignette
        normarr = me.linspace(-1, 1, s.phasesamples, dtype=dtype)
        image_circle_modifier = s.p.get('v_slr', 1.0) * 0.6
        gridx, gridy = me.meshgrid(normarr - x_displacement_mm / SENSOR_WIDTH * 1e-3 * image_circle_modifier,
                                   normarr - y_displacement_mm / SENSOR_WIDTH * 1e-3 * image_circle_modifier)
        vignette_radius_grid = (gridx ** 2 + gridy ** 2) ** 0.5
        vignette_crop_circle_radius = s.p.get('v_rad', 1.0)
        vignette_mask = vignette_radius_grid < vignette_crop_circle_radius
        vignette_mask = np.clip((vignette_crop_circle_radius - vignette_radius_grid) * smoothfactor + 0.5, 0, 1)
        mask *= vignette_mask

        normarr = me.linspace(-1, 1, s.phasesamples, dtype=dtype)
        image_circle_modifier = s.p.get('v_slr', 1.0) * 0.6
        gridx, gridy = me.meshgrid(normarr + x_displacement_mm / SENSOR_WIDTH * 1e-3 * image_circle_modifier,
                                   normarr + y_displacement_mm / SENSOR_WIDTH * 1e-3 * image_circle_modifier)
        vignette_radius_grid = (gridx ** 2 + gridy ** 2) ** 0.5
        vignette_crop_circle_radius = s.p.get('v_rad', 1.0) * 1.0
        vignette_mask = vignette_radius_grid < vignette_crop_circle_radius
        vignette_mask = np.clip((vignette_crop_circle_radius - vignette_radius_grid) * smoothfactor + 0.5, 0, 1)
        mask *= vignette_mask

    if plot or s.id_or_hash == -1:
        if engine is cp:
            print(square_grid)
            plt.imshow(cp.asnumpy(mask))
            plt.colorbar()
            plt.show()
        else:
            print(square_grid)
            plt.imshow(mask)
            plt.colorbar()
            plt.show()
    return mask


def plot_pixel_vignetting_loss():
    fstops = 2.0 ** np.linspace(0.0, 1.5, 6)
    test_fstops = (1, 2**0.16667, 1.222, 2**0.5, 1.4 * 2**0.166667, 2, 2 * 2 ** 0.166667, 2 * 2**0.5)
    benefits_exp = (1.93,1.90, 1.81, 1.70, 1.5, 0.93, 0.62, 0)

    s = TestSettings(0, dict(base_fstop=1.0, fstop=2.0 * 2**0.5))
    s.phasesamples = 256
    s.pixel_vignetting = False
    baseline = mask_pupil(s, np).mean()

    s.pixel_vignetting = True

    if "optimise" and 0:
        def callable(x):
            error = 0
            for testfstop, benefit_exp in zip(test_fstops, benefits_exp):
                s.p = dict(base_fstop=1, fstop=testfstop)
                s.p['a'], s.p['b'] = x
                benefit = np.log2(mask_pupil(s, np).mean() / baseline)
                print(x[0], x[1], testfstop, benefit)
                error += (benefit - benefit_exp) ** 2
            print()
            return error

        opt = optimize.minimize(callable, (0, 0))  # ,bounds=((-20,20), (-30, 30)
        a, b = opt.x
    else:
        a, b = 0, 0

    plot_fstops = 2**np.linspace(0, 1.5, 6)
    benefits = []
    benefits_stop = []
    for testfstop in plot_fstops:
        s.p = dict(base_fstop=testfstop, fstop=testfstop)
        s.p['a'], s.p['b'] = a, b
        benefit = np.log2(mask_pupil(s, np, plot=True).mean()/(testfstop**2) / baseline)
        benefits.append(benefit)
        s.p = dict(base_fstop=1, fstop=testfstop)
        s.p['a'], s.p['b'] = a, b
        benefit = np.log2(mask_pupil(s, np, plot=True).mean() / baseline)
        benefits_stop.append(benefit)

    plt.plot(plot_fstops, benefits)
    plt.plot(plot_fstops, benefits_stop)
    plt.plot(test_fstops, benefits_exp)
    plt.plot()
    plt.show()
    exit()


def plot_lens_vignetting_loss(base_fstop=1.4):
    fstops = 2.0 ** np.linspace(0.0, 2, 4)
    for stop in fstops:
        s = TestSettings(0, dict(base_fstop=base_fstop, fstop=stop * base_fstop))
        s.phasesamples = 128
        s.pixel_vignetting = True
        s.lens_vignetting = True
        s.p['v_mag'] = 0.8
        s.p['v_rad'] = 1.3
        s.p['v_x'] = -0.8
        s.p['v_y'] = -0.8
        baseline = mask_pupil(s, np).mean()
        heights = np.linspace(0, 1, 16)
        losses = []
        s.x_loc = 0
        s.y_loc = 0
        mask_pupil(s, np, plot=True)
        for height in heights:
            s.x_loc = 3000 + height * IMAGE_WIDTH / 2
            s.y_loc = 2000 + height * IMAGE_HEIGHT / 2
            # losses.append(np.log2(mask_pupil(s, np).mean() / baseline))
            losses.append(mask_pupil(s, np).mean() / baseline)
        plt.plot(heights, losses)
    plt.show()

# from lentil import wavefront_test
# s = TestSettings(0, dict(base_fstop=2.6, fstop=2.6))
# s.phasesamples = 256
# s.return_type = RETURN_MTF
# s.p['tca_slr'] = 0
#
# s.p['v_slr'] =2.6
# s.x_loc = 5600
# s.y_loc = 3800
# s.x_loc = 6000-5600
# s.y_loc = 4000-3800
# s.p['z5'] = -1
# s.p['df_offset'] = 0
# s.p['df_step'] = 1
# s.default_exit_pupil_position_mm = 1000
# s.p['ellip'] = 0
# s.allow_cuda = False
# s.p['loca'] = 0.0000001
# s.defocus = 0
# s.mono = True

# mask_pupil(s, plot=True)
# tr = try_wavefront(s)
# plt.plot(tr.otf[0], label='sag')
# plt.plot(tr.otf[1], label='tan')
# plt.legend()
# print(tr.otf[0])
# psf = tr.psf
# psf.plot2d(axlim=30)
# plt.show()
# exit()y



pool = multiprocessing.Pool(processes=8)

def testmul(samples, total_loops=2*5, Q=2):
    s = TestSettings(0, dict(fstop=1.4, base_fstop=1.4))
    s.phasesamples = 256
    s.fftsize = 512
    s.allow_cuda = False
    pool.starmap(try_wavefront, [(s, )] * 16 * 5)
