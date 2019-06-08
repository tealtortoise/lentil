import time
from collections import deque
import cupy as cp
import cupyx.scipy.ndimage
import cupyx.scipy.fftpack
import numpy as np
import prysm
from matplotlib import pyplot
from scipy import interpolate, ndimage, fftpack, optimize
import matplotlib.pyplot as plt

from lentil import wavefront_config
from lentil.wavefront_config import SPACIAL_FREQS, BASE_WAVELENGTH, MODEL_WVLS, DEFAULT_SAMPLES
from lentil.constants_utils import *


# prysm.config.backend = 'cuda'
# prysm.config.precision = wavefront_config.PRECISION

zcache = prysm.zernike.zcache.regular
cupyzcache = {}

RETURN_MTF = 1
RETURN_OTF = 2
RETURN_LSF = 3
RETURN_PSF = 4
RETURN_WITH_PROCESSING_DETAILS = 5

settings_cache = {}

mask_cache = deque(maxlen=wavefront_config.MASK_CACHE_SIZE)

class TestSettings:
    def __init__(self, defocus, p):
        self.defocus = defocus
        self.p = p
        self.mono = False
        self.plot = False
        self.dummy = False
        self.allow_cuda = wavefront_config.USE_CUDA
        self.id_or_hash = 0
        self.strehl_estimate = 1.0
        self.fftsize = None
        self.phasesamples = None
        self.return_type = RETURN_OTF
        self.cpu_gpu_fftsize_boundary = wavefront_config.CPU_GPU_FFTSIZE_BOUNDARY
        self.effective_q = None
        self.guide_mtf = None
        self.samples_autosize_scalar = 1.0
        self.fftsize_autosize_scalar = 1.0
        self.cache_sizes = True
        self.x_loc = IMAGE_WIDTH / 2
        self.y_loc = IMAGE_HEIGHT / 2
        self.pixel_vignetting = True
        self.lens_vignetting = True
        self.default_exit_pupil_position_mm = 90
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

# def try_wavefront(defocus, p, mono=False, plot=False, dummy=False, use_cuda=True, index=None,
#                   strehl_estimate=1.0, mtf_mean=0.7, fftsize_override=None, samples_override=None, lsf=False, otf=True,
#                   return_psf=False, boundary=wavefront_config.CPU_GPU_FFTSIZE_BOUNDARY, mtf=None, gpd=None):
#     pass


def get_z9(p, modelwavelength):
    rel_wv = modelwavelength / BASE_WAVELENGTH
    spcaz9 = (modelwavelength / BASE_WAVELENGTH - 1.0) * p.get('spca', 0.0) * 2
    spca2z9 = (rel_wv - 1.0) ** 2 * p.get('spca2', 0.0) * 10
    return (p.get('z9', 0.0) + spcaz9 + spca2z9) * wavefront_config.BASE_WAVELENGTH


def get_z4(defocus, p, modelwavelength):
    fstop_base_ratio = p['fstop'] / p['base_fstop']
    rel_wv = modelwavelength / BASE_WAVELENGTH
    locadefocus = (rel_wv - 1.0) ** 2 * 10 * p.get('loca', 0.0)
    loca1defocus = (rel_wv - 1.0) * 2 * p.get('loca1', 0.0)
    return -((defocus - p['df_offset']) * p['df_step'] * fstop_base_ratio ** 2 - locadefocus - loca1defocus) * wavefront_config.BASE_WAVELENGTH


strehl_estimate_cache = None
mtf_cache = None
return_cache = None


def get_processing_details(s: TestSettings):
    if s.id_or_hash is not None and s.id_or_hash in settings_cache:
        stup = settings_cache[s.id_or_hash]
        if s.fftsize is None:
            s.fftsize = stup[0]
        if s.phasesamples is None:
            s.phasesamples = stup[1]
        if s.effective_q is None:
            s.effective_q = stup[2]
        s.allow_cuda = s.allow_cuda and s.fftsize > s.cpu_gpu_fftsize_boundary
        return s

    if s.return_type == RETURN_LSF:
        s.fftsize = 128
        s.phasesamples = 64
        s.effective_q = 2
        return s

    minimum_q = np.clip((0.5 + s.strehl_estimate * 3.5), 1.0, 4) * s.samples_autosize_scalar
    min_samples = -np.inf
    f_stopped_down = s.p['fstop'] / s.p['base_fstop']
    if s.guide_mtf is None:
        min_samples = 384
    else:
        for otf in s.guide_mtf:
            pad = np.pad(otf, (0, 64-len(otf)), mode='constant')

            lsfshifted = np.real(fftpack.ifft(np.concatenate((pad, np.flip(pad)))))
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

            a, b = optimize.minimize(cost, (1.0, 1.0,), bounds=((0.01, 100), (0.1, 30),)).x
            c = 0
            cutoff = 0.03

            needed_width = -10 / a * np.log((cutoff - c) / b)

            # plt.plot(cost((a, b), return_curve=True))
            # plt.plot(lsfmax)
            # plt.ylim(0, 0.1)
            # plt.hlines([cutoff], 0, 64)
            # plt.show()
            min_samples_this_axis = needed_width * s.fftsize_autosize_scalar * 9
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

    s.allow_cuda = s.allow_cuda and fftsize >= s.cpu_gpu_fftsize_boundary

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

    return s


# def try_wavefront(defocus, p, mono=False, plot=False, dummy=False, use_cuda=True, index=None,
#                   strehl_estimate=1.0, mtf_mean=0.7, fftsize_override=None, samples_override=None, lsf=False, otf=True,
#                   return_psf=False, boundary=wavefront_config.CPU_GPU_FFTSIZE_BOUNDARY, mtf=None, gpd=None):


def try_wavefront(s: TestSettings):
    t = time.time()

    tr = TestResults()
    tr.copy_important_settings(s)

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

    use_prysm = False
    if use_prysm:
        use_cuda = False
        if prysm.config.backend != np:
            prysm.config.backend = np
            prysm.config.precision = wavefront_config.PRECISION

    prysm_path_q = s.fftsize / s.phasesamples

    if use_cuda:
        fft2 = cupyx.scipy.fftpack.fft2
        affine_transform = cupyx.scipy.ndimage.affine_transform
    else:
        fft2 = fftpack.fft2
        affine_transform = ndimage.affine_transform

    tr.used_cuda = use_cuda

    realdtype = "float32" if wavefront_config.PRECISION == 32 else "float64"
    complexdtype = "complex64" if wavefront_config.PRECISION == 32 else "complex128"

    me = cp if use_cuda else np


    eval_wavelengths = [BASE_WAVELENGTH] if s.mono else MODEL_WVLS

    if "join psfs" and 1:
        join_psfs = 1
        if s.return_type == RETURN_PSF or s.return_type == RETURN_LSF:
            mono_psf_stack = me.zeros(s.fftshape, dtype="float64")

        psfx = me.zeros((s.fftsize, ), dtype="float64")
        psfy = me.zeros((s.fftsize, ), dtype="float64")
        psf_lst = []
    else:
        join_psfs = 0

    samplelst = []

    polychromatic_weights = me.array([float(photopic_fn(wv * 1e3) * d50_interpolator(wv)) for wv in eval_wavelengths])

    t_misc = 0
    t_pupils = 0
    t_get_phases = 0
    t_get_fcns = 0
    t_pads = 0
    t_ffts = 0
    t_affines = 0
    t_mtfs = 0
    t_init = time.time() - t
    t = time.time()

    mask = mask_pupil(s, engine=me, dtype=realdtype)

    t_maskmaking = time.time() - t

    for wvl_num, (model_wvl, polych_weight) in enumerate(zip(eval_wavelengths, polychromatic_weights)):
        t = time.time()
        rel_wv = model_wvl / BASE_WAVELENGTH

        zkwargs = {}
        for key, value in s.p.items():

            if key.upper() == 'Z9':
                zkwargs[key] = get_z9(s.p, model_wvl)
                continue
            if key.upper() == 'Z4':
                raise ValueError("No Z4 separately!")
            if key.lower().startswith('z') and key[1].isdigit():
                zkwargs[key] = value * mul * wavefront_config.BASE_WAVELENGTH

        z4 = get_z4(s.defocus, s.p, model_wvl)
        samplelst.append(s.phasesamples)
        t_misc += time.time() - t
        if s.phasesamples not in zcache:
            t = time.time()
            pupil = prysm.FringeZernike(np.ones(36) * 0.5,
                                        dia=10, norm=False,
                                        wavelength=model_wvl,
                                        opd_unit="um",
                                        mask_target='none',
                                        samples=s.phasesamples, )
            t_pupils += time.time() - t
            t = time.time()
            if me is cp:
                if s.phasesamples not in cupyzcache:
                    cupyzcache[s.phasesamples] = {}
                for key, val in zcache[s.phasesamples].items():
                    cupyzcache[s.phasesamples][key] = me.array(val)
            t_get_phases += time.time() - t
        t = time.time()
        pupil = prysm.FringeZernike(dia=10, wavelength=model_wvl, norm=False,
                                    opd_unit="um",
                                    mask_target='none',
                                    samples=s.phasesamples, )
        t_pupils += time.time() - t
        t = time.time()
        z_arr = np.zeros(36)
        for key, value in zkwargs.items():
            idx = int(key[1:]) - 1
            z_arr[idx] = value
        z_arr[3] = z4

        if me is cp:
            cache = cupyzcache
        else:
            cache = zcache
        phase = me.zeros(s.phaseshape, dtype=realdtype)
        for (key, val) in cache[s.phasesamples].items():
            phase += val * float(z_arr[key])
        phase /= model_wvl
        t_get_phases += time.time() - t

        t = time.time()
        if join_psfs:
            if use_prysm:
                pupil.mask(mask, target='fcn')
                mono_psf = prysm.PSF.from_pupil(pupil, efl=s.p['base_fstop'] * 10, Q=prysm_path_q, norm='radiometric')
                # plt.plot(np.log10(mono_psf.data[int(mono_psf.data.shape[0]/2), :]))
                # mono_psf.plot2d()
                # plt.show()
                psf_lst.append(mono_psf)
            else:
                # phase = me.array(pupil.change_phase_unit(to='waves', inplace=False),
                #                  dtype=realdtype)
                # phase = me.zeros((samples, samples), dtype=realdtype)
                wavefunction = me.exp(1j * 2 * me.pi * phase)
                wavefunction *= mask

                t_get_fcns += time.time() - t

                if model_wvl == min(eval_wavelengths):
                    if s.plot and not use_prysm:
                        mono_psf = prysm.PSF.from_pupil(pupil, efl=s.p['base_fstop'] * 10, Q=prysm_path_q)

                    x, y = prysm.propagation.prop_pupil_plane_to_psf_plane_units(wavefunction,
                                                                                 pupil.sample_spacing,
                                                                                 s.p['base_fstop'] * 10, model_wvl,
                                                                                 prysm_path_q)
                    x = x[:s.fftsize]
                    y = y[:s.fftsize]

                t = time.time()
                padpx = int((s.fftsize - s.phasesamples) / 2)
                if padpx > 0:
                    pt = padpx, padpx
                    padded_cropped_pupil_fcn = me.pad(me.array(wavefunction, dtype=complexdtype), (pt, pt), mode="constant")
                elif padpx < 0:
                    padded_cropped_pupil_fcn = me.array(wavefunction[-padpx:-padpx+s.fftsize, -padpx:-padpx+s.fftsize], dtype=complexdtype)
                else:
                    padded_cropped_pupil_fcn = me.array(wavefunction, dtype=complexdtype)

                pad = me.fft.fftshift(padded_cropped_pupil_fcn)

                t_pads += time.time() - t
                t = time.time()

                # pad = me.fft.fft2(pad, norm='ortho')
                pad = fft2(pad, overwrite_x=True)
                shifted = me.fft.ifftshift(pad)
                impulse_response = me.absolute(shifted)
                impulse_response **= 2
                impulse_response *= (1.0 / impulse_response.sum() * polych_weight)

                impx = impulse_response.sum(axis=0)
                impy = impulse_response.sum(axis=1)

                t_ffts += time.time() - t
                t = time.time()

                psf_size = impulse_response.shape[0]
                zoom_factor = model_wvl / min(eval_wavelengths)

                if s.return_type == RETURN_PSF:
                    transform = me.array(((1.0 / zoom_factor, 0), (0, 1.0 / zoom_factor)))
                    offset = (psf_size - 1) / 2 * (1.0 - 1.0 / zoom_factor)
                    zoomed_mono_psf = affine_transform(impulse_response, transform, offset, order=1)
                    mono_psf_stack += zoomed_mono_psf

                transform = me.array((1.0 / zoom_factor,))
                offset = (psf_size - 1) / 2 * (1.0 - 1.0 / zoom_factor)
                zoomx = affine_transform(impx, transform, offset, order=1)
                zoomy = affine_transform(impy, transform, offset, order=1)
                psfx += zoomx
                psfy += zoomy

                t_affines += time.time() - t

                if s.plot:
                    psf_lst.append(mono_psf)
        else:
            m = prysm.MTF.from_pupil(pupil, efl=s.p['base_fstop'] * 10, Q=prysm_path_q)
            if use_cuda:
                sag_x, sag_mod = m.sag
                interpfn = interpolate.InterpolatedUnivariateSpline(cp.asnumpy(sag_x), cp.asnumpy(sag_mod), k=1)
                a = interpfn(SPACIAL_FREQS / DEFAULT_PIXEL_SIZE * 1e-3)
            else:
                a = m.exact_sag(SPACIAL_FREQS / DEFAULT_PIXEL_SIZE * 1e-3)
            mtfs.append(a)

        pupilslices.append(pupil.slice_x[1])

        metric = np.abs(rel_wv - 1)
        if metric < bestpupil[0]:
            bestpupil = metric, pupil, model_wvl, zkwargs

    t = time.time()

    if join_psfs:
        if use_prysm:
            psf = prysm.PSF.polychromatic(psf_lst, me.array(polychromatic_weights))
            mtf = prysm.MTF.from_psf(psf)
            final_mtf_array = mtf.exact_sag(SPACIAL_FREQS / DEFAULT_PIXEL_SIZE * 1e-3)
        else:

            if s.return_type == RETURN_PSF:
                psf = prysm.PSF(x=cp.asnumpy(x), y=cp.asnumpy(y), data=cp.asnumpy(mono_psf_stack))
                tr.psf = psf
                return tr

            if s.return_type == RETURN_LSF:
                lsf_sag = mono_psf_stack.sum(axis=0)
                lsf_sag /= lsf_sag.max()
                lsf_tan = mono_psf_stack.sum(axis=1)
                lsf_tan /= lsf_tan.max()
                tr.lsf = lsf_sag, lsf_tan
                return tr
            else:
                centre = int(s.fftsize / 2)
                x = prysm.fttools.forward_ft_unit((x[1] - x[0]) / 1e3, s.fftsize)  # 1e3 for microns => mm
                sag_x = x[centre:]
                tan_x = x[centre:]
                if 0:
                    if otf:
                        mtf = me.fft.fftshift(me.fft.fft2(me.fft.ifftshift(mixedpsf)))
                    else:
                        mtf = me.absolute(me.fft.fftshift(me.fft.fft2(mixedpsf)))
                    # y = prysm.fttools.forward_ft_unit((x[1] - x[0]) / 1e3, len(y))
                    mtf = mtf / np.abs(mtf[centre, centre])
                    sag_mod = mtf[centre, centre:]
                    tan_mod = mtf[centre:, centre]

                sag_mod = me.fft.fftshift(me.fft.fft(me.fft.ifftshift(psfx)))[centre:]
                tan_mod = me.fft.fftshift(me.fft.fft(me.fft.ifftshift(psfy)))[centre:]

                sag_mod /= np.abs(sag_mod)[0]
                tan_mod /= np.abs(tan_mod)[0]

                if me is cp:
                    sag_x = cp.asnumpy(sag_x)
                    tan_x = cp.asnumpy(tan_x)
                    sag_mod = cp.asnumpy(sag_mod)
                    tan_mod = cp.asnumpy(tan_mod)

                if s.return_type == RETURN_MTF:
                    interpfn = interpolate.InterpolatedUnivariateSpline(sag_x, np.abs(sag_mod), k=1)
                    sagmtf = interpfn(SPACIAL_FREQS / DEFAULT_PIXEL_SIZE * 1e-3)
                    interpfn = interpolate.InterpolatedUnivariateSpline(tan_x, np.abs(tan_mod), k=1)
                    tanmtf = interpfn(SPACIAL_FREQS / DEFAULT_PIXEL_SIZE * 1e-3)
                    tr.otf = sagmtf, tanmtf

                if s.return_type == RETURN_OTF:
                    interpfn = interpolate.InterpolatedUnivariateSpline(sag_x, np.real(sag_mod), k=1)
                    sagmtf = interpfn(SPACIAL_FREQS / DEFAULT_PIXEL_SIZE * 1e-3)
                    interpfn = interpolate.InterpolatedUnivariateSpline(tan_x, np.real(tan_mod), k=1)
                    tanmtf = interpfn(SPACIAL_FREQS / DEFAULT_PIXEL_SIZE * 1e-3)
                    interpfn = interpolate.InterpolatedUnivariateSpline(sag_x, np.imag(sag_mod), k=1)
                    sagmtf_i = interpfn(SPACIAL_FREQS / DEFAULT_PIXEL_SIZE * 1e-3)
                    interpfn = interpolate.InterpolatedUnivariateSpline(tan_x, np.imag(tan_mod), k=1)
                    tanmtf_i = interpfn(SPACIAL_FREQS / DEFAULT_PIXEL_SIZE * 1e-3)
                    sagmtf = sagmtf + sagmtf_i * 1j
                    tanmtf = tanmtf + tanmtf_i * 1j

                    tr.otf = sagmtf, tanmtf

    t_mtfs += time.time() - t

    timings = dict(t_init=t_init,
                   t_maskmaking=t_maskmaking,
                   t_pupils=t_pupils,
                   t_get_phases=t_get_phases,
                   t_get_fcns=t_get_fcns,
                   t_pads=t_pads,
                   t_ffts=t_ffts,
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
                plt.plot(cp.asnumpy(slice), label="{} : {:.3f} λRMS".format(key, rms / wavefront_config.BASE_WAVELENGTH))
        slice_ = bestpupil[1].slice_x[1]
        rms = (slice_[me.isfinite(slice_)] ** 2).mean() ** 0.5
        plt.plot(slice_, label="All : {:.3f} λRMS".format(rms / wavefront_config.BASE_WAVELENGTH))
        pupil = prysm.FringeZernike(z4=z4,
                                    dia=10, norm=False,
                                  wavelength=bestpupil[2],
                                  opd_unit="um",
                                  samples=samples,
                                  **bestpupil[3])

        pupil = mask_pupil(pupil, p['base_fstop'], p['fstop'])
        slice_ = pupil.slice_x[1]
        rms = (slice_[me.isfinite(slice_)] ** 2).mean() ** 0.5
        plt.plot(slice_, '--', label="ZeroZ4 : {:.3f} λRMS".format(rms / wavefront_config.BASE_WAVELENGTH), color='black')
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


def mask_pupil(s: TestSettings, engine=np, dtype="float64", plot=False):
    hashtuple = (s.p['base_fstop'],
                 s.p['fstop'],
                 s.x_loc,
                 s.y_loc,
                 s.phasesamples,
                 s.p.get('a', 0.0),
                 s.p.get('b', 0.0),
                 s.p.get('v_amt', 1.0),
                 s.p.get('v_rad', 1.0),
                 s.p.get('squariness', 0.5),
                 "np" if engine is np else "cp",
                 dtype)

    hash = hashtuple.__hash__()

    for cachehash, mask in mask_cache:
        if cachehash == hash:
            return mask

    print("Building mask", os.getpid(), hashtuple)
    me = engine

    aperture_stop_norm_radius = s.p['base_fstop'] / s.p['fstop']

    na = 1 / (2.0 * s.p['base_fstop'])
    onaxis_peripheral_ray_angle = me.arcsin(na, dtype=dtype)

    pupil_radius_mm = me.tan(onaxis_peripheral_ray_angle, dtype=dtype) * s.default_exit_pupil_position_mm

    x_displacement_mm = (s.x_loc - IMAGE_WIDTH / 2) * DEFAULT_PIXEL_SIZE * 1e3
    y_displacement_mm = (s.y_loc - IMAGE_HEIGHT / 2) * DEFAULT_PIXEL_SIZE * 1e3

    if s.fix_pupil_rotation:
        x_displacement_mm = (x_displacement_mm ** 2 + y_displacement_mm ** 2) ** 0.5
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

    stopmask = pupil_norm_radius_grid < aperture_stop_norm_radius

    a = s.p.get('a', 0.0)
    b = s.p.get('b', 0.0)

    if not s.pixel_vignetting:
        mask = stopmask
    else:

        coeff_4 = -18.73 * a
        corff_6 = 485 * b

        square_grid = 1.0 / (1.0 + (pixel_angle_grid**4 * coeff_4 + pixel_angle_grid ** 6 * corff_6))

        mask = stopmask * square_grid

        # mask = stopmask

    if s.lens_vignetting:
        # Lens vignette
        normarr = me.linspace(-1, 1, s.phasesamples, dtype=dtype)
        image_circle_modifier = s.p.get('v_amt', 1.0) * 1.0
        gridx, gridy = me.meshgrid(normarr - x_displacement_mm / SENSOR_WIDTH * 1e-3 * image_circle_modifier,
                                   normarr - y_displacement_mm / SENSOR_WIDTH * 1e-3 * image_circle_modifier)
        vignette_radius_grid = (gridx ** 2 + gridy ** 2) ** 0.5
        vignette_crop_circle_radius = s.p.get('v_rad', 1.0)
        vignette_mask = vignette_radius_grid < vignette_crop_circle_radius
        mask *= vignette_mask

        normarr = me.linspace(-1, 1, s.phasesamples, dtype=dtype)
        image_circle_modifier = s.p.get('v_amt', 1.0) * 1.0
        gridx, gridy = me.meshgrid(normarr + x_displacement_mm / SENSOR_WIDTH * 1e-3 * image_circle_modifier,
                                   normarr + y_displacement_mm / SENSOR_WIDTH * 1e-3 * image_circle_modifier)
        vignette_radius_grid = (gridx ** 2 + gridy ** 2) ** 0.5
        vignette_crop_circle_radius = s.p.get('v_rad', 1.0) * 1.1
        vignette_mask = vignette_radius_grid < vignette_crop_circle_radius
        mask *= vignette_mask

    # plt.plot(mask[int(shape[0] / 2), :])
    # plt.show()

    mask_cache.appendleft((hash, mask))

    if plot:
        print(square_grid)
        plt.imshow(mask)
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
        s.p['vignette'] = 0.8
        s.p['vcr'] = 1
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