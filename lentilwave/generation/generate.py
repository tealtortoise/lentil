import time

import numpy as np
import prysm
try:
    import cupy as cp
    import cupyx.scipy.ndimage
    import cupyx.scipy.fftpack
except ImportError:
    cp = None

from scipy import interpolate, ndimage, fftpack
import matplotlib.pyplot as plt

from lentil import constants_utils as lentilconf
from lentilwave import config, helpers
from lentilwave.generate import masks, caches


def sanitycheck(s):
    if s.p['fstop'] < s.p['base_fstop']:
        raise ValueError("Base_fstop must be wider (lower) than fstop {} < {}".format(s.p['fstop'], s.p['base_fstop']))


defaultcaches = {'np': caches.GenerateCache(),
                 'cp': caches.GenerateCache()}


def get_phase_cache_cube(s: helpers.TestSettings, me=np, realdtype="float64"):
    zarr = s.zernike_flags.copy()
    zarr[3] = 1
    zarr[8] = 1

    prysm.FringeZernike(zarr*0.1,
                        dia=10, norm=False,
                        opd_unit="um",
                        mask_target='none',
                        samples=s.phasesamples, )  # This is just here to fill the coeff cache

    # Assign empty array ready to populate
    arr = me.empty((s.phasesamples, s.phasesamples, s.zernike_flags.sum()), dtype=realdtype)

    # Now pack the phases into a 3d array
    for z in s.used_zernikes:
        idx = s.zernike_index[z - 1]  # Zero based
        singlez = prysm.zernike.zcache.regular[s.phasesamples][z - 1]
        arr[:, :, idx] = me.array(singlez, dtype=realdtype)
    return arr


def generate(s: helpers.TestSettings, cache_=defaultcaches):
    t = time.time()

    tr = helpers.TestResults()
    tr.copy_important_settings(s)

    if s.dummy:
        return tr

    sanitycheck(s)

    s.get_processing_details()
    use_cuda = s.allow_cuda and cp is not None

    if use_cuda:
        fft2 = cupyx.scipy.fftpack.fft2
        affine_transform = cupyx.scipy.ndimage.affine_transform
    else:
        fft2 = fftpack.fft2
        affine_transform = ndimage.affine_transform

    tr.used_cuda = use_cuda

    # Normalise PSF/LSF after scaling, rather than before
    SAM_RADIOMETRIC_MODEL = True

    FORCE_PSF = False

    def sync():
        # Option to sync cuda device after each stage for profiling
        pass

    realdtype = "float32" if config.PRECISION == 32 else "float64"
    complexdtype = "complex64" if config.PRECISION == 32 else "complex128"

    me, engine_string = (cp, 'cp') if use_cuda else (np, 'np')

    # We only care about cached items on appropriate device
    engcache = cache_[engine_string]

    eval_wavelengths = [config.BASE_WAVELENGTH] if s.mono else config.MODEL_WVLS

    build_psf = s.return_psf or FORCE_PSF or s.return_prysm_mtf

    if build_psf:
        # Get zero array for building full polychromatic 2d PSF
        psf_stack_sum = me.zeros(s.fftshape, dtype="float64")

    # Get 2 1D LSFs
    lsf_sag = np.zeros((s.fftsize, ), dtype="float64")
    lsf_tan = np.zeros((s.fftsize, ), dtype="float64")

    polychromatic_weights = np.array([float(lentilconf.photopic_fn(wv * 1e3) *
                                            lentilconf.d50_interpolator(wv)) for wv in eval_wavelengths])

    # Plan pupil distortion
    ellip = s.p.get('ellip', 0)
    xellip = np.clip(1.0 + ellip, 0.5, 1.0)
    yellip = np.clip(1.0 - ellip, 0.5, 1.0)

    # Timing accumulators
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
    mask = masks.build_mask(s, engine=me, dtype=realdtype)
    if s.return_mask:
        tr.mask = mask
    t_maskmaking = time.time() - t

    # Analysis p dictionary to get Z usage
    s.get_used_zernikes()
    zusedhash = hash(s.used_zernikes)
    zhash = hash(tuple(s.zernike_array))

    for wvl_num, (model_wvl, polych_weight) in enumerate(zip(eval_wavelengths, polychromatic_weights)):
        t = time.time()
        rel_wv = model_wvl / config.BASE_WAVELENGTH

        z4 = helpers.get_z4(s.defocus, s.p, model_wvl)
        z9 = helpers.get_z9(s.p, model_wvl)
        t_misc += time.time() - t

        if s.phasesamples not in engcache.cubes or engcache.cubes[s.phasesamples][0] != zusedhash:
            # New phase size so we need to call prysm and build some phases
            t = time.time()

            # Get a 3d array (one 2d phase for each coefficient in use)
            cube = get_phase_cache_cube(s, me=me)

            # Cache it
            engcache.cubes[s.phasesamples] = (zusedhash, cube)

            sync()
            t_get_phases += time.time() - t
        else:
            assert engcache.cubes[s.phasesamples][0] == zusedhash
            # We have a valid cached cube
            _, cube = engcache.cubes[s.phasesamples]

        # Get a blank pupil to get unit data from
        t = time.time()
        pupil = prysm.FringeZernike(dia=10, wavelength=model_wvl, norm=False,
                                    opd_unit="um",
                                    mask_target='none',
                                    samples=s.phasesamples, )
        t_pupils += time.time() - t

        t = time.time()
        basephase = None

        # Do we already have a base phase without Z4 and Z9
        if s.phasesamples in engcache.basephases:
            cached_base = engcache.basephases[s.phasesamples]
            if cached_base[0] == zhash:
                # Already done
                basephase = cached_base[1]

        if basephase is None:
            # We need to build one
            _, cube = engcache.cubes[s.phasesamples]  # This should already be populated from earlier

            # Zero out any Z4 and Z9 otherwise wouldn't be a base
            indexed_no_z4_no_z9 = s.zernike_array_indexed.copy()
            indexed_no_z4_no_z9[s.zernike_index[4 - 1]] = 0
            indexed_no_z4_no_z9[s.zernike_index[9 - 1]] = 0
            if me is cp:
                indexed_no_z4_no_z9 = me.array(indexed_no_z4_no_z9)

            # Run dot product
            basephase = cube @ indexed_no_z4_no_z9

            # Cache it
            cached_base = zhash, basephase
            engcache.basephases[s.phasesamples] = cached_base

        # Now we have basephase add Z4 and Z9 to taste
        phase = basephase.copy()
        z4_phase = cube[:, :, s.zernike_index[4 - 1]]
        phase += z4_phase * z4
        z9_phase = cube[:, :, s.zernike_index[9 - 1]]
        phase += z9_phase * z9

        phase /= model_wvl
        sync()
        t_get_phases += time.time() - t

        t = time.time()
        # Get complex wavefunction
        wavefunction = me.exp(1j * 2 * me.pi * phase)
        # Apply mask
        wavefunction *= mask
        sync()
        t_get_fcns += time.time() - t

        if model_wvl == min(eval_wavelengths):
            # This is our shortest wavelength, samples spacing will be normalised to this
            psf_sample_spacing = prysm.propagation.pupil_sample_to_psf_sample(pupil_sample=pupil.sample_spacing,
                                                          samples=s.fftsize,
                                                          wavelength=model_wvl,
                                                          efl=s.p['base_fstop'] * 10) * 1e-3
            psf_units = np.arange(-s.fftsize / 2, s.fftsize / 2) * psf_sample_spacing

        # Process wavefunction
        resized_wavefunction = pad_and_distort(s, wavefunction, affine_transform=affine_transform, me=me, complexdtype=complexdtype)

        t = time.time()
        # FFTs
        fftarr = me.fft.fftshift(resized_wavefunction)
        fftarr = fft2(fftarr, overwrite_x=True)
        shifted = me.fft.ifftshift(fftarr)

        # Get PSF for incoherent imaging
        mono_psf = me.absolute(shifted)
        mono_psf **= 2

        if not SAM_RADIOMETRIC_MODEL:
            mono_psf /= mono_psf.sum()

        # Sum down to two 1D LSFs
        impx = mono_psf.sum(axis=1)
        impy = mono_psf.sum(axis=0)
        sync()
        t_ffts += time.time() - t

        # Clipping to ensure a nice continuous function (avoid NOP at zoom == 1.0)
        zoom_factor = float(np.clip(model_wvl / min(eval_wavelengths), 1.001, np.inf))
        shift_x, shift_y = helpers.get_lca_shifts(s, model_wvl, psf_sample_spacing)
        if build_psf:
            t = time.time()

            # Resample PSF to fit minimum wavelength sample spacing
            scaled_mono_psf = helpers.zoom2d(mono_psf,
                                             zoom_factor / xellip, zoom_factor / yellip,
                                             shift_x, shift_y,
                                             affine_transform=affine_transform,
                                             me=me)

            if SAM_RADIOMETRIC_MODEL:
                # Renormalise to ensure radiometry
                scaled_mono_psf *= polych_weight / scaled_mono_psf.sum()
            else:
                scaled_mono_psf *= polych_weight

            # Add to stack
            psf_stack_sum += scaled_mono_psf
            sync()
            t_affines += time.time() - t

        t = time.time()
        if me is cp:
            # Bring back from GPU
            impx = cp.asnumpy(impx)
            impy = cp.asnumpy(impy)
        sync()
        t_cudasyncs += time.time() - t

        t = time.time()

        # Resample LSFs to match minimum wavelength sample spacing
        scaled_sag_lsf = helpers.zoom1d(impx, zoom_factor / xellip, shift_x)
        scaled_tan_lsf = helpers.zoom1d(impy, zoom_factor / yellip, shift_y)

        mul = 1.0 * zoom_factor

        if SAM_RADIOMETRIC_MODEL:
            # Normalise PSF intensity to ensure radiometry
            lsf_sag += scaled_sag_lsf / scaled_sag_lsf.sum() * polych_weight
            lsf_tan += scaled_tan_lsf / scaled_tan_lsf.sum() * polych_weight
        else:
            lsf_sag += scaled_sag_lsf * polych_weight * mul
            lsf_tan += scaled_tan_lsf * polych_weight * mul
        sync()
        t_affines += time.time() - t

    t = time.time()

    if build_psf:
        # Replace LSFs with LSFs from PSF since we have it
        lsf_sag = psf_stack_sum.sum(axis=1)
        lsf_tan = psf_stack_sum.sum(axis=0)
        if me is cp:
            lsf_sag = cp.asnumpy(lsf_sag)
            lsf_tan = cp.asnumpy(lsf_tan)

    # Extra TCA blur
    if shift_x != 0 and 0:
        lsf_tan = ndimage.gaussian_filter1d(lsf_tan, shift_x / 10, 0, mode="constant", cval=0.0)

    if build_psf:
        # Since we have full psf put data into prysm PSF object
        npunits = cp.asnumpy(psf_units)
        numpystack = cp.asnumpy(psf_stack_sum)
        if shift_x != 0:
            tca_blurred = ndimage.gaussian_filter1d(numpystack, shift_x / 10, 1, mode="constant", cval=0.0)
        else:
            tca_blurred = numpystack
        prysm_psf = prysm.PSF(x=npunits, y=npunits, data=tca_blurred)
        if s.return_psf:
            tr.psf = prysm_psf

    centre = s.fftsize // 2
    # Get OTF x units from prysm
    mtf_x_units = prysm.fttools.forward_ft_unit(psf_sample_spacing * 1e-3, s.fftsize)
    sag_x = mtf_x_units[centre:]  # We don't need negative frequency data as LSF was real
    tan_x = mtf_x_units[centre:]

    mtf_mapper_fft_halfwindowsize_um = 16 * lentilconf.DEFAULT_PIXEL_SIZE * 1e6

    # Get hashable to help caching
    tukeykey = (s.fftsize, psf_sample_spacing)
    try:
        tukey_window = cache_['np'].windows[tukeykey]
    except KeyError:
        if len(cache_['np'].windows) > 300:
            cache_['np'].windows = {}
        tukey_window = lentilconf.tukey(psf_units / mtf_mapper_fft_halfwindowsize_um, 0.6)
        cache_['np'].windows[tukeykey] = tukey_window

    # Run FFT on LSFs to get MTF (with phase normalisation)
    sag_mod = lentilconf.normalised_centreing_fft(lsf_sag * tukey_window, fftpack=fftpack, engine=np)[:centre]
    tan_mod = lentilconf.normalised_centreing_fft(lsf_tan * tukey_window, fftpack=fftpack, engine=np)[:centre]

    if s.return_prysm_mtf:
        prysm_mtf = prysm.MTF.from_psf(prysm_psf)
        tr.prysm_mtf = prysm_mtf

    get_x_freqs = config.SPACIAL_FREQS / lentilconf.DEFAULT_PIXEL_SIZE * 1e-3
    interpolator = interpolate.InterpolatedUnivariateSpline
    order = 2
    if s.return_otf and s.return_otf_mtf:
        sagmtf = interpolator(sag_x, np.abs(sag_mod), k=order)(get_x_freqs)
        tanmtf = interpolator(tan_x, np.abs(tan_mod), k=order)(get_x_freqs)
        tr.otf = sagmtf, tanmtf

    if s.return_otf and not s.return_otf_mtf:
        sagmtf = interpolator(sag_x, np.real(sag_mod), k=order)(get_x_freqs)
        tanmtf = interpolator(tan_x, np.real(tan_mod), k=order)(get_x_freqs)
        sagmtf_i = interpolator(sag_x, np.imag(sag_mod), k=order)(get_x_freqs)
        tanmtf_i = interpolator(tan_x, np.imag(tan_mod), k=order)(get_x_freqs)
        tr.otf = sagmtf + 1j * sagmtf_i,\
                 tanmtf + 1j * tanmtf_i
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

    return tr


def pad_and_distort(s, wavefunction, affine_transform=ndimage.affine_transform, me=np, complexdtype="complex128"):
    # How many array elements need adding or removing from each side of phase before FFT
    padpx = int((s.fftsize - s.phasesamples) / 2)

    ellip = None    # Plan pupil distortion
    # ellip = s.p.get('ellip', 0)
    # xellip = np.clip(1.0 + ellip, 0.5, 1.0)
    # yellip = np.clip(1.0 - ellip, 0.5, 1.0)

    if padpx > 0:
        # Our phase needs padding to satisfy oversampling criteria
        pt = padpx, padpx
        if config.ENABLE_PUPIL_DISTORTION and ellip is not None:
            wavefunction = helpers.zoom2d(wavefunction, xellip, yellip, affine_transform=affine_transform, me=me)
        t = time.time()
        resized_wavefunction = me.pad(me.array(wavefunction, dtype=complexdtype), (pt, pt), mode="constant")
    elif padpx < 0:
        # Our phase is over padded
        resized_wavefunction = me.array(wavefunction[-padpx:-padpx + s.fftsize, -padpx:-padpx + s.fftsize],
                                        dtype=complexdtype)
        if config.ENABLE_PUPIL_DISTORTION and ellip is not None:
            resized_wavefunction = helpers.zoom2d(resized_wavefunction, xellip, yellip, affine_transform=affine_transform,
                                                  me=me)
    else:
        # Our phase is just right
        resized_wavefunction = wavefunction
        if config.ENABLE_PUPIL_DISTORTION and ellip is not None:
            resized_wavefunction = helpers.zoom2d(resized_wavefunction, xellip, yellip, affine_transform=affine_transform,
                                                  me=me)
    return resized_wavefunction


