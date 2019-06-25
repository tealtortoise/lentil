import numpy as np
from scipy import interpolate, fftpack, optimize

from lentil import constants_utils as lentilconf

from lentilwave import config
from lentilwave.generation import caches


def get_processing_details(s, cache_: caches.GeneratorCache = None):
    if cache_ is not None and s.id_or_hash is not None and s.id_or_hash in cache_.settings:
        stup = cache_.settings[s.id_or_hash]
        if s.fftsize is None:
            s.fftsize = stup[0]
        if s.phasesamples is None:
            s.phasesamples = stup[1]
        if s.effective_q is None:
            s.effective_q = stup[2]
        s.allow_cuda = s.allow_cuda and s.fftsize > s.cpu_gpu_arraysize_boundary
        return s

    minimum_q = np.clip((s.strehl_estimate * 4) * s.q_autosize_scalar, 2, 3)
    min_samples = -np.inf
    f_stopped_down = s.p['fstop'] / s.p['base_fstop']
    if s.guide_mtf is None:
        min_samples = 384
    else:
        for otf in s.guide_mtf:
            freqs = np.arange(0, 65) / 64
            zero_plus_spacial_freqs = np.concatenate(([0], config.SPACIAL_FREQS, [1.0, 2.0]))
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

    for power in range(4, 10):
        samples = 2 ** power
        if samples > min_samples:
            break
        samples = int((2 ** power * 1.5) / 2 + 0.5) * 2
        if samples > min_samples:
            break

    effective_q_without_padding = f_stopped_down

    min_fftsize = minimum_q * samples / effective_q_without_padding

    for fftsize in (lentilconf.CUDA_GOOD_FFT_SIZES if s.allow_cuda else lentilconf.CPU_GOOD_FFT_SIZES):
        if fftsize >= min_fftsize:
            break

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

    if cache_ is not None and s.id_or_hash is not None:
        cache_.settings[s.id_or_hash] = fftsize, samples, effective_q

    # s.fftsize = 1024
    # s.phasesamples = 512

    return s
