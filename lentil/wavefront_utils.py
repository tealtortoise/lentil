import random
import multiprocessing
import lentil.constants_utils
from lentil.constants_utils import *
from lentil.focus_set import FocusSet, read_wavefront_data
# from lentil.focus_set import estimate_focus_jitter

# use_cuda = wavefront_config.USE_CUDA
from lentilwave.encode_decode import convert_wavefront_dicts_to_p_dicts
from lentilwave import config, TestSettings, TestResults
from lentilwave.generation import generate


class TerminateOptException(Exception):
    pass


def cauchy_fit(x, y):
    meanpeak_idx = np.argmax(y)
    meanpeak_pos = x[meanpeak_idx]
    meanpeak = y[meanpeak_idx]
    # highest_data_y = y_values[highest_data_x_idx]

    # print(highest_data_x_idx)

    if meanpeak_idx > 0:
        x_inc = x[meanpeak_idx] - x[meanpeak_idx - 1]
    else:
        x_inc = x[meanpeak_idx + 1] - x[meanpeak_idx]

    # y_values = np.cos(np.linspace(-6, 6, len(x))) + 1
    absgrad = np.abs(np.gradient(y)) / meanpeak
    gradsum = np.cumsum(absgrad)
    distances_from_peak = np.abs(gradsum - np.mean(gradsum[meanpeak_idx:meanpeak_idx + 1]))
    shifted_distances = interpolate.InterpolatedUnivariateSpline(x, distances_from_peak, k=1)(
        x - x_inc * 0.5)
    weights = np.clip(1.0 - shifted_distances * 1.3, 1e-1, 1.0) ** 5

    fitfn = cauchy

    optimise_bounds = fitfn.bounds(meanpeak_pos, meanpeak, x_inc)

    sigmas = 1. / weights
    initial = fitfn.initial(meanpeak_pos, meanpeak, x_inc)
    fitted_params, _ = optimize.curve_fit(fitfn, x, y,
                                          bounds=optimise_bounds, sigma=sigmas, ftol=1e-5, xtol=1e-5,
                                          p0=initial)
    return fitted_params


def get_weights(shape, focus_values, centre):
    focus_deviations = np.abs(focus_values - centre)
    max_focus_deviation = focus_deviations.max()
    focusweights = 1.0 - focus_deviations / max_focus_deviation * (1.0 - config.EXTREME_FOCUS_WEIGHT)

    freqrange = config.HIGH_FREQUENCY_WEIGHT
    freqweights = np.linspace(1.0, freqrange, shape[0]).reshape((shape[0], 1))
    expanded = np.repeat(focusweights[np.newaxis, :], shape[0], axis=0)
    weights = expanded * freqweights
    return weights ** 2


def _process_focusset(num):
    # ob_ = 2
    # fs_slices_ = 0
    # skip_ = 0
    focusset = focussets_[num]

    if type(focusset) is str:
        focusset = FocusSet(rootpath=focusset, use_calibration=True, include_all=True, load_complex=complex_otf_)
        return_focusset = True
    else:
        return_focusset = False

    wfd = [("", {})]
    ps = None
    if not from_scratch_ and num == 0:
        wfd = focusset.read_wavefront_data(overwrite=True, x_loc=x_loc_, y_loc=y_loc_)
        if wfd[-1][1] != {}:
            try:
                ps = convert_wavefront_dicts_to_p_dicts(wfd[-1][1])
                p = ps[0]
                if 'df_step' in ps[0] and 'df_offset' in ps[0]:
                    hints_needed = False
                else:
                    hints_needed = True
            except IndexError:
                p = None
                hints_needed = True
        else:
            p = None
            hints_needed = True

    elif not from_scratch_ and all_ps_ is not None:
        try:
            p = all_ps_[num]
            hints_needed = False
        except IndexError:
            hints_needed = True
    else:
        hints_needed = True

    # print("wfd ", wfd)
    data = lentil.constants_utils.FocusSetData()
    data.wavefront_data = wfd
    sag_ob = focusset.get_interpolation_fn_at_point(IMAGE_WIDTH / 2, IMAGE_HEIGHT / 2, AUC, SAGITTAL)
    focus_values = sag_ob.focus_data[:]

    if hints_needed:
        tup = focusset.find_best_focus(IMAGE_WIDTH / 2, IMAGE_HEIGHT / 2, axis=MERIDIONAL, _return_step_data_only=True,
                                       _step_estimation_posh=True)
        est_defocus_rms_wfe_step, longitude_defocus_step_um, coc_step, image_distance,\
            subject_distance, fit_peak_y, prysm_offset = tup
        data.hints['df_step'] = est_defocus_rms_wfe_step
        data.hints['df_offset'] = prysm_offset
    else:
        if p is not None:
            if 'df_step' in p:
                data.hints['df_step'] = p['df_step']
            if 'df_offset' in p:
                data.hints['df_offset'] = p['df_offset']
    # exit()
    mtf_means = sag_ob.sharp_data

    fitted_params = cauchy_fit(focus_values, mtf_means)

    cauchy_peak_x = fitted_params[1]
    cauchy_peak_y = fitted_params[0]
    print("Found peak {:.3f} at {:.3f}".format(cauchy_peak_y, cauchy_peak_x))
    data.cauchy_peak_x = cauchy_peak_x

    if len(wfd) == 0:
        data.hints['df_offset'] = (min(focus_values) - 2, cauchy_peak_x, max(focus_values) + 2)

    # Move on to get full frequency data

    # Find centre index
    centre_idx = int(interpolate.InterpolatedUnivariateSpline(focus_values,
                                                              range(len(focus_values)),
                                                              k=1)(cauchy_peak_x) + 0.5)
    if type(fs_slices_) is int:
        size = fs_slices_
    else:
        size = fs_slices_[num]
    slicelow = max(avoid_ends_, int(centre_idx - size * skip_ / 2 + 1))
    slicehigh = min(slicelow + size, len(mtf_means) - avoid_ends_)
    limit = (slicelow, slicehigh)
    print("Limit", limit)

    sag_data = []
    mer_data = []

    if complex_otf_:
        sagaxis = SAGITTAL_COMPLEX
        meraxis = MERIDIONAL_COMPLEX
    else:
        sagaxis = SAGITTAL
        meraxis = MERIDIONAL

    if x_loc_ is not None and y_loc_ is not None:
        x_test_loc = x_loc_
        y_test_loc = y_loc_
    else:
        x_test_loc = ob_.x_loc
        y_test_loc = ob_.y_loc
    for freq in config.SPACIAL_FREQS:
        print(freq)
        sag_ob = focusset.get_interpolation_fn_at_point(x_test_loc, y_test_loc, freq, sagaxis, limit=limit, skip=skip_)
        mer_ob = focusset.get_interpolation_fn_at_point(x_test_loc, y_test_loc, freq, meraxis, limit=limit, skip=skip_)
        sag_data.append(sag_ob.sharp_data)
        mer_data.append(mer_ob.sharp_data)

    data.x_loc = x_test_loc
    data.y_loc = y_test_loc

    sag_mtf_values = np.array(sag_data)
    mer_mtf_values = np.array(mer_data)
    merged_mtf_values = (sag_mtf_values + mer_mtf_values) * 0.5
    mtf_means = np.abs(merged_mtf_values).mean(axis=0)
    focus_values = sag_ob.focus_data
    max_pos = focus_values[np.argmax(mtf_means)]

    diff_mtf = diffraction_mtf(config.SPACIAL_FREQS, focusset.exif.aperture)
    diff_mtf_mean = diff_mtf.mean()
    strehl_ests = mtf_means / diff_mtf_mean

    data.merged_mtf_values = merged_mtf_values
    data.sag_mtf_values = sag_mtf_values
    data.mer_mtf_values = mer_mtf_values
    data.mtf_means = mtf_means
    data.focus_values = focus_values
    data.max_pos = max_pos
    data.strehl_ests = strehl_ests
    if num == 0:
        data.all_ps = []
    weights = get_weights(merged_mtf_values.shape, focus_values, cauchy_peak_x)

    assert weights.shape == merged_mtf_values.shape

    weightmean = np.mean(weights)
    data.weights = weights / weightmean

    data.exif = focusset.exif
    if return_focusset:
        return data, focusset
    else:
        return data


def pre_process_focussets(focussets, fs_slices, skip, avoid_ends=1, from_scratch=True, x_loc=None, y_loc=None,
                          complex_otf=True):
    # ob = focussets[0].find_sharpest_location()
    ob = None

    def init():
        global focussets_
        global fs_slices_
        global skip_
        global ob_
        global from_scratch_
        global avoid_ends_
        global x_loc_
        global y_loc_
        global complex_otf_
        global all_ps_
        focussets_ = focussets
        ob_ = ob
        skip_ = skip
        fs_slices_ = fs_slices
        from_scratch_ = from_scratch
        avoid_ends_ = avoid_ends
        x_loc_ = x_loc
        y_loc_ = y_loc
        complex_otf_ = complex_otf
        all_ps_ = all_ps

    if type(focussets[0]) is str:
        wfd = read_wavefront_data(focusset_path=focussets[0], x_loc=x_loc, y_loc=y_loc)
        try:
            dct = wfd[-1][1]
            all_ps = convert_wavefront_dicts_to_p_dicts(dct)
        except IndexError:
            all_ps = None


    if not config.DISABLE_MULTIPROCESSING:
        pool = multiprocessing.Pool(initializer=init)
        datas = pool.map(_process_focusset, range(len(focussets)))
    else:
        init()
        datas = [_process_focusset(_) for _ in range(len(focussets))]

    if type(focussets[0]) is str:
        datas, focussets = zip(*datas)

    # if 'all_ps' in datas[0]:
    #     for p, data in zip(datas[0].all_ps[1:], datas[1:]):
    #         data.hints = [("", p)]
    #
    # If data is old and saved before fstop data masking compensated in try_wavefront()
    #
    df_steps = [(data, data.wavefront_data[-1][1].get('p.opt:df_step')) for data in datas if 'p.opt:df_step' in data.wavefront_data[-1][1]]
    print(df_steps)
    if len(df_steps) > 1:
        data_with_steps, steps = zip(*df_steps)
        if np.all(np.diff(steps) > 0):
            base_fstop = datas[0].exif.aperture
            for data, step in df_steps[1:]:
                fstop = data.exif.aperture
                data.wavefront_data[-1][1]['p.opt:df_step'] *= (base_fstop / fstop) ** 2
    return datas, focussets


def jitterstats():
    errs = 0
    max = 0
    hints = 0

    random.seed(145)
    np.random.seed(145)
    num = 20
    for a in range(num):
        data = build_synthetic_dataset(subsets=1, test_stopdown=2, base_aperture=1.4, slices_per_fstop=19)[0]
        err = data.jittererr
        maxerr = data.jittererrmax
        hint = data.hintjit
        if maxerr > max:
            max = maxerr
        errs += err
        hints += hint
    print(errs / num)
    print(maxerr)
    print(hints / num)


def plot_nominal_psf(*args, wfdd={}, x_loc=IMAGE_WIDTH/2, y_loc=IMAGE_HEIGHT/2):
    # plt.cla()
    # plt.close()
    disable_plot = False
    defocuses = [-2.4, 0, 2.4, 4.8]
    defocus_amount = 2
    defocuses = np.linspace(-defocus_amount, defocus_amount, 5)
    if not disable_plot:
        f, axes = plt.subplots(len(args), len(defocuses), sharey=True, sharex=True)
        if len(args) == 1:
            axes = axes
    min_fstop = min(p['fstop'] for p in args)
    for na, dct in enumerate(args):

        df_offset = dct["df_offset"]
        for nd, defocus in enumerate(defocuses):
            alter = (dct['fstop'] / min_fstop) ** 2
            # alter = 1
            s = TestSettings(dct, defocus=defocus / alter + df_offset)
            # s.p = dict(base_fstop=1.2, fstop=1.2 * 2 ** (na / 2), df_offset=dct['df_offset'], df_step=dct['df_step'],
            #            v_scr=1, lca_slr=0, spca2=0.0)
            # s.p['v_y'] = -0.6
            # s.p['v_slr'] = 0
            s.x_loc = x_loc
            s.y_loc = y_loc
            # s.p['loca'] = 0
            # s.p['loca1'] = 0
            # s.p['spca2'] = 0
            # s.p['spca'] = 0
            # s.p['z9'] += 0.08
            # s.p['z10'] = 0
            # s.p['z11'] = 0
            # s.p['tca_slr'] = 1
            s.return_psf = True
            s.pixel_vignetting = True
            s.lens_vignetting = True
            s.phasesamples = 384
            s.fftsize = 768
            psf = generate(s).psf
            # zs = {}
            # for key, value in dct.items():
            #     if key[0].lower() == 'z' and key[1].isdigit():
            #         zs[key] = value
            #     elif key[0:7].lower() == 'p.opt:z' and key[7].isdigit():
            #         zs[key[6:]] = value
            # zs['z4'] = defocus
            # print(zs)
            # pupil = prysm.FringeZernike(**zs, norm=True, dia=10)
            # psf = prysm.PSF.from_pupil(pupil, efl=30, Q=5)
            if not disable_plot:
                if len(args) == 1:
                    ax = axes[nd]
                else:
                    ax = axes[na, nd]
                psf.plot2d(ax=ax, fig=f, axlim=defocus_amount*6)
    plt.show()


def build_normalised_scale_dictionary(gradients, ordering, target=1.0):
    listdct = {}
    for gradient, (pname, applies, _) in zip(gradients, ordering):
        if pname not in listdct:
            listdct[pname] = []
        for _ in applies:
            listdct[pname].append(target * gradient ** 0.5)
    dct = {}
    for k, v in listdct.items():
        dct[k] = abs(np.array(v).mean())
    return dct
