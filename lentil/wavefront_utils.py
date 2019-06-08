import random
import multiprocessing
import prysm
import lentil.constants_utils
from lentil.constants_utils import *
from lentil import wavefront_config
from lentil.focus_set import FocusSet
# from lentil.focus_set import estimate_focus_jitter
from lentil.wavefront_config import SPACIAL_FREQS, MODEL_WVLS, EXTREME_FOCUS_WEIGHT, HIGH_FREQUENCY_WEIGHT, CHEAP_LOCA_CUTOFF, USE_EXISTING_PARAMETER_IF_FIXED

# use_cuda = wavefront_config.USE_CUDA
from lentil.wavefront_test import try_wavefront
from lentil import wavefront_test


class TerminateOptException(Exception):
    pass


def convert_wavefront_dicts_to_p_dicts(wfdd):
    ps = []
    nomfstops = wfdd['fstops']
    min_fstop = np.inf
    for stop in nomfstops:
        p = {}
        for key, value in wfdd.items():
            split = key.split(":")
            if split[0] == "p.opt":
                if "@" in split[1]:
                    param, nomf = split[1].split("@")
                    if nomf == str(stop):
                        p[param] = value
                else:
                    p[split[1]] = value

        if p['fstop'] < min_fstop:
            min_fstop = p['fstop']
        ps.append(p)
    for p in ps:
        p['base_fstop'] = min_fstop
    return ps


def encode_parameter_tuple(dataset, use_initial=False, use_existing=True,
                           params=(wavefront_config.OPTIMISE_PARAMS, wavefront_config.FIXED_PARAMS)):
    passed_options_ordering = []
    initial_guess = []
    optimise_bounds = []

    prepend = "p.initial:" if use_initial else "p.opt:"

    if use_existing:
        try:
            existing_dict = dataset[0].wavefront_data[-1][1]
        except (AttributeError, IndexError):
            existing_dict = {}
    else:
        existing_dict = {}

    if 'fstop' in params[0]:
        fstop_first_params = []
        fstop_first_params.append('fstop')
        for param in params[0]:
            if param != 'fstop':
                fstop_first_params.append(param)
    else:
        fstop_first_params = params[0]
    fstops = [data.exif.aperture for data in dataset]
    base_fstop = min(fstops)

    for pnum, pname in enumerate(fstop_first_params):
        paramconfigtup = wavefront_config.PARAMS_OPTIONS[pname]
        config_low , config_initial, config_high, f_lambda, optim_per_focusset, scale = paramconfigtup
        scale *= wavefront_config.SCALE_EXTRA[pnum]

        if optim_per_focusset == wavefront_config.OPT_PER_FOCUSSET:
            loops = len(dataset)
        else:
            loops = 1
            passed_options_ordering.append((pname, tuple(range(len(dataset)), ), None))

        for a in range(loops):
            if optim_per_focusset == wavefront_config.OPT_PER_FOCUSSET:
                passed_options_ordering.append((pname, (a,), None))
                existingkey = prepend + pname + "@{}".format(dataset[a].exif.aperture)
                try:
                    hint_dict = dataset[a].hints
                except (AttributeError, IndexError):
                    hint_dict = {}
            else:
                existingkey = prepend + pname
                hint_dict = {}

            fmul = f_lambda(fstops[a], base_fstop)

            if pname == "df_offset":
                if existingkey in existing_dict:
                    initial = existing_dict[existingkey] / fmul
                    low = initial - 6
                    high = initial + 6
                elif pname in hint_dict:
                    initial = hint_dict[pname] / fmul
                    low = initial - 30
                    high = initial + 30
                else:
                    low = min(dataset[a].focus_values) - 2
                    high = max(dataset[a].focus_values) + 2
                    initial = (low + high) * 0.5

            elif pname == 'df_step':
                    if existingkey in existing_dict:
                        initial = existing_dict[existingkey] / fmul
                    elif pname in hint_dict:
                        r = dataset[a].exif.aperture / base_fstop
                        initial = hint_dict[pname] * base_fstop * r
                    else:
                        initial = config_initial / fmul
                    low = initial / wavefront_config.DF_STEP_TOLERANCE
                    high = initial * wavefront_config.DF_STEP_TOLERANCE

            elif pname == 'fstop':
                if existingkey in existing_dict:
                    existfstop = existing_dict[existingkey]
                    initial = existfstop / fmul
                    if optim_per_focusset == wavefront_config.OPT_SHARED:
                        corr = existfstop / fstops[a]
                        for ix in range(len(fstops)):
                            fstops[ix] *= corr
                    else:
                        fstops[a] = existfstop
                    if a == 0:
                        base_fstop = existfstop
                else:
                    initial = dataset[a].exif.aperture / fmul
                low = config_low
                high = config_high

            else:
                if existingkey in existing_dict:
                    initial = existing_dict[existingkey] / fmul
                elif pname in hint_dict:
                    initial = hint_dict[pname] / fmul
                else:
                    initial = config_initial / fmul
                low = min(config_low, initial)
                high = max(config_high, initial)
            initial_guess.append(initial * scale)
            optimise_bounds.append((low * scale, high * scale))

    return initial_guess, optimise_bounds, passed_options_ordering


def decode_parameter_tuple(tup, passed_options_ordering, dataset, use_existing_fixed=USE_EXISTING_PARAMETER_IF_FIXED,
                           params=(wavefront_config.OPTIMISE_PARAMS, wavefront_config.FIXED_PARAMS)):
    # print(use_existing_fixed, USE_EXISTING_PARAMETER_IF_FIXED)
    # exit()
    ps = []
    for _ in dataset:
        ps.append({})
    popt = {}
    opt_fstops = [data.exif.aperture for data in dataset]
    nominal_fstops = opt_fstops.copy()
    if 'fstop' in wavefront_config.OPTIMISE_PARAMS:
        # Avoid large gradients due to large pupil change with fstop
        for tix, ((name, setapplies, fieldapplies), val) in enumerate(zip(passed_options_ordering, tup)):
            config_low , config_initial, config_high, f_lambda, optim_per_focusset, scale = wavefront_config.PARAMS_OPTIONS[name]
            scale *= wavefront_config.SCALE_EXTRA[tix]
            if name == 'fstop':
                for a in setapplies:
                    opt_fstops[a] = val / scale * dataset[a].exif.aperture

    base_fstop = min(opt_fstops)

    for tix, ((name, setapplies, fieldapplies), val) in enumerate(zip(passed_options_ordering, tup)):
        config_low , config_initial, config_high, f_lambda, optim_per_focusset, scale = wavefront_config.PARAMS_OPTIONS[name]
        scale *= wavefront_config.SCALE_EXTRA[tix]
        for a in setapplies:
            if name == "fstop":
                fmul = f_lambda(nominal_fstops[a], base_fstop)
            else:
                fmul = f_lambda(opt_fstops[a], base_fstop)
            if fieldapplies is None:
                ps[a][name] = val * fmul / scale
                if len(setapplies) == 1:
                    popt["{}{}".format(name, a)] = val * fmul / scale
            else:
                ps[a]["{}.{}".format(name, fieldapplies)] = val * fmul / scale
                popt["{}{}.{}".format(name, a, fieldapplies)] = val * fmul / scale
        if len(setapplies) > 1:
            if name == "fstop":
                popt["FSTOP_CORR"] = val / scale
                for p in ps:
                    p['fstop_corr'] = val / scale
            else:
                popt["{}".format(name)] = val * fmul / scale

    pfix = {}
    for name in params[1]:
        config_low , config_initial, config_high, f_lambda, optim_per_focusset, scale = wavefront_config.PARAMS_OPTIONS[name]
        for a, data in enumerate(dataset):
            existing_dict = data.wavefront_data[-1][1]
            existingkey = "p.opt:" + name

            if use_existing_fixed and existingkey in existing_dict:
                value = existing_dict[existingkey]
                ps[a][name] = value
                pfix["{}.{}".format(name, a)] = value
            else:
                fmul = f_lambda(opt_fstops[a], base_fstop)
                # print(ps[a], name, config_initial, fmul)
                ps[a][name] = config_initial * fmul
                pfix["{}.{}".format(name, a)] = config_initial * fmul
    return ps, popt, pfix


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
    focusweights = 1.0 - focus_deviations / max_focus_deviation * (1.0 - EXTREME_FOCUS_WEIGHT)

    # focusrange = (1.0 - EXTREME_FOCUS_WEIGHT) ** 0.5
    # focusweights = 1.0 - np.linspace(-focusrange, focusrange, shape[1]) ** 2

    freqrange = HIGH_FREQUENCY_WEIGHT
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

    if not from_scratch_:
        wfd = focusset.read_wavefront_data(overwrite=True, x_loc=x_loc_, y_loc=y_loc_)
    else:
        wfd = [("", {})]
    # print("wfd ", wfd)
    data = lentil.constants_utils.FocusSetData()
    data.wavefront_data = wfd
    sag_ob = focusset.get_interpolation_fn_at_point(IMAGE_WIDTH / 2, IMAGE_HEIGHT / 2, AUC, SAGITTAL)
    focus_values = sag_ob.focus_data[:]

    if wfd[-1][1] == {} and num == 0:
        tup = focusset.find_best_focus(IMAGE_WIDTH / 2, IMAGE_HEIGHT / 2, axis=MERIDIONAL, _return_step_data_only=True,
                                       _step_estimation_posh=True)
        est_defocus_rms_wfe_step, longitude_defocus_step_um, coc_step, image_distance,\
            subject_distance, fit_peak_y, prysm_offset = tup
        data.hints['df_step'] = est_defocus_rms_wfe_step
        data.hints['df_offset'] = prysm_offset
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
    print(focus_values)
    # print(list(range(len(focus_values))))
    # print(cauchy_peak_x)
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
    for freq in SPACIAL_FREQS:
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

    diff_mtf = diffraction_mtf(SPACIAL_FREQS, focusset.exif.aperture)
    diff_mtf_mean = diff_mtf.mean()
    strehl_ests = mtf_means / diff_mtf_mean

    data.merged_mtf_values = merged_mtf_values
    data.sag_mtf_values = sag_mtf_values
    data.mer_mtf_values = mer_mtf_values
    data.mtf_means = mtf_means
    data.focus_values = focus_values
    data.max_pos = max_pos
    data.strehl_ests = strehl_ests
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
        focussets_ = focussets
        ob_ = ob
        skip_ = skip
        fs_slices_ = fs_slices
        from_scratch_ = from_scratch
        avoid_ends_ = avoid_ends
        x_loc_ = x_loc
        y_loc_ = y_loc
        complex_otf_ = complex_otf

    if not wavefront_config.DISABLE_MULTIPROCESSING:
        pool = multiprocessing.Pool(initializer=init)
        datas = pool.map(_process_focusset, range(len(focussets)))
    else:
        init()
        datas = [_process_focusset(_) for _ in range(len(focussets))]

    if type(focussets[0]) is str:
        datas, focussets = zip(*datas)

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


def build_synthetic_dataset(subsets=1, test_stopdown=0, base_aperture=1.4, stop_inc=1, slices_per_fstop=10, seed=None):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    cauchy_fit_x = float("NaN")
    z_ranges = lambda z: 1.0 - z / 20
    focus_motor_error = 0.08
    defocus_step = 0.55 / base_aperture
    defocus_offset = slices_per_fstop / 2
    valid = False
    with multiprocessing.Pool(processes=wavefront_config.CUDA_PROCESSES) as pool:
        valids = [1.0]
        zfix = 1
        while sum(valids) > 0:
            print("graagl")
            valids = [0.0]
            strehls = []
            base_aperture_error = random.random() * 0.1 + 0.95
            if not zfix:
                z = {}
                zsum = 0.0
                zmaxsum = 0.0
                for paramname in ['z5', 'z6','z7','z8','z9', 'z16', 'z25', 'z36']:
                    if paramname.lower()[0] == 'z' and paramname[1].isdigit():
                        z_range = z_ranges(int(paramname[1:]))
                        z_random = (random.random() * (z_range * 2) - z_range)
                        z[paramname] = z_random / base_aperture
                        zsum += np.abs(z_random)
                        zmaxsum += z_range
                if not 0.32 < zsum < 0.65:
                    print("Invali WFE {:.3f}! repeating...".format(zsum))
                    valids.append(1.0)
                    if not zfix:
                        continue
            else:
                z = {'z9': 0.022253602152529917, 'z16': 0.24229720434201457, 'z25': 0.02173054601719872, 'z36': -0.014062440213119979}

                z = {}
                for z_ in range(5, 37):
                    range_ = 1 * (7.0 / z_) ** 3
                    z["z{}".format(z_)] = random.random() * range_ - range_ / 2

            nom_focus_values = np.arange(slices_per_fstop)
            dataset = []
            tests = max(test_stopdown, subsets)
            fstops = [base_aperture * 2 ** (n / 2 * stop_inc) for n in range(tests)]
            for fstop in fstops:
                if fstop != base_aperture:
                    individual_fstop_error = 1.0 # random.random() * 0.05 + 0.975
                else:
                    individual_fstop_error = 1.0
                actual_focus_values = nom_focus_values + np.random.normal(0.0, focus_motor_error * base_aperture / fstop, nom_focus_values.shape)
                individual_df_step_error = random.random() * 0.2 + 0.9
                model_fstop = fstop * individual_fstop_error * base_aperture_error
                model_step = defocus_step / model_fstop / base_aperture_error * individual_df_step_error
                print("Nominal f#: {:.3f}, actual f#: {:.3f}".format(fstop, model_fstop))
                data = FocusSetData()
                strehls.append(0.0)
                for test_strehl in [False, True]:
                    if test_strehl:
                        arr_sag = np.array(through_focus_sag).T
                        arr_mer = np.array(through_focus_mer).T
                        mtf_means = np.abs(arr_sag + arr_mer).mean(axis=0) / 2
                        cauchy_fit_x = cauchy_fit(nom_focus_values, mtf_means)[1]
                        focus_values_to_test = [cauchy_fit_x]
                    else:
                        focus_values_to_test = actual_focus_values
                    arglst = []
                    for focus in focus_values_to_test:
                        p = {}
                        p['df_offset'] = defocus_offset
                        p['df_step'] = model_step
                        p['fstop'] = model_fstop
                        p['base_fstop'] = base_aperture * base_aperture_error
                        p.update(z)
    # def try_wavefront(defocus, p, mono=False, plot=False, dummy=False, use_cuda=True, index=None,
    #                   strehl_estimate=1.0, mtf_mean=0.7, fftsize_override=None, samples_override=None):
                        args = (focus,
                                p, True, False, False, False, 0, 1.0, 1.0, 384*4, 384)
                        arglst.append(args)

                    if test_strehl:
                        tup = try_wavefront(*args)
                        strehls[-1] = np.mean(np.abs(tup[1]) + np.abs(tup[2])) * 0.5
                    else:
                        outs = pool.starmap(try_wavefront, arglst)
                        # outs = [try_wavefront(*args) for args in arglst]
                        through_focus_sag = [out[1] for out in outs]
                        through_focus_mer = [out[2] for out in outs]
                # print("Fstop {} strehl {}".format(fstop, strehls[-1]))

                data.merged_mtf_values = (arr_sag + arr_mer) / 2
                data.sag_mtf_values = arr_sag
                data.mer_mtf_values = arr_mer
                data.focus_values = nom_focus_values
                data.mtf_means = mtf_means
                data.max_pos = nom_focus_values[np.argmax(data.mtf_means)]
                data.cauchy_peak_x = cauchy_fit_x
                diff_mtf_mean = diffraction_mtf(SPACIAL_FREQS, fstop).mean()
                data.strehl_ests = mtf_means / diff_mtf_mean
                data.weights = get_weights(data.merged_mtf_values.shape, nom_focus_values, cauchy_fit_x)
                data.hints = {'df_step': model_step,# * fstop ,
                              'df_offset': defocus_offset,
                              'loca': 0,
                              'focus_errors': actual_focus_values - nom_focus_values}

                if not 1 < cauchy_fit_x < slices_per_fstop:
                    print(cauchy_fit_x)
                    valids.append(1.0)

                exif = EXIF()
                exif.aperture = fstop
                exif.focal_length_str = "28 mm"
                data.exif = exif

                # estimate_focus_jitter(data)

                dataset.append(data)

                secret_ground_truth = {}
                secret_ground_truth.update(z)
                secret_ground_truth['df_step'] = model_step
                secret_ground_truth['df_offset'] = defocus_offset
                secret_ground_truth['base_fstop'] = base_aperture * base_aperture_error
                secret_ground_truth['fstop'] = model_fstop
                data.secret_ground_truth = secret_ground_truth
            strehls = np.array(strehls)
            monotonic = np.all(np.diff(strehls) > -0.1)
            print("Strehls {}".format(str(strehls)))
            print("Monotonic ish? {}".format(str(monotonic)))
            print("Strehls {:.3f} -> {:.3f}".format(min(strehls), max(strehls)))
            if 1 or monotonic and min(strehls) > -1 and max(strehls) > 0.8:
                pass
            else:
                print("Not valid, rerunning!")
                valids.append(1.0)

        print("Actual focus values: {}".format(str(actual_focus_values)))
        print("Zs: {}".format(str(z)))
        print("F-stops: {}".format(str(fstops)))

    return dataset[:subsets]


# def remove_last_saved_wavefront_data(focusset):
#     focusset.read_wavefront_data(overwrite=True)
#     new_wfd = focusset.wavefront_data[:-1]
#     save_wafefront_data(focusset.get_wavefront_data_path(), new_wfd, overwrite=True)



def get_loca_kernel(p, normalise_shift, soft_limit=True):
    if p == 0:
        return np.array([1]), 0, 0
    fabsoka = np.clip(np.abs(p['loca']) * 1e2, 1e-12, np.inf)

    e_cutoff = CHEAP_LOCA_CUTOFF
    inc = 2.0 / fabsoka ** 0.2 * p['df_step'] * p['base_fstop'] ** 0.5

    if inc < 0.1 and soft_limit:
        inc = 0.03 + 0.03 / (0.06 / inc) ** 2
    else:
        if inc < 0.05:
            raise ValueError("Too much LOCA!")

    raw_kernel_size = max(13.5, -np.log(e_cutoff) / inc)
    required_kernel_size = int(raw_kernel_size + 1.0)

    # print(raw_kernel_size)

    centred_xvals = np.arange(required_kernel_size * 2 -1) - required_kernel_size + 1
    # print(centred_xvals)

    e = np.exp(-np.arange(0, required_kernel_size * inc + 1, inc))[:int(required_kernel_size)]
    if required_kernel_size < len(e) - 1:
        e[int(required_kernel_size):] = 0

    kernel_raw = np.concatenate((np.zeros((required_kernel_size-1,)), e))
    # print(len(kernel_raw))
    hann = np.cos(np.clip(centred_xvals / raw_kernel_size * np.pi * 1.1, -np.pi, np.pi)) + 1
    kernel_windowed = kernel_raw * hann
    kernel = kernel_windowed / kernel_windowed.sum()  # Normalise

    # plt.plot(centred_xvals, hann)
    # plt.plot(centred_xvals, kernel_windowed)
    # plt.plot(centred_xvals, kernel_raw)
    # plt.show()

    xvals = np.arange(len(kernel))
    if p['loca'] < 0:
        kernel = np.flip(kernel)
    if normalise_shift:
        interpxvals = np.arange(-10, len(kernel) + 10)
        shift = - ((xvals - required_kernel_size + 1) * kernel).sum()
        pad = np.zeros((10,))
        padded_kernel = np.concatenate((pad, kernel, pad))

        shift_ay = interpolate.InterpolatedUnivariateSpline(interpxvals,
                                                            padded_kernel, k=1)(xvals - shift)
    else:
        shift_ay = kernel
    # print(xvals)
    # print(centred_xvals)
    # print(2,kernel)
    # print(p['loca'], shift_ay)

    useful = shift_ay > 0
    # print(useful)
    # print(3,shift_ay)
    # print(4,useful)
    needed_coeffs = centred_xvals[useful]

    # print(5,needed_coeffs)
    # print(needed_coeffs)

    # print(min(needed_coeffs), max(needed_coeffs))
    clipped_kernel = shift_ay[useful]
    # print(clipped_kernel)
    addleft, addright = max(needed_coeffs), -min(needed_coeffs)
    # print(addleft, addright, p['loca'])
    # print()
    # print()
    # print()
    # print()
    # print()

    # print()_
    # exit()
    return clipped_kernel, addleft, addright



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


def optimise_loca_colvolution_coeffs():
    colournumber = 0
    slices = 32

    def convolve():
        kernel, _, _ = get_loca_kernel(p, True)

        output = []
        for row in baseline_data:
            output.append(np.convolve(row, kernel, 'valid'))
        output = np.array(output)
        return output

    def calccost_e(params):
        param, b = params
        cost = ((convolve(param, b) - fancy_loca_data) ** 2).mean() * 1e3
        print(param, b, cost)
        return cost

    def calccost(conv):
        output = []
        # kernel = np.zeros((len(conv)+1))
        # kernel[int(len(conv)/2)] = 1.0 - conv.sum()
        # print(kernel, kernel.shape)
        actual_conv_array = conv
        for row in baseline_data:
            # print(row)
            output.append(np.convolve(row, actual_conv_array, 'valid'))
            # print(output[-1])
        output = np.array(output)
        cost = ((output - fancy_loca_data) ** 2).mean() * 1e3
        print(cost)
        return cost



    # Make our null kernel
    # nullkernel = np.zeros((sidelen*2+1,))
    # nullkernel[sidelen] = 1

    # Get a mp pool
    pool = multiprocessing.Pool()

    if len(MODEL_WVLS) < 15:
        raise Exception("Not really enough frequencies to do this properly!")
    loca = 0

    for loca in np.linspace(0.02,6, 2):
        fstop = 1.25
        p = {'df_offset': slices/2,
             'df_step': 0.42,
             'fstop': fstop,
             # 'z9': 0.17,
             'base_fstop': fstop,
             'samples': 256}
        pass_p = p.copy()
        pass_p['loca'] = loca
        _, lside, rside = get_loca_kernel(pass_p, True)
        # continue

        baseline_data = []
        fancy_loca_data = []
        fancy_loca_focus_values = list(np.arange(slices))
        baseline_focus_values = list(np.arange(-lside, slices + rside))

        print("Generating baseline data....")
        print(baseline_focus_values)
        arglst = list(zip(baseline_focus_values,
                          [p] * len(baseline_focus_values),
                          [True] * len(baseline_focus_values)))
        outs = pool.starmap(try_wavefront, arglst)
        baseline_data, _, _, _ = zip(*outs)

        # Reference alterations
        p['loca'] = loca

        print("Generating loca data....")
        print(fancy_loca_focus_values)
        arglst = list(zip(fancy_loca_focus_values, [p] * len(fancy_loca_focus_values)))
        outs = pool.starmap(try_wavefront, arglst)
        fancy_loca_data, _, _, _ = zip(*outs)
        fancy_loca_data = np.array(fancy_loca_data).T
        baseline_data = np.array(baseline_data).T

        # r = []
        # for row , b in zip(baseline_data[-1:], fancy_loca_data[-1:]):
        # if 1:
            # row = baseline_data.mean(axis=0)
            # b = fancy_loca_data.mean(axis=0)
            # stacklst = []
            # for rolln in range(sidelen*2+1):
            #     stacklst.append(baseline_data[:,rolln:rolln+slices])
            # stack = np.array(stacklst)
            # print(row)
            # print(stack)
            # print(b)
            # r.append(np.linalg.lstsq(stack.T, fancy_loca_data, rcond=None)[0])
        # r = np.array(r)
        # plt.plot(r.mean(axis=0))
        # plt.plot(r.mean(axis=0), 's')
        # plt.show()
        # exit()


        if 0:
            opt = optimize.minimize(calccost_e, [0.33, 0.0], method="BFGS")
            a, b = opt.x
            # opt = optimize.minimize(calccost, nullkernel, method='BFGS')
            # print(opt)
            # plt.plot(make_actual_kernel(opt.x))
            # plt.plot(opt.x)
        else:
            a, b = 0.32, 0
        # e = np.exp(-np.arange(0, sidelen * x / p['loca'], x / p['loca']))
        # e = e / e.sum()
        colour = COLOURS[colournumber % len(COLOURS)]
        colournumber += 1
        print(fancy_loca_focus_values)
        print(convolve().mean(axis=0).shape)
        plt.plot(fancy_loca_focus_values,
                 convolve().mean(axis=0),
                 '--',
                 label="foca {}".format(fstop),
                 color=colour)
        plt.plot(np.array(fancy_loca_focus_values),
                 fancy_loca_data.mean(axis=0),
                 '-',
                 label="loka {}".format(fstop),
                 color=colour)
        # plt.plot(np.array(baseline_focus_values),
        #          baseline_data.mean(axis=0),
        #          '-.',
        #          label="noloka {}".format(fstop),
        #          color=colour)

    pool.close()

    # plt.plot(np.concatenate((np.zeros((sidelen,)), e)))
    # plt.plot(np.concatenate((np.zeros((sidelen,)), e)))
    plt.legend()
    plt.show()
    pass


def plot_nominal_psf(*args, wfd={}):
    # plt.cla()
    # plt.close()
    defocuses = [-2.4, 0, 2.4, 4.8]
    defocuses = np.linspace(-10, 10, 5)
    f, axes = plt.subplots(len(args), len(defocuses), sharey=True, sharex=True)
    if len(args) == 1:
        axes = axes
    min_fstop = min(p['fstop'] for p in args)
    for na, dct in enumerate(args):
        df_offset = dct["df_offset"]
        for nd, defocus in enumerate(defocuses):
            alter = (dct['fstop'] / min_fstop) ** 2
            # alter = 1
            s = wavefront_test.TestSettings(defocus / alter + df_offset, dct)
            s.return_type = wavefront_test.RETURN_PSF
            s.pixel_vignetting = True
            s.lens_vignetting = True
            psf = try_wavefront(s).psf
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
            if len(args) == 1:
                ax = axes[nd]
            else:
                ax = axes[na, nd]
            psf.plot2d(ax=ax, fig=f, axlim=60)
    plt.show()
