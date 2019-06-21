import time
import random
import signal
import multiprocessing
from multiprocessing.pool import ThreadPool
from collections import OrderedDict
import cupy
import prysm
import numpy as np
# import nlopt

import lentil.wavefront_test
from lentil import wavefront_utils
from lentil.constants_utils import *
from lentil.wavefront_utils import TerminateOptException, decode_parameter_tuple, encode_parameter_tuple, get_loca_kernel
from lentil.wavefront_test import try_wavefront, plot_pixel_vignetting_loss, get_processing_details, TestSettings, TestResults
from lentil import wavefront_test
from lentil.wavefront_config import SPACIAL_FREQS, MODEL_WVLS, OPTIMISE_PARAMS, PARAMS_OPTIONS, FIXED_PARAMS, COST_MULTIPLIER, SAVE_RESULTS, DISABLE_MULTIPROCESSING, MAXITER, USE_CHEAP_LOCA_MODEL, CHEAP_LOCA_NORMALISE_FOCUS_SHIFT
from lentil import wavefront_config
from lentil.focus_set import save_wafefront_data, scan_path, read_wavefront_file

keysignal = ""
exit_signal = False

# plot_pixel_vignetting_loss()
# exit()


def testme(samples, loops=16*5, Q=2, *args, **kwargs):
    for _ in range(loops):
        psflst = []
        for wl in np.linspace(0.4, 0.7, 9):
            pupil = prysm.FringeZernike(np.ones(48)*0.1, samples=samples, opd_unit="um", wavelength=wl)
            psflst.append(prysm.PSF.from_pupil(pupil, efl=3, Q=Q))
        allpsf = prysm.PSF.polychromatic(psflst)
        mtf = prysm.MTF.from_psf(allpsf)
        mtf.exact_sag(np.linspace(0, 10, 14))
        mtf.exact_tan(np.linspace(0, 10, 14))

def testmul(samples, total_loops=16*5, Q=2):

    pool.starmap(testme, [(samples, int(total_loops / 8), Q)] * 8)

pool = multiprocessing.Pool(processes=8)

def run_model():
    defocuses = 16
    fnos = 5
    wvls = 9
    mtfs = []
    for i in range(defocuses):
        for j in range(fnos):
            for k in range(wvls):
                pu = prysm.FringeZernike(np.random.rand(48), samples=128)
                mt = prysm.MTF.from_pupil(pu, 1)
                # mtfs.append(mt.slices().x[1][20])


# run_model()
# exit()
def testq():
    for q in np.linspace(2,5,4):
        psflst = []
        for wl in np.linspace(0.4, 0.7, 9):
            pupil = prysm.FringeZernike(np.zeros(48), samples=64, opd_unit="um", wavelength=wl)
            psf = prysm.PSF.from_pupil(pupil, efl=16, Q=q)
            psflst.append(psf)
        allpsf = prysm.PSF.polychromatic(psflst)
        x, y = prysm.MTF.from_psf(allpsf).sag
        plt.plot(x,y, label="{}".format(q))
    plt.xlim(0,125)
    plt.legend()
    plt.title("MTF vs Q (f/16 zero WFE)")
    plt.xlabel("Spacial Frequency (lp/mm)")
    plt.ylabel("MTF")
    plt.ylim(0,1)
    plt.show()

# testq()
# exit()




def _wait_for_keypress():
    global keysignal
    global exit_signal
    while not exit_signal and keysignal.lower() not in ['x', 's']:
        f = input()
        keysignal = f


def _save_data(ps, initial_ps, set, dataset, fun, nit, nfev, success, starttime, autosave=False, quiet=False):
    if not quiet:
        print("Writing wavefront data...")
    all_p = {}
    all_p_init = {}
    for p_list, dct in [(ps, all_p), (initial_ps, all_p_init)]:
        for p, data in zip(ps[:], set[:]):
            for key, value in p.items():
                if key in PARAMS_OPTIONS:
                    opt_per_focusset = PARAMS_OPTIONS[key][4]
                elif key == 'fstop_corr':
                    opt_per_focusset = wavefront_config.OPT_PER_FOCUSSET
                else:
                    opt_per_focusset = wavefront_config.OPT_SHARED
                if opt_per_focusset == wavefront_config.OPT_PER_FOCUSSET:
                    fstop = data.exif.aperture
                    dct["{}@{}".format(key, fstop)] = value
                elif opt_per_focusset == wavefront_config.OPT_SHARED:
                    if p is ps[0]:
                        dct[key] = value

    # for p, pinit, focusset, data in zip(ps, initial_ps, set, dataset):
    outdict = {}
    extra = {'zernike.numbering': wavefront_config.ZERNIKE_SCHEME,
             'prysm.samples': wavefront_config.DEFAULT_SAMPLES,
             'fstops': [_.exif.aperture for _ in set],
             'frequencies': ["{:.6f}".format(_) for _ in SPACIAL_FREQS],
             'wavelengths': list(MODEL_WVLS),
             'wfe.unit': "wavelengths (575nm)",
             'final.cost': fun,
             'num.iterations': nit,
             'num.fevals': nfev,
             'success': str(success),
             'runtime': time.time()-starttime,
             'x.loc': dataset[0].x_loc,
             'y.loc': dataset[0].y_loc,
             'cheap.loca.model':USE_CHEAP_LOCA_MODEL,
             'cheap.loca.model.normalise':CHEAP_LOCA_NORMALISE_FOCUS_SHIFT,
             'cost.multiplier': COST_MULTIPLIER,
             'total.num.fevals': count}
    # for freq, arr in zip(SPACIAL_FREQS, data.weights):
    #     extra["Weights.frequency(cy/px):{:.6f}".format(freq)] = list(arr)
    # for freq, arr in zip(SPACIAL_FREQS, data.chart_mtf_values):
    #     extra["Chart.MTFs.frequency(cy/px):{:.6f}".format(freq)] = list(arr)

    for data in dataset:
        extra["fields@{}".format(data.exif.aperture)] = list(data.focus_values)

    for key, value in all_p_init.items():
        outdict["p.initial:"+key] = value

    for key, value in all_p.items():
        outdict["p.opt:"+key] = value

    if hasattr(dataset[0], 'secret_ground_truth'):
        for key, value in data.secret_ground_truth.items():
            outdict["p.truth:"+key] = value

    for key in ps[0].keys():
        nodigitskey = key.split(".")[0]
        try:
            outdict["p.opt.per.fstop:"+nodigitskey] = str(PARAMS_OPTIONS[nodigitskey][4])
        except KeyError:
            pass

    outdict.update(extra)

    if autosave:
        wavefront_data = [("Autosaved wavefront data", outdict)]
    else:
        wavefront_data = [("Optimised wavefront data", outdict)]
    try:
        path = set[0].get_wavefront_data_path(seed=wavefront_config.RANDOM_SEED)
    except AttributeError:
        path = "wavefront_results/"
    # path = "/home/sam/"
    if 1 and SAVE_RESULTS:
        suffix = "x{}.y{}".format(dataset[0].x_loc, dataset[0].y_loc)
        if autosave:
            suffix += ".autosave"
        save_wafefront_data(path, wavefront_data, suffix=suffix, quiet=False)
    else:
        log.warning("Results not saved!")


def _split_array(array, split):
    subs = []
    lastsize = 0

    for size in split:
        subs.append(array[:, lastsize:lastsize + size])
        lastsize += size
    return subs


def _calculate_cost(modelall, chartall, split, weightsall, count=1):
    # Calculate line gradients
    skewcosts = []
    # meansquares = (((np.abs(modelall) - np.abs(chartall)) * 1e3) ** 2 * weightsall).mean() * 1e-4
    magdiffs = (abs(modelall) - abs(chartall)) * 2
    realdiffs = np.real(modelall - chartall)
    imagdiffs = np.imag(modelall - chartall)
    complex_sq = (realdiffs**2 + imagdiffs**2)
    # meansquares = ((np.abs(modelall - chartall) * 1e3) ** 2 * weightsall).mean() * 1e-4
    meansquares = ((complex_sq + magdiffs ** 2) * weightsall).mean()
    return meansquares * wavefront_config.COST_WEIGHT_MEAN_SQUARES, 0, meansquares * wavefront_config.COST_WEIGHT_MEAN_SQUARES,0
    models = _split_array(modelall, split)
    charts = _split_array(chartall, split)
    line_trends = []
    modelpeaks = []
    chartpeaks = []
    quadfits = []
    for model, chart in zip(models, charts):
        x = np.linspace(-1, 1, model.shape[1])
        line_err_grads = []
        line_err_cs = []
        derivdiffs = []
        for modelline, chartline in zip(model, chart):
            derivdiffs.append(np.diff(modelline) - np.diff(chartline))
            line_diff = modelline - chartline
            line_err_fit = np.polyfit(x, line_diff, 1)
            quad_err_fit = np.polyfit(x, modelline - chartline, 2)
            quadfits.append(quad_err_fit*np.array([1,1,1]))
            linegradcost = (line_err_fit[0]**2).sum()
            blinecost = (line_err_fit[1]**2).sum()
            linegrad = (line_err_fit[0]**2).sum()
            linec = (line_err_fit[1]**2).sum()
            line_err_grads.append(linegrad)
            line_err_cs.append(linec)

            modelpeak = np.roots(np.polyder(np.polyfit(x, modelline, 2, w=chartline**2)))[0]
            chartpeak = np.roots(np.polyder(np.polyfit(x, chartline, 2, w=chartline**2)))[0]
            # print(modelpeak, chartpeak, line_err_fit)
            # plt.gca()
            # plt.plot(x, modelline)
            # plt.plot(x, chartline)
            # plt.show()
            modelpeaks.append(modelpeak)
            chartpeaks.append(chartpeak)
        # modelskew = np.polyfit(SPACIAL_FREQS, np.clip(modelpeaks, -2, 2), 1)
        # chartskew = np.polyfit(SPACIAL_FREQS, np.clip(chartpeaks, -2, 2), 1)
        # skewcost = ((modelskew - chartskew) ** 2).sum() * 1e0

        # skewcosts.append(skewcost)

        line_trend = np.polyfit(SPACIAL_FREQS, line_err_grads, 1)
        line_trends.append(line_trend[0])

    # linegradcost = (np.array(line_err_grads) * 1e3).mean()
    linegradcost = (line_trends[0] * 1e3)**2  # Gradient of trend

    # linegradcost = ((np.array(derivdiffs)*1e3) ** 2).mean() * 1e-6

    linegradcost = ((np.array(quadfits)*100)**2).mean()
    # if count % 100 == 0:
    #     for a, b in zip(modelpeaks, chartpeaks):
    #         print(a * 100, b* 100)

    chartweights = chartall / (chartall.mean())

    peak_mnsq = (((np.array(modelpeaks) - np.array(chartpeaks))*1e3)**2).mean() * 1e-6

    final = (peak_mnsq * wavefront_config.COST_WEIGHT_PEAK_LOCATIONS +
             linegradcost * wavefront_config.COST_WEIGHT_LINE_DIFF
             + meansquares * wavefront_config.COST_WEIGHT_MEAN_SQUARES)
    return (final, linegradcost * wavefront_config.COST_WEIGHT_LINE_DIFF,
             meansquares * wavefront_config.COST_WEIGHT_MEAN_SQUARES,
            peak_mnsq * wavefront_config.COST_WEIGHT_PEAK_LOCATIONS,)


def _jiggle_zeds(x, passed_options_ordering):
    # print("Jiggling Zeds!")
    # print("In params:", list(x))
    zswaplst = [ix for ix, (s, tup) in enumerate(passed_options_ordering) if s.startswith('z')]
    zswaprandom = zswaplst.copy()
    random.shuffle(zswaprandom)
    new = list(x)
    for ix in range(len(zswaplst)):
        new[zswaplst[ix]] = x[zswaprandom[ix]]
    # print("Jiggled params:", new)
    return new


def _randomise_zeds(x, passed_options_ordering):
    # print("Jiggling Zeds!")
    # print("In params:", list(x))
    zswaplst = [ix for ix, (s, tup, _) in enumerate(passed_options_ordering) if s.startswith('z')]
    sum_ = 0
    zswaprandom = zswaplst.copy()
    # random.shuffle(zswaprandom)
    # valid = False
    max_ = -np.inf
    for ix in zswaplst:
        sum_ += x[ix]
        if np.abs(x[ix]) > max_:
            max_ = np.abs(x[ix])
    for _ in range(1000):
        new = list(x)
        newsum = 0
        newmax_ = -np.inf
        for ix in zswaplst:
            rn = (random.random()-0.5) * max_ * 2.2
            new[ix] = rn
            newsum += rn
        if np.abs(newsum / sum_ - 1) < 0.1:
            break

    # print("Jiggled params:", new)
    return new


def estimate_wavefront_errors(set, fs_slices=16, skip=1, from_scratch=False, processes=None, plot_gradients_initial=None,
                              x_loc=None, y_loc=None, complex_otf=False, avoid_ends=1):
    if hasattr(set[0], 'merged_mtf_values'):
        dataset = set
        if not from_scratch:
            for data in dataset:
                try:
                    pass
                    entry, number = scan_path(data.get_wavefront_data_path(wavefront_config.RANDOM_SEED))
                except FileNotFoundError:
                    break
                # entry, number = scan_path('wavefront_results/')
                wfd = read_wavefront_file(entry.path)
                if len(wfd):
                    select_wfd = {}
                    dct = wfd[-1][1]
                    for key, val in dct.items():
                        if 1 or key.lower().startswith('p.opt:z') and key[7].isdigit():
                            select_wfd[key] = val
                        # elif key.lower().startswith('p.opt:df_') and key[7].isdigit():
                        #     select_wfd[key] = val

                data.wavefront_data = [("", select_wfd)]
        input = "DATASET"
    elif hasattr(set[0], 'fields') or type(set[0]) is str:
        dataset, focussets = wavefront_utils.pre_process_focussets(set, fs_slices, skip, avoid_ends=avoid_ends, from_scratch=from_scratch,
                                                        x_loc=x_loc, y_loc=y_loc, complex_otf=complex_otf)
        if type(set[0]) is str:
            set = focussets
        input = "FOCUSSET"
    else:
        raise ValueError("Unknown input!")

    count = 0
    it_count = 0
    iterations = 0
    prev_iterations = -1
    first_it_evals = 0
    total_iterations = 0
    timings = {}
    t_prep = 0
    t_calc = 0
    t_run = 0
    cpu_gpu_fftsize_boundary = wavefront_config.CPU_GPU_ARRAYSIZE_BOUNDARY
    # extend_model = 15
    last_x = None
    lastcost = np.inf
    allsubprocesstimes = []
    allevaltimes = []
    process_details_cache = []
    starttime = time.time()

    chart_lst_sag = []
    chart_lst_mer = []
    for data in dataset:
        chart_lst_sag.append(data.sag_mtf_values)
        chart_lst_mer.append(data.mer_mtf_values)
    strehlest_lst = [data.strehl_ests for data in dataset]
    chart_sag_concat = np.concatenate(chart_lst_sag, axis=1)
    chart_mer_concat = np.concatenate(chart_lst_mer, axis=1)
    chart_mtf_means_concat = np.abs((chart_mer_concat + chart_sag_concat) * 0.5).mean(axis=0)
    strehl_est_concat = np.concatenate(strehlest_lst, axis=0)

    split = [ay.shape[0] for ay in strehlest_lst]
    print("Focusset sizes:", split)

    focus_values_sequenced = [dataset[0].focus_values]
    weights_concat = np.concatenate([f.weights for f in dataset], axis=1)
    for f in dataset[1:]:
        new_focus_values = f.focus_values + max(focus_values_sequenced[-1] - f.focus_values[0])
        focus_values_sequenced.append(new_focus_values)
    focus_values_concat = np.concatenate(focus_values_sequenced)

    ######################
    # Set up live plotting

    if wavefront_config.LIVE_PLOTTING and plot_gradients_initial is None:
        plt.show()
        fig, axesarray = plt.subplots(2, 2, sharey=False, sharex=True, gridspec_kw={'height_ratios': [3, 1]})
        subplots = [[{}, {}], [{}, {}]]
        chart_axes = [[SAGITTAL, MERIDIONAL], [SAGITTAL, MERIDIONAL]]
        fns = [[np.abs, np.abs], [np.imag, np.imag]]
        titles = [["Sagittal MTF", "Meridional MTF"], ["Sagittal Imag", "Meridional Imag"]]
        chart_values = [[chart_sag_concat, chart_mer_concat], [chart_sag_concat, chart_mer_concat]]
        limits = [[(0, 1)]*2, [(-0.3, 0.3)]*2]

        for nrow, (axespair, plotdictpair, chartaxespair, fnpair, titlepair, chartvalpair, limitspair) in enumerate(zip(axesarray, subplots, chart_axes, fns, titles, chart_values, limits)):
            for nplot, (axes, plotdict, chart_axis, fn, title, chart_vals, limit) in enumerate(zip(axespair, plotdictpair, chartaxespair, fnpair, titlepair, chartvalpair, limitspair)):
                plotdict['lines'] = []
                axes.set_title(title)
                axes.set_ylim(*limit)
                axes.hlines(0, min(focus_values_concat), max(focus_values_concat))
                plotdict['plot_process_fn'] = fn
                for n, (freq, vals, alpha) in enumerate(zip(wavefront_utils.SPACIAL_FREQS[wavefront_config.PLOT_LINES],
                                                                chart_vals[wavefront_config.PLOT_LINES],
                                                                wavefront_config.PLOT_ALPHAS)):
                    color = NICECOLOURS[n % 4]
                    axes.plot(focus_values_concat, fn(vals), '--', label="Chart {:.2f}".format(freq), color=color, alpha=0.5)
                    line,  = axes.plot(focus_values_concat, fn(vals), '-', label="Model {:.2f}".format(freq), color=color, alpha=alpha)
                    plotdict['lines'].append(line)


    total_slices = len(focus_values_concat) + len(dataset)

    #####################
    # Set up process pools

    if processes is None and not DISABLE_MULTIPROCESSING:
        prysm.zernike.cupyzcache = {}
        multi = True
        optimal_processes = multiprocessing.cpu_count()
        processes_opts = np.arange(4, 15)
        loop_ops = np.ceil(total_slices / processes_opts)
        efficiency = total_slices / (loop_ops * processes_opts)
        cpu_efficiency_favour = 1 - processes_opts-optimal_processes**2
        print(processes_opts)
        print(loop_ops)
        print(efficiency)
        processes = max(zip(efficiency, cpu_efficiency_favour, processes_opts))[2]
        print("Using {} processes (for {} slices)".format(processes, total_slices))
        # cpupool = multiprocessing.Pool(processes=processes if wavefront_config.USE_CUDA else processes)

        def init():
            def shutupshop(*args, **kwargs):
                pass

            signal.signal(signal.SIGTERM, shutupshop)
            signal.signal(signal.SIGINT, shutupshop)
            signal.signal(signal.SIGQUIT, shutupshop)

        if wavefront_config.CPU_ONLY_PROCESSES is not None:
            processes = wavefront_config.CPU_ONLY_PROCESSES
        # pool = multiprocessing.pool.ThreadPool
        pool = multiprocessing.Pool
        cpupool = pool(processes=wavefront_config.CUDA_CPU_PROCESSES if wavefront_config.USE_CUDA else processes, initializer=init)
        cudapool = pool(processes=wavefront_config.CUDA_PROCESSES, initializer=init)


    else:
        multi = False

    initial_ps = []

    last_params = None

    def prysmfit(*params, plot=False, return_timing_only=False, no_process_details_cache=False):
        """
        Provide inner loop for scipy optimise

        :param params: Iterable of parameters
        :param plot: Explicitly plot results
        :return: cost (unless return_dicts) is True
        """
        # Use outer scope for passing progress parameters
        t = time.time()

        nonlocal count
        nonlocal it_count
        nonlocal initial_ps
        nonlocal prev_iterations
        nonlocal lastcost
        nonlocal first_it_evals
        nonlocal last_params
        nonlocal t_prep
        nonlocal t_run
        nonlocal t_calc

        # Check deltas
        orders = []
        names = []
        if last_params is None:
            last_params = params[0]

        for oldval, val, (pname, _, _) in zip(last_params, params[0], passed_options_ordering):
            try:
                if val - oldval != 0:
                    try:
                        orders.append(int(np.log10((val - oldval) * 0.33)))
                    except FloatingPointError:
                        orders.append("")
                else:
                    orders.append("")
            except (ZeroDivisionError, OverflowError, ValueError):
                orders.append("")
            names.append(pname)

        last_params = params[0]

        # print(repr(params[0]))
        ps, popt, pfix = decode_parameter_tuple(params[0], passed_options_ordering, dataset)

        try:
            onlyset = [params[1]]
        except IndexError:
            onlyset = list(range(len(dataset)))

        arglists = [list() for _ in range(len(dataset))]
        all_arg_lst = []
        gpu_arg_lst = []
        cpu_arg_lst = []
        data_for_args = []
        all_focus_offsets = []

        evalstart = time.time()
        refs = []
        gpu_fftsizes = []
        cpu_fftsizes = []
        qs = []
        loop_base_fstop = min((p['fstop'] for p in ps))
        index_counter = 0
        slice_counter = -1

        for nd, (data, p) in enumerate(zip(dataset, ps)):
            p['cauchy_peak_x'] = data.cauchy_peak_x
            try:
                mono = data.hints['loca'] == 0
            except (KeyError, AttributeError):
                mono = False
            dummy = nd not in onlyset
            p['base_fstop'] = loop_base_fstop

            if 'df_each' in OPTIMISE_PARAMS:
                focus_offsets = np.zeros((len(data.focus_values),))
                for key, value in p.items():
                    if key.startswith("df_each."):
                        num = int(key.split(".")[1])
                        focus_offsets[num] = value * 10
                all_focus_offsets.append(focus_offsets)
                focus_values = np.add(data.focus_values, focus_offsets)
            else:
                focus_offsets = np.zeros((len(data.focus_values),))
                all_focus_offsets.append(focus_offsets)
                focus_values = data.focus_values
            expanded_focus_values = list(focus_values)
            if p['fstop'] == loop_base_fstop:
                sub_focus_values = expanded_focus_values + [data.cauchy_peak_x]
            else:
                sub_focus_values = expanded_focus_values

            for nf, defocus in enumerate(sub_focus_values):
                if defocus != data.cauchy_peak_x:
                    slice_counter += 1
                plottry = plot and nf == (len(sub_focus_values) - 1) and p is ps[0]
                float_p = {k_: float(v_) for k_, v_ in p.items()}
                index_add = 10000 if defocus == data.cauchy_peak_x else 0

                s = TestSettings(defocus, float_p)
                s.mono = mono
                s.plot = plottry
                s.dummy = dummy
                s.id_or_hash = index_counter + index_add
                s.strehl_estimate = strehl_est_concat[slice_counter]
                s.return_otf = True
                s.return_otf_mtf = not complex_otf
                s.cpu_gpu_arraysize_boundary = cpu_gpu_fftsize_boundary
                s.guide_mtf = chart_sag_concat.T[slice_counter], chart_mer_concat.T[slice_counter]
                s.x_loc = x_loc
                s.y_loc = y_loc
                s.exif = data.exif

                if s.get_processing_details().allow_cuda:
                    gpu_arg_lst.append((s,))
                else:
                    cpu_arg_lst.append((s,))
                all_arg_lst.append(s)

                data_for_args.append(data)
                index_counter += 1
        t_prep += time.time() - t
        t = time.time()

        # f = wavefront_test._try_wavefront_prysmref
        f = wavefront_test.try_wavefront

        if multi:
            cpures = cpupool.starmap_async(f, cpu_arg_lst)
            outcuda = cudapool.starmap(f, gpu_arg_lst)
            cpustart = time.time()
            out = cpures.get()
            cpuwait = time.time() - cpustart
            out.extend(outcuda)
        else:
            out = [f(args) for args in all_arg_lst]
            cpuwait = 0

        t_run += time.time() - t
        t = time.time()

        out.sort(key=lambda tr: tr.id_or_hash)

        evalrealtime = time.time() - evalstart
        if return_timing_only:
            return evalrealtime, cpuwait

        allevaltimes.append(evalrealtime)

        for using_cuda in [False, True]:
            timingdicts = [tr.timings for tr in out if bool(tr.used_cuda) is using_cuda]
            if using_cuda not in timings:
                timings[using_cuda] = {}
            if len(timingdicts):
                timingkeys = list(timingdicts[0].keys())
                if timings[using_cuda] == {}:
                    for key in timingkeys:
                        timings[using_cuda][key] = 0
                for dct in timingdicts:
                    for key in timingkeys:
                        timings[using_cuda][key] += dct[key]


        # _, out_sag, out_tan, times, peakinesss, strehls, fftsizes = zip(*out)

        bestfocuspeakiness = 1#np.clip(peakinesss[-1], 1.4, 10.0)

        # (t_init, t_pupils, t_get_phases, t_get_fcns, t_pads, t_ffts, t_affines, t_mtfs, t_misc) = zip(*times)

        # print(sum(t_init), sum(t_pupils), sum(t_get_phases),sum(t_get_fcns), sum(t_pads), sum(t_ffts), sum(t_affines), sum(t_mtfs), sum(t_misc))
        # print(np.array(times).sum())

        strehl = 1#strehls[-1]

        # Strip out cauchy_x_peak test

        out_sag, out_tan = zip(*[tr.otf for tr in out][:-1])

        model_sag_values = np.array(out_sag).T
        model_mer_values = np.array(out_tan).T

        gpu_fftsizes = [tr.fftsize for tr in out[:-1] if tr.used_cuda]
        cpu_fftsizes = [tr.fftsize for tr in out[:-1] if not tr.used_cuda]

        # Run cost calculations
        if p['zero'] != 0:
            offset_model_sag_values = p['zero'] + model_sag_values * (1.0 - p['zero'])
            offset_model_mer_values = p['zero'] + model_mer_values * (1.0 - p['zero'])
        else:
            offset_model_sag_values = model_sag_values
            offset_model_mer_values = model_mer_values
        cost_sag, _, _, _ = _calculate_cost(offset_model_sag_values, chart_sag_concat, split, weights_concat, count)
        cost_mer, _, _, _ = _calculate_cost(offset_model_mer_values, chart_mer_concat, split, weights_concat, count)
        cost = (cost_sag**2 + cost_mer**2) ** 0.5
        # cost = cost_mer

        # Check for parameters near bounds
        for arg, (low, high), order in zip(params[0], optimise_bounds, passed_options_ordering):
            try:
                ratio = (arg - low) / (high - low)
            except (RuntimeWarning, FloatingPointError):
                ratio = 0.5
            if ratio < 0.03 or ratio > 0.97:
                if iterations > prev_iterations or count % 50 == 51:
                    scale = wavefront_config.PARAMS_OPTIONS[order[0]][5]
                    log.warning(
                        "{} ({}) if at {:.3f} very close to bounds {:.3f} {:.3f}".format(order[0],
                                                                                         order[1],
                                                                                         arg / scale,
                                                                                         low / scale,
                                                                                         high / scale))
            if wavefront_config.ENFORCE_BOUNDS_IN_COST and np.abs(ratio - 0.5) * 2 > 1.01:
                bounds_cost = float("inf")
            else:
                bounds_cost = 0

        cost += bounds_cost

        t_calc += time.time() - t

        if iterations == 1 and prev_iterations == 0:
            first_it_evals = count

        if iterations > prev_iterations or count % 2 == 0:
            # print(strehls)
            evaltime = sum(allevaltimes)
            displaystrlst = []
            headerstrlst = []
            summarydict = OrderedDict()
            summarydict["evals"] = it_count
            summarydict["nit"] = iterations
            if iterations < total_iterations:
                summarydict["tot.nit"] = total_iterations
            summarydict["   cost  "] = cost
            # summarydict["   lgcost  "] = linegradcost
            # summarydict["   mscost  "] = meansquarecost
            # summarydict["  skewcost "] = skewcost
            # summarydict["  xmodscost "] = disp_xmods_cost
            summarydict['ev/it'] = int((count - first_it_evals) / total_iterations) if total_iterations > 0 else count
            summarydict['t.eval'] = allevaltimes[-1]
            summarydict['cpuwait'] = cpuwait
            # summarydict["peak"] = bestfocuspeakiness
            # summarydict["strl"] = strehl

            endsummarydict = OrderedDict()
            endsummarydict["cpu.q"] = len(cpu_arg_lst)
            endsummarydict["gpu.q"] = len(gpu_arg_lst)
            # endsummarydict["MPratio"] = singlethread_loop_time * (len(cpu_arg_lst )+len(gpu_arg_lst)) * count / evaltime
            try:
                endsummarydict['cpu.fft'] = (np.array(cpu_fftsizes)**2).mean() ** 0.5
            except (FloatingPointError, RuntimeWarning):
                pass
            try:
                endsummarydict['gpu.fft'] = (np.array(gpu_fftsizes)**2).mean() ** 0.5
            except (FloatingPointError, RuntimeWarning):
                pass
            # endsummarydict['tot.loca.ext'] = int(sum(loca_split) - sum(split))
            endsummarydict["tot.t"] = int(time.time() - starttime)

            for order, name in zip(orders, names):
                endsummarydict['delta.{}'.format(name)] = order

            for key, value in list(summarydict.items()) + \
                              list(popt.items()) + \
                              list(endsummarydict.items()) + \
                              [("  |", 0)] + list(pfix.items()):
                if key.startswith("df_"):
                    key = key[3:]
                if type(value) is int:
                    vallen = len("{:d}".format(-np.abs(value)))
                    valtype = "int"
                elif type(value) is str:
                    vallen = len(value)
                    valtype = "str"
                else:
                    vallen = len("{:.3f}".format(-np.abs(value)))
                    valtype = "float"
                width = max(vallen, len(key)) + 0
                headformatstr = r"{:^"+str(width)+r"}"
                if valtype == "int":
                    valformatstr = r"{:"+str(width)+r"d}"
                elif valtype == "str":
                    valformatstr = r"{:"+str(width)+r"}"
                else:
                    valformatstr = r"{:"+str(width)+r".3f}"

                headerstrlst.append(headformatstr.format(key.lower()))
                displaystrlst.append(valformatstr.format(value))
            if count % 24 == 0:
                print()
                for using_cuda in [False, True]:
                    strlist = ["GPU" if using_cuda else "CPU"]
                    total = 0
                    tlist = list(timings[using_cuda].items())
                    tlist.append(("t_prep", t_prep))
                    tlist.append(("t_run", t_run))
                    tlist.append(("t_calc", t_calc))
                    for k, v in tlist:
                        stri = "{}: {:.0f}".format(k, v)
                        strlist.append(stri.ljust(15))
                        if k not in ['t_run']:
                            total += v
                    strlist.insert(0, "Total: {:.0f}".format(total).ljust(20))
                    if using_cuda is False:
                        strlist.append("")
                    print(" ".join(strlist))
                np.set_printoptions(linewidth=1000)
                print(repr(params[0]))
                print()
                print("  ".join(headerstrlst))
            print("  ".join(displaystrlst))

        if plot or (wavefront_config.LIVE_PLOTTING and iterations > prev_iterations  or count % 2 == 0) and plot_gradients_initial is None:
            # for nplot, (axes, plotdict) in enumerate(zip(axesarray, subplots)):
            #         for n, (freq, model_sag, model_mer, line_sag, line_mer) in enumerate(zip(
            #                                                     wavefront_utils.SPACIAL_FREQS[wavefront_config.PLOT_LINES],
            #                                                     offset_model_sag_values[wavefront_config.PLOT_LINES],
            #                                                     offset_model_mer_values[wavefront_config.PLOT_LINES],
            #                                                     plotdict['linessag'], plotdict['linesmer'])):
                        # color = COLOURS[n % 8]
                        # axes.plot(focus_values, chart, '-', label="Chart", color=color )
                        # plot_process_fn = plotdict['plot_process_fn']
                        # line_sag.set_ydata(plot_process_fn(model_sag))
                        # line_mer.set_ydata(plot_process_fn(model_mer))
                    # plt.draw()
                    # plt.pause(1e-6)

            for chartaxespair, plotdictpair in zip(chart_axes, subplots):
                for chart_axis, plotdict in zip(chartaxespair, plotdictpair):
                    lines = plotdict['lines']
                    plot_process_fn = plotdict['plot_process_fn']
                    if chart_axis == SAGITTAL:
                        modelvals = offset_model_sag_values[wavefront_config.PLOT_LINES]
                    else:
                        modelvals = offset_model_mer_values[wavefront_config.PLOT_LINES]
                    for n, (line, vals) in enumerate(zip(lines, modelvals)):
                        line.set_ydata(plot_process_fn(vals))
            plt.draw()
            plt.pause(1e-6)
        count += 1
        it_count += 1
        prev_iterations = iterations
        lastcost = cost
        return cost * wavefront_config.HIDDEN_COST_SCALE

    initial_guess, optimise_bounds, passed_options_ordering = encode_parameter_tuple(dataset)
    # exit()
    if plot_gradients_initial is not None and plot_gradients_initial is not False:
        baseline = np.array(plot_gradients_initial)# * wavefront_config.SCALE_EXTRA[:len(plot_gradients_initial)]
        deltainc = 1e-8
        numvals = 4
        deltas = np.linspace(-deltainc * (numvals-1) / 2, deltainc * (numvals-1) / 2, numvals)
        deltas = np.linspace(0, deltainc * (numvals-1) / 2, numvals)
        # deltas = np.linspace(0, deltainc * (numvals), numvals)
        costs_lst = []
        legends = []
        gradients = []
        jiggles = []

        test_axes = np.arange(0, len(plot_gradients_initial))

        for axis in test_axes:
            print("Testing {}".format(passed_options_ordering[axis][0]))
            costs = []
            param_array = deltas + baseline[axis]
            for param in param_array:
                params = baseline.copy()
                params[axis] = param
                cost = prysmfit(params)
                costs.append(cost)
                # time.sleep(1)
            costs_lst.append(costs)
            poly = np.polyfit(deltas, costs, 2)
            polyvals = np.polyval(poly, deltas)
            mean_diff = np.diff(costs).mean()
            polyfit_rmse = ((polyvals - costs)**2).mean() ** 0.5 / mean_diff
            # print(polyfit_rmse)
            # plt.plot(polyvals)
            # plt.plot(costs)
            # plt.show()
            gradients.append(poly[1])
            jiggles.append(polyfit_rmse)
            legends.append("{}".format(passed_options_ordering[axis][0]))
        gradients = np.array(gradients) / np.mean(np.abs(gradients))
        print(repr(gradients))
        print(passed_options_ordering)
        for axis, grad, jiggle in zip(test_axes, gradients, jiggles):
            print("Axis {}, gradient {:.3f}, jiggles {:.2f}"
                  .format("{}".format(passed_options_ordering[axis][0]), grad, jiggle))

        plt.cla()
        for costs, legend in zip(costs_lst, legends):
            plt.plot(deltas, costs, marker='v', label=legend)
        plt.legend()
        plt.show()
        return gradients, passed_options_ordering

    print(passed_options_ordering)
    print("Initial guess", repr(np.array(initial_guess)))
    # decoded = decode_parameter_tuple(initial_guess, passed_options_ordering, dataset)
    # print(repr(initial_guess))
    # print(repr(decoded))
    # wavefront_config.SCALE_EXTRA = np.arange(2, 70)

    # initial_guess2, optimise_bounds, passed_options_ordering = encode_parameter_tuple(dataset)
    # decoded2 = decode_parameter_tuple(initial_guess2, passed_options_ordering, dataset)
    # print(np.array(initial_guess2) / np.array(initial_guess))
    # for name, applies, _ in passed_options_ordering:
    #     a = decoded[0][0][name]
    #     b = decoded2[0][0][name]
    #     print(name, b/a)
    # exit()
    prysmfit(initial_guess)
    # plt.show()
    # exit()

    # Profile and fine tune
    if wavefront_config.USE_CUDA and wavefront_config.CPU_GPU_FFTSIZE_BOUNDARY_FINETUNE:
        for _ in range(5):
            prysmfit(initial_guess, return_timing_only=True)
        # hithigh  = False
        # hitlow = False
        # while 0:
        #     cpuwaits = []
        #     evals = []
        #     print("Trying fftsize boundary {}".format(cpu_gpu_fftsize_boundary))
        #     for _ in range(3):
        #         eval, wait = prysmfit(initial_guess, return_timing_only=True, no_process_details_cache=True)
        #         cpuwaits.append(wait)
        #         evals.append(eval)
        #     meaneval = np.mean(evals)
        #     print("   Time {:.3f}".format(meaneval))
        #     meanwait = np.mean(cpuwaits)
        #     print(meaneval, meanwait)
        #     if meanwait < 0.1:
        #         cpu_gpu_fftsize_boundary += 16
        #         if hithigh:
        #             break
        #         hitlow = True
        #     else:
        #         cpu_gpu_fftsize_boundary -= 16
        #         if hitlow:
        #             break
        #         hithigh = True

        # tries = np.linspace(-120,120, 7) + wavefront_config.CPU_GPU_FFTSIZE_BOUNDARY
        tries = CUDA_GOOD_FFT_SIZES[(CUDA_GOOD_FFT_SIZES < wavefront_config.FINETUNE_MAX) * (CUDA_GOOD_FFT_SIZES >= wavefront_config.FINETUNE_MIN)]
        # tries = tries[tries >= 0]

        times = []
        for cpu_gpu_fftsize_boundary in tries:
            cpuwaits = []
            evals = []
            print("Trying fftsize boundary {}".format(cpu_gpu_fftsize_boundary))
            prysmfit(initial_guess, return_timing_only=True, no_process_details_cache=True)
            prysmfit(initial_guess, return_timing_only=True, no_process_details_cache=True)
            for _ in range(1):
                eval, wait = prysmfit(initial_guess, return_timing_only=True, no_process_details_cache=True)
                cpuwaits.append(wait)
                evals.append(eval)
            times.append(np.mean(evals))
            # try:
            #     if times[-1] > times[-2] > times[-3]:
            #         break
            # except IndexError:
            #     pass
        times = np.array(times)
        valid = times < min(times)*1.35
        if sum(valid) > 2000:
            poly = np.polyfit(tries[valid], times[valid], 2)
            root = np.roots(np.polyder(poly))[0]
        else:
            root = tries[np.argmin(times)]
        zipped = list(zip(times, tries))
        # zipped.sort(key=lambda t:t[0])
        print(zipped)
        cpu_gpu_fftsize_boundary = int(root)



        print("Using {} FFTsize boundary".format(cpu_gpu_fftsize_boundary))


    starttime = time.time()
    success = None
    nfev = None
    initial_ps, _, _ = decode_parameter_tuple(initial_guess, passed_options_ordering, dataset)

    # _save_data(initial_ps,initial_ps,set, dataset, 0,0,0,0,0)
    for p in initial_ps:
        print(p)
    # exit()
    #     p['base_fstop'] = min(p['fstop'] for p in initial_ps)
    #     plt.plot(strehl_est_concat)
    #     plt.plot(chart_mtf_means_concat)
    #     plt.plot([get_processing_details(strehl, mtfm, p, mtf=(sag, mer))[0] / 500 for
    #               strehl, mtfm, sag, mer in
    #               zip(strehl_est_concat, chart_mtf_means_concat, chart_sag_concat.T, chart_mer_concat.T)],
    #              label="fftsize")
    #     plt.plot([get_processing_details(strehl, mtfm, p, mtf=(sag, mer))[1] / 500 for
    #               strehl, mtfm, sag, mer in
    #               zip(strehl_est_concat, chart_mtf_means_concat, chart_sag_concat.T, chart_mer_concat.T)],
    #              label="samples")
    #     plt.legend()
    #     plt.show()
    # exit()

    # print(initial_ps[0])
    # initial_ps[0]['base_fstop'] = initial_ps[0]['fstop']
    # print(dataset[0].cauchy_peak_x)
    # wavefront_utils.plot_nominal_psf(initial_ps[0])
    # exit()
    fun = None

    # Plot Zs
    # y = []
    # x = []
    # for key, val in initial_ps[0].items():
    #     if key.startswith('z') and key[1].isdigit():
    #         x.append(int(key[1:]))
    #         y.append(val)
    # plt.cla()
    # plt.plot(x, y)
    # plt.show()
    # exit()
    def compare_hidden(ps):
        print("{:14}: {:>9} {:>9} {:>9}".format("Param", "Fit", "Truth", "Error"))
        print("-------------------------------------------------")

        for key in ps[0].keys():
            fits = [p[key] for p in ps]

            if key == 'df_step':
                est_defocus_rms_wfe_step = fits[0]

            nodigitskey = key.split(".")[0]
            if key == 'fstop_corr':
                per_focusset = PARAMS_OPTIONS['fstop'][4]
            else:
                per_focusset = PARAMS_OPTIONS[nodigitskey][4]
            try:
                truths = [data.secret_ground_truth[key] for data in dataset]
            except KeyError:
                truths = [0] * len(dataset)

            for fit, truth in zip(fits, truths):
                error = fit - truth
                print("{:14}: {:9.3f} {:9.3f} {:9.3f}".format(key, fit, truth, error))
                if not per_focusset:
                    break
        else:
            for fit in fits:
                print("{:14}: {:9.3f}".format(key, fit))
                if not per_focusset:
                    break

    if 1:
        options = {#'ftol': 1e-6,
                   # 'eps': 1e-06 * wavefront_config.DEFAULT_SCALE,
                   #'gtol': 1e-2,
                   #  'xtol':1e-4,
                   # 'maxcor':100,
                   'maxiter': MAXITER}

        def callback(x, *args):
            nonlocal total_iterations
            nonlocal iterations
            nonlocal last_x
            global keysignal
            nonlocal it_count
            ps, popt, pfix = decode_parameter_tuple(x, passed_options_ordering, dataset)
            _save_data(ps, initial_ps,set, dataset, lastcost,iterations+1, count, False, starttime, True, True)
            last_x = x
            total_iterations += 1
            it_count = 0
            iterations += 1
            if lastcost < 0.02 or keysignal.lower() in ['s', 'a', 'x']:
                if keysignal.lower() == 'a':
                    keysignal = ""
                print(keysignal)
                raise TerminateOptException()
            return  # lastcost > 1.0

        fun = np.inf
        bestfun = np.inf
        while fun > 0.02 and keysignal.lower() not in ['x', 's']:

            iterations = 0
            try:
                # raise TerminateOptException()
                # opt = optimize.basinhopping(prysmfit,
                #                             initial_guess,
                #                             minimizer_kwargs={'method':'L-BFGS-B', 'options':options, 'callback':callback},
                #                             callback=callback_b)
                                            # )

                # opto = nlopt.opt(nlopt.LD_SLSQP)
                # opto.set_min_objective(prysmfit)

                # watcher = threading.Thread(target=_wait_for_keypress)
                # watcher.start()
                hidden = False
                for data in dataset:
                    if hasattr(data, 'secret_ground_truth'):
                        print("Dataset has secret ground truth!")
                        print(data.secret_ground_truth)
                        hidden = True

                def close_pools_and_exit(*args, **kwargs):
                    cudapool.close()
                    cpupool.close()
                    cudapool.terminate()
                    cpupool.terminate()
                    cudapool.join()
                    cpupool.join()
                    exit()

                def raise_exit_flag(*args, **kwargs):
                    global keysignal
                    global exit_signal
                    keysignal = "s"
                    print("EXITING!!")

                signal.signal(signal.SIGTERM, raise_exit_flag)
                signal.signal(signal.SIGINT, raise_exit_flag)
                signal.signal(signal.SIGQUIT, raise_exit_flag)

                initial_ps, _, _ = decode_parameter_tuple(initial_guess, passed_options_ordering, dataset)

                # opt = optimize.basinhopping(prysmfit, initial_guess,
                #                             minimizer_kwargs=dict(method='L-BFGS-b', options=options, bounds=optimise_bounds,callback=callback),
                #                             callback=callback_b)
                # opt = optimize.minimize(prysmfit, initial_guess, method="SLSQP", bounds=optimise_bounds,
                #                     options=options, callback=callback)
                # opt = optimize.minimize(prysmfit, initial_guess, method="Nelder-Mead",# bounds=optimise_bounds,
                #                         options=dict(maxiter=5000, maxfev=15000), callback=callback)
                # opt = optimize.minimize(prysmfit, initial_guess, method="COBYLA", bounds=optimise_bounds,
                #                         options=dict(), callback=callback)
                opt = optimize.minimize(prysmfit, initial_guess, method="L-BFGS-B", bounds=optimise_bounds,
                                        options=options, callback=callback)
                # opt = optimize.minimize(prysmfit, initial_guess, method="trust-constr", bounds=optimise_bounds,
                #                         options=dict(), callback=callback)
                fun = opt.fun / wavefront_config.HIDDEN_COST_SCALE
                x = opt.x
                try:
                    nit = opt.nit
                except AttributeError:
                    nit = iterations
                nfev = opt.nfev
                print(opt)
                success = opt.success
            except TerminateOptException:
                fun = lastcost
                nit = iterations
                nfev = count
                x = last_x
                success = True

            print('==== FINISHED ====')

            ps, popt, pfix = decode_parameter_tuple(x, passed_options_ordering, dataset)

            # print(initial_ps)

            if (fun < bestfun or keysignal.lower() in ['s']) and keysignal.lower() not in ['a', 'x']:
                bestfun = fun
                _save_data(ps, initial_ps, set, dataset, fun, nit, nfev, success, starttime)

            if hidden:
                compare_hidden(ps)
                wavefront_utils.plot_nominal_psf(ps[0], dataset[0].secret_ground_truth)
            # Shuffle zs around
            old_initial = initial_guess.copy()

            initial_guess = _randomise_zeds(x, passed_options_ordering)

            new_initial_p = {}
            old_initial_p = {}
            inc = 0
            for paramname, applies, fieldapplies in passed_options_ordering:
                if len(applies) > 1:
                    add = ""
                else:
                    add = "{:d}".format(applies[0])
                new_initial_p[paramname+add] = initial_guess[inc]
                old_initial_p[paramname+add] = old_initial[inc]
                inc += 1

            # print("Not Jiggled:", old_initial_p)
            # print("Jiggled:", new_initial_p)

            close_pools_and_exit()

    else:
        fit, _ = optimize.curve_fit(prysmfit, list(focus_values) * len(wavefront_utils.SPACIAL_FREQS), chart_mtf_values.flatten(),
                                    p0=initial_guess, sigma=1.0 / weights.flatten(), bounds=curve_fit_bounds)
        print(fit)
        prysmfit(0, *fit, plot=1)

        est_defocus_rms_wfe_step = fit[1]

    if opt.success and keysignal.lower() not in ['x', 'a']:
        est_defocus_pv_wfe_step = est_defocus_rms_wfe_step * 2 * 3 ** 0.5

        log.info("--- Focus step size estimates ---")
        log.info("    RMS Wavefront defocus error {:8.4f} λ".format(est_defocus_rms_wfe_step))

        longitude_defocus_step_um = est_defocus_pv_wfe_step * dataset[0].exif.aperture**2 * 8 * 0.55
        log.info("    Image side focus shift      {:8.3f} µm".format(longitude_defocus_step_um))

        na = 1 / (2.0 * dataset[0].exif.aperture)
        theta = np.arcsin(na)
        coc_step = np.tan(theta) * longitude_defocus_step_um * 2

        focal_length_m = dataset[0].exif.focal_length * 1e-3

        def get_opposide_dist(dist):
            return 1.0 / (1.0 / focal_length_m - 1.0 / dist)

        lens_angle_of_view = dataset[0].exif.angle_of_view
        # print(lens_angle_of_view)
        subject_distance = CHART_DIAGONAL * 0.5 / np.sin(lens_angle_of_view / 2)
        image_distance = get_opposide_dist(subject_distance)

        log.info("    Subject side focus shift    {:8.2f} mm".format((get_opposide_dist(image_distance-longitude_defocus_step_um *1e-6) - get_opposide_dist(image_distance)) * 1e3))
        log.info("    Blur circle  (CoC)          {:8.2f} µm".format(coc_step))

        log.info("Nominal subject distance {:8.2f} mm".format(subject_distance * 1e3))
        log.info("Nominal image distance   {:8.2f} mm".format(image_distance * 1e3))

    cpupool.close()
    global exit_signal
    exit_signal = True
    return
    return est_defocus_rms_wfe_step, longitude_defocus_step_um, coc_step, image_distance, subject_distance# , dataset[0].cauchy_peak_y
