import time
import random
import math
import multiprocessing
from collections import OrderedDict
import prysm
from scipy import optimize
from lentil import wavefront_utils
from lentil.constants_utils import *
from lentil.wavefront_utils import try_wavefront
from lentil.wavefront_config import freqs, modelwavelengths, optimise_params, params_options, fixed_params


lst = list(np.linspace(0, 1, 5e7))
lenlst = len(lst)


def try_wavefront_fake(defocus=0, defocus_offset=0, defocus_step=0.1, loca=0, spca=0, basewv=0.575, fstop=2.8,
                  base_fstop=2.8, locaref=0.575, z={}, zero_offset=0.0, plot=False, only_strehl=False):

    start = time.time()
    # for p in range(10):
    #     power = 1.01 + p * 0.0011
    #     q = range(99999)

    sum_ = 0
    use = [random.randint(0, lenlst-2) for _ in range(99999)]
    for ix in use:
        x = lst[ix]
        sum_ += x ** 0.0012
        sum_ *= math.log(1.23)

    return tuple([0.5]*len(freqs)), time.time()-start, 2, 0.8


if MULTIPROCESSING > 1:
    globalpool = None
    

def estimate_wavefront_errors(set, fs_slices=16, skip=1, processes=None, plot=True):
    print(set[0].chart_mtf_values)
    if hasattr(set[0], 'chart_mtf_values'):
        dataset = set
    elif hasattr(set[0], 'fields'):
        dataset = wavefront_utils.pre_process_focussets(set, fs_slices, skip)
    else:
        raise ValueError("Unknown input!")

    count = 0
    allsubprocesstimes = []
    allevaltimes = []
    starttime = time.time()

    plt.show()
    axes = plt.gca()
    axes.set_title("Model MTF vs Chart MTF")
    lines = []
    lineskip = 3
    chart_mtf_values_concat = np.concatenate([data.chart_mtf_values for data in dataset], axis=1)
    focus_values_sequenced = [dataset[0].focus_values]
    weights_concat = np.concatenate([f.weights for f in dataset], axis=1)
    base_fstop = min((f.exif.aperture for f in dataset))

    for f in dataset[1:]:
        new_focus_values = f.focus_values + max(focus_values_sequenced[-1] + 1)
        focus_values_sequenced.append(new_focus_values)

    focus_values_concat = np.arange(chart_mtf_values_concat.shape[1])
    for n, (freq, chart) in enumerate(zip(wavefront_utils.freqs[::lineskip], chart_mtf_values_concat[::lineskip])):
        color = COLOURS[n % 8]
        axes.plot(focus_values_concat, chart, '-', label="Chart {:.2f}".format(freq), color=color )
        line,  = axes.plot(focus_values_concat, chart, '--', label="Model {:.2f}".format(freq), color=color)
        lines.append(line)
    axes.legend()
    total_slices = len(focus_values_concat) + len(dataset)

    def init():
        pass
        # global globalfocussets
        # globalfocussets = focussets

    if processes is None:
        optimal_processes = multiprocessing.cpu_count()
        processes_opts = np.arange(4, 15)
        loop_ops = np.ceil(total_slices / processes_opts)
        efficiency = total_slices / (loop_ops * processes_opts)
        cpu_efficiency_favour = 1 - (processes_opts-optimal_processes)**2
        print(processes_opts)
        print(loop_ops)
        print(efficiency)
        processes = max(zip(efficiency, cpu_efficiency_favour, processes_opts))[2]
    print("Using {} processes (for {} slices)".format(processes, total_slices))
    # exit()
    pool = multiprocessing.Pool(processes=processes, initializer=init)

    # Benchmark
    print("Benchmarking!")
    totaltime = 0
    primeloops = 3
    testloops = 10
    looplst = [False] * primeloops + [True] * testloops
    for use in looplst:
        _, time_, _, _ = try_wavefront(z=dict(z4=0.1, z5=0.1, z6=0.1, z7=0.1))
        if use:
            totaltime += time_
    singlethread_loop_time = totaltime / testloops
    print("Average loop time {:.1f}ms".format(singlethread_loop_time * 1e3))

    passed_options_ordering = []

    def prysmfit(*params, multi=True, plot=False):
        if len(params) == 1:
            ps = []
            for _ in dataset:
                ps.append({})
            popt = {}
            for (name, applies), val in zip(passed_options_ordering, params[0]):
                tup = params_options[name]
                for a in applies:
                    if name == "fstop":
                        mult = dataset[a].exif.aperture
                    elif name == 'df_step':
                        mult = dataset[a].exif.aperture #* (dataset[a].exif.aperture / base_fstop) ** 2
                    else:
                        mult = tup[3](base_fstop)
                    ps[a][name] = val * mult
                    if len(applies) == 1:
                        popt["{}.{}".format(name, a)] = val * mult
                if len(applies) > 1:
                    if name == "fstop":
                        popt["FSTOP_CORR"] = val
                    else:
                        popt["{}".format(name)] = val * mult

        else:
            raise ValueError("BORK!")
            p = {}
            # popt = {}
            # for name, val in zip(optimise_params, params[1:]):
            #     tup = params_options[name]
            #     mult = tup[3](data.exif.aperture)
            #     p[name] = val * mult
            #     popt[name] = val * mult
        pfix = {}
        for param in fixed_params:
            tup = params_options[param]
            for a, data in enumerate(dataset):
                mult = tup[3](data.exif.aperture)
                ps[a][param] = tup[1] * mult
                pfix["{}.{}".format(param, a)] = tup[1] * mult

        # exit()

        out = []
        basewv = 0.575
        arglst = []
        addarglst = []
        data_for_args = []

        evalstart = time.time()

        loop_base_fstop = min((p['fstop'] for p in ps))

        for data, p in zip(dataset, ps):
            z = {}
            for key, value in p.items():
                if key[0] == 'z' and key[1].isdigit():
                    z[key.upper()] = value

            sub_focus_values = list(data.focus_values) + [p['df_offset']]
            for n, defocus in enumerate(sub_focus_values):
                plottry = plot and n == (len(sub_focus_values) - 1)
                args = [float(_) for _ in [defocus,
                                           p['df_offset'],
                                           p['df_step'],
                                           p['loca'],
                                           p['spca'],
                                           basewv,
                                           p['fstop'],
                                           loop_base_fstop,
                                           p['locaref']]] + [z, None, plottry]

                # for a in args:
                #     print(type(a))
                # exit()
                # if multi > 0:/
                arglst.append(args)
                data_for_args.append(data)
                # else:
                #     raise ValueError("Bork!")
                #     out.append(try_wavefront(*args))
            addarglst.append(arglst.pop(-1))

        arglst.extend(addarglst)
        # print("Number of multiprocess items {}".format(len(arglst)))
        if multi > 0:
            out = pool.starmap(wavefront_utils.try_wavefront, arglst)
        else:
            out = []
            for args in arglst:
                out.append(try_wavefront(*args))
        evalrealtime = time.time() - evalstart
        allevaltimes.append(evalrealtime)

        out, times, peakinesss, strehls = zip(*out)
        bestfocuspeakiness = np.clip(peakinesss[-1], 2.0, 5.0)
        allsubprocesstimes.extend(times)
        strehl = strehls[-1]
        if 1 or plot:
            plotout = out[-len(addarglst)]
            out = out[:-len(addarglst)]
        # print(len(addarglst))
        # print(np.array(out).shape)
        # print(out)
        # exit()
        model_mtf_values = np.array(out).T
        if plot:
            # pass
            # pupil.plot2d()
            # plt.show()
            # plt.plot(model_mtf_values.flatten(), label="Prysm Model {:.3f}λ Z4 step size, {:.3f}λ Z11".format(defocus_step, aberr))
            plt.plot(model_mtf_values.flatten(), label="Prysm Model {:.3f}λ Z4 step size, {:.3f}λ Z11, {:.3f}λ Z22"
                                                       "".format(p['df_step'], p['z9'], p['z16']))
            plt.plot(chart_mtf_values_concat.flatten(), label="MTF Mapper data")
            plt.xlabel("Focus position (arbitrary units)")
            plt.ylabel("MTF")
            plt.title("Prysm model vs chart ({:.32}-{:.2f}cy/px AUC) {}".format(LOWAVG_NOMBINS[0] / 64, LOWAVG_NOMBINS[-1] / 64, data.exif.summary))
            plt.ylim(0, 1)
            plt.legend()
            plt.show()
        global count
        count += 1

        offset_model_mtf_values = p['zero'] + model_mtf_values * (1.0 - p['zero'])

        cost = np.sum((offset_model_mtf_values - chart_mtf_values_concat) ** 2 * weights_concat) * 1e2  # + (bestfocuspeakiness-2.0)**2

        # if count % 20 == 0:
        #     strehl = try_wavefront(*([cauchy_peak_x, p['df_offset']] + args[2:] + [True]))
        #     print("Strehl {:.3f}".format(strehl))
        if count % 5 == 0:
            displaystrlst = ["#{:.0f}: {:.3f} ({:.3f}PK, {:.3f}st)".format(count, cost, bestfocuspeakiness, strehl)]
            for key, value in popt.items():
                substr = "{} {:.3f}".format(key.upper(), value)
                displaystrlst.append(substr)
            displaystrlst.append("  |  ")
            for key, value in pfix.items():
                substr = "{} {:.3f}".format(key.upper(), value)
                displaystrlst.append(substr)
            print(", ".join(displaystrlst))

        if count % 20 == 19:
            realtime = time.time() - starttime
            cputime = sum(allsubprocesstimes)
            evaltime = sum(allevaltimes)
            print("   Runtime {:.2f}s, Core CPU t: {:.2f}s, MPratio {:.1f}-{}slices, Eval Proportion {:.3f}, mean eval {:.3f}s, mean MP-loop {:.3f}s"
                  "".format(realtime,
                            cputime,
                            singlethread_loop_time * (len(arglst)) * count / evaltime,
                            len(arglst),
                            evaltime / realtime,
                            evaltime / count,
                            np.mean(allsubprocesstimes)))

        if len(params) > 1:
            return model_mtf_values.flatten()

        if plot or count % 10 == 1:
            for n, (freq, model, line) in enumerate(zip(wavefront_utils.freqs[::lineskip], offset_model_mtf_values[::lineskip], lines)):
                # color = COLOURS[n % 8]
                # axes.plot(focus_values, chart, '-', label="Chart", color=color )
                line.set_ydata(model)
            # plt.ion()
            plt.draw()
            plt.pause(1e-6)
        return cost

    initial_guess = []
    optimise_bounds = []
    for paramname in optimise_params:
        per_focusset = params_options[paramname][4]
        tup = params_options[paramname]
        low = tup[0]
        initial = tup[1]
        high = tup[2]
        if per_focusset:
            for a, data in enumerate(dataset):
                passed_options_ordering.append((paramname, (a,)))
                if paramname == 'df_offset':
                    initial = data.cauchy_peak_x
                    low = min(data.focus_values) - 3
                    high = max(data.focus_values) + 3
                initial_guess.append(initial)
                optimise_bounds.append((low, high))

        else:
            passed_options_ordering.append((paramname, tuple(range(len(dataset)))))
            initial_guess.append(initial)
            optimise_bounds.append((low, high))

    print(initial_guess)
    print(optimise_bounds)
    print(passed_options_ordering)
    # exit()
    curve_fit_bounds = list(zip(*optimise_bounds))
    # print(curve_fit_bounds)
        # print(paramname, mult, tup)
    # print(initial_guess)
    # print(optimise_bounds)
    # exit()
    # exit()

    # initial_guess = [11.10561494,  0.39775171,  0.09595456, -0.0986406 , -0.17167902,
    #    -2.32231102,  0.64717264,  0.58967162]
    # 595: 2.311 in 22.59s, DEFOCUS_OFFSET 11.324, DEFOCUS_STEP 0.107, Z9 -0.088, Z16 -0.011, Z25 -0.009, Z36 0.024, LOCA 0.265, LOCAREF 0.541,   |  , Z7 0.000, Z8 0.000, APERTURE 2.940, SPCA 0.004, ZERO_OFFSET 0.005
    # 56mm all [14.29142161,  8.96481699,  4.86504373,  0.54432487,  0.36725587,
    #    0.23819082, -0.03285181,  0.05561526, -0.68575977, -0.15878104,
    #   -2.25549849, -2.5       , -0.93305931,  0.5       ]
    # 4156: 19.033 (2.000PK, 0.554st) in 82.75s, DEFOCUS_OFFSET(0) 14.291, DEFOCUS_OFFSET(1) 8.965, DEFOCUS_OFFSET(2) 4.865, DEFOCUS_STEP(0) 0.653, DEFOCUS_STEP(1) 1.028, DEFOCUS_STEP(2) 1.334, Z9(0) -0.027, Z9(1) -0.027, Z9(2) -0.027, Z16(0) 0.046, Z16(1) 0.046, Z16(2) 0.046, Z25(0) -0.571, Z25(1) -0.571, Z25(2) -0.571, Z36(0) -0.132, Z36(1) -0.132, Z36(2) -0.132, LOCA(0) -1.880, LOCA(1) -2.083, LOCA(2) -0.778, LOCAREF(0) 0.500, LOCAREF(1) 0.500, LOCAREF(2) 0.500,   |  , Z7(0) 0.000, Z7(1) 0.000, Z7(2) 0.000, Z8(0) 0.000, Z8(1) 0.000, Z8(2) 0.000, APERTURE(0) 1.200, APERTURE(1) 2.800, APERTURE(2) 5.600, SPCA(0) 0.008, SPCA(1) 0.004, SPCA(2) 0.002, ZERO_OFFSET(0) 0.005, ZERO_OFFSET(1) 0.005, ZERO_OFFSET(2) 0.005


    #90mm
    # 4382: 10.469 (2.000PK, 0.903st) in 55.65s, DEFOCUS_OFFSET(0) 16.777, DEFOCUS_OFFSET(1) 11.752, DEFOCUS_OFFSET(2) 7.617, DEFOCUS_STEP(0) 0.272, DEFOCUS_STEP(1) 0.380, DEFOCUS_STEP(2) 0.527, Z9(0) 0.057, Z9(1) 0.057, Z9(2) 0.057, Z16(0) -0.067, Z16(1) -0.067, Z16(2) -0.067, Z25(0) -0.044, Z25(1) -0.044, Z25(2) -0.044, Z36(0) -0.091, Z36(1) -0.091, Z36(2) -0.091, LOCA(0) -0.943, LOCA(1) -0.943, LOCA(2) -0.943, LOCAREF(0) 0.601, LOCAREF(1) 0.601, LOCAREF(2) 0.601,   |  , Z7(0) 0.000, Z7(1) 0.000, Z7(2) 0.000, Z8(0) 0.000, Z8(1) 0.000, Z8(2) 0.000, APERTURE(0) 2.000, APERTURE(1) 2.800, APERTURE(2) 4.000, SPCA(0) 0.005, SPCA(1) 0.004, SPCA(2) 0.003, ZERO_OFFSET(0) 0.005, ZERO_OFFSET(1) 0.005, ZERO_OFFSET(2) 0.005
#[16.77715379, 11.75201761,  7.61670147,  0.13618264,  0.13563434,
        #0.1318735 ,  0.11470272, -0.1332087 , -0.0883821 , -0.18195644,
       #-1.88663101,  0.60059712]
    starttime = time.time()
    if 1:
        if 1:
            options = {'ftol': 1e-6,
                       #'eps': 1e-06,
                       'gtol': 1e-2,
                       'maxiter': 1}
        else:
            options = {}
        # opt = optimize.minimize(prysmfit, initial_guess, method="trust-constr", bounds=bounds)
        # prysmfit([14.39457095,  0.47370952,  0.04539592, -0.119297  , -0.20086028,
       # -1.28074449,  0.        ,  0.5       ], plot=True)
       #  exit()
       #  initial_guess = [15.45797755,  9.07694616,  9.67888576,  0.02916234,  0.05286887, # 60mm
       #  0.04945902,  0.04119428, -0.15280071,  0.01580217,  0.14503988,
       #  1.178283  , -3.62798686]
       #  initial_guess = [20.03858813, 11.03007915,  7.96193796,  6.67643635,  0.16533184, #16mm x 4
       #  0.15092223,  0.16195945,  0.16758065, -0.14156922, -0.39328133,
       #  0.22280391, -0.09829078,  1.23939573,  2.95968507]
        opt = optimize.minimize(prysmfit, initial_guess, method="L-BFGS-B", bounds=optimise_bounds,
                                options=options)
        print('==== FINISHED ====')
        print('==== FINISHED ====')
        print('==== FINISHED ====')
        print('==== FINISHED ====')

        for data in dataset:
            if hasattr(data, 'secret_ground_truth'):
                print("Dataset has secret ground truth!")
                print(data.secret_ground_truth)
        print(opt)
        plt.show()

        prysmfit(opt.x, multi=False, plot=True)
        for solve, (name, applies) in zip(opt.x, passed_options_ordering):
            print(solve, name, applies)
            if name == 'df_step':
                est_defocus_rms_wfe_step = solve
                break
    else:
        fit, _ = optimize.curve_fit(prysmfit, list(focus_values)*len(wavefront_utils.freqs), chart_mtf_values.flatten(),
                                    p0=initial_guess, sigma=1.0 / weights.flatten(), bounds=curve_fit_bounds)
        print(fit)
        prysmfit(0, *fit, plot=1)

        est_defocus_rms_wfe_step = fit[1]



    # log.debug("Fn fit peak is {:.3f} at {:.2f}".format(fitted_params[0], fitted_params[1]))
    # log.debug("Fn sigma: {:.3f}".format(fitted_params[2]))

    # ---
    # Estimate defocus step size
    # ---
    # if "_fixed_defocus_step_wfe" in dir(focusset):
    #     est_defocus_rms_wfe_step = focusset._fixed_defocus_step_wfe
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

    return est_defocus_rms_wfe_step, longitude_defocus_step_um, coc_step, image_distance, subject_distance, dataset[0].cauchy_peak_y
