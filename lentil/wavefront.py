import time
import multiprocessing
from collections import OrderedDict
import prysm
from scipy import optimize
from lentil.constants_utils import *


freqs = np.arange(2, 30, 2) / 64
modelwavelengths = np.linspace(0.450, 0.650, 70)

DIVIDE_BY_APERTURE = lambda f: 1.0 / f
MULTIPLY_BY_APERTURE = lambda f: f
NOP = lambda f: 1.0


params_options = OrderedDict(
    # param_name=(Min, Initial, Max, alter_with_aperture)
    defocus_offset=(None, None, None, NOP),
    defocus_step=(0.1, 0.3, 1, DIVIDE_BY_APERTURE),
    z7=(-1, 0, 1, DIVIDE_BY_APERTURE),
    z8=(-1, 0, 1, DIVIDE_BY_APERTURE),
    z9=(-1, 0, 1, DIVIDE_BY_APERTURE),
    z16=(-1, 0, 1, DIVIDE_BY_APERTURE),
    z25=(-1, 0, 1, DIVIDE_BY_APERTURE),
    z36=(-1, 0, 1, DIVIDE_BY_APERTURE),
    aperture=(1/1.2, 1.0, 1.2, MULTIPLY_BY_APERTURE),
    loca=(-2.5, 0, 2.5, DIVIDE_BY_APERTURE),
    spca=(-1, 0.01, 1, DIVIDE_BY_APERTURE),
    locaref=(0.500, 0.550, 0.680, NOP),
    zero_offset=(0, 0.005, 0.05, NOP),
)

optimise_params = [
    'defocus_offset',
    'defocus_step',
    #'z7',
    #'z8',
    'z9',
    'z16',
    'z25',
    'z36',
    # 'aperture',
    'loca',
    # 'spca',
    'locaref',
    # 'zero_offset',
]

fixed_params = []
for key in params_options.keys():
    if key not in optimise_params:
        fixed_params.append(key)


def try_wavefront(defocus=0, defocus_offset=0, defocus_step=0.1, loca=0, spca=0, basewv=0.575, aperture=2.8,
                  locaref=0.575, z={}, zero_offset=0.0, plot=False, only_strehl=False):
    # print(defocus, only_strehl)
    start = time.time()
    mul = 1
    mtfs = []
    weights = []
    bestpupil = (np.inf, None, None)
    for modelwavelength in [0.575] if only_strehl else modelwavelengths:
        rel_wv = modelwavelength / locaref
        locadefocus = (rel_wv - 1.0) ** 2 * 25 * loca
        spcaz9 = (modelwavelength / 0.575 - 1.0) * spca * 5
        # spcaz9 = (modelwavelength / 0.575 - 1.0) ** 2 * 75 * spca

        zkwargs = {}
        for key, value in z.items():
            if key == 'Z9':
                zkwargs[key] = (value * mul + spcaz9) * basewv
                continue
            if key == 'Z4':
                raise ValueError("No Z4 separately!")
            zkwargs[key] = value * mul * basewv

        pupil = prysm.FringeZernike(Z4=-((defocus - defocus_offset) * defocus_step * mul - locadefocus) * basewv,
                                  dia=10, norm=False,
                                  wavelength=modelwavelength,
                                  opd_unit="um",
                                  samples=128,
                                  **zkwargs)

        m = prysm.MTF.from_pupil(pupil, efl=aperture * 10)
        a = m.exact_sag(freqs / DEFAULT_PIXEL_SIZE * 1e-3)
        mtfs.append(a)
        weights.append(photopic_fn(modelwavelength * 1e3))
        metric = np.abs(rel_wv - 1)
        if metric < bestpupil[0]:
            bestpupil = metric, pupil, modelwavelength, zkwargs
    if only_strehl:
        return bestpupil[1].strehl
        # return pupilall.strehl
    avg = np.average(mtfs, axis=0, weights=weights)

    slice_ = bestpupil[1].slice_x[1]
    slicedv = np.abs(np.diff(slice_[np.isfinite(slice_)]))
    peakiness = slicedv.max() / slicedv.mean()
    strehl = bestpupil[1].strehl
    # print("Peakiness {:.3f}".format(peakiness))

    if plot:
        print(defocus, defocus_offset, bestpupil[2], bestpupil[1].strehl, 1234)
        bestpupil[1].plot2d()
        plt.show()
        for key, value in bestpupil[3].items():
            if value != 0:
                pupil = prysm.FringeZernike(dia=10, norm=False,
                                          wavelength=bestpupil[2],
                                          opd_unit="um",
                                          samples=128,
                                          **{key: value})
                slice = pupil.slice_x[1]
                rms = (slice[np.isfinite(slice)] ** 2).mean() ** 0.5
                plt.plot(slice, label="{} : {:.3f} λRMS".format(key, rms / basewv))
        slice_ = bestpupil[1].slice_x[1]
        rms = (slice_[np.isfinite(slice_)] ** 2).mean() ** 0.5
        plt.plot(slice_, label="All : {:.3f} λRMS".format(rms / basewv))
        print(222, bestpupil[3])
        pupil = prysm.FringeZernike(dia=10, norm=False,
                                  wavelength=bestpupil[2],
                                  opd_unit="um",
                                  samples=128,
                                  **bestpupil[3])

        slice_ = pupil.slice_x[1]
        rms = (slice_[np.isfinite(slice_)] ** 2).mean() ** 0.5
        plt.plot(slice_, '--', label="ZeroZ4 : {:.3f} λRMS".format(rms / basewv), color='black')
        plt.legend()
        plt.show()

        slicedv = np.abs(np.diff(slice_[np.isfinite(slice_)]))
        peakiness = slicedv.max() / slicedv.mean()

        plt.plot(np.diff(slice_), label="Pupil slice derivative at best focus {:.3f}".format(peakiness))
        plt.legend()
        plt.show()

    return tuple(avg), time.time()-start, peakiness, strehl


if MULTIPROCESSING > 1:
    globalpool = None
    

def estimate_wavefront_errors(focusset, size=11, skip=1, plot=True):
    # data = np.array(data)
    # plot = FieldPlot()
    # plot.zdata = data
    # plot.xticks = np.linspace(0,1, data.shape[0])
    # plot.yticks = np.linspace(0,1, data.shape[1])
    # plot.smooth2d(show=1)

    use_minimize = True

    pos = focusset.get_interpolation_fn_at_point(IMAGE_WIDTH/2, IMAGE_HEIGHT/2, AUC, SAGITTAL)
    focus_values = pos.focus_data[:]

    mtf_means = pos.sharp_data
    # chart_mtf_values = mtf_means
    # data = np.array(data)

    meanpeak_idx = np.argmax(mtf_means)
    meanpeak_pos = focus_values[meanpeak_idx]
    meanpeak = mtf_means[meanpeak_idx]
    # highest_data_y = y_values[highest_data_x_idx]

    # print(highest_data_x_idx)

    if meanpeak_idx > 0:
        x_inc = focus_values[meanpeak_idx] - focus_values[meanpeak_idx-1]
    else:
        x_inc = focus_values[meanpeak_idx+1] - focus_values[meanpeak_idx]

    # y_values = np.cos(np.linspace(-6, 6, len(focus_values))) + 1
    absgrad = np.abs(np.gradient(mtf_means)) / meanpeak
    gradsum = np.cumsum(absgrad)
    distances_from_peak = np.abs(gradsum - np.mean(gradsum[meanpeak_idx:meanpeak_idx+1]))
    shifted_distances = interpolate.InterpolatedUnivariateSpline(focus_values, distances_from_peak, k=1)(focus_values - x_inc*0.5)
    weights = np.clip(1.0 - shifted_distances * 1.3 , 1e-1, 1.0) ** 5

    fitfn = cauchy

    optimise_bounds = fitfn.bounds(meanpeak_pos, meanpeak, x_inc)

    sigmas = 1. / weights
    initial = fitfn.initial(meanpeak_pos, meanpeak, x_inc)
    fitted_params, _ = optimize.curve_fit(fitfn, focus_values, mtf_means,
                                          bounds=optimise_bounds, sigma=sigmas, ftol=1e-5, xtol=1e-5,
                                          p0=initial)
    cauchy_peak_x = fitted_params[1]
    cauchy_peak_y = fitted_params[0]
    print("Found peak {:.3f} at {:.3f}".format(cauchy_peak_y, cauchy_peak_x))

    # Move on to get full frequency data
    slicelow = max(0, int(cauchy_peak_x - size * skip / 2))
    slicehigh = slicelow + size
    limit = (slicelow, slicehigh)
    print("Limit", limit)
    datalst = []
    for freq in freqs:
        pos = focusset.get_interpolation_fn_at_point(IMAGE_WIDTH/2, IMAGE_HEIGHT/2, freq, SAGITTAL, limit=limit, skip=skip)
        pos1 = focusset.get_interpolation_fn_at_point(IMAGE_WIDTH/2, IMAGE_HEIGHT/2, freq, MERIDIONAL, limit=limit, skip=skip)
        datalst.append((pos.sharp_data[:] + pos1.sharp_data)*0.5)

    chart_mtf_values = np.array(datalst)  # [:,::skip]
    mtf_means = chart_mtf_values.mean(axis=0)  # [::skip]
    focus_values = pos.focus_data # [::skip]
    max_pos = focus_values[np.argmax(mtf_means)]


    count = 0
    weights = np.clip((chart_mtf_values + 3), 0.01, 1.0) * 10
    weights = 10.5 - chart_mtf_values
    weightmean = np.mean(weights)
    allsubprocesstimes = []
    allevaltimes = []
    starttime = time.time()

    plt.show()
    axes = plt.gca()
    axes.set_title("Model MTF vs Chart MTF")
    lines = []
    lineskip = 3
    for n, (freq, chart) in enumerate(zip(freqs[::lineskip], chart_mtf_values[::lineskip])):
        color = COLOURS[n % 8]
        axes.plot(focus_values, chart, '-', label="Chart {:.2f}".format(freq), color=color )
        line,  = axes.plot(focus_values, chart, '--', label="Model {:.2f}".format(freq), color=color)
        lines.append(line)
    axes.legend()

    def init():
        global globalfocusset
        globalfocusset = focusset
    pool = multiprocessing.Pool(processes=8, initializer=init)

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

    def prysmfit(*params, plot=False):
        if len(params) == 1:
            p = {}
            popt = {}
            for name, val in zip(optimise_params, params[0]):
                tup = params_options[name]
                mult = tup[3](focusset.exif.aperture)
                p[name] = val * mult
                popt[name] = val * mult
        else:
            p = {}
            popt = {}
            for name, val in zip(optimise_params, params[1:]):
                tup = params_options[name]
                mult = tup[3](focusset.exif.aperture)
                p[name] = val * mult
                popt[name] = val * mult

        pfix = {}
        for param in fixed_params:
            tup = params_options[param]
            mult = tup[3](focusset.exif.aperture)
            p[param] = tup[1] * mult
            pfix[param] = tup[1] * mult

        out = []
        basewv = 0.575
        arglst = []
        multi = 1

        z = {}
        for key, value in p.items():
            if key[0] == 'z' and key[1].isdigit():
                z[key.upper()] = value

        if 1 or plot:
            sub_focus_values = list(focus_values) + [p['defocus_offset']]
        else:
            sub_focus_values = focus_values
        evalstart = time.time()
        for n, defocus in enumerate(sub_focus_values):
            plottry = plot and n == (len(sub_focus_values) - 1)
            args = [float(_) for _ in [defocus,
                                       p['defocus_offset'],
                                       p['defocus_step'],
                                       p['loca'],
                                       p['spca'],
                                       basewv,
                                       p['aperture'],
                                       p['locaref']]] + [z, int(max_pos), plottry]

            # for a in args:
            #     print(type(a))
            # exit()
            if multi > 0:
                arglst.append(args)
            else:
                out.append(try_wavefront(*args))
        if multi > 0:
            out = pool.starmap(try_wavefront, arglst)
        evalrealtime = time.time() - evalstart
        allevaltimes.append(evalrealtime)

        out, times, peakinesss, strehls = zip(*out)
        bestfocuspeakiness = np.clip(peakinesss[-1], 2.0, 5.0)
        allsubprocesstimes.extend(times)
        strehl = strehls[-1]

        if 1 or plot:
            plotout = out[-1]
            out = out[:-1]

        model_mtf_values = np.array(out).T
        if plot:
            # pass
            # pupil.plot2d()
            # plt.show()
            # plt.plot(model_mtf_values.flatten(), label="Prysm Model {:.3f}λ Z4 step size, {:.3f}λ Z11".format(defocus_step, aberr))
            plt.plot(model_mtf_values.flatten(), label="Prysm Model {:.3f}λ Z4 step size, {:.3f}λ Z11, {:.3f}λ Z22"
                                                       "".format(p['defocus_step'], p['z9'], p['z16']))
            plt.plot(chart_mtf_values.flatten(), label="MTF Mapper data")
            plt.xlabel("Focus position (arbitrary units)")
            plt.ylabel("MTF")
            plt.title("Prysm model vs chart ({:.32}-{:.2f}cy/px AUC) {}".format(LOWAVG_NOMBINS[0] / 64, LOWAVG_NOMBINS[-1] / 64, focusset.exif.summary))
            plt.ylim(0, 1)
            plt.legend()
            plt.show()
        global count
        count += 1

        offset_model_mtf_values = p['zero_offset'] + model_mtf_values * (1.0 - p['zero_offset'])

        cost = np.sum((offset_model_mtf_values - chart_mtf_values) ** 2 * weights) * 1e2 / weightmean + (bestfocuspeakiness-2.0)**2

        # if count % 20 == 0:
        #     strehl = try_wavefront(*([cauchy_peak_x, p['defocus_offset']] + args[2:] + [True]))
        #     print("Strehl {:.3f}".format(strehl))

        displaystrlst = ["#{:.0f}: {:.3f} ({:.3f}PK, {:.3f}st) in {:.2f}s".format(count, cost, bestfocuspeakiness, strehl,  evalrealtime)]
        for key, value in popt.items():
            substr = "{} {:.3f}".format(key.upper(), value)
            displaystrlst.append(substr)
        displaystrlst.append("  |  ")
        for key, value in pfix.items():
            substr = "{} {:.3f}".format(key.upper(), value)
            displaystrlst.append(substr)

        if count % 10 == 0:
            realtime = time.time() - starttime
            cputime = sum(allsubprocesstimes)
            evaltime = sum(allevaltimes)
            print("Runtime {:.2f}s, Subprocess CPU time: {:.2f}s, MP ratio {:.2f} ({}+1 slices), Eval Proportion {:.3f}, mean eval {:.3f}s, mean MP-loop {:.3f}s"
                  "".format(realtime,
                            cputime,
                            singlethread_loop_time * (1+len(mtf_means)) * count / evaltime,
                            len(mtf_means),
                            evaltime / realtime,
                            evaltime / count,
                            np.mean(allsubprocesstimes)))

        print(", ".join(displaystrlst))
        if len(params) > 1:
            return model_mtf_values.flatten()

        if plot or count % 10 == 1:
            for n, (freq, model, line) in enumerate(zip(freqs[::lineskip], offset_model_mtf_values[::lineskip], lines)):
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
        if paramname == 'defocus_offset':
            initial_guess.append(cauchy_peak_x)
            optimise_bounds.append((min(focus_values)-3, max(focus_values)+3))
            continue
        tup = params_options[paramname]
        mult = 1.0
        initial_guess.append(tup[1] * mult)
        optimise_bounds.append((tup[0] * mult, tup[2] * mult))

    curve_fit_bounds = list(zip(*optimise_bounds))
    print(curve_fit_bounds)
        # print(paramname, mult, tup)
    # print(initial_guess)
    # print(optimise_bounds)
    # exit()
    # exit()

    # initial_guess = [11.10561494,  0.39775171,  0.09595456, -0.0986406 , -0.17167902,
    #    -2.32231102,  0.64717264,  0.58967162]
    # 595: 2.311 in 22.59s, DEFOCUS_OFFSET 11.324, DEFOCUS_STEP 0.107, Z9 -0.088, Z16 -0.011, Z25 -0.009, Z36 0.024, LOCA 0.265, LOCAREF 0.541,   |  , Z7 0.000, Z8 0.000, APERTURE 2.940, SPCA 0.004, ZERO_OFFSET 0.005
    starttime = time.time()
    if 1:
        if 1:
            options = {'ftol': .2e-6,
                       #'eps': 1e-06,
                       'gtol': .2e-2,
                       'maxiter': 400}
        else:
            options = {}
        # opt = optimize.minimize(prysmfit, initial_guess, method="trust-constr", bounds=bounds)
        # prysmfit([14.39457095,  0.47370952,  0.04539592, -0.119297  , -0.20086028,
       # -1.28074449,  0.        ,  0.5       ], plot=True)
       #  exit()
        opt = optimize.minimize(prysmfit, initial_guess, method="L-BFGS-B" , bounds=optimise_bounds,
                                options=options)
        print('==== FINISHED ====')
        print('==== FINISHED ====')
        print('==== FINISHED ====')
        print('==== FINISHED ====')
        plt.show()
        print(opt)
        prysmfit(opt.x, plot=True)
        est_defocus_rms_wfe_step = opt.x[1]
        # print(len(list(focus_values)*len(freqs)))
        # print(len(chart_mtf_values.flatten()))
        # exit()
        est_defocus_rms_wfe_step = opt.x[1]
    else:
        fit, _ = optimize.curve_fit(prysmfit, list(focus_values)*len(freqs), chart_mtf_values.flatten(),
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

    longitude_defocus_step_um = est_defocus_pv_wfe_step * focusset.exif.aperture**2 * 8 * 0.55
    log.info("    Image side focus shift      {:8.3f} µm".format(longitude_defocus_step_um))

    na = 1 / (2.0 * focusset.exif.aperture)
    theta = np.arcsin(na)
    coc_step = np.tan(theta) * longitude_defocus_step_um * 2

    focal_length_m = focusset.exif.focal_length * 1e-3

    def get_opposide_dist(dist):
        return 1.0 / (1.0 / focal_length_m - 1.0 / dist)

    lens_angle_of_view = focusset.exif.angle_of_view
    # print(lens_angle_of_view)
    subject_distance = CHART_DIAGONAL * 0.5 / np.sin(lens_angle_of_view / 2)
    image_distance = get_opposide_dist(subject_distance)

    log.info("    Subject side focus shift    {:8.2f} mm".format((get_opposide_dist(image_distance-longitude_defocus_step_um *1e-6) - get_opposide_dist(image_distance)) * 1e3))
    log.info("    Blur circle  (CoC)          {:8.2f} µm".format(coc_step))

    log.info("Nominal subject distance {:8.2f} mm".format(subject_distance * 1e3))
    log.info("Nominal image distance   {:8.2f} mm".format(image_distance * 1e3))

    return est_defocus_rms_wfe_step, longitude_defocus_step_um, coc_step, image_distance, subject_distance, cauchy_peak_y
