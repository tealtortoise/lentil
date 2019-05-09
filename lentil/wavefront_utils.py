import time
import random
import multiprocessing
import numpy as np
from scipy import optimize
import prysm
from lentil.constants_utils import *
from lentil import wavefront_config
from lentil.wavefront_config import freqs, modelwavelengths


class FocusSetData:
    def __init__(self):
        self.chart_mtf_values = None
        self.mtf_means = None
        self.focus_values = None
        self.max_pos = None
        self.weights = None
        self.exif = None
        self.cauchy_peak_x = None


def mask_pupil(pupil, radius):
    x = np.linspace(-1, 1, pupil.phase.shape[0])
    y = np.linspace(-1, 1, pupil.phase.shape[0])
    grid = np.meshgrid(x, y)
    radius_grid = (grid[0]**2 + grid[1]**2)**0.5
    mask = radius_grid < radius
    pupil.mask(mask, 'both')
    return pupil


def try_wavefront(defocus=0, defocus_offset=0, defocus_step=0.1, loca=0, spca=0, basewv=0.575, fstop=2.8,
                  base_fstop=2.8, locaref=0.575, z={}, zero_offset=0.0, plot=False, only_strehl=False, samples=128):
    if fstop < base_fstop:
        raise ValueError("Base_fstop must be wider (lower) than fstop ")
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
                                  dia=10, norm=True,
                                  wavelength=modelwavelength,
                                  opd_unit="um",
                                  samples=samples,
                                  **zkwargs)

        pupil = mask_pupil(pupil, base_fstop / fstop)

        m = prysm.MTF.from_pupil(pupil, efl=base_fstop * 10)
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
                pupil = prysm.FringeZernike(dia=10, norm=True,
                                          wavelength=bestpupil[2],
                                          opd_unit="um",
                                          samples=samples,
                                          **{key: value})
                pupil = mask_pupil(pupil, base_fstop / fstop)
                slice = pupil.slice_x[1]
                rms = (slice[np.isfinite(slice)] ** 2).mean() ** 0.5
                plt.plot(slice, label="{} : {:.3f} λRMS".format(key, rms / basewv))
        slice_ = bestpupil[1].slice_x[1]
        rms = (slice_[np.isfinite(slice_)] ** 2).mean() ** 0.5
        plt.plot(slice_, label="All : {:.3f} λRMS".format(rms / basewv))
        print(222, bestpupil[3])
        pupil = prysm.FringeZernike(dia=10, norm=True,
                                  wavelength=bestpupil[2],
                                  opd_unit="um",
                                  samples=samples,
                                  **bestpupil[3])

        pupil = mask_pupil(pupil, base_fstop / fstop)
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

def pre_process_focussets(focussets, freqs, fs_slices, skip):
    ob = focussets[0].find_sharpest_location()
    datas = []
    for focusset in focussets:
        data = FocusSetData()
        pos = focusset.get_interpolation_fn_at_point(IMAGE_WIDTH / 2, IMAGE_HEIGHT / 2, AUC, SAGITTAL)
        focus_values = pos.focus_data[:]

        mtf_means = pos.sharp_data
        # chart_mtf_values = mtf_means
        # data = np.array(data)

        fitted_params = cauchy_fit(focus_values, mtf_means)

        cauchy_peak_x = fitted_params[1]
        cauchy_peak_y = fitted_params[0]
        print("Found peak {:.3f} at {:.3f}".format(cauchy_peak_y, cauchy_peak_x))
        data.cauchy_peak_x = cauchy_peak_x

        # Move on to get full frequency data
        size = fs_slices - 1
        slicelow = max(0, int(cauchy_peak_x - size * skip / 2))
        slicehigh = slicelow + size
        limit = (slicelow, slicehigh)
        print("Limit", limit)
        datalst = []
        for freq in freqs:
            pos = focusset.get_interpolation_fn_at_point(ob.x_loc, ob.y_loc, freq, SAGITTAL, limit=limit, skip=skip)
            pos1 = focusset.get_interpolation_fn_at_point(ob.x_loc, ob.y_loc, freq, MERIDIONAL, limit=limit, skip=skip)
            datalst.append((pos.sharp_data[:] + pos1.sharp_data) * 0.5)

        chart_mtf_values = np.array(datalst)  # [:,::skip]
        mtf_means = chart_mtf_values.mean(axis=0)  # [::skip]
        focus_values = pos.focus_data  # [::skip]
        max_pos = focus_values[np.argmax(mtf_means)]

        data.chart_mtf_values = chart_mtf_values
        data.mtf_means = mtf_means
        data.focus_values = focus_values
        data.max_pos = max_pos
        weights = np.ones(chart_mtf_values.shape)
        weightmean = np.mean(weights)
        data.weights = weights / weightmean
        data.exif = focusset.exif
        datas.append(data)
    return datas


def build_synthetic_dataset(subsets=4, test_stopdown=4, base_aperture=2.0, slices_per_fstop=15):
    cauchy_fit_x = float("NaN")
    z_ranges = dict(z7=1.2, z8=1.2, z9=0.6, z16=0.4, z25=0.2, z36=0.1)
    focus_motor_error = 0.0005
    defocus_step = 0.41 / base_aperture
    defocus_offset = slices_per_fstop / 2
    valid = False
    pool = multiprocessing.Pool()
    while not valid:
        strehls = []
        base_aperture_error = 1.0 #random.random() * 0.1 + 0.95

        z = {}
        zsum = 0.0
        zmaxsum = 0.0
        for paramname in wavefront_config.optimise_params:
            if paramname.lower()[0] == 'z' and paramname[1].isdigit():
                z_range = z_ranges.get(paramname, 0.5)
                z_random = (random.random() * (z_range * 2) - z_range)
                z[paramname] = z_random / base_aperture
                zsum += np.abs(z_random)
                zmaxsum += z_ranges[paramname]
        if not 0.2 < zsum < 0.75:
            print("Invali WFE! repeating...")
            continue

        nom_focus_values = np.arange(slices_per_fstop)
        dataset = []
        fstops = [base_aperture * 2 ** (n / 2) for n in range(test_stopdown)]
        for fstop in fstops:
            if fstop != base_aperture:
                individual_fstop_error = 1.0 # random.random() * 0.05 + 0.975
            else:
                individual_fstop_error = 1.0
            actual_focus_values = nom_focus_values + np.random.normal(0.0, focus_motor_error * base_aperture / fstop, nom_focus_values.shape)
            individual_df_step_error = random.random() * 0.2 + 0.9
            model_fstop = fstop * individual_fstop_error * base_aperture_error
            model_step = defocus_step * model_fstop / base_aperture / base_aperture_error * individual_df_step_error
            print("Nominal f#: {:.3f}, actual f#: {:.3f}".format(fstop, model_fstop))
            data = FocusSetData()
            through_focus = []
            strehls.append(0.0)
            for test_strehl in [False, True]:
                if test_strehl:
                    arr = np.array(through_focus).T
                    mtf_means = arr.mean(axis=0)
                    cauchy_fit_x = cauchy_fit(nom_focus_values, mtf_means)[1]
                    focus_values_to_test = [cauchy_fit_x]
                else:
                    focus_values_to_test = actual_focus_values
                arglst = []
                for focus in focus_values_to_test:
                    args = (focus,
                            defocus_offset,
                            model_step,
                            0.0, #loca
                            0.0, # spca
                            0.575,
                            model_fstop,
                            base_aperture * base_aperture_error,
                            0.575,
                            z,
                            0,
                            False,
                            False,
                            128 if test_strehl else 1024)
                    arglst.append(args)

                if test_strehl:
                    strehls[-1] = try_wavefront(*args)[3]
                else:
                    outs = pool.starmap(try_wavefront, arglst)
                    through_focus = [out[0] for out in outs]
            # print("Fstop {} strehl {}".format(fstop, strehls[-1]))

            data.chart_mtf_values = arr
            data.focus_values = nom_focus_values
            data.mtf_means = mtf_means
            data.weights = np.ones(data.chart_mtf_values.shape)
            data.max_pos = nom_focus_values[np.argmax(data.mtf_means)]
            data.cauchy_peak_x = cauchy_fit_x

            exif = EXIF()
            exif.aperture = fstop
            exif.focal_length_str = "28 mm"
            data.exif = exif

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
        if monotonic and min(strehls) > 0.15 and max(strehls) > 0.8:
            valid = True
        else:
            print("Not valid, rerunning!")
        valid=True

    print("Actual focus values: {}".format(str(actual_focus_values)))
    print("Zs: {}".format(str(z)))
    print("F-stops: {}".format(str(fstops)))

    return dataset[:subsets]