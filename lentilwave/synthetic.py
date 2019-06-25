import multiprocessing
import random

import numpy as np

from lentil import wavefront_config__old, FocusSetData, diffraction_mtf, EXIF
from lentil.wavefront_config__old import SPACIAL_FREQS
from lentil.wavefront_test__old import try_wavefront
from lentil.wavefront_utils import cauchy_fit, get_weights


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
    with multiprocessing.Pool(processes=wavefront_config__old.CUDA_PROCESSES) as pool:
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