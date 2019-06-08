#!/usr/bin/python3
import os
import logging
from matplotlib import pyplot as plt

from lentil import *
from lentil import focus_set
from lentil.plot_utils import COLOURS
from lentil.wavefront import estimate_wavefront_errors
from lentil import wavefront_utils
from lentil.wavefront_analysis import plot_wfe_data, plot_nominal_psfs
from lentil import wavefront_analysis
from lentil.wavefront_config import RANDOM_SEED
import lentil.wavefront_utils
from lentil import wavefront_test

BASE_PATH = "/home/sam/nashome/MTFMapper Stuff/"

PATHS = [
    # "Bernard/",

    # "56mm/f1.2/",
    # "56mm/f2.8/",
    # "56mm/f5.6/",
    # "56mm/f8/",

    # '16-55mm/16mm f2.8/',
    # '16-55mm/16mm f5.6/',

    # '16mm/f1.4/',
    # '16mm/f2/',
    # '16mm/f2.8/',
    # '16mm/f4/',
    # '16mm/f5.6/',
    # '16mm/f8/',
    # '16mm/f11/',

    # '16-55mm/16mm/f2.8/',
    # '16-55mm/16mm/f4fine/',
    # '16-55mm/16mm/f5.6/',
    # '16-55mm/16mm/f8/',
    # '16-55mm/16mm/f11/',
    #
    # '16-55mm/18mm/f2.8/',
    # '16-55mm/18mm/f4/',
    # '16-55mm/18mm/f5.6/',
    # '16-55mm/18mm/f8/',
    # '16-55mm/18mm/f11/',
    #

    '16-55mm/27mm/f2.8/',
    '16-55mm/27mm/f4/',
    # '16-55mm/27mm/f5.6/',
    # '16-55mm/27mm/f8/',

    # "16-55mm/55mm/f2.8/",
    # "16-55mm/55mm/f4/",
    # "16-55mm/55mm/f5.6/",
    # "16-55mm/55mm/f8/",
    # "16-55mm/55mm/f11/",

    # "55-200mm/55mm/f3.5/",
    # "55-200mm/55mm/f5.6/",
    # "55-200mm/55mm/f8/",
    # "55-200mm/55mm/f11/",

    # "55-200mm/95mm/f4/",
    # "55-200mm/95mm/f5.6/",
    # "55-200mm/95mm/f8/",
    # "55-200mm/95mm/f11/",

    # "55-200mm/200mm/f4.8/",
    # "55-200mm/200mm/f5.6/",
    # "55-200mm/200mm/f8/",
    # "55-200mm/200mm/f11/",
    # "55-200mm/200mm/f16/",

    # "100-400mm/100mm/f4.5/",
    # "100-400mm/100mm/f5.6/",
    # "100-400mm/100mm/f8/",

    # "100-400mm/230mm/f5b/",
    # "100-400mm/230mm/f5.6b/",
    # "100-400mm/230mm/f6.3b/",
    # "100-400mm/230mm/f7.1b/",
    # "100-400mm/230mm/f8/",
    # "100-400mm/230mm/f8b/",
    # "100-400mm/230mm/f11b/",
    # "100-400mm/230mm/f16/",

    # "60mm/f2.4/",
    # "60mm/f4/",
    # "60mm/f5.6/",
    # "60mm/f8/delay/",
    # "60mm/f11/",
    #
    # "90mm/f2/",
    # "90mm/f2.8/",
    # "90mm/f4/",
    # "90mm/f5.6/",
    # "90mm/f8/",
    # "90mm/f11/",
    # "90mm/f16/",
    #
    # "18-55mm/55mm/f4/",
    # "18-55mm/55mm/f5.6/",
    # "18-55mm/55mm/f8/",
    # "18-55mm/55mm/f11/",
    # '23mm f1.4/',
]

ax = None
recalibrate = 0
calibration = 1  # None if recalibrate else True
names = []
# PATHS.reverse()
# focusset = FocusSet(PATHS[0], include_all=1, use_calibration=1, rescan=0)
# focusset.fields[7].plot_points(0.45, MERIDIONAL)
# focusset.build_calibration(8, writetofile=False)
# exit()
# focusset.find_compromise_focus(axis=MERIDIONAL);exit()
# field = SFRField(pathname=os.path.join(PATHS[0], "mtfmappertemp_426", SFRFILENAME), calibration=focusset.base_calibration)
# field.plot(freq=0.35, plot_type=CONTOUR2D)
# field = SFRField(pathname=os.path.join(PATHS[1], "DSCF0004.RAF.no_corr.sfr"), calibration=focusset.base_calibration)
# field.plot(freq=0.35, plot_type=CONTOUR2D)
# field = SFRField(pathname=os.path.join(PATHS[2], "DSCF0004.RAF.ca_only.sfr"), calibration=focusset.base_calibration)
# field.plot(freq=0.35, plot_type=CONTOUR2D)
# field = SFRField(pathname=os.path.join(PATHS[3], "DSCF0004.RAF.ca_and_distortion.sfr"), calibration=focusset.base_calibration)
# field.plot(freq=0.35, plot_type=CONTOUR2D)
#
# exit()

apertures = []
fns = {}
freq = 0.28
# field = SFRField(pathname=os.path.join(BASE_PATH, "18-55mm/55mm/f11/mtfm3/DSCF8556.RAF.no_corr.sfr"), get_phase=True)
# exit()
# fs = SFRField(pathname="/home/sam/nashome/MTFMapper Stuff/56mm/f5.6/mtfm3/DSCF8260.RAF.no_corr.sfr", load_complex=True)
# exit()

fallbackpaths = [fallback_results_path(os.path.join(BASE_PATH, path), 3) for path in PATHS[:]]

# focussets = [FocusSet(path, include_all=0, use_calibration=1, load_complex=False) for path in fallbackpaths[:1]]
# wavefront_analysis.plot_chromatic_aberration(focussets[0])
# focussets[0].read_wavefront_data(overwrite=True)
# wavefront_analysis.plot_nominal_psfs(focussets[0], stop_downs=[0,1,2,3])
# exit()
# focussets[0].plot_sfr_vs_freq_at_point_for_each_field(2500, 2000)
# exit()
# focus_set.clear_numbered_autosaves(focussets[0].get_wavefront_data_path())
#focussets[0].read_wavefront_data(overwrite=True, read_autosave=True)
# exit()
# wavefront_analysis.plot_nominal_psfs(focussets[0])
# exit()
# focussets[0].fields[13].plot_sfr_at_point(4500,2000,MERIDIONAL_IMAJ)
# exit()
# focussets[0].fields[15].plot_points(freq=0.25, axis=MERIDIONAL_ANGLE)
initial = np.array([1.91342048e-03,  2.48280903e-02,  1.67862687e-02,  1.42241087e-02,  2.72742148e-03,  1.13075087e-03,  6.85835415e-04, -1.14663112e-04,  4.15021973e-05, -2.65469031e-03,  6.32943724e-03, -4.76860553e-04,  2.69996364e-03,  3.20183870e-04, -1.81168606e-03,  2.84055262e-04,  7.59997655e-04, -5.52678153e-03, -6.03509949e-03,  1.80867226e-05,  1.02042402e-03, -1.14488022e-03, -9.51656281e-06, -2.32969179e-03, -5.78473956e-04,  4.20201538e-04, -3.03927220e-03, -5.60703629e-03, -1.04249616e-03,  5.83945741e-04,  3.08436089e-03,  9.05787010e-04, -1.80680001e-03, -1.92969569e-04, -9.74365542e-04, -9.74365536e-04, -1.17940292e-04, -1.03126267e-03, -1.70651666e-03, -3.37188287e-04,  4.26668090e-05, -1.14910906e-03, -2.49515759e-03, -3.94710456e-03, -7.28123535e-03,  2.00000000e-03,  2.90897471e-04])

# estimate_wavefront_errors(fallbackpaths, fs_slices=8, from_scratch=False, x_loc=700, y_loc=701,
#                          plot_gradients_initial=None, complex_otf=False)
estimate_wavefront_errors(fallbackpaths, fs_slices=(25,20,20,20,20), from_scratch=True, x_loc=700, y_loc=701,
                          plot_gradients_initial=None, complex_otf=True)
# estimate_wavefront_errors(fallbackpaths, fs_slices=3, from_scratch=False, x_loc=2000, y_loc=2000,
#                           plot_gradients_initial=None, complex_otf=True)
# plot_wfe_data(focussets[0])
# (_.remove_duplicated_fields() for _ in focussets)
# for f in focussets:
#     wavefront_utils.remove_last_saved_wavefront_data(f)
# wavefront_utils.jitterstats()
# focus_values = focus_set.FocusPositionReader(fallbackpaths[0])
# print(focus_values['DSCF0004.RAF.no_corr.sfr'])
# print(focus_values[4])
# print(focus_values[44])
# print(np.array(focus_values))
# focus_set.estimate_focus_jitter(fallbackpaths[1], plot=2)
# focus_set.save_focus_jitter(fallbackpaths[0], None)
# wavefront_utils.optimise  _loca_colvolution_coeffs()
exit()

# aperture = focusset.exif.aperture
# fn = focusset.get_mtf_vs_image_height(freq=freq)
# ident = focusset.exif.lens_model + " " + focusset.exif.focal_length
# if ident in fns:
#     fns[ident].append((aperture, fn))
# else:
#     fns[ident] = [(aperture, fn)]

# sharps.append(focusset.find_best_focus(3000, 2000, 0.08).sharp)
# mtf50s.append(focusset.find_best_focus(3000, 2000, MTF50).sharp)
# focusset.find_best_focus(2000, 2000, plot=1)
# focusset.find_relevant_fields();exit()
# focusset.build_calibration(focusset.exif.aperture*1.1, writetofile=0)
# focusset.plot_sfr_vs_freq_at_point_for_each_field(323.364411, 3957.360777, MERIDIONAL, waterfall=1)
# focusset.find_best_focus(323.364411, 3957.360777, AUC, MERIDIONAL)
# focusset.find_best_focus(2000,2000, AUC, MERIDIONAL, plot=True, fitfn=cauchy)
# focusset.find_best_focus(2000,2000, AUC, MERIDIONAL, plot=True, fitfn=fastgauss)
# exit()
# focusset.find_compromise_focus(plot_type=SMOOTH2D, detail=2)
# exit()
# pos = focusset.find_compromise_focus(freq=AUC)
# focusset.plot_mtf_vs_image_height(show_diffraction=True, show=0)

# skew = False
# freq = 0.08
# ax, skew = focusset.plot_ideal_focus_field(freq=freq, plot_type=PROJECTION3D, plot_curvature=True, skewplane=skew, show=False)
# ax, skew = focusset.plot_ideal_focus_field(freq=freq, axis=SAGITTAL, alpha=0.2, plot_type=PROJECTION3D, plot_curvature=True, skewplane=skew, ax=ax, show=False)
# ax, skew = focusset.plot_ideal_focus_field(freq=freq, axis=MERIDIONAL, alpha=0.2,  plot_type=PROJECTION3D, plot_curvature=True, skewplane=skew, ax=ax, show=False)
# plt.show()

# focusset.plot_ideal_focus_field(freq=AUC, plot_type=SMOOTH2D, plot_curvature=False, axis=MEDIAL)
# focusset.fields[14].plot_sfr_at_point(5688.378935, 3939.958362, MERIDIONAL)
# exit()
# focusset.plot_sfr_vs_freq_at_point_for_each_field(3750.982357181818, 143.092597, MERIDIONAL, waterfall=1)
# focusset.plot_sfr_vs_freq_at_point_for_each_field(2782.284068272727, 3939.958362, MERIDIONAL, waterfall=1)
# focusset.plot_sfr_vs_freq_at_point_for_each_field(5688.378935, 3939.958362, MERIDIONAL, waterfall=1)
"""56mm f/1.2 mtfm3 high fit errors
    3750.982357181818, 143.092597, 0.3 MERIDIONAL
High fit error: 0.066
2782.284068272727, 3939.958362, 0.3 MERIDIONAL
5688.378935, 3939.958362, 0.3 MERIDIONAL"""
"""
27mm f/2.8 323.364411, 3957.360777, AUC, MERIDIONAL Bimodal"""
# focusset.find_best_focus(2782.284068272727, 3939.958362, 0.3, MERIDIONAL, plot=True)
# focusset.find_best_focus(2782.284068272727, 3939.958362, 0.3, MERIDIONAL, plot=True)
# focusset.find_best_focus(5688.378935, 3939.958362, 0.3, MERIDIONAL, plot=True)
# plt.plot(apertures, sharps, color=COLOURS[0], label="0.08 cy/px")
# plt.plot(apertures, sharps, '.')
# calibrator.average_calibrations()
# calibrator.write_calibration()
exit()
print(fns)
heights = [0.1, 0.45, 0.75]
for nmodel, (model, lst) in enumerate(fns.items()):
    apertures, fnlist = zip(*lst)
    for nheight, height in enumerate(heights):
            linestyles = ['-', '--', ':']
            colours = [0, 3, 4]
            plt.plot(apertures, [_(height, 0.1) for _ in fnlist], linestyles[nheight], color=COLOURS[colours[nmodel]],
                     label="{} {:.2f}".format(model, height), alpha=0.35)
            plt.plot(apertures, [_(height, 0.1) for _ in fnlist], 's', color=COLOURS[colours[nmodel]], alpha=0.8)
plt.ylim(diffraction_mtf(freq, LOW_BENCHMARK_FSTOP), diffraction_mtf(freq, HIGH_BENCHBARK_FSTOP))
# plt.plot(apertures, mtf50s, color=COLOURS[2], label="MTF50")
# plt.plot(apertures, mtf50s, '.')
plt.legend()
plt.xscale("log")
plt.title(focusset.exif.summary)
plt.show()
