#!/usr/bin/python3
import os
import logging
from matplotlib import pyplot as plt

from lentil import *
from lentil.plot_utils import COLOURS

BASE_PATH = "/home/sam/nashome/MTFMapper Stuff/"

PATHS = [
    # "Bernard/",

    # "56mm/f1.2/",
    "56mm/f2.8/",
    "56mm/f5.6/",
    "56mm/f8/",

    # '16-55mm/16mm f2.8/',
    # '16-55mm/16mm f5.6/',

    # '16mm/f1.4/',
    # '16mm/f2/',
    # '16mm/f2.8/',
    # '16mm/f4/',
    # '16mm/f5.6/',
    '16mm/f8/',
    # '16mm/f11/',

    # '16-55mm/16mm/f2.8/',
    '16-55mm/16mm/f4fine/',
    '16-55mm/16mm/f5.6/',
    '16-55mm/16mm/f8/',
    '16-55mm/16mm/f11/',
    #
    # '16-55mm/18mm/f2.8/',
    # '16-55mm/18mm/f4/',
    # '16-55mm/18mm/f5.6/',
    # '16-55mm/18mm/f8/',
    # '16-55mm/18mm/f11/',
    #

    # '16-55mm/27mm/f2.8/',
    # '16-55mm/27mm/f4/',
    # '16-55mm/27mm/f5.6/',
    # '16-55mm/27mm/f8/',

    # "16-55mm/55mm/f2.8/",
    "16-55mm/55mm/f4/",
    # "16-55mm/55mm/f5.6/",
    "16-55mm/55mm/f8/",
    "16-55mm/55mm/f11/",

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
    "100-400mm/100mm/f5.6/",
    "100-400mm/100mm/f8/",

    # "100-400mm/230mm/f5b/",
    # "100-400mm/230mm/f5.6b/",
    # "100-400mm/230mm/f6.3b/",
    # "100-400mm/230mm/f7.1b/",
    # "100-400mm/230mm/f8/",
    "100-400mm/230mm/f8b/",
    "100-400mm/230mm/f11b/",
    # "100-400mm/230mm/f16/",

    # "60mm/f2.4/",
    "60mm/f4/",
    "60mm/f5.6/",
    "60mm/f8/delay/",
    "60mm/f11/",
    #
    # "90mm/f2/",
    # "90mm/f2.8/",
    # "90mm/f4/",
    "90mm/f5.6/",
    "90mm/f8/",
    "90mm/f11/",
    "90mm/f16/",
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
calibrator = Calibrator()
for nheight, path in enumerate(PATHS[:]):
    if "56mm" in path and "f2" in path:
        focusset = FocusSet(fallback_results_path(os.path.join(BASE_PATH, path), 3), include_all=0, use_calibration=1)
        # calibrator.add_focusset(focusset)
        focusset.estimate_wavefront_error(max_fnumber_error=0.0)
    # focusset = FocusSet(os.path.join(BASE_PATH, path), include_all=1, use_calibration=1)
    # focusset.exif.focal_length = 45
    # focusset.exif.aperture = 1.8
    # focusset.build_calibration(writetofile=1)
    # print(focusset.fields[0].points[0].calibration)
    # focusset.estimate_wavefront_error()
    # focusset.plot_ideal_focus_field(axis=MERIDIONAL)
    # focusset.find_best_focus(4500, 4500, LOWAVG, axis=MERIDIONAL, plot=1)
    # print(focusset.exif.aperture)
    # print(focusset.exif.focal_length)
    # exit()
    # focusset.attempt_to_calibrate_focus(4500, 3500, unit=FOCUS_SCALE_FOCUS_SHIFT)
    # focusset.plot_field_curvature_strip_contour(axis=SAGITTAL)
    # exit()
    # focusset.find_best_focus(3000, 2000, 0.5, MERIDIONAL, plot=1)
    # exit()
    # ax, sk = focusset.plot_ideal_focus_field(axis=MERIDIONAL, show=False)
    # focusset.plot_ideal_focus_field(axis=SAGITTAL, ax=ax, show=False)
    # ax.set_zlim(-0.4, 0.4)
    # plt.show()
    # exit()
    # focusset.find_best_focus(3000,2000,LOWAVG,plot=1, axis=SAGITTAL)
    # focusset.find_best_focus(3000,2000,0.3,plot=1, axis=SAGITTAL)
    # exit()


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
calibrator.average_calibrations()
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
