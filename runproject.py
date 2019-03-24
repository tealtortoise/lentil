#!/usr/bin/python3
import os
import logging
from matplotlib import pyplot as plt

from lentil import *


PATHS = [
    # "/mnt/mtfm/Bernard/",

    # "/mnt/mtfm/56mm/f1.2/mtfm/",
    "/mnt/mtfm/56mm/f2.8/mtfm/",
    # "/mnt/mtfm/56mm/f5.6/mtfm/",
    # "/mnt/mtfm/56mm/f8/mtfm/",
    # "/mnt/mtfm/56mm/f8/mtfm/",

    # '/mnt/mtfm/16-55mm/16mm f2.8/mtfm/',
    # '/mnt/mtfm/16-55mm/16mm f5.6/',
    #
    # '/mnt/mtfm/16-55mm/27mm f2.8/',
    # '/mnt/mtfm/16-55mm/27mm f4/mtfm/',
    # '/mnt/mtfm/16-55mm/27mm f5.6/',
    # '/mnt/mtfm/16-55mm/27mm f8/',

    # "/mnt/mtfm/16-55mm/55mm/f2.8/mtfm/",  # dodgy?
    # "/mnt/mtfm/16-55mm/55mm/f4/mtfm/",
    # "/mnt/mtfm/16-55mm/55mm/f5.6/mtfm/",
    # "/mnt/mtfm/16-55mm/55mm/f8/mtfm/",
    # "/mnt/mtfm/16-55mm/55mm/f11/mtfm/",

    # "/mnt/mtfm/55-200mm/200mm/f4.8/mtfm/",
    # "/mnt/mtfm/55-200mm/200mm/f5.6/mtfm/",
    # "/mnt/mtfm/55-200mm/200mm/f8/mtfm/",
    # "/mnt/mtfm/55-200mm/200mm/f11/mtfm/",
    # "/mnt/mtfm/55-200mm/200mm/f16/mtfm/",

    # "/mnt/mtfm/55-200mm/55mm/f3.5/mtfm/",
    # "/mnt/mtfm/55-200mm/55mm/f5.6/mtfm/",
    # "/mnt/mtfm/55-200mm/55mm/f8/mtfm/",
    # "/mnt/mtfm/55-200mm/55mm/f11/mtfm/",

    # "/mnt/mtfm/60mm/f2.4/mtfm/",
    # "/mnt/mtfm/60mm/f4/mtfm/",
    # "/mnt/mtfm/60mm/f5.6/mtfm/",
    # "/mnt/mtfm/60mm/f8/delay/mtfm/",
    # "/mnt/mtfm/60mm/f8/nodelay/mtfm/",
    # "/mnt/mtfm/60mm/f11/mtfm/",

    # "/mnt/mtfm/90mm/f2/mtfm/",
    # "/mnt/mtfm/90mm/f2.8/mtfm/",
    # "/mnt/mtfm/90mm/f4/mtfm/",
    # "/mnt/mtfm/90mm/f5.6/mtfm/",
    # "/mnt/mtfm/90mm/f8/mtfm/",
    # "/mnt/mtfm/90mm/f11/mtfm/",
    # "/mnt/mtfm/90mm/f16/mtfm/",

    # "/mnt/mtfm/18-55mm/55mm/f4/mtfm/",
    # "/mnt/mtfm/18-55mm/55mm/f5.6/mtfm/"
    # "/mnt/mtfm/18-55mm/55mm/f8/mtfm/"
    # "/mnt/mtfm/18-55mm/55mm/f11/mtfm/"
    # '/mnt/mtfm/23mm f1.4/mtmf/'
]
ax = None
recalibrate = 0
calibration = 1#None if recalibrate else True
# PATHS.reverse()
for path in PATHS:

    focusset = FocusSet(path, rescan=0, include_all=0, use_calibration=calibration)
    # focusset.find_compromise_focus(detail=0.8, axis=MERIDIONAL, plot_freq=0.35)
    # bestpoint = focusset.get_peak_sfr(plot=1, show=0)
    # print(1)
    # focusset.set_calibration_sharpen(18.8, 0.3, stack=True)
    # focusset.set_calibration_sharpen(1.55, 1.6, stack=True)
    # focusset.set_calibration_sharpen(0.4, 8.0, stack=True)
    # print(2)
    # bestpoint = focusset.get_peak_sfr(plot=1, show=1)

    # exit()
    # bestpoint = focusset.get_peak_sfr()
    # bestpoint.plot_acutance_vs_printsize()
    # focusset.fields[0].points[100].get_acutance(1.0, 0.27)
    # exit()
    # focusset.get_peak_sfr()
    # calibration = focusset.build_calibration(fstop=5.6, writetofile=recalibrate)
    # for field in focusset.fields:
    #     field.plot_points(AUC, SAGITTAL, autoscale=True)
    # for x in np.linspace(0, 6000, 12):
    #     print(focusset.find_best_focus(x, x*2/3, 0.1, SAGITTAL, plot=True))
    #     plt.show()
    # print(focusset.find_best_focus(800, 1800, 0.05, SAGITTAL, plot=True))
    # plt.show()
    # focusset.find_best_focus(1800, 1800, 0.5, SAGITTAL, plot=True)
    # plt.show()
    # exit()
    # if 0:
    skewplane = 0
    detail = 0.6
    plot_type = CONTOUR2D
    plot_type = PROJECTION3D
    plot_curvature = 1
    freq = AUC
    ax = None
    ax, skewplane = focusset.plot_ideal_focus_field(detail=detail, show=False, freq=freq,
                                                    ax=ax, axis=MERIDIONAL,
                                                    plot_type=plot_type, plot_curvature=plot_curvature,
                                                    skewplane=skewplane, alpha=0.6, title=focusset.lens_name)

    # plt.show()
    # ax, skewplane = focusset.plot_ideal_focus_field(detail=detail, show=False, freq=freq,
    #                                                 ax=ax, axis=MERIDIONAL,
    #                                                 plot_type=plot_type, plot_curvature=1,
    #                                                 skewplane=skewplane, alpha=0.2)
    # ax, skewplane = focusset.plot_ideal_focus_field(detail=detail, show=False, freq=freq,
    #                                                 ax=ax, axis=SAGITTAL,
    #                                                 plot_type=plot_type, plot_curvature=1,
    #                                                 skewplane=skewplane, alpha=0.2)
plt.show()

# focusset.find_best_focus(2000, 3000, 0.25, SAGGITAL, plot=True)
# focusset.find_best_focus(4597.787801708333, 1536.6772678750003, 0.25, SAGGITAL, plot=True)
