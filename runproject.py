#!/usr/bin/python3
import os
import logging
from matplotlib import pyplot as plt

from lentil import *


PATHS = [
    "/mnt/mtfm/Bernard/",

    # "/mnt/mtfm/56mm/f1.2/mtfm/"
    # "/mnt/mtfm/56mm/f2.8/mtfm/",
    # "/mnt/mtfm/56mm/f5.6/mtfm/",
    # "/mnt/mtfm/56mm/f8/mtfm/",

    # '/mnt/mtfm/16-55mm/16mm f5.6/'
    #
    # '/mnt/mtfm/16-55mm/27mm f2.8/',
    # '/mnt/mtfm/16-55mm/27mm f4/mtfm/',
    # '/mnt/mtfm/16-55mm/27mm f5.6/',
    # '/mnt/mtfm/16-55mm/27mm f8/',

    # "/mnt/mtfm/16-55mm/55mm/f2.8/mtfm/" # dodgy?
    # "/mnt/mtfm/16-55mm/55mm/f4/mtfm/",
    # "/mnt/mtfm/16-55mm/55mm/f5.6/mtfm/",
    # "/mnt/mtfm/16-55mm/55mm/f8/mtfm/",
    # "/mnt/mtfm/16-55mm/55mm/f11/mtfm/"

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
for path in PATHS:

    focusset = FocusSet(path, rescan=0, include_all=0)

    # for field in focusset.fields:
    #     field.plot_points(AUC, SAGITTAL, autoscale=True)
    # exit()

    # for x in np.linspace(0, 6000, 12):
    #     print(focusset.find_best_focus(x, x*2/3, 0.1, SAGITTAL, plot=True))
    #     plt.show()
    # print(focusset.find_best_focus(800, 1800, 0.05, SAGITTAL, plot=True))
    # plt.show()
    # focusset.find_best_focus(1800, 1800, 0.5, SAGITTAL, plot=True)
    # plt.show()
    # exit()


    skewplane = 1
    detail = 0.8
    plot_type = CONTOUR2D
    plot_curvature = 0
    freq = AUC
    ax, skewplane = focusset.plot_ideal_focus_field(detail=detail, show=False, freq=freq,
                                                    ax=ax, axis=MEDIAL,
                                                    plot_type=plot_type, plot_curvature=plot_curvature,
                                                    skewplane=skewplane, alpha=0.6)
    plt.show()
    # ax, skewplane = focusset.plot_ideal_focus_field(detail=detail, show=False, freq=freq,
    #                                                 ax=ax, axis=MERIDIONAL,
    #                                                 plot_type=plot_type, plot_curvature=1,
    #                                                 skewplane=skewplane, alpha=0.2)
    # ax, skewplane = focusset.plot_ideal_focus_field(detail=detail, show=False, freq=freq,
    #                                                 ax=ax, axis=SAGITTAL,
    #                                                 plot_type=plot_type, plot_curvature=1,
    #                                                 skewplane=skewplane, alpha=0.2)
    # plt.show()

    # focusset.find_best_focus(2000, 3000, 0.25, SAGGITAL, plot=True)
    # focusset.find_best_focus(4597.787801708333, 1536.6772678750003, 0.25, SAGGITAL, plot=True)
