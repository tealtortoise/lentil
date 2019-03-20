#!/usr/bin/python3
import os
import logging
from matplotlib import pyplot as plt

from lentil import *


PATHS = [
    # "/mnt/mtfm/Bernard/",

    # "/mnt/mtfm/56mm/f1.2/mtfm/"
    # "/mnt/mtfm/56mm/f2.8/mtfm/",
    # "/mnt/mtfm/56mm/f5.6/mtfm/"
    "/mnt/mtfm/56mm/f8/mtfm/",

    # '/mnt/mtfm/16-55mm/16mm f5.6/'

    # '/mnt/mtfm/16-55mm/27mm f2.8/'
    # '/mnt/mtfm/16-55mm/27mm f4/mtfm/'
    '/mnt/mtfm/16-55mm/27mm f8/'

    # "/mnt/mtfm/16-55mm/55mm/f2.8/mtfm/"
    # "/mnt/mtfm/16-55mm/55mm/f4/mtfm/",
    # "/mnt/mtfm/16-55mm/55mm/f5.6/mtfm/"
    # "/mnt/mtfm/16-55mm/55mm/f8/mtfm/"
    # "/mnt/mtfm/16-55mm/55mm/f11/mtfm/"

    # "/mnt/mtfm/18-55mm/55mm/f4/mtfm/"
    # "/mnt/mtfm/18-55mm/55mm/f5.6/mtfm/"
    "/mnt/mtfm/18-55mm/55mm/f8/mtfm/"
    # "/mnt/mtfm/18-55mm/55mm/f11/mtfm/"
    # '/mnt/mtfm/23mm f1.4/mtmf/'
]
ax = None
for path in PATHS:

    focusset = FocusSet(path, include_all=0)
    # field = focusset.fields[9]
    # points = field.points
    # axis = MERIDIONAL
    # freq01 = [point.get_freq(0.2) for point in points]
    # mtf50 = [point.get_freq(-1) for point in points]
    # auc = [point.auc for point in points]
    # plt.scatter(freq01, auc)
    # plt.show()
    # focusset.fields[1].plot(0.3, detail=2.0, axis=SAGITTAL, show=True, plot_type=0)
    # focusset.fields[1].plot_fit_errors_2d(freqs=[0.3], by_percent=False, axis=MEDIAL)
    # focusset.fields[1].summarise_fit_errors(freqs=[0.3], by_percent=False)
    # exit()
    # dups = [4,7,9, 12, 16,23, 27.1, 32, 36 ]
    # focusset.plot_field_curvature_strip_contour(AUC, SAGITTAL)
    # focusset.plot_field_curvature_strip_contour(AUC, MERIDIONAL)
    # focusset.remove_duplicated_fields()
    # exit()
    # focusset.find_relevant_fields(detail=0.3, writepath=path)
    # focusset.fields[8].plot_points(0.05, MERIDIONAL) #, plot_type=1, detail=2)
    # focusset.remove_duplicated_fields(plot=1)
    # exit()
    # focusset.find_best_focus(1800, 1800, 0.3, MERIDIONAL, plot=True)
    # plt.show()
    # focusset.find_best_focus(1800, 1800, 0.05, SAGITTAL, plot=True)
    # plt.show()
    # exit()
    # plt.plot(focusset.fields[0].np_dict[MERIDIONAL]['np_x'], focusset.fields[0].np_dict[MERIDIONAL]['np_y'], '.')
    # plt.show()
    # exit()
    # focusset.find_best_focus(500, 2000, 0.26, SAGITTAL, plot=True)
    # plt.show()
    # focusset.find_best_focus(500, 2000, 0.26, MEDIAL, plot=True)
    # plt.show()
    # exit()
    # ax = focusset.fields[8].plot(0.15, MERIDIONAL, show=False, ax=ax, alpha=0.4)
    # ax = focusset.fields[8].plot(0.15, SAGITTAL, show=False, ax=ax, alpha=0.4)
    # exit()
    #
    # y = []
    # x = np.linspace(0, 1 - 1/64, 64)
    #
    # plt.plot(x, GOOD)
    # plt.plot(x, diffraction_mtf(x*1.3))
    # plt.show()
    # exit()
    # for field in focusset.fields:
    #     field.plot_points(0.5, axis=SAGGITAL)
        # field.plot(0.3, axis=SAGGITAL, show=True)
        # field.plot(0.3, axis=MERIDIONAL, show=True)
    # x_vals = np.linspace(0, 6000, 30)
    # freqs = np.linspace(0,0.9, 30)
    # for n in x_vals:
    #     y_vals = []
    #     for f in freqs:
    #         y_vals.append(focusset.fields[7].interpolate_value(n, 0, f, SAGGITAL))
    #     plt.plot(freqs, y_vals)
    #     plt.show()
    #     focusset.find_best_focus(n, 0, 0.3, SAGGITAL, plot=True); plt.show()
    # exit()
    # focusset.find_best_focus(541.0863437083334, 71.681156, 0.3, SAGGITAL, plot=True); plt.show(); exit()
    # focusset.fields[8].plot_points();exit()
    # focusset.plot_field_curvature_strip(0.3);exit()
    skewplane = 0
    detail = 0.7
    plot_type = 0
    plot_curvature = 0
    freq = 15/250
    ax, skewplane = focusset.plot_ideal_focus_field(detail=detail, show=False, freq=freq,
                                                    ax=ax, axis=MEDIAL,
                                                    plot_type=plot_type, plot_curvature=plot_curvature,
                                                    color=[0.0, 0.0, 1.0, 0.5],
                                                    skewplane=skewplane, alpha=0.6)
    # ax, skewplane = focusset.plot_ideal_focus_field(detail=detail, show=False, freq=freq,
    #                                                 ax=ax, axis=MERIDIONAL,
    #                                                 plot_type=1, plot_curvature=1,
    #                                                 color=[0.0, 0.0, 1.0, 0.5],
    #                                                 skewplane=skewplane, alpha=0.2)
    # ax, skewplane = focusset.plot_ideal_focus_field(detail=detail, show=False, freq=freq,
    #                                                 ax=ax, axis=SAGITTAL,
    #                                                 plot_type=plot_type, plot_curvature=1,
    #                                                 color=[0.0, 0.0, 1.0, 0.5],
    #                                                 skewplane=skewplane, alpha=0.2)
    plt.show()

    # focusset.find_best_focus(2000, 3000, 0.25, SAGGITAL, plot=True)
    # focusset.find_best_focus(4597.787801708333, 1536.6772678750003, 0.25, SAGGITAL, plot=True)
