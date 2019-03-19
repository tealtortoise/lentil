#!/usr/bin/python3
import os
from matplotlib import pyplot as plt

from lentil import *

sfrfilename = 'edge_sfr_values.txt'
PATHS = [
    "/mnt/mtfm/Bernard/",
    # "/mnt/mtfm/56mm/f2.8/mtfm/",
    # "/mnt/mtfm/56mm/f8/mtfm/"
    # "/mnt/mtfm/56mm/f5.6/mtfm/"
    # "/mnt/mtfm/56mm/f1.2/mtfm/"

    # '/mnt/mtfm/16-55mm/16mm f5.6/'
    # '/mnt/mtfm/16-55mm/27mm f2.8/'
    # '/mnt/mtfm/16-55mm/27mm f4/mtfm/'
    # '/mnt/mtfm/16-55mm/27mm f8/'
    # '/mnt/mtfm/23mm f1.4/mtmf/'
]
ax = None
for path in PATHS:
    filenames = []

    with os.scandir(path) as it:
        for entry in it:
            print(entry.path)
            try:
                entrynumber = int("".join([s for s in entry.name if s.isdigit()]))
            except ValueError:
                continue

            if entry.is_dir():
                fullpathname = os.path.join(path, entry.path, sfrfilename)
                print(fullpathname)
                sfr_file_exists = os.path.isfile(fullpathname)
                if not sfr_file_exists:
                    continue
            elif entry.is_file():
                fullpathname = entry.path
            else:
                continue
            filenames.append((entrynumber, fullpathname))

    filenames.sort()
    _, filenames = zip(*filenames)

    focusset = FocusSet(filenames[:])
    # focusset.fields[1].plot(0.3, detail=2.0, axis=SAGITTAL, show=True, plot_type=0)
    focusset.fields[1].plot_fit_errors_2d(freqs=[0.3], by_percent=False, axis=BOTH_AXES)
    # focusset.fields[1].summarise_fit_errors(freqs=[0.3], by_percent=False)
    exit()
    dups = [4,7,9, 12, 16,23, 27.1, 32, 36 ]
    # focusset.plot_field_curvature_strip_contour(0.18, SAGITTAL)
    # focusset.plot_field_curvature_strip_contour(0.18, MERIDIONAL)
    # exit()
    # focusset.remove_duplicated_fields()
    # exit()
    # focusset.fields[7].plot(0.1, SAGITTAL, plot_type=0, detail=2)
    # focusset.find_best_focus(1000, 2000, 0.16, MERIDIONAL, plot=False)
    # plt.plot(focusset.fields[0].np_dict[MERIDIONAL]['np_x'], focusset.fields[0].np_dict[MERIDIONAL]['np_y'], '.')
    # plt.show()
    exit()
    # focusset.find_best_focus(500, 2000, 0.26, SAGITTAL, plot=True)
    # plt.show()
    # focusset.find_best_focus(500, 2000, 0.26, BOTH_AXES, plot=True)
    # plt.show()
    # exit()
    # ax = focusset.fields[8].plot(0.15, MERIDIONAL, show=False, ax=ax, alpha=0.4)
    # ax = focusset.fields[8].plot(0.15, SAGITTAL, show=False, ax=ax, alpha=0.4)
    # plt.show()
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
    detail = 1.5
    plot_type = 1
    # ax, skewplane = focusset.plot_ideal_focus_field(detail=detail, show=False, freq=0.14,
    #                                                 ax=ax, axis=BOTH_AXES,
    #                                                 plot_type=1, plot_curvature=1,
    #                                                 color=[0.0, 0.0, 1.0, 0.5],
    #                                                 skewplane=skewplane, alpha=0.6)
    # ax, skewplane = focusset.plot_ideal_focus_field(detail=detail, show=False, freq=0.14,
    #                                                 ax=ax, axis=MERIDIONAL,
    #                                                 plot_type=1, plot_curvature=1,
    #                                                 color=[0.0, 0.0, 1.0, 0.5],
    #                                                 skewplane=skewplane, alpha=0.2)
    ax, skewplane = focusset.plot_ideal_focus_field(detail=detail, show=False, freq=0.14,
                                                    ax=ax, axis=SAGITTAL,
                                                    plot_type=plot_type, plot_curvature=1,
                                                    color=[0.0, 0.0, 1.0, 0.5],
                                                    skewplane=skewplane, alpha=0.6)

    # focusset.find_best_focus(2000, 3000, 0.25, SAGGITAL, plot=True)
    # focusset.find_best_focus(4597.787801708333, 1536.6772678750003, 0.25, SAGGITAL, plot=True)
plt.show()
# focusset.plot_ideal_focus_field(0.3)
exit()
# print(focusset.fields[2].interpolate_value(4000, 2000, 0.2, axis =MERIDIONAL));exit()
# print(focusset.fields[2].plot(axis=MERIDIONAL)); exit()
axis = SAGITTAL

focusset = FocusSet(filenames)
# focusset.find_best_focus(1400, 2000, 0.1, axis, plot=True)
plt.show()

# field.plot(SAGGITAL, 1, detail=1.5)
# field.plot(MERIDIONAL, 1, detail=1.5)
