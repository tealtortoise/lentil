#!/usr/bin/python3
import os
import logging
from matplotlib import pyplot as plt

from lentil import *


PATHS = [
    # "/media/sf_mtfm/Bernard/",

    # "/media/sf_mtfm/56mm/f1.2/",
    # "/media/sf_mtfm/56mm/f2.8/",
    # "/media/sf_mtfm/56mm/f5.6/",
    # "/media/sf_mtfm/56mm/f8/",

    # '/media/sf_mtfm/16-55mm/16mm f2.8/',
    # '/media/sf_mtfm/16-55mm/16mm f5.6/',

    # '/media/sf_mtfm/16mm/f1.4/',
    # '/media/sf_mtfm/16mm/f2/',
    # '/media/sf_mtfm/16mm/f2.8/',
    # '/media/sf_mtfm/16mm/f4/',
    # '/media/sf_mtfm/16mm/f5.6/',
    # '/media/sf_mtfm/16mm/f8/',
    # '/media/sf_mtfm/16mm/f11/',

    # '/media/sf_mtfm/16-55mm/16mm/f2.8/',
    # '/media/sf_mtfm/16-55mm/16mm/f4/',
    # '/media/sf_mtfm/16-55mm/16mm/f4fine/",
    # '/media/sf_mtfm/16-55mm/16mm/f5.6/',
    # '/media/sf_mtfm/16-55mm/16mm/f8/',
    # '/media/sf_mtfm/16-55mm/16mm/f11/',

    # '/media/sf_mtfm/16-55mm/18mm/f2.8/',
    # '/media/sf_mtfm/16-55mm/18mm/f4/',
    # '/media/sf_mtfm/16-55mm/18mm/f5.6/",
    # '/media/sf_mtfm/16-55mm/18mm/f8/',
    # '/media/sf_mtfm/16-55mm/18mm/f11/',
    #

    # '/media/sf_mtfm/16-55mm/27mm/f2.8/',
    # '/media/sf_mtfm/16-55mm/27mm/f4/',
    # '/media/sf_mtfm/16-55mm/27mm/f5.6/',
    # '/media/sf_mtfm/16-55mm/27mm/f8/',

    # "/media/sf_mtfm/16-55mm/55mm/f2.8/",
    # "/media/sf_mtfm/16-55mm/55mm/f4/",
    # "/media/sf_mtfm/16-55mm/55mm/f4/mtfm3/",
    # "/media/sf_mtfm/16-55mm/55mm/f4/mtfm4/",
    # "/media/sf_mtfm/16-55mm/55mm/f4/mtfm5/",
    # "/media/sf_mtfm/16-55mm/55mm/f5.6/",
    # "/media/sf_mtfm/16-55mm/55mm/f8/",
    # "/media/sf_mtfm/16-55mm/55mm/f11/",

    "/media/sf_mtfm/55-200mm/200mm/f4.8/",
    # "/media/sf_mtfm/55-200mm/200mm/f5.6/",
    # "/media/sf_mtfm/55-200mm/200mm/f8/",
    # "/media/sf_mtfm/55-200mm/200mm/f11/",
    # "/media/sf_mtfm/55-200mm/200mm/f16/",

    # "/media/sf_mtfm/55-200mm/55mm/f3.5/",
    # "/media/sf_mtfm/55-200mm/55mm/f5.6/",
    # "/media/sf_mtfm/55-200mm/55mm/f8/",
    # "/media/sf_mtfm/55-200mm/55mm/f11/",

    # "/media/sf_mtfm/60mm/f2.4/",
    # "/media/sf_mtfm/60mm/f4/",
    # "/media/sf_mtfm/60mm/f4/mtfm3",
    # "/media/sf_mtfm/60mm/f4/mtfm4",
    # "/media/sf_mtfm/60mm/f4/mtfm5",
    # "/media/sf_mtfm/60mm/f5.6/",
    # "/media/sf_mtfm/60mm/f8/delay/",
    # "/media/sf_mtfm/60mm/f8/nodelay/",
    # "/media/sf_mtfm/60mm/f11/",
    #
    # "/media/sf_mtfm/90mm/f2/",
    # "/media/sf_mtfm/90mm/f2.8/",
    # "/media/sf_mtfm/90mm/f4/",
    # "/media/sf_mtfm/90mm/f5.6/",
    # "/media/sf_mtfm/90mm/f8/",
    # "/media/sf_mtfm/90mm/f11/",
    # "/media/sf_mtfm/90mm/f16/",

    # "/media/sf_mtfm/18-55mm/55mm/f4/",
    # "/media/sf_mtfm/18-55mm/55mm/f5.6/",
    # "/media/sf_mtfm/18-55mm/55mm/f8/",
    # "/media/sf_mtfm/18-55mm/55mm/f11/",
    # '/media/sf_mtfm/23mm f1.4/',
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
for n, path in enumerate(PATHS[:]):

    focusset = FocusSet(fallback_results_path(path, 5), include_all=1, use_calibration=1)
    # focusset.plot_sfr_vs_freq_at_point_for_each_field(323.364411, 3957.360777, MERIDIONAL, waterfall=1)
    # focusset.find_best_focus(323.364411, 3957.360777, AUC, MERIDIONAL)
    # focusset.find_best_focus(2000,2000, AUC, MERIDIONAL, plot=True, fitfn=cauchy)
    # focusset.find_best_focus(2000,2000, AUC, MERIDIONAL, plot=True, fitfn=fastgauss)
    # exit()
    # focusset.find_compromise_focus(plot_type=SMOOTH2D, detail=2)
    # exit()
    focusset.plot_ideal_focus_field(plot_type=SMOOTH2D, plot_curvature=False)
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