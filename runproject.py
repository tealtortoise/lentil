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

    # '/media/sf_mtfm/16-55mm/16mm f2.8/mtfm/',
    # '/media/sf_mtfm/16-55mm/16mm f5.6/',

    # '/media/sf_mtfm/16mm/f1.4/',
    # '/media/sf_mtfm/16mm/f2/mtfm/',
    '/media/sf_mtfm/16mm/f2.8/',
    # '/media/sf_mtfm/16mm/f4/',
    # '/media/sf_mtfm/16mm/f4/mtfm3/',
    # '/media/sf_mtfm/16mm/f4/mtfm4/',
    # '/media/sf_mtfm/16mm/f5.6/',
    # '/media/sf_mtfm/16mm/f5.6/mtfm3/',
    # '/media/sf_mtfm/16mm/f5.6/mtfm4/',
    # '/media/sf_mtfm/16mm/f8/mtfm/',
    # '/media/sf_mtfm/16mm/f11/mtfm/',

    # '/media/sf_mtfm/16-55mm/16mm/f2.8/',
    # '/media/sf_mtfm/16-55mm/16mm/f4/',
    # '/media/sf_mtfm/16-55mm/16mm/f4/mtfm4/',
    # '/media/sf_mtfm/16-55mm/16mm/f4fine/mtfm/',
    # '/media/sf_mtfm/16-55mm/16mm/f4fine/mtfm2/',
    # '/media/sf_mtfm/16-55mm/16mm/f5.6/',
    # '/media/sf_mtfm/16-55mm/16mm/f5.6/mtfm3/',
    # '/media/sf_mtfm/16-55mm/16mm/f5.6/mtfm4/',
    # '/media/sf_mtfm/16-55mm/16mm/f5.6/mtfm_distortion/',
    # '/media/sf_mtfm/16-55mm/16mm/f8/mtfm/',
    # '/media/sf_mtfm/16-55mm/16mm/f8/mtfm3/',
    # '/media/sf_mtfm/16-55mm/16mm/f8/mtfm4/',
    # '/media/sf_mtfm/16-55mm/16mm/f8/mtfm5/',
    # '/media/sf_mtfm/16-55mm/16mm/f11/mtfm/',

    # '/media/sf_mtfm/16-55mm/18mm/f2.8/mtfm/',
    # '/media/sf_mtfm/16-55mm/18mm/f4/mtfm/',
    # '/media/sf_mtfm/16-55mm/18mm/f5.6/mtfm/',
    # '/media/sf_mtfm/16-55mm/18mm/f8/mtfm/',
    # '/media/sf_mtfm/16-55mm/18mm/f11/mtfm/',
    #

    # '/media/sf_mtfm/16-55mm/27mm/f2.8/',
    # '/media/sf_mtfm/16-55mm/27mm/f4/',
    # '/media/sf_mtfm/16-55mm/27mm/f5.6/',
    # '/media/sf_mtfm/16-55mm/27mm/f8/',

    # "/media/sf_mtfm/16-55mm/55mm/f2.8/",
    # "/media/sf_mtfm/16-55mm/55mm/f4/mtfm/",
    # "/media/sf_mtfm/16-55mm/55mm/f4/mtfm3/",
    # "/media/sf_mtfm/16-55mm/55mm/f4/mtfm4/",
    # "/media/sf_mtfm/16-55mm/55mm/f4/mtfm5/",
    # "/media/sf_mtfm/16-55mm/55mm/f5.6/",
    # "/media/sf_mtfm/16-55mm/55mm/f8/mtfm/",
    # "/media/sf_mtfm/16-55mm/55mm/f11/mtfm/",

    # "/media/sf_mtfm/55-200mm/200mm/f4.8/mtfm/",
    # "/media/sf_mtfm/55-200mm/200mm/f5.6/mtfm/",
    # "/media/sf_mtfm/55-200mm/200mm/f8/mtfm/",
    # "/media/sf_mtfm/55-200mm/200mm/f11/mtfm/",
    # "/media/sf_mtfm/55-200mm/200mm/f16/mtfm/",

    # "/media/sf_mtfm/55-200mm/55mm/f3.5/mtfm/",
    # "/media/sf_mtfm/55-200mm/55mm/f5.6/mtfm/",
    # "/media/sf_mtfm/55-200mm/55mm/f8/mtfm/",
    # "/media/sf_mtfm/55-200mm/55mm/f11/mtfm/",

    # "/media/sf_mtfm/60mm/f2.4/mtfm/",
    # "/media/sf_mtfm/60mm/f4/mtfm/",
    # "/media/sf_mtfm/60mm/f4/mtfm3",
    # "/media/sf_mtfm/60mm/f4/mtfm4",
    # "/media/sf_mtfm/60mm/f4/mtfm5",
    # "/media/sf_mtfm/60mm/f5.6/mtfm/",
    # "/media/sf_mtfm/60mm/f8/delay/mtfm/",
    # "/media/sf_mtfm/60mm/f8/nodelay/mtfm/",
    # "/media/sf_mtfm/60mm/f11/mtfm/",
    #
    # "/media/sf_mtfm/90mm/f2/mtfm/",
    # "/media/sf_mtfm/90mm/f2.8/mtfm/",
    # "/media/sf_mtfm/90mm/f2.8/mtfm3/",
    # "/media/sf_mtfm/90mm/f4/mtfm/",
    # "/media/sf_mtfm/90mm/f4/mtfm3/",
    # "/media/sf_mtfm/90mm/f5.6/mtfm/",
    # "/media/sf_mtfm/90mm/f8/mtfm/",
    # "/media/sf_mtfm/90mm/f11/mtfm/",
    # "/media/sf_mtfm/90mm/f16/mtfm/",

    # "/media/sf_mtfm/18-55mm/55mm/f4/mtfm/",
    # "/media/sf_mtfm/18-55mm/55mm/f5.6/mtfm/",
    # "/media/sf_mtfm/18-55mm/55mm/f8/mtfm/",
    # "/media/sf_mtfm/18-55mm/55mm/f11/mtfm/",
    # '/media/sf_mtfm/23mm f1.4/mtmf/'
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
for n, path in enumerate(PATHS[0:1]):
    path = os.path.join(path, "mtfm3/")
    focusset = FocusSet(path, include_all=1, use_calibration=1)
    focusset.plot_ideal_focus_field()
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
    # focusset.find_best_focus(2782.284068272727, 3939.958362, 0.3, MERIDIONAL, plot=True)
    # focusset.find_best_focus(2782.284068272727, 3939.958362, 0.3, MERIDIONAL, plot=True)
    # focusset.find_best_focus(5688.378935, 3939.958362, 0.3, MERIDIONAL, plot=True)