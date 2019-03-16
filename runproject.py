#!/usr/bin/python3
import os
from matplotlib import pyplot as plt

from lentil import *

PATH = '/mnt/mtfm/16-55mm/27mm f2.8/mtfmappertemp_{:.0f}/'
numberrange = range(14, 24)
PATH = '/mnt/mtfm/16-55mm/27mm f5.6/mtfmappertemp_{:.0f}/'
numberrange = range(43, 52)
sfrfilename = 'edge_sfr_values.txt'
PATHS = [
    # "/mnt/mtfm/56mm/f2.8/mtfm/",
    "/mnt/mtfm/56mm/f5.6/mtfm/"
    # "/mnt/mtfm/56mm/f1.2/mtfm/"

    # '/mnt/mtfm/16-55mm/16mm f5.6/'
# '/mnt/mtfm/16-55mm/27mm f2.8/'
# '/mnt/mtfm/16-55mm/27mm f8/'
# '/mnt/mtfm/23mm f1.4/Results/'
]
ax = None
for path in PATHS:
    imagedirs = os.listdir(path)
    filenames = []
    for dir_ in imagedirs:
        if dir_[:9] == 'mtfmapper':
            filename = os.path.join(path, dir_, sfrfilename)
            filenames.append(filename)

    focusset = FocusSet(filenames[:])
    # focusset.plot_field_curvature_strip(0.1);exit()
    ax = focusset.plot_ideal_focus_field(detail=0.5, show=False, freq=0.05, ax=ax, axis=MERIDIONAL,
                                         plot_curvature=True, color=[0.8, 0, 0, 0.5])
    ax = focusset.plot_ideal_focus_field(detail=0.5, show=False, freq=0.05, ax=ax, axis=SAGGITAL,
                                         plot_curvature=True, color=[0.0, 0.0, 1.0, 0.5])

    # ax = focusset.fields[9].plot(0.15, MERIDIONAL, show=False, ax=ax)
    # ax = focusset.fields[9].plot(0.15, SAGGITAL, show=False, ax=ax)
    # focusset.find_best_focus(2000, 3000, 0.25, SAGGITAL, plot=True)
    # focusset.find_best_focus(4597.787801708333, 1536.6772678750003, 0.25, SAGGITAL, plot=True)
plt.show()
# focusset.plot_ideal_focus_field(0.3)
exit()
# print(focusset.fields[2].interpolate_value(4000, 2000, 0.2, axis =MERIDIONAL));exit()
# print(focusset.fields[2].plot(axis=MERIDIONAL)); exit()
axis = SAGGITAL

focusset = FocusSet(filenames)
# focusset.find_best_focus(1400, 2000, 0.1, axis, plot=True)
plt.show()

# field.plot(SAGGITAL, 1, detail=1.5)
# field.plot(MERIDIONAL, 1, detail=1.5)
