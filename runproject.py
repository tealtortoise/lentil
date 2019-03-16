#!/usr/bin/python3
import os
from matplotlib import pyplot as plt

from lentil import *
from lentil.constants_utils import SAGGITAL, MERIDIONAL

PATH = '/mnt/mtfm/16-55mm/27mm f2.8/mtfmappertemp_{:.0f}/'
numberrange = range(14, 24)
PATH = '/mnt/mtfm/16-55mm/27mm f5.6/mtfmappertemp_{:.0f}/'
numberrange = range(43, 52)
sfrfilename = 'edge_sfr_values.txt'
PATH = "/mnt/mtfm/56mm/f2.8/mtfm/"
# PATH = '/mnt/mtfm/16-55mm/16mm f5.6/'
# PATH = '/mnt/mtfm/16-55mm/27mm f2.8/'
# PATH = '/mnt/mtfm/23mm f1.4/Results/'

imagedirs = os.listdir(PATH)
filenames = []
for dir in imagedirs:
    if dir[:9] == 'mtfmapper':
        filename = os.path.join(PATH, dir, sfrfilename)
        filenames.append(filename)

focusset = FocusSet(filenames[:])
for field in focusset.fields:
    pass# field.plot(0.1)
focusset.find_best_focus(2000, 3000, 0.25, SAGGITAL, plot=True)
# focusset.find_best_focus(4597.787801708333, 1536.6772678750003, 0.25, SAGGITAL, plot=True)
plt.show()
# focusset.plot_ideal_focus_field(0.3)
exit()
# print(focusset.fields[2].interpolate_value(4000, 2000, 0.2, axis =MERIDIONAL));exit()
# print(focusset.fields[2].plot(axis=MERIDIONAL)); exit()
axis = SAGGITAL

focusset = FocusSet(filenames)
# focusset.find_best_focus(1400, 2000, 0.1, axis, plot=True)
sag = []
sagl = []
sagh = []
mer = []
merl = []
merh = []
x_rng = range(100, 3900, 200)
for n in x_rng:
    x = 3000
    y= n
    f = 0.05
    focuspos, sharpness, l, h = focusset.find_best_focus(x, y, f, SAGGITAL)
    sag.append(focuspos)
    sagl.append(l)
    sagh.append(h)
    focuspos, sharpness, l, h = focusset.find_best_focus(x, y, f, MERIDIONAL)
    mer.append(focuspos)
    merl.append(l)
    merh.append(h)

plt.plot(x_rng, sag, color='green')
plt.plot(x_rng, sagl, '--', color='green')
plt.plot(x_rng, sagh, '--', color='green')
plt.plot(x_rng, mer, color='blue')
plt.plot(x_rng, merl, '--', color='blue')
plt.plot(x_rng, merh, '--', color='blue')
plt.show(); exit()
plt.show()

# field.plot(SAGGITAL, 1, detail=1.5)
# field.plot(MERIDIONAL, 1, detail=1.5)
