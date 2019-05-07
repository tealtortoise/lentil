#!/usr/bin/env python
# coding: utf-8

# In[2]:


import inspect

import numpy as np
from scipy import  interpolate, optimize
from prysm import NollZernike, MTF, PSF, FringeZernike

from matplotlib import pyplot as plt
from lentil import FocusSet
from lentil.constants_utils import *
# 9, 16, 25, 36
for z in range(3,60):
    print(z)
    p = FringeZernike(**{"Z{}".format(z):0.3})
    print(z)
    p.plot2d()
    plt.show()
exit()

plt.style.use('bmh')
if 0:
    dia = 10
    fnos = [1, 4]
    opds = [.4, .1]
    mtfs = []
    for fno, opd in zip(fnos, opds):
        z = NollZernike(Z11=opd, norm=True, dia=dia, opd_unit='um', wavelength=0.1, samples=1024)
        ps = PSF.from_pupil(z, efl=dia * fno)
        mt = MTF.from_psf(ps)
        mtfs.append(mt.tan)

    fig, ax = plt.subplots()
    for curve, fno, opd in zip(mtfs, fnos, opds):
        # each element in mtfs is a tuple of (frequency, MTF).
        # * unpacks this and provides the x, y positional arguments to ax.plot
        ax.plot(*curve, label=f'F/{fno}, opd={opd}')

    ax.legend()
    ax.set(xlim=(0, 120), xlabel='Spatial Frequency [cy/mm]',
           ylim=(0, 1), ylabel='MTF [Rel. 1.0]')
    plt.show()
    exit()

freqs = np.arange(0, 0.5, 1/64) * 250
freqsint = np.arange(0.05, 0.3, 1/64) * 250

addabers = np.linspace(0.1, 0.2, 7)
fstop = 2.8
wl = 0.55
for add in addabers:
    pupil = NollZernike(Z11=add, dia=10, norm=True, opd_unit='waves', wavelength=wl, samples=1024)
    m = MTF.from_pupil(pupil, efl=10*fstop)
    sharp = np.mean(m.exact_xy(freqsint))
    print(add, sharp)

    plt.plot(freqs / 250, m.exact_xy(freqs), label="RMS WFE Z11 {:.2f}µm f/1".format(add))
# fstop = 4
# addabers *= 0.25
# for add in addabers:
#     pupil = NollZernike(Z11=add*0.5, dia=10, norm=True, opd_unit='um', wavelength=wl, samples=1024)
#     m = MTF.from_pupil(pupil, efl=10*fstop)
#     plt.plot(freqs, m.exact_xy(freqs), label="RMS WFE Z11 {:.2f}µm f/4".format(add))
plt.legend()
plt.xlabel("Spacial Frequency (cy/mm)")
plt.ylabel("MTF")
plt.ylim(0,1)
# plt.title("MTF vs Z11 Spherical Aberration f/{:.1f} (λ={:.0f}nm)".format(fstop, wl*1000))
plt.title("MTF vs Z11 Spherical Aberration(λ={:.0f}nm)".format(wl*1000))

plt.show()
