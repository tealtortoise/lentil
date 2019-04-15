#!/usr/bin/env python
# coding: utf-8

# In[2]:


import inspect

import numpy as np
from scipy import  interpolate, optimize
from prysm import NollZernike, MTF, PSF

from matplotlib import pyplot as plt
from lentil import FocusSet
from lentil.constants_utils import *

plt.style.use('bmh')
freqs = np.arange(0.0, 0.5, 1/64) * 250
for z in [0,4,11,22, 37, -1]:
# for z in range(30,50):
    add = {0:0, 4:0.21, 5:0.4, 11:0.2, 22:0.2, 36:0.2, -1:0}
    if z == 0:
        pupil = NollZernike(dia=10, norm=True, wavelength=0.55, opd_unit="um")
    elif z == -1:
        pupil = NollZernike(dia=10, norm=True, wavelength=0.55*3.7, opd_unit="um")
    else:
        pupil = NollZernike(dia=10, norm=False, wavelength=0.55, opd_unit="waves", **{"Z{}".format(z): 0.4})
    m = MTF.from_pupil(pupil, efl=10*2.8)
    # sharp = np.mean(m.exact_xy(freqs))
    # pupil.plot2d()
    # plt.show()
    print(z)
    plt.plot(freqs / 250, m.exact_xy(freqs), label="Z{:.0f}".format(z))
plt.ylim(0,1)
plt.xlim(0,0.5)
plt.legend()
plt.show()
exit()
# m = MTF.from_pupil(pupil, efl=80)
# plt.plot(freqs, m.exact_xy(freqs))
# plt.ylim(0, 1)
# plt.show()
# pupil = NollZernike(Z11=0.1, dia=10, norm=True, wavelength=0.15, opd_unit="um")
# pupil.plot2d()
# plt.show()
# m = MTF.from_pupil(pupil, efl=14)
# plt.plot(freqs, m.exact_xy(freqs))
# plt.ylim(0, 1)
# plt.show()
# defocus_ar = np.linspace(-5, 5, 21)
defocus_ar = np.linspace(-1.5, 1.5, 21)
# defocus_ar = np.array([0])
model_step_size = defocus_ar[1] - defocus_ar[0]
print("Defocus step size {:.3f}".format(model_step_size))
# for zed in range(9,11   ):
zed = 11
fl = 100
fstop = 2.8
wl = 0.55
normterms = []
addaberrs = np.linspace(0, 0.3, 10)
offsets = [10, 0.2, 0.1, 0.0, -0.1, -0.2, -0.3,]
# offsets = [0.0]
for offset in offsets:
    offset = offset / 2
    spacingresults = []
    peak_ys = []
    mtfs = []
    for add in addaberrs:
        zedstr = "Z{:.0f}".format(zed)
        print(zedstr)
        if zed == 4:
            continue
        mtf_ar = []
        peak = 0
        for nd, z4defocus in enumerate(defocus_ar):
            pupil = NollZernike(Z4=z4defocus*wl + add*2.0, dia=fl/fstop, norm=True, **{zedstr:add}, wavelength=wl, opd_unit="um")
            # pupil.plot2d()
            # plt.show()
            # print(pupil.strehl)
            m = MTF.from_pupil(pupil, efl=fl)
            if 0:
                plt.plot(freqs, m.exact_xy(freqs))
                plt.ylim(0, 1)
            # continue
            # psf = PSF.from_pupil(pupil, efl=10*fstop)
            # psf.plot2d()
            # plt.show()
            # continue
            # print(pupil.strehl)
            if nd >= 990:
                def optfn(inp):
                    return psf.encircled_energy(inp) - 0.8
                try:
                    coc = optimize.newton(optfn, 2, tol=0.01)
                    print("CoC from PSF {:.2f}".format(coc * 2))
                except RuntimeError:
                    pass
            focusshift_um = z4defocus * 3.5 * 0.55 * (fstop ** 2) * 8
            na = 1 / (2.0 * fstop)
            theta = np.arcsin(na)
            coce = np.tan(theta) * focusshift_um * 2
            # print("Focus shift {:.3f}um".format(focusshift_um))
            # print("Theta {}".format(theta * 180 / np.pi))
            # print("CoC est {:.2f} um".format(coce))
            # sharp = m.exact_xy([freqs[32]])[0]
            # mtf_ar.append(sharp)

            sharp = np.mean(m.exact_xy(freqs))
            # plt.plot(freqs, m.exact_xy(freqs))
            # plt.show()
            mtf_ar.append(sharp)
            # sharp = np.mean(m.exact_xy([10]))
            if sharp > peak:
                peak = sharp
            # print("MTF {:.3f}".format(sharp))
        # continue

        if 0 and "normalise":
            mtf_ar = np.array(mtf_ar) / np.max(mtf_ar)
        peak_ys.append(peak)
        focusset = FocusSet()
        # focusset._fixed_defocus_step_wfe = defocus_ar[1] - defocus_ar[0]
        exif = EXIF()
        exif.aperture = fstop
        exif.focal_length = 28.0
        focusset.exif = exif
        posob = lambda x: 0
        posob.focus_data = np.arange(len(mtf_ar))
        posob.sharp_data = np.array(mtf_ar)
        print(mtf_ar)
        posob.interpfn = interpolate.InterpolatedUnivariateSpline(posob.focus_data, mtf_ar, k=2)
        # print(posob.focus_data, mtf_ar)
        # focusset.find_best_focus(0, 0, _pos=posob, plot=1)
        # continue
        est_defocus_rms_wfe_step, longitude_defocus_step_um, coc_step, image_distance, subject_distance, peak_y = \
            focusset.find_best_focus(0, 0, _pos=posob, plot=0, _return_step_data_only=True, _step_est_offset=offset)
        spacingresults.append(est_defocus_rms_wfe_step)
        # peak_ys.append(peak_y)

        # plt.plot(defocus_ar, mtf_ar, label="{} {:.2f}".format(zedstr, add))
    # plt.show()

    print(model_step_size)
    print(addaberrs)
    print(spacingresults)

    normterms.append(spacingresults[0] / model_step_size)
    spacingresults_norm = np.array(spacingresults)  / spacingresults[np.argmax(addaberrs == 0)]

    if offset > 3:
        plt.plot(addaberrs, (spacingresults_norm - 1) * 100, label="No Correction".format(offset))
    else:
        plt.plot(addaberrs, (spacingresults_norm - 1) * 100, label="Correction {:.2f}".format(offset))
# plt.plot(addaberrs, spacingresults / model_step_size)
# plt.plot(addaberrs, peak_ys)
print(offsets)
print("norm", normterms)
# plt.ylim(-20, 5)
plt.legend()
plt.xlim(0, 0.3)
plt.ylabel("Error in defocus step estimation (%)")
plt.xlabel("Additional Z11 Spherical aberration (λ RMS WFE Norm)")
plt.title("Estimation errors in determining defocus from MTF (f/2.8)")
plt.show()
plt.plot(addaberrs, peak_ys)
plt.xlim(0, 0.5)
plt.ylim(0.0, 1.0)
plt.xlabel("Additional Z11 Spherical aberration (λ RMS WFE Norm)")
plt.ylabel("MTF at 10 cy/mm")
plt.title("Best focus MTF performance for varying spherical aberration")
plt.show()
# plt.legend()
# plt.xlabel("Defocus (waves)")
# plt.ylabel("Area under MTF (from 0-125 cy/mm)")
# plt.title("MTF vs defocus with addition of 0.45 waves of other aberrations")
# plt.show()
# Now we will calculate its Strehl using the function defined above, and via the MTF equation
exit()
# In[24]:


efl = 28  # F/2.8, arbitrary and meaningless here
pupil_diffraction = NollZernike(dia=10)
mtf_diffraction = MTF.from_pupil(pupil_diffraction, efl)
mtf_real = MTF.from_pupil(pupil, efl)

# convenience function, needed since the samples are log-spaced
def integrate_mtf(mtf):
    return np.trapz(np.trapz(mtf.data, x=mtf.unit_x, axis=1), x=mtf.unit_y, axis=0)

int_real = integrate_mtf(mtf_real)
int_diff = integrate_mtf(mtf_diffraction)
strehl_mtf = int_real / int_diff

pupil.strehl, strehl_mtf**2, pupil.strehl / strehl_mtf**2


# The ratio of the two is .977, a good fit.  Especially considering Welford's equation is only approximate.  We can see (perhaps) how the area under the MTF might make a good metric, given how much is known about RMS wavefront error as an indicator of imaging or focusing equality.
#
# Let's check if the two move together when we increase wavefront error by adding focus

# In[39]:


def make_mtf(focus, return_p=False):
    p = NollZernike(dia=10, Z11=0.05, Z4=focus, norm=True)
    m = MTF.from_pupil(p, efl=28)
    if return_p:
        return p, m
    else:
        return m

def make_rmswfe_area_under_mtf_tuple(focus):
    p, m = make_mtf(focus, return_p=True)
    m_strehl_proxy = integrate_mtf(m)
    rmswfe = p.rms
    return rmswfe, m_strehl_proxy

def plot_ratio_of_metrics(values):
    plt.plot(*values.swapaxes(0,1))
    ax = plt.gca()
    ax.set(xlabel='RMS WFE, λ', ylabel='Area Under MTF')

focuses = np.linspace(-2, 2, 50)  # +/- 2 waves RMS of focus error is a pretty big range, but this is no problem.
values = np.asarray([make_rmswfe_area_under_mtf_tuple(f) for f in focuses])
plot_ratio_of_metrics(values)


# We see two lines because there are both postiive-valued and negative-valued focus contributions, but the RMS is always positive.  The curve looks a lot like $1/x^2$, what if we manipulate things to undo that?

# In[41]:


vals2 = values.copy()
vals2[:,1] = 1 / np.sqrt(vals2[:, 1])
plot_ratio_of_metrics(vals2)


# Now things are proportional, less some wiggles. We can conclude (with perhaps a lack of some mathematical rigor in how easily we were convinced) that the area under the (entire) MTF curve is inversely proportional to the square of the RMS wavefront error.  This makes sense, given our bridge through the Strehl ratio, and knowledge that a high MTF needs low wavefront error.
#
# If we maximize the area under the MTF, or minimize this manipulation of it, we achieve minimum RMS wavefront error focus.
#
# Now, there may be ripples in the MTF as a function of focus.  Let's plot an image to see them.

# In[50]:


# make_mtf returns a prysm.otf.MTF object, tan returns (abscissa, coordinate) arrays, we just take the values for now
source_of_units = make_mtf(0)
mtf_values = [make_mtf(f).tan[1] for f in focuses]

mtf_im = np.asarray(mtf_values)
plt.imshow(mtf_im, extent=[0, 1e3 / pupil.wavelength / 2.8, -2, 2], aspect='auto', interpolation='lanczos')
plt.gca().set(xlabel='Spatial Frequency, [cy/mm]', ylabel='Defocus [λ RMS]')
plt.grid(False)


# Plenty of ripples.  What if we plot the MTF at a single spatial frequency, say 75 lp/mm?

# In[53]:


idx_75 = np.searchsorted(source_of_units.tan[0], 75)
plt.plot(focuses, mtf_im[:, idx_75])
plt.gca().set(ylabel='MTF, [Rel. 1.0]', xlabel='Defocus [λ RMS]')


# Clear ripples.  What about the area under the curve overlaid with this?  Our wavefronts are rotationally symmetric, so the area under the tangential MTF (or any other radial slice) is proportional to the area under the entire thing.

# In[58]:


mtf_im_auc = np.trapz(mtf_im, x=source_of_units.tan[0], axis=1)
plt.plot(focuses, mtf_im[:, idx_75], label='MTF @ 75 cy/mm')
plt.plot(focuses, mtf_im_auc/mtf_im_auc.max() * mtf_im[:, idx_75].max(), label='AUC')  # arbitrary normalization
plt.gca().set(ylabel='metric', xlabel='Defocus [λ RMS]')
plt.legend(title='metric')


# It misses a little bit at this frequency, but does a nice job of smoothing out the ripples.  It also looks similar to a Lorentzian or other curve-fittable form.  To demonstrate the bad idea-ness of using a single frequency as a focus metric, let's crank up the amount of spherical aberration and put a bend in the MTF vs Frequency.

# In[65]:


def make_mtf(focus, return_p=False):
    p = NollZernike(dia=10, Z11=0.15, Z4=focus, norm=True)
    m = MTF.from_pupil(p, efl=28)
    if return_p:
        return p, m
    else:
        return m

def make_rmswfe_area_under_mtf_tuple(focus):
    p, m = make_mtf(focus, return_p=True)
    m_strehl_proxy = integrate_mtf(m)
    rmswfe = p.rms
    return rmswfe, m_strehl_proxy


# In[66]:


mtf_values = [make_mtf(f).tan[1] for f in focuses]

mtf_im = np.asarray(mtf_values)
plt.imshow(mtf_im, extent=[0, 1e3 / pupil.wavelength / 2.8, -2, 2], aspect='auto', interpolation='lanczos')
plt.gca().set(xlabel='Spatial Frequency, [cy/mm]', ylabel='Defocus [λ RMS]')
plt.grid(False)


# We see a characteristic curvature now.  What is best focus?  Depending on the frequency we look at, we will give a different answer.

# In[67]:


mtf_im_auc = np.trapz(mtf_im, x=source_of_units.tan[0], axis=1)
plt.plot(focuses, mtf_im_auc)
plt.gca().set(ylabel='AUC', xlabel='Defocus [λ RMS]')#, xlim=(-0.25, 0.25))


# The area under the curve gives a single answer (and a smooth function!), and it's very close to zero (the "right" one, as chosen by minimum RMS wavefront).
