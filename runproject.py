#!/usr/bin/python3

from lentil import *
from lentil.plot_utils import COLOURS
from lentilwave.retrieval import estimate_wavefront_errors
from lentilwave import analysis

BASE_PATH = "/home/sam/nashome/MTFMapper Stuff/"

analysis.plot_tstops()
exit()

PATHS = [
    # "Bernard/",

    "56mm/f1.2/",
    "56mm/f2.8/",
    "56mm/f5.6/",
    # "56mm/f8/",

#     '16mm/f1.4/',
#     '16mm/f2/',
#     '16mm/f2.8/',
#     '16mm/f4/',
#     '16mm/f5.6/',
    # '16mm/f8/',
    # '16mm/f11/',

    # '16-55mm/16mm/f2.8/',
    # '16-55mm/16mm/f4/',
    # '16-55mm/16mm/f5.6/',
    # '16-55mm/16mm/f8/',
    # '16-55mm/16mm/f11/',

    # '16-55mm/18mm/f2.8/',
    # '16-55mm/18mm/f4/',
    # '16-55mm/18mm/f5.6/',
    # '16-55mm/18mm/f8/',
    # '16-55mm/18mm/f11/',
    #

    # '16-55mm/27mm/f2.8/',
    # '16-55mm/27mm/f4/',
    # '16-55mm/27mm/f5.6/',
    # '16-55mm/27mm/f8/',

    # "16-55mm/55mm/f2.8/",
    # "16-55mm/55mm/f4/",
    # "16-55mm/55mm/f5.6/",
    # "16-55mm/55mm/f8/",
    # "16-55mm/55mm/f11/",

    # "55-200mm/55mm/f3.5/",
    # "55-200mm/55mm/f5.6/",
    # "55-200mm/55mm/f8/",
    # "55-200mm/55mm/f11/",

    # "55-200mm/95mm/f4/",
    # "55-200mm/95mm/f5.6/",
    # "55-200mm/95mm/f8/",
    # "55-200mm/95mm/f11/",
    #
    # "55-200mm/200mm/f4.8/",
    # "55-200mm/200mm/f5.6/",
    # "55-200mm/200mm/f8/",
    # "55-200mm/200mm/f11/",
    # "55-200mm/200mm/f16/",

    # "100-400mm/100mm/f4.5/",
    # "100-400mm/100mm/f5.6/",
    # "100-400mm/100mm/f8/",

    # "100-400mm/230mm/f5b/",
    # "100-400mm/230mm/f5.6b/",
    # "100-400mm/230mm/f6.3b/",
    # "100-400mm/230mm/f7.1b/",
    # "100-400mm/230mm/f8/",
    # "100-400mm/230mm/f8b/",
    # "100-400mm/230mm/f11b/",
    # "100-400mm/230mm/f16/",

    # "60mm/f2.4/",
    # "60mm/f4/",
    # "60mm/f5.6/",
    # "60mm/f8/delay/",
    # "60mm/f11/",
    #
    # "90mm/f2/",
    # "90mm/f2.8/",
    # "90mm/f4/",
    # "90mm/f5.6/",
    # "90mm/f8/",
    # "90mm/f11/",
    # "90mm/f16/",
    #
    # "18-55mm/55mm/f4/",
    # "18-55mm/55mm/f5.6/",
    # "18-55mm/55mm/f8/",
    # "18-55mm/55mm/f11/",
    # '23mm f1.4/',
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
# if 1:
#     for x in np.linspace(-0.4, 0.4, 9):
#         s = wavefront_test.TestSettings(0, dict(base_fstop=2, fstop=2, v_x=-0.21, v_y=x))
#         s.x_loc = 600
#         s.phasesamples = 128
#         s.y_loc = 600
#
#
#         wavefront_test.mask_pupil(s,np, plot=True)
# exit()

# f = SFRField(pathname="/home/sam/mtfmtemp/edge_sfr_values.txt", load_complex=True)
# f2 = SFRField(pathname="/home/sam/mtfmtemp/edge_sfr_values.txt", load_complex=False)
# f = SFRField(pathname="/home/sam/nashome/MTFMapper Stuff/90mm/f2.8/mtfm3/DSCF0041.RAF.no_corr.sfr", load_complex=True)
# print(len(f.points))
# f.plot_edge_angle()
#f.plot_sfr_at_point(3000,2000)
# f.plot(4/64, axis=SAGITTAL)
# f2.plot(axis=SAGITTAL)
# f.plot(4/64, axis=MERIDIONAL)
# exit()
# f2.plot(axis=MERIDIONAL)
# f.plot_sfr_at_point(500,2500, SAGITTAL)
# f.plot_sfr_at_point(500,2500, MERIDIONAL)
# f.plot(AUC, SAGITTAL)
# exit()
# f.plot(0.1, SAGITTAL_IMAG, detail=2)
# f.plot(0.1, MERIDIONAL_IMAG, detail=2)
# f.plot_points(0.1, axis=SAGITTAL_IMAG)
# f.plot_points(0.15, axis=MERIDIONAL_IMAG)
# exit()

apertures = []
fns = {}
freq = 0.28
# field = SFRField(pathname=os.path.join(BASE_PATH, "90mm/f4/mtfm3/DSCF0008.RAF.no_corr.sfr"), load_complex=True)
# field.plot()
# exit()
# exit()
# fs = SFRField(pathname="/home/sam/nashome/MTFMapper Stuff/56mm/f5.6/mtfm3/DSCF8260.RAF.no_corr.sfr", load_complex=True)
# exit()
# wavefront_test.plot_lens_vignetting_loss(1.4)
# exit()
fallbackpaths = [fallback_results_path(os.path.join(BASE_PATH, path), 3) for path in PATHS[:]]
#
# focussets2 = [FocusSet(path, include_all=0, use_calibration=1, load_complex=False) for path in fallbackpaths[:1]]
focussets = [FocusSet(path, include_all=0, use_calibration=1, load_complex=False) for path in fallbackpaths[:1]]
# for x, y in [(5600, 3700), (200, 3800), (200, 200), (5800, 200)]:
    # x, y = 5800, 3800
    # x, y = 200, 3800
    # x, y = 200, 200
    # x, y = 5800, 200
    # focussets[0].plot_sfr_vs_freq_at_point_for_each_field(x, y, SAGITTAL)
    # focussets[0].plot_sfr_vs_freq_at_point_for_each_field(x, y, MERIDIONAL)
# exit()
#focussets[0].fields[9].plot_points(8/64,SAGITTAL, add_corners=True)
#focussets[0].fields[9].plot_points(8/64,MERIDIONAL, add_corners=True)
# ax = plt.gca()
# fig = ax.figure
# ax.set_xlim(0,0.5)
# ax.set_ylim(0.5,1)
# focussets[0].fields[19].plot(freq=4/64, axis=SAGITTAL)
# focussets2[0].fields[19].plot(4/64, axis=SAGITTAL)
# focussets2[0].plot_sfr_vs_freq_at_point_for_each_field(4800, 3400, SAGITTAL)
# focussets[0].plot_sfr_vs_freq_at_point_for_each_field(4800, 3400, SAGITTAL)
# focussets2[0].plot_best_sfr_vs_freq_at_point(4800, 3400, axis=SAGITTAL, ax=ax, fig=fig)
# exit()
# focussets[0].plot_ideal_focus_field()
# focussets[0].fields[15].plot(0.09, axis=MERIDIONAL_IMAG)
# exit()
# focussets1 = [FocusSet(path, include_all=0, use_calibration=1, load_complex=True) for path in fallbackpaths[:1]]
# print(focussets[0].focus_data)
# focussets[0].
# focussets[0].plot_best_sfr_vs_freq_at_point(5500, 3800, secondline_fn=lambda f: diffraction_mtf(f, 2.8), x_values=RAW_SFR_FREQUENCIES[:32:2])
# focussets[0].fields[9].plot_points(add_corners=True, autoscale=True)
# focussets[0].fields[0].plot_points(add_corners=True, autoscale=True)
# exit()
# focussets[0].plot_best_sfr_vs_freq_at_point(2500, 2800, secondline_fn=lambda f: diffraction_mtf(f, 2.8))
# focussets[0].plot_best_sfr_vs_freq_at_point(5500, 3800)
# focussets[0].find_best_focus(5500, 3800, axis=SAGITTAL, plot=True, freq=0.06)
# focussets[0].plot_mtf_vs_image_height(show_diffraction=focussets[0].exif.aperture)
# focussets[0].fields[9].plot(freq=0.05)
# focussets[0].plot_mtf_vs_image_height(show_diffraction=focussets[0].exif.aperture)
# focussets[0].plot_best_sfr_vs_freq_at_point(x=2500, y=2500, axis=SAGITTAL)
#focussets[0].fields[9].plot_fit_errors_2d()
# analysis.plot_nominal_psfs(focussets[0], x_loc=3400, y_loc=1600, stop_downs=(0,1,2,3))
# analysis.plot_chromatic_aberration(focussets[0])
# exit()
# /
initial = np.array([0.2       , 2.46468085, 1.61460029, 1.36471719, 0.0875    ,
       0.10284683, 0.06707652, 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.2       , 0.        , 0.        ])
# initial[:10] *= 2
initial = None
# grads, ordering = estimate_wavefront_errors(fallbackpaths, fs_slices=(20, 16, 14, 13, 12), from_scratch=True, x_loc=4800, y_loc=3400,
#                           plot_gradients_initial=initial, complex_otf=True, avoid_ends=1)
# grads = np.array([-1.36072816e+01,  1.69517186e-01,  5.66556794e-02,  1.07142371e-01,  6.73504528e-02,  7.64492433e-02, -1.21165080e+01, -8.23269748e+00, -5.84318337e+00, -2.90676978e+00, -1.16881759e+00,  9.83174243e-02, -2.32998051e-04,  4.93216159e-01,  2.42837039e-01, -8.06567277e-01,  1.95430222e-01, -6.62891194e-02, -4.86538038e-02,  6.61537089e-04, -4.03025633e-01, -3.27707870e-01,  4.02323164e-01, -4.63855289e-02,  4.54533814e-04, -1.71025883e-01,  1.49387170e-01, -4.52590505e-02, -1.51844004e-03, -5.89063372e-02,  1.08724961e-01, -3.97145341e-01,  1.15179117e-02, -2.25768424e-02,  9.07004911e-02, -6.25102395e-04,  7.15260058e-02,  7.15260058e-02, -3.01056248e-02, -7.19909981e-03, -1.21362845e-01, -8.03395493e-02,  5.62270768e-01, -4.35906331e-03,  6.46375807e-05,  6.33907675e-02,  5.55825523e-02, -2.56299716e-02, -3.45226735e-05, -9.26226302e-02, -3.86147789e-02, -1.78741781e-01, -3.99309336e-04,  6.79413080e-02,  1.62836500e-01, -4.71954079e-02, -6.63889213e-02, -3.99380827e-02,
#        -2.61142324e-01,  2.20676714e-01, -8.99599785e-02, -1.10982883e+01])
# ordering = [('fstop', (0, 1, 2, 3, 4), None), ('df_offset', (0,), None), ('df_offset', (1,), None), ('df_offset', (2,), None), ('df_offset', (3,), None), ('df_offset', (4,), None), ('df_step', (0,), None), ('df_step', (1,), None), ('df_step', (2,), None), ('df_step', (3,), None), ('df_step', (4,), None), ('z5', (0, 1, 2, 3, 4), None), ('z6', (0, 1, 2, 3, 4), None), ('z7', (0, 1, 2, 3, 4), None), ('z8', (0, 1, 2, 3, 4), None), ('z9', (0, 1, 2, 3, 4), None), ('z10', (0, 1, 2, 3, 4), None), ('z11', (0, 1, 2, 3, 4), None), ('z12', (0, 1, 2, 3, 4), None), ('z13', (0, 1, 2, 3, 4), None), ('z14', (0, 1, 2, 3, 4), None), ('z15', (0, 1, 2, 3, 4), None), ('z16', (0, 1, 2, 3, 4), None), ('z17', (0, 1, 2, 3, 4), None), ('z18', (0, 1, 2, 3, 4), None), ('z19', (0, 1, 2, 3, 4), None), ('z20', (0, 1, 2, 3, 4), None), ('z21', (0, 1, 2, 3, 4), None), ('z22', (0, 1, 2, 3, 4), None), ('z23', (0, 1, 2, 3, 4), None), ('z24', (0, 1, 2, 3, 4), None), ('z25', (0, 1, 2, 3, 4), None), ('z26', (0, 1, 2, 3, 4), None), ('z27', (0, 1, 2, 3, 4), None), ('z28', (0, 1, 2, 3, 4), None), ('z29', (0, 1, 2, 3, 4), None), ('z30', (0, 1, 2, 3, 4), None), ('z31', (0, 1, 2, 3, 4), None), ('z32', (0, 1, 2, 3, 4), None), ('z33', (0, 1, 2, 3, 4), None), ('z34', (0, 1, 2, 3, 4), None), ('z35', (0, 1, 2, 3, 4), None), ('z36', (0, 1, 2, 3, 4), None), ('z37', (0, 1, 2, 3, 4), None), ('z38', (0, 1, 2, 3, 4), None), ('z39', (0, 1, 2, 3, 4), None), ('z40', (0, 1, 2, 3, 4), None), ('z41', (0, 1, 2, 3, 4), None), ('z42', (0, 1, 2, 3, 4), None), ('z43', (0, 1, 2, 3, 4), None), ('z44', (0, 1, 2, 3, 4), None), ('z45', (0, 1, 2, 3, 4), None), ('z46', (0, 1, 2, 3, 4), None), ('z47', (0, 1, 2, 3, 4), None), ('z48', (0, 1, 2, 3, 4), None), ('loca1', (0, 1, 2, 3, 4), None), ('loca', (0, 1, 2, 3, 4), None), ('spca', (0, 1, 2, 3, 4), None), ('spca2', (0, 1, 2, 3, 4), None), ('v_slr', (0, 1, 2, 3, 4), None), ('tca_slr', (0, 1, 2, 3, 4), None), ('ellip', (0, 1, 2, 3, 4), None)]

# dct = wavefront_utils.build_normalised_scale_dictionary(grads, ordering)
# print(dct)
# exit()
# estimate_wavefront_errors(fallbackpaths, fs_slices=(25,23,21,19), from_scratch=False, x_loc=3400, y_loc=1600,
#                           plot_gradients_initial=None, complex_otf=True, avoid_ends=1)
estimate_wavefront_errors(fallbackpaths, fs_slices=(33, 29, 13), from_scratch=False, x_loc=5200, y_loc=3750,
                          plot_gradients_initial=initial, complex_otf=True, avoid_ends=1)
# estimate_wavefront_errors(fallbackpaths, fs_slices=(35,26,22,22,15), from_scratch=False, x_loc=2300, y_loc=1600,
#                           plot_gradients_initial=None, complex_otf=True)
# estimate_wavefront_errors(fallbackpaths, fs_slices=(22,20,18,15,15), from_scratch=False, x_loc=4800, y_loc=3400,
#                           plot_gradients_initial=None, complex_otf=True)
# estimate_wavefront_errors(fallbackpaths, fs_slices=(35,25,25,25), from_scratch=False, x_loc=2300, y_loc=1600,
#                           plot_gradients_initial=None, complex_otf=True)
# plot_wfe_data(focussets[0])
# (_.remove_duplicated_fields() for _ in focussets)
# for f in focussets:
#     wavefront_utils.remove_last_saved_wavefront_data(f)
# wavefront_utils.jitterstats()
# focus_values = focus_set.FocusPositionReader(fallbackpaths[0])
# print(focus_values['DSCF0004.RAF.no_corr.sfr'])
# print(focus_values[4])
# print(focus_values[44])
# print(np.array(focus_values))
# focus_set.estimate_focus_jitter(fallbackpaths[1], plot=2)
# focus_set.save_focus_jitter(fallbackpaths[0], None)
# wavefront_utils.optimise  _loca_colvolution_coeffs()
exit()

# aperture = focusset.exif.aperture
# fn = focusset.get_mtf_vs_image_height(freq=freq)
# ident = focusset.exif.lens_model + " " + focusset.exif.focal_length
# if ident in fns:
#     fns[ident].append((aperture, fn))
# else:
#     fns[ident] = [(aperture, fn)]

# sharps.append(focusset.find_best_focus(3000, 2000, 0.08).sharp)
# mtf50s.append(focusset.find_best_focus(3000, 2000, MTF50).sharp)
# focusset.find_best_focus(2000, 2000, plot=1)
# focusset.find_relevant_fields();exit()
# focusset.build_calibration(focusset.exif.aperture*1.1, writetofile=0)
# focusset.plot_sfr_vs_freq_at_point_for_each_field(323.364411, 3957.360777, MERIDIONAL, waterfall=1)
# focusset.find_best_focus(323.364411, 3957.360777, AUC, MERIDIONAL)
# focusset.find_best_focus(2000,2000, AUC, MERIDIONAL, plot=True, fitfn=cauchy)
# focusset.find_best_focus(2000,2000, AUC, MERIDIONAL, plot=True, fitfn=fastgauss)
# exit()
# focusset.find_compromise_focus(plot_type=SMOOTH2D, detail=2)
# exit()
# pos = focusset.find_compromise_focus(freq=AUC)
# focusset.plot_mtf_vs_image_height(show_diffraction=True, show=0)

# skew = False
# freq = 0.08
# ax, skew = focusset.plot_ideal_focus_field(freq=freq, plot_type=PROJECTION3D, plot_curvature=True, skewplane=skew, show=False)
# ax, skew = focusset.plot_ideal_focus_field(freq=freq, axis=SAGITTAL, alpha=0.2, plot_type=PROJECTION3D, plot_curvature=True, skewplane=skew, ax=ax, show=False)
# ax, skew = focusset.plot_ideal_focus_field(freq=freq, axis=MERIDIONAL, alpha=0.2,  plot_type=PROJECTION3D, plot_curvature=True, skewplane=skew, ax=ax, show=False)
# plt.show()

# focusset.plot_ideal_focus_field(freq=AUC, plot_type=SMOOTH2D, plot_curvature=False, axis=MEDIAL)
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
# plt.plot(apertures, sharps, color=COLOURS[0], label="0.08 cy/px")
# plt.plot(apertures, sharps, '.')
# calibrator.average_calibrations()
# calibrator.write_calibration()
exit()
print(fns)
heights = [0.1, 0.45, 0.75]
for nmodel, (model, lst) in enumerate(fns.items()):
    apertures, fnlist = zip(*lst)
    for nheight, height in enumerate(heights):
            linestyles = ['-', '--', ':']
            colours = [0, 3, 4]
            plt.plot(apertures, [_(height, 0.1) for _ in fnlist], linestyles[nheight], color=COLOURS[colours[nmodel]],
                     label="{} {:.2f}".format(model, height), alpha=0.35)
            plt.plot(apertures, [_(height, 0.1) for _ in fnlist], 's', color=COLOURS[colours[nmodel]], alpha=0.8)
plt.ylim(diffraction_mtf(freq, LOW_BENCHMARK_FSTOP), diffraction_mtf(freq, HIGH_BENCHBARK_FSTOP))
# plt.plot(apertures, mtf50s, color=COLOURS[2], label="MTF50")
# plt.plot(apertures, mtf50s, '.')
plt.legend()
plt.xscale("log")
plt.title(focusset.exif.summary)
plt.show()
