#!/usr/bin/python3
import os
import logging
from matplotlib import pyplot as plt

from lentil import *


PATHS = [
    # "/mnt/mtfm/Bernard/",

    # "/mnt/mtfm/56mm/f1.2/mtfm/",
    # "/mnt/mtfm/56mm/f2.8/mtfm/",
    # "/mnt/mtfm/56mm/f5.6/mtfm/",
    # "/mnt/mtfm/56mm/f5.6/mtfm3/",
    # "/mnt/mtfm/56mm/f8/mtfm/",
    # "/mnt/mtfm/56mm/f8/mtfm/",

    # '/mnt/mtfm/16-55mm/16mm f2.8/mtfm/',
    # '/mnt/mtfm/16-55mm/16mm f5.6/',

    # '/mnt/mtfm/16mm/f1.4/mtfm/',
    # '/mnt/mtfm/16mm/f2/mtfm/',
    # '/mnt/mtfm/16mm/f2.8/mtfm/',
    # '/mnt/mtfm/16mm/f4/mtfm/',
    # '/mnt/mtfm/16mm/f4/mtfm3/',
    # '/mnt/mtfm/16mm/f4/mtfm4/',
    # '/mnt/mtfm/16mm/f5.6/mtfm/',
    '/mnt/mtfm/16mm/f5.6/mtfm3/',
    '/mnt/mtfm/16mm/f5.6/mtfm4/',
    # '/mnt/mtfm/16mm/f8/mtfm/',
    # '/mnt/mtfm/16mm/f11/mtfm/',

    # '/mnt/mtfm/16-55mm/16mm/f2.8/mtfm/',
    # '/mnt/mtfm/16-55mm/16mm/f4/mtfm3/',
    # '/mnt/mtfm/16-55mm/16mm/f4/mtfm4/',
    # '/mnt/mtfm/16-55mm/16mm/f4fine/mtfm/',
    # '/mnt/mtfm/16-55mm/16mm/f4fine/mtfm2/',
    # '/mnt/mtfm/16-55mm/16mm/f5.6/mtfm/',
    # '/mnt/mtfm/16-55mm/16mm/f5.6/mtfm3/',
    # '/mnt/mtfm/16-55mm/16mm/f5.6/mtfm4/',
    # '/mnt/mtfm/16-55mm/16mm/f5.6/mtfm_distortion/',
    # '/mnt/mtfm/16-55mm/16mm/f8/mtfm/',
    # '/mnt/mtfm/16-55mm/16mm/f11/mtfm/',

    # '/mnt/mtfm/16-55mm/18mm/f2.8/mtfm/',
    # '/mnt/mtfm/16-55mm/18mm/f4/mtfm/',
    # '/mnt/mtfm/16-55mm/18mm/f5.6/mtfm/',
    # '/mnt/mtfm/16-55mm/18mm/f8/mtfm/',
    # '/mnt/mtfm/16-55mm/18mm/f11/mtfm/',
    #

    # '/mnt/mtfm/16-55mm/27mm/f2.8/mtfm/',
    # '/mnt/mtfm/16-55mm/27mm/f4/mtfm/',
    # '/mnt/mtfm/16-55mm/27mm/f5.6/mtfm/',
    # '/mnt/mtfm/16-55mm/27mm/f8/mtfm/',

    # "/mnt/mtfm/16-55mm/55mm/f2.8/mtfm/",  # dodgy?
    # "/mnt/mtfm/16-55mm/55mm/f4/mtfm/",
    # "/mnt/mtfm/16-55mm/55mm/f5.6/mtfm/",
    # "/mnt/mtfm/16-55mm/55mm/f8/mtfm/",
    # "/mnt/mtfm/16-55mm/55mm/f11/mtfm/",

    # "/mnt/mtfm/55-200mm/200mm/f4.8/mtfm/",
    # "/mnt/mtfm/55-200mm/200mm/f5.6/mtfm/",
    # "/mnt/mtfm/55-200mm/200mm/f8/mtfm/",
    # "/mnt/mtfm/55-200mm/200mm/f11/mtfm/",
    # "/mnt/mtfm/55-200mm/200mm/f16/mtfm/",

    # "/mnt/mtfm/55-200mm/55mm/f3.5/mtfm/",
    # "/mnt/mtfm/55-200mm/55mm/f5.6/mtfm/",
    # "/mnt/mtfm/55-200mm/55mm/f8/mtfm/",
    # "/mnt/mtfm/55-200mm/55mm/f11/mtfm/",

    # "/mnt/mtfm/60mm/f2.4/mtfm/",
    # "/mnt/mtfm/60mm/f4/mtfm/",
    # "/mnt/mtfm/60mm/f4/mtfm_distortion/",
    # "/mnt/mtfm/60mm/f5.6/mtfm/",
    # "/mnt/mtfm/60mm/f8/delay/mtfm/",
    # "/mnt/mtfm/60mm/f8/nodelay/mtfm/",
    # "/mnt/mtfm/60mm/f11/mtfm/",
    #
    # "/mnt/mtfm/90mm/f2/mtfm/",
    # "/mnt/mtfm/90mm/f2.8/mtfm/",
    # "/mnt/mtfm/90mm/f2.8/mtfm3/",
    # "/mnt/mtfm/90mm/f4/mtfm/",
    # "/mnt/mtfm/90mm/f4/mtfm3/",
    # "/mnt/mtfm/90mm/f5.6/mtfm/",
    # "/mnt/mtfm/90mm/f8/mtfm/",
    # "/mnt/mtfm/90mm/f11/mtfm/",
    # "/mnt/mtfm/90mm/f16/mtfm/",

    # "/mnt/mtfm/18-55mm/55mm/f4/mtfm/",
    # "/mnt/mtfm/18-55mm/55mm/f5.6/mtfm/",
    # "/mnt/mtfm/18-55mm/55mm/f8/mtfm/",
    # "/mnt/mtfm/18-55mm/55mm/f11/mtfm/",
    # '/mnt/mtfm/23mm f1.4/mtmf/'
]
ax = None
recalibrate = 0
calibration = 1  # None if recalibrate else True
names = []
# PATHS.reverse()
# focusset = FocusSet(PATHS[0], include_all=1)
# focusset.find_compromise_focus()
field = SFRField(pathname=os.path.join(PATHS[0], "DSCF0005.RAF.sfr"), calibration=None)
field.plot(freq=0.35, plot_type=CONTOUR2D)
field = SFRField(pathname=os.path.join(PATHS[1], "DSCF0005.RAF.corr.sfr"), calibration=None)
field.plot(freq=0.35, plot_type=CONTOUR2D)

exit()
for n, path in enumerate(PATHS):
    focusset = FocusSet(path, include_all=1, use_calibration=0)
    focusset.plot_best_sfr_vs_freq_at_point(3000, 2000, axis=MERIDIONAL)
    # focusset.fields[5].points[100].plot()
    # focusset.skip_fields_and_check_accuracy()
    # pos = focusset.find_compromise_focus(plot_type=None, detail=1, weighting_fn=EVEN_WEIGHTED)
    # focusset.plot_mtf_vs_image_height(freqs=np.array((10, 30, 50))*1.5 / 250, analysis_pos=pos, detail=1)
    # exit()
    # focusset.find_best_focus(200, 200, freq=0.3, axis=SAGITTAL, plot=1)
    # focusset.build_calibration(4, writetofile=recalibrate, opt_freq=AUC)
    # exit()

    # data_sfr = focusset.get_peak_sfr(freq=AUC, axis=BOTH_AXES).raw_sfr_data[:]
    # data_sfr2 = focusset.find_sharpest_raw_points_avg_sfr() * focusset.base_calibration
    # data_sfr3 = diffraction_mtf(RAW_SFR_FREQUENCIES, 5.6, calibration=None)
    # print(data_sfr3)
    # plt.plot(data_sfr[:32], color='black')
    # names.append("16-55mm f/2.8 at f/4")
    # plt.plot(RAW_SFR_FREQUENCIES[:32], data_sfr2[:32], color='black')
    # names.append("f/4 diffraction limit")
    # plt.plot(RAW_SFR_FREQUENCIES[:32], data_sfr3[:32], '--', color='orange')
# plt.ylim(0, 1)
# plt.xlim(0, 0.5)
# plt.xlabel("Frequency (cy/px)")
# plt.ylabel("Spacial frequency response")
# plt.title("SFR of Fujifilm 16-55mm vs diffraction")
#
# plt.legend(names)
# plt.show()
if 0:
    # exit()
    # focusset.plot_ideal_focus_field(freq=AUC, skewplane=False, plot_type=CONTOUR2D, plot_curvature=False, detail=1.3)
    focusset.plot_ideal_focus_field(freq=AUC, skewplane=False, plot_type=PROJECTION3D, plot_curvature=1, detail=1.3)
    # pos =focusset.find_compromise_focus(freq=0.14, weighting_fn=CENTRE_WEIGHTED)
    # focusset.plot_best_focus_vs_frequency(3000, 2000)
    # exit()
    # focusset.plot_ideal_focus_field(plot_curvature=1, plot_type=PROJECTION3D, skewplane=True)#
    # axis = SAGITTAL
    # pos = focusset.find_sharpest_location(axis=axis)
    # focusset.plot_best_sfr_vs_freq_at_point(pos.x_loc, pos.y_loc,
    #                                         secondline_fn=lambda x: diffraction_mtf(x, 4), axis=axis)
    # focusset.find_best_focus(3000,2000, 0.4, plot=True, show=True)
    # plt.show()
    # focusset.guess_focus_shift_field()
    # focusset.plot_field_curvature_strip_contour()
    # pos = focusset.find_compromise_focus(detail=0.5, axis=MEDIAL, weighting_fn=EDGE_WEIGHTED)
    # focusset.plot_mtf_vs_image_height(pos, detail=0.7, freqs=(0.04, 0.16, 0.32), axis=MEDIAL)

    # focusset.plot_mtf_vs_image_height(detail=0.7, freqs=(0.04, 0.16,0.32), axis=MEDIAL)
    # exit()
    # focusset.plot_mtf_vs_image_height(pos, detail=0.7, freqs=(0.16,))
    # exit()
    # for field in focusset.fields:
    #     field.np_dict_cache[SAGITTAL]['np_x'] = None
    #     field.np_dict_cache[MERIDIONAL]['np_x'] = None
    # focusset.plot_field_curvature_strip_contour(0.25, SAGITTAL)
    # for field in focusset.fields[6:10]:
    #     field.plot(0.25, axis=SAGITTAL)
    # focusset.fields[8].plot(detail=2)
    # exit()
    # def weightfn(height):
    #     return 1 if (height < 0.1) else 0.0001
    #     return 1 if (0.68 < height < 0.72) else 0.0001
    # focusset.find_compromise_focus(freq=ACUTANCE, detail=0.7, axis=MEDIAL, plot_freq=None, weighting_fn=EVEN_WEIGHTED)
    # exit()
    # bestpoint = focusset.get_peak_sfr(plot=1, show=0)
    # print(1)
    # focusset.set_calibration_sharpen(18.8, 0.3, stack=True)
    # focusset.set_calibration_sharpen(1.55, 1.6, stack=True)
    # focusset.set_calibration_sharpen(0.4, 8.0, stack=True)
    # print(2)
    # bestpoint = focusset.get_peak_sfr(plot=1, show=1)

    # exit()
    # bestpoint = focusset.get_peak_sfr()
    # bestpoint.plot_acutance_vs_printsize()
    # focusset.fields[0].points[100].get_acutance(1.0, 0.27)
    # exit()
    # focusset.get_peak_sfr()
    # calibration = focusset.build_calibration(fstop=5.6, writetofile=recalibrate)
    # for field in focusset.fields:
    #     field.plot_points(AUC, SAGITTAL, autoscale=True)
    # for x in np.linspace(0, 6000, 12):
    #     print(focusset.find_best_focus(x, x*2/3, 0.1, SAGITTAL, plot=True))
    #     plt.show()
    # print(focusset.find_best_focus(800, 1800, 0.05, SAGITTAL, plot=True))
    # plt.show()
    # focusset.find_best_focus(1800, 1800, 0.5, SAGITTAL, plot=True)
    # plt.show()
    # exit()
if 0:
    skewplane = 0
    detail = 1.2
    plot_type = CONTOUR2D
    plot_type = PROJECTION3D
    plot_curvature = 1
    freq = AUC
    ax = None
    ax, skewplane = focusset.plot_ideal_focus_field(detail=detail, show=False, freq=freq,
                                                    ax=ax, axis=SAGITTAL,
                                                    plot_type=plot_type, plot_curvature=plot_curvature,
                                                    skewplane=skewplane, alpha=0.6, title=focusset.lens_name)

    # plt.show()
    # ax, skewplane = focusset.plot_ideal_focus_field(detail=detail, show=False, freq=freq,
    #                                                 ax=ax, axis=MERIDIONAL,
    #                                                 plot_type=plot_type, plot_curvature=1,
    #                                                 skewplane=skewplane, alpha=0.2)
    # ax, skewplane = focusset.plot_ideal_focus_field(detail=detail, show=False, freq=freq,
    #                                                 ax=ax, axis=SAGITTAL,
    #                                                 plot_type=plot_type, plot_curvature=1,
    #                                                 skewplane=skewplane, alpha=0.2)
plt.show()

# focusset.find_best_focus(2000, 3000, 0.25, SAGGITAL, plot=True)
# focusset.find_best_focus(4597.787801708333, 1536.6772678750003, 0.25, SAGGITAL, plot=True)
