import colorsys
import csv
import math
import os
import logging
from logging import getLogger

import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from scipy import optimize, interpolate, stats

from lentil.sfr_point import SFRPoint
from lentil.sfr_field import SFRField
from lentil.constants_utils import *

log = getLogger(__name__)
log.setLevel(logging.INFO)

ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
log.addHandler(ch)

class FocusSet:
    """
    A range of fields with stepped focus, in order
    """

    def __init__(self, rootpath, rescan=False, include_all=False, use_calibration=True):
        self.fields = []
        self.lens_name = rootpath
        calibration = None
        try:
            if len(use_calibration) == 32:
                calibration = use_calibration
                use_calibration = True
        except TypeError:
            pass

        self.use_calibration = use_calibration
        filenames = []

        if use_calibration and calibration is None:
            try:
                with open("calibration.csv", 'r') as csvfile:
                    reader = csv.reader(csvfile, delimiter=',', quotechar='|')
                    reader.__next__()
                    calibration = np.array([float(cell) for cell in reader.__next__()])
            except FileNotFoundError:
                pass

        try:
            # Attempt to open lentil_data
            with open(os.path.join(rootpath, "slfjsadf" if rescan or include_all else "lentil_data.csv"), 'r')\
                    as csvfile:
                print("Found lentildata")
                csvreader = csv.reader(csvfile, delimiter=',', quotechar='|')
                for row in csvreader:
                    if row[0] == "Relevant filenames":
                        stubnames = row[1:]
                    elif row[0] == "lens_name":
                        self.lens_name = row[1]
                pathnames = [os.path.join(rootpath, stubname) for stubname in stubnames]
                for pathname in pathnames:
                    self.fields.append(SFRField(pathname=pathname, calibration=calibration))
        except FileNotFoundError:
            print("Did not find lentildata, finding files...")
            with os.scandir(rootpath) as it:
                for entry in it:
                    print(entry.path)
                    try:
                        entrynumber = int("".join([s for s in entry.name if s.isdigit()]))
                    except ValueError:
                        continue

                    if entry.is_dir():
                        fullpathname = os.path.join(rootpath, entry.path, SFRFILENAME)
                        print(fullpathname)
                        sfr_file_exists = os.path.isfile(fullpathname)
                        if not sfr_file_exists:
                            continue
                        stubname = os.path.join(entry.name, SFRFILENAME)
                    elif entry.is_file():
                        fullpathname = entry.path
                        stubname = entry.name
                    else:
                        continue
                    filenames.append((entrynumber, fullpathname, stubname))

                    filenames.sort()

            for entrynumber, pathname, filename in filenames:
                print("Opening file {}".format(pathname))
                field = SFRField(pathname=pathname, calibration=calibration)
                field.filenumber = entrynumber
                field.filename = filename
                self.fields.append(field)
            if not include_all:
                self.remove_duplicated_fields()
                self.find_relevant_fields(writepath=rootpath, freq=AUC)

    def find_relevant_fields(self, freq=AUC, detail=1, writepath=None):
        min_ = float("inf")
        max_ = float("-inf")
        x_values, y_values = self.fields[0].build_axis_points(24 * detail, 16 * detail)
        totalpoints = len(x_values) * len(y_values) * 2
        done = 0
        for axis in MERIDIONAL, SAGITTAL:
            for x in x_values:
                for y in y_values:
                    print("Finding relevant field for point {} of {}".format(done, totalpoints))
                    done += 1
                    fopt, _, _, _, _, _ = self.find_best_focus(x, y, freq=freq, axis=axis)
                    if fopt > max_:
                        max_ = fopt
                    if fopt < min_:
                        min_ = fopt
        print("Searched from {} fields".format(len(self.fields)))
        print("Found {} to {} contained all peaks".format(min_, max_))
        minmargin = int(max(0, min_ - 2.0))
        maxmargin = int(min(len(self.fields)-1, max_+ 2.0)+1.0)
        print("Keeping {} to {}".format(minmargin, maxmargin))
        filenumbers = []
        filenames = []
        for field in self.fields[minmargin:maxmargin+1]:
            filenumbers.append(field.filenumber)
            filenames.append(field.filename)
        if writepath:
            with open(os.path.join(writepath, "lentil_data.csv"), 'w') as datafile:
                csvwriter = csv.writer(datafile, delimiter=',', quotechar='|',)
                csvwriter.writerow(["Relevant filenames"]+filenames)
                print("Data saved to lentil_data.csv")

    def plot_ideal_focus_field(self, freq=AUC, detail=1.0, axis=MEDIAL, plot_curvature=False,
                               plot_type=0, show=True, ax=None, skewplane=False, alpha=0.7, title=None):
        """
        Plots peak sharpness / curvature at each point in field across all focus

        :param freq: Frequency of interest in cy/px (-1 for MTF50)
        :param detail: Alters number of points in plot (relative to 1.0)
        :param axis: SAGGITAL or MERIDIONAL or MEDIAL
        :param plot_curvature: Show curvature if 1, else show mtf/sfr
        :param plot_type: CONTOUR2D or PROJECTION3D
        :param show: Displays plot if True
        :param ax: Pass existing matplotlib axis to use
        :param skewplane: True to build and use deskew plane, or pass existing plane
        :return: matplotlib axis for reuse, skewplane for reuse
        """

        num_sheets = 0  # Number of acceptable focus sheets each side of peak focus
        if title is None:
            title = self.lens_name

        gridit, focus_posits, x_values, y_values = self._get_grids(detail)
        sharps = focus_posits.copy()
        if plot_curvature:
            colours = focus_posits.copy()
            z_values_low = focus_posits.copy()
            z_values_high = focus_posits.copy()

        tot_locs = len(focus_posits)
        locs = 1

        for x_idx, y_idx, x, y in gridit:
            print("Finding best focus for location {} / {}".format(locs, tot_locs))
            locs += 1
            peak, sharp, low, high, _, _ = self.find_best_focus(x, y, freq, axis)

            sharps[y_idx, x_idx] = sharp
            if plot_curvature:
                focus_posits[y_idx, x_idx] = peak
                z_values_low[y_idx, x_idx] = low
                z_values_high[y_idx, x_idx] = high

        if plot_curvature and skewplane:
            if "__call__" not in dir(skewplane):
                x_int, y_int = np.meshgrid(x_values, y_values)
                print(x_values)
                print(x_int)
                print(y_int)
                print(focus_posits.flatten())
                skewplane = interpolate.SmoothBivariateSpline(x_int.flatten(), y_int.flatten(),
                                                              focus_posits.flatten(), kx=1, ky=1, s=float("inf"))

            for x_idx, x in enumerate(x_values):
                for y_idx, y in enumerate(y_values):
                    sheet = skewplane(x, y)
                    focus_posits[y_idx, x_idx] -= sheet
                    z_values_low[y_idx, x_idx] -= sheet
                    z_values_high[y_idx, x_idx] -= sheet

        if freq == ACUTANCE:
            low_perf = 0.4
            high_perf = 1.0
        else:
            low_perf = diffraction_mtf(freq, LOW_BENCHMARK_FSTOP)
            high_perf = diffraction_mtf(freq, HIGH_BENCHBARK_FSTOP)

        if plot_type == CONTOUR2D:
            fig, ax = plt.subplots()
            if plot_curvature:
                contours = np.arange(int(np.amin(focus_posits)*2)/2.0 - 0.5, np.amax(focus_posits)+0.5, 0.5)
                z_values = focus_posits
            else:
                inc = np.clip((high_perf - low_perf) / 20, 0.002, 0.05)
                # inc = 0.01

                contours = np.arange(int(low_perf/inc)*inc - inc, high_perf + inc, inc)
                # contours = np.arange(0.0, 1.0, 0.005)
                z_values = sharps
            colors = []
            linspaced = np.linspace(0.0, 1.0, len(contours))
            for lin, line in zip(linspaced, contours):
                colors.append(plt.cm.jet(1.0 - lin))
                # colors.append(colorsys.hls_to_rgb(lin * 0.8, 0.4, 1.0))
            print(z_values)
            print(contours)
            print(colors)

            ax.set_ylim(np.amax(y_values), np.amin(y_values))
            CS = ax.contourf(x_values, y_values, z_values, contours, colors=colors, extend='both')
            CS2 = ax.contour(x_values, y_values, z_values, contours, colors=('black',))
            plt.xlabel("Image position x")
            plt.ylabel("Image position y")
            plt.clabel(CS2, inline=1, fontsize=10)
            plt.title(title)
        else:
            if ax is None:
                fig = plt.figure()
            passed_ax = ax

            if ax is None:
                ax = fig.add_subplot(111, projection='3d')
                if not plot_curvature:
                    ax.set_zlim(0.0, 1.0)
            ax.set_ylim(np.amax(y_values), np.amin(y_values))
            # ax.zaxis.set_major_locator(LinearLocator(10))
            # ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

            x, y = np.meshgrid(x_values, y_values)
            print(x.flatten().shape)
            print(y.flatten().shape)
            print(focus_posits.shape)

            if plot_curvature:
                if num_sheets > 0:
                    sheets = []
                    for n in range(num_sheets):
                        sheets.append(focus_posits * (n/num_sheets) + z_values_low * (1 - (n/num_sheets)))

                    for n in range(num_sheets+1):
                        ratio = n / max(1, num_sheets)  # Avoid divide by zero
                        sheets.append(z_values_high * (ratio) + focus_posits * (1 - (ratio)))

                    sheet_nums = np.linspace(-1, 1, len(sheets))
                else:
                    sheets = [focus_posits]
                    sheet_nums = [0]
            else:
                sheets = [sharps]
                sheet_nums = [0]
            for sheet_num, sheet in zip(sheet_nums, sheets):

                cmap = plt.cm.jet  # Base colormap
                my_cmap = cmap(np.arange(cmap.N))  # Read colormap colours
                my_cmap[:, -1] = 0.52 - (sheet_num ** 2) ** 0.5 * 0.5  # Set colormap alpha
                # print(my_cmap[1,:].shape);exit()
                new_cmap = np.ndarray((256, 4))

                #new_color = [color[0], color[1], color[2], 0.5 - (sheet_num ** 2) ** 0.5 * 0.4]
                #new_facecolor = [color[0], color[1], color[2], 0.3 - (sheet_num ** 2) ** 0.5 * 0.24]

                # print(low_perf, high_perf)
                new_facecolor = plt.cm.jet(np.clip(1.0 - ((sharps - low_perf) / (high_perf - low_perf)), 0.0, 1.0))  # linear gradient along the t-axis
                new_edgecolor = 'b'
                new_facecolor[:,:,3] = alpha
                # new_color = new_facecolor = np.repeat(col1[np.newaxis, :, :], focus_posits.shape[0], axis=0)  # expand over the theta-    axis
                # print(col1);print(col1.shape);exit()
                # print(focus_posits.shape)
                # print(new_facecolor)
                # print(new_facecolor.shape);exit()

                for a in range(256):
                    mod = 0.5 - math.cos(a / 256 * math.pi) * 0.5
                    new_cmap[a, :] = my_cmap[int(mod * 256), :]

                mycmap = ListedColormap(new_cmap)
                if plot_curvature:
                    norm = matplotlib.colors.Normalize(vmin=low_perf, vmax=high_perf)
                else:
                    norm = matplotlib.colors.Normalize(vmin=0, vmax=1)

                surf = ax.plot_surface(x, y, sheet, facecolors=new_facecolor, norm=norm, edgecolors=new_edgecolor,
                                       rstride=1, cstride=1, linewidth=1, antialiased=True)
                ax.set_xlabel("Image position x")
                ax.set_ylabel("Image position y")
                ax.set_title(title)
            if passed_ax is None:
                pass# fig.colorbar(surf, shrink=0.5, aspect=5)
        if show:
            plt.show()
        return ax, skewplane

    def plot_field_curvature_strip_contour(self, freq=0.1, axis=MERIDIONAL):
        heights = np.linspace(-1, 1, 41)
        plot_focuses = np.linspace(0, len(self.fields), 40)
        field_focus_locs = np.arange(0, len(self.fields))
        sfrs = np.ndarray((len(plot_focuses), len(heights)))
        for hn, height in enumerate(heights):
            x = (IMAGE_WIDTH / 2) + height * (IMAGE_WIDTH / 2)
            y = (IMAGE_HEIGHT / 2) + height * (IMAGE_HEIGHT / 2)
            height_sfr = [field.interpolate_value(x, y, freq, axis=axis) for field in self.fields]
            height_sfr_fn = interpolate.InterpolatedUnivariateSpline(field_focus_locs, height_sfr, k=1)
            height_sfr_interpolated = height_sfr_fn(plot_focuses)
            sfrs[:,hn] = height_sfr_interpolated

        low_perf = 0.0
        high_perf = diffraction_mtf(min(1.0, freq * 1), 11.0)

        fig, ax = plt.subplots()

        contours = np.arange(int(low_perf * 20) / 50.0 - 0.02, high_perf + 0.02, 0.02)
        colors = []
        linspaced = np.linspace(0.0, 1.0, len(contours))
        for lin, line in zip(linspaced, contours):
            colors.append(plt.cm.jet(lin))
            # colors.append(colorsys.hls_to_rgb(lin * 0.8, 0.4, 1.0))

        ax.set_ylim(np.amin(plot_focuses), np.amax(plot_focuses))
        CS = ax.contourf(heights, plot_focuses, sfrs, contours, colors=colors)
        CS2 = ax.contour(heights, plot_focuses, sfrs, contours, colors=('black',), linewidths=0.8)
        plt.clabel(CS2, inline=1, fontsize=10)
        plt.title('Simplest default with labels')
        plt.show()

    def find_best_focus(self, x, y, freq, axis=MEDIAL, plot=False, show=False, strict=False):
        """
        Get peak SFR at specified location and frequency vs focus, optionally plot.

        :param x: x loc
        :param y: y loc
        :param freq: spacial freq (or MTF50 or AUC)
        :param axis: MERIDIONAL, SAGITTAL or MEDIAL
        :param plot: plot to pyplot if True
        :param strict: do not raise exception if peak cannot be determined definitively
        :return: peak_focus_loc, peak_sfr, low_bound_focus, high_bound_focus, spline interpolation fn, curve_fn
        """
        # Go recursive if both planes needed
        if axis == MEDIAL:
            sag_x, sag_y, _, _, sag_interp, _ = self.find_best_focus(x, y, freq, SAGITTAL)
            mer_x, mer_y, _, _, mer_interp, _ = self.find_best_focus(x, y, freq, MERIDIONAL)
            # return 3, 0.5, 0, 0, None, None

            med_x = (sag_x + mer_x) * 0.5
            if math.isnan(med_x):
                return float("NaN"), float("NaN"), float("NaN"), float("NaN"), None, None
            # prop = med_x - math.floor(med_x)
            # sag_midpeak = self.fields[int(med_x)].interpolate_value(x, y, freq, SAGITTAL) * (1 - prop) + \
            #               self.fields[int(med_x + 1)].interpolate_value(x, y, freq, SAGITTAL) * prop
            # mer_midpeak = self.fields[int(med_x)].interpolate_value(x, y, freq, MERIDIONAL) * (1 - prop) + \
            #               self.fields[int(med_x + 1)].interpolate_value(x, y, freq, MERIDIONAL) * prop
            # med_y = ((sag_midpeak + mer_midpeak) * 0.5)

            def interp_merge(in_x):
                return (sag_interp(in_x) + mer_interp(in_x)) * 0.5

            med_y = interp_merge(med_x)

            log.debug(str((med_x, med_y)))
            return med_x, med_y, med_y, med_y, interp_merge, None

        # Get SFR from each field
        y_values = []
        for field in self.fields:
            y_values.append(field.interpolate_value(x, y, freq, axis))
        y_values = np.array(y_values)
        x_values = np.arange(0, len(y_values), 1)  # Arbitrary focus units
        # return 3, 0.5, 0,0,None, None


        # Define gaussian curve function
        def fastgauss(gaussx, a,b,c):
            return a * np.exp(-(gaussx - b) ** 2 / (2 * c ** 2))

        def twogauss(gaussx, a, b, c, peaky):
            peaky = peaky * np.clip((c - 0.7) / 2.0, 0.0, 1.0)  # No peaky at low sigma
            a1 = 1 / (1 + peaky)
            a2 = peaky / (1 + peaky)
            c1 = c * 1.8
            c2 = c / 1.4
            wide = a1 * np.exp(-(gaussx - b) ** 2 / (2 * c1 ** 2))
            narrow = a2 * np.exp(-(gaussx - b) ** 2 / (2 * c2 ** 2))
            both = (wide + narrow) * a
            return both

        # Initial fit guesses
        highest_data_x = np.argmax(y_values)
        highest_data_y = y_values[highest_data_x]
        highest_within_tolerance = y_values > highest_data_y * 0.95
        filtered_x_values = (x_values * highest_within_tolerance)[highest_within_tolerance]
        mean_peak_x = filtered_x_values.mean()

        # Define optimisation bounds
        bounds = ((highest_data_y * 0.95, mean_peak_x - 0.9, 0.8, -0.3),
                  (highest_data_y * 1.15, mean_peak_x + 0.9, 50.0, 1.3))
        bounds = ((highest_data_y * 0.98, mean_peak_x - 0.9, 0.7,),
                  (highest_data_y * 1.15, mean_peak_x + 0.9, 50.0,))

        widen = 1.0
        totaltarget = 1.0
        for round in range(1):  # Disable
            offsets = np.arange(len(y_values)) - mean_peak_x  # x-index vs peak estimate
            weights_a = np.clip(1.2 - np.abs(offsets) / 11 / widen, 0.1, 1.0) ** 2  # Weight small offsets higher
            norm_y_values = y_values / y_values.max()  # Normalise to 1.0
            weights_b = np.clip(norm_y_values - 0.4, 0.0001, 1.0)  # Weight higher points higher, ignore below 0.4
            weights = weights_a * weights_b  # Merge
            weights = weights / weights.max()  # Normalise
            total = (weights * y_values).sum()
            # print(weights_a)
            # print(weights_b)
            # print(weights)
            log.debug("Widen {} Total {:.3f}".format(widen, total))
            if total < totaltarget:
                widen *= totaltarget / total
            else:
                break
        log.debug("Fit weightings {}".format(weights))
        sigmas = 1. / weights

        fitted_params, _ = optimize.curve_fit(fastgauss, x_values, y_values,
                                              bounds=bounds, sigma=sigmas, ftol=0.05, xtol=0.05,
                                              p0=(highest_data_y, mean_peak_x, 2.0,))

        log.debug("Gaussian fit peak is {:.3f} at {:.2f}".format(fitted_params[0], fitted_params[1]))
        log.debug("Gaussian sigma: {:.3f}".format(fitted_params[2]))
        # log.debug("Gaussian peaky: {:.3f}".format(fitted_params[3]))

        gaussfit_peak_x = fitted_params[1]
        gaussfit_peak_y = fitted_params[0]

        def curvefn(xvals):
            return twogauss(xvals, *fitted_params)

        fit_y = curvefn(x_values)
        errorweights = np.clip((y_values - y_values.max() * 0.8), 0.000001, 1.0)**1
        mean_abs_error = np.average(np.abs(fit_y - y_values), weights=errorweights)
        mean_abs_error_rel = mean_abs_error / highest_data_y

        log.debug("RMS fit error (normalised 1.0): {:.3f}".format(mean_abs_error_rel))

        if mean_abs_error_rel > 0.12:
            errorstr = "Very high fit error: {:.3f}".format(mean_abs_error_rel)
            log.warning(errorstr)

            # plot = True
            # show = True
            # return float("NaN"), float("NaN"), float("NaN"), float("NaN"), None, None
        elif mean_abs_error_rel > 0.06:
            errorstr = "High fit error: {:.3f}".format(mean_abs_error_rel)
            log.warning(errorstr)
            # plot = True
            # show = True
            if strict:
                errorstr = "Strict mode, fit aborted".format(mean_abs_error_rel)
                log.warning(errorstr)
                return float("NaN"), float("NaN"), float("NaN"), float("NaN"), None, None

        interp_fn = interpolate.InterpolatedUnivariateSpline(x_values, y_values, k=2)

        if plot:
            # Plot original data
            plt.plot(x_values, y_values, '.', color='black')
            plt.plot(x_values, y_values, '-', color='black')

            # Plot fit curve
            x_plot = np.linspace(0, x_values.max(), 100)
            y_gaussplot = curvefn(x_plot)
            plt.plot(x_plot, y_gaussplot, color='green')
            plt.plot(x_values, errorweights / errorweights.max() * y_values.max(), '--', color='gray')

            # Plot interpolation curve
            y_interpplot = interp_fn(x_plot)
            plt.plot(x_plot, y_interpplot, color='orange')

            # Plot weights
            plt.plot(x_values, weights * gaussfit_peak_y, '--', color='magenta')
            if show:
                plt.show()

        return gaussfit_peak_x, gaussfit_peak_y, 0, 0, interp_fn, curvefn

    def plot_field_curvature_strip(self, freq, show=True):
        sag = []
        sagl = []
        sagh = []
        mer = []
        merl = []
        merh = []
        x_rng = range(100, IMAGE_WIDTH, 200)
        for n in x_rng:
            x = n
            y = IMAGE_HEIGHT - n * IMAGE_HEIGHT / IMAGE_WIDTH
            focuspos, sharpness, l, h = self.find_best_focus(x, y, freq, SAGITTAL)
            sag.append(focuspos)
            sagl.append(l)
            sagh.append(h)
            focuspos, sharpness, l, h = self.find_best_focus(x, y, freq, MERIDIONAL)
            mer.append(focuspos)
            merl.append(l)
            merh.append(h)

        plt.plot(x_rng, sag, color='green')
        plt.plot(x_rng, sagl, '--', color='green')
        plt.plot(x_rng, sagh, '--', color='green')
        plt.plot(x_rng, mer, color='blue')
        plt.plot(x_rng, merl, '--', color='blue')
        plt.plot(x_rng, merh, '--', color='blue')
        if show:
            plt.show()

    def remove_duplicated_fields(self, plot=False, train=[]):
        fields = self.fields[:]
        prev = fields[0].points
        new_fields = [fields[0]]
        for n, field in enumerate(fields[1:]):
            tuplist=[]
            dup = (n+1) in train
            for pointa, pointb in zip(prev, field.points):
                tup = pointa.is_match_to(pointb)
                tuplist.append([dup] + [n + 1] + list(tup))
            prev = field.points
            tuplist = np.array(tuplist)
            duplikely = np.percentile(tuplist[:,6], 80) < 0.15
            if not duplikely:
                new_fields.append(field)
            else:
                log.info("Field {} removed as duplicate".format(n+1))
        log.info("Removed {} out of {} field as duplicates".format(len(self.fields) - len(new_fields), len(self.fields)))
        self.fields = new_fields
        return


        tuplist = np.array(tuplist)
        log.info(np.percentile(tuplist[tuplist[:,0] == 0], [50], axis=0))
        log.info()
        log.info(np.percentile(tuplist[tuplist[:,0] == 1], [50], axis=0))
        exit()
        log.info(tuplist[tuplist[:,1] == 1][:].mean(axis=0))
        exit()
        dup=0
        print("{:.0f} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f} ".format(dup, np.array(x_dif).mean(),
                                                                  np.array(y_dif).mean(),
                                                                  np.array(angle_dif).mean(),
                                                                  np.array(radang_dif).mean(),
                                                                  np.array(sfrsum).mean()))
        dup=1
        x_dif, y_dif, angle_dif, radang_dif, sfrsum = zip(*duplist)
        print("{:.0f} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f} ".format(dup, np.array(x_dif).mean(),
                                                                  np.array(y_dif).mean(),
                                                                  np.array(angle_dif).mean(),
                                                                  np.array(radang_dif).mean(),
                                                                  np.array(sfrsum).mean()))

    def plot_best_sfr_vs_freq_at_point(self, x, y, x_values=None, secondline_fn=None, show=True):
        if x_values is None:
            x_values = np.linspace(0, 0.5, 30)
        y = [self.find_best_focus(x, y, f)[1] for f in x_values]
        plt.plot(x_values, y)
        if secondline_fn:
            plt.plot(x_values, secondline_fn(x_values))
        if show:
            plt.show()

    def get_peak_sfr(self, x=(IMAGE_WIDTH / 2), y=(IMAGE_HEIGHT / 2), freq=AUC, axis=MEDIAL, plot=False, show=False):
        """
        Get entire SFR at specified location at best focus determined by 'focus' passed

        :param x:
        :param y:
        :param freq:
        :param axis:
        :param plot:
        :param show:
        :return:
        """
        focuspos, _, _, _, _, _ = self.find_best_focus((IMAGE_WIDTH / 2), (IMAGE_HEIGHT / 2), AUC, axis=axis)

        # Build sfr at that point
        data_sfr = []
        for f in RAW_SFR_FREQUENCIES[:32]:
            _, _, _, _, interpfn, _ = self.find_best_focus((IMAGE_WIDTH / 2), (IMAGE_HEIGHT / 2), f, axis=axis)
            data_sfr.append(interpfn(focuspos))

        data_sfr = np.array(data_sfr)
        print("Acutance {:.3f}".format(calc_acutance(data_sfr)))
        if plot:
            plt.plot(RAW_SFR_FREQUENCIES[:32], data_sfr)
            plt.ylim(0.0, data_sfr.max())
            if show:
                plt.show()
        return SFRPoint(rawdata=data_sfr)

    def build_calibration(self, fstop, plot=True, writetofile=False):
        """
        Assume diffraction limited lens to build calibration data

        :param fstop: Taking f-stop
        :param plot: Plot if True
        :param writetofile: Write to calibration.csv file
        :return: Numpy array of correction data
        """
        if self.use_calibration:
            if writetofile:
                raise AttributeError("Focusset must be loaded without existing calibration")
            else:
                log.warning("Existing calibration loaded (will compare calibrations)")
        # Get best AUC focus postion

        f_range = RAW_SFR_FREQUENCIES[:32]
        data_sfr = self.get_peak_sfr(freq=AUC).raw_sfr_data[:32]

        diffraction_sfr = diffraction_mtf(f_range, fstop)  # Ideal sfr

        correction = diffraction_sfr / data_sfr

        if writetofile:
            with open("calibration.csv", 'w') as csvfile:
                csvwriter = csv.writer(csvfile, delimiter=',', quotechar='|')
                csvwriter.writerow(list(f_range))
                csvwriter.writerow(list(correction))

        if plot:
            plt.ylim(0, max(correction))
            plt.plot(f_range, data_sfr)
            plt.plot(f_range, diffraction_sfr, '--')
            plt.plot(f_range, correction)
            plt.title(self.lens_name)
            plt.show()
        return correction

    def set_calibration_sharpen(self, amount, radius, stack=False):
        for field in self.fields:
            field.set_calibration_sharpen(amount, radius, stack)

    def _get_grids(self, detail=0.3):
        x_values, y_values = self.fields[0].build_axis_points(24 * detail, 16 * detail)
        mesh = np.meshgrid(np.arange(len(x_values)), np.arange(len(y_values)))
        mesh2 = np.meshgrid(x_values, y_values)
        meshes = [grid.flatten() for grid in (mesh+mesh2)]
        return list(zip(*meshes)), np.zeros((len(y_values), len(x_values))), x_values, y_values

    def find_compromise_focus(self, freq=AUC, axis=MEDIAL, detail=1.0, plot_freq=None):
        gridit, numpyarray, x_values, y_values = self._get_grids(detail)
        n_fields = len(self.fields)
        field_locs = np.arange(0, n_fields, 0.1)  # Analyse with 0.1 field accuracy

        # Made sfr data array
        sharps = np.repeat(numpyarray[:,:,np.newaxis], len(field_locs), axis=2)

        xm, ym = np.meshgrid(x_values, y_values)
        heights = ((xm - (IMAGE_WIDTH / 2))**2 + (ym - (IMAGE_HEIGHT / 2))**2)**0.5 / ((IMAGE_WIDTH / 2)**2+(IMAGE_HEIGHT / 2)**2)**0.5
        weights = np.ndarray((len(y_values), len(x_values)))

        # Iterate over all locations
        for n_x, n_y, x, y in gridit:
            # Get sharpness data at location
            _, _, _, _, interpfn, _ = self.find_best_focus(x, y, freq, axis=axis)

            # Gather data at all sub-points
            pos_sharps = interpfn(field_locs)
            sharps[n_y, n_x] = pos_sharps

            # Build weighting
            img_height = calc_image_height(x, y)
            weight = (1.0 - img_height) ** 2
            # weight = img_height ** 2
            weight = 1.0
            weights[n_y, n_x] = weight

        # Build kernal density model to de-bias mid-field due to large number of points
        height_kde = stats.gaussian_kde(heights.flatten(), bw_method=0.8)
        height_density_weight_mods = height_kde(heights.flatten()).reshape(weights.shape)

        plot_kde_info = 0
        if plot_kde_info:
            px = np.linspace(0, 1, 30)
            plt.plot(px, height_kde(px))
            plt.show()
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(xm.flatten(), ym.flatten(), (weights / height_density_weight_mods).flatten(), c='b', marker='.')
            plt.show()

        weights = np.repeat((weights / height_density_weight_mods)[:, :, np.newaxis], len(field_locs), axis=2)
        average = np.average(sharps, axis=(0, 1), weights=weights)

        slice_ = average > (average.max() * 0.8)
        poly = np.polyfit(field_locs[slice_], average[slice_], 2)
        polyder = np.polyder(poly, 1)
        peak_focus_pos = np.roots(polyder)[0]
        print("Found peak focus at position {:.2f}".format(peak_focus_pos))

        if 1 or not 0 < peak_focus_pos < (n_fields - 1):
            _fit = np.polyval(poly, field_locs)
            plt.plot(field_locs, average)
            plt.plot(field_locs, _fit)
            plt.show()

        if plot_freq:
            sharpfield = numpyarray
            for n_x, n_y, x, y in gridit:
                # Get sharpness data at location
                _, _, _, _, interpfn, _ = self.find_best_focus(x, y, plot_freq, axis=axis)
                sharpfield[n_y, n_x] = interpfn(peak_focus_pos)
        else:
            sharpfield = sharps[:, :, int(peak_focus_pos*10 + 0.5)]

        # Move on to plotting results
        fig, ax = plt.subplots()

        low_perf = diffraction_mtf(freq, LOW_BENCHMARK_FSTOP)
        high_perf = diffraction_mtf(freq, HIGH_BENCHBARK_FSTOP)

        inc = np.clip((high_perf - low_perf) / 20, 0.002, 0.05)
        contours = np.arange(int(low_perf/inc)*inc - inc, high_perf + inc, inc)
        # contours = np.arange(0.0, 1.0, 0.005)
        z_values = sharpfield
        colors = []
        linspaced = np.linspace(0.0, 1.0, len(contours))
        for lin, line in zip(linspaced, contours):
            colors.append(plt.cm.jet(1.0 - lin))
            # colors.append(colorsys.hls_to_rgb(lin * 0.8, 0.4, 1.0))

        ax.set_ylim(np.amax(y_values), np.amin(y_values))
        CS = ax.contourf(x_values, y_values, z_values, contours, colors=colors, extend='both')
        CS2 = ax.contour(x_values, y_values, z_values, contours, colors=('black',))
        plt.xlabel("Image position x")
        plt.ylabel("Image position y")
        plt.clabel(CS2, inline=1, fontsize=10)
        plt.title("Compromise focus sharpness")
        plt.show()
        exit()

"photos@scs.co.uk"
"S 0065-3858491"