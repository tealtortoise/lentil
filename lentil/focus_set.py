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
ch.setLevel(logging.INFO)
log.addHandler(ch)

SFRFILENAME = 'edge_sfr_values.txt'


class FocusSet:
    """
    A range of fields with stepped focus, in order
    """

    def __init__(self, rootpath, rescan=False, include_all=False):
        self.fields = []

        filenames = []

        try:
            with open(os.path.join(rootpath, "slfjsadf" if rescan or include_all else "lentil_data.csv"), 'r')\
                    as csvfile:
                print("Found lentildata")
                csvreader = csv.reader(csvfile, delimiter=',', quotechar='|')
                for row in csvreader:
                    if row[0] == "Relevant filenames":
                        stubnames = row[1:]
                pathnames = [os.path.join(rootpath, stubname) for stubname in stubnames]
                for pathname in pathnames:
                    self.fields.append(SFRField(pathname=pathname))
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
                    else:
                        continue
                    filenames.append((entrynumber, fullpathname, stubname))

                    filenames.sort()

            for entrynumber, pathname, filename in filenames:
                print("Opening file {}".format(pathname))
                field = SFRField(pathname=pathname)
                field.filenumber = entrynumber
                field.filename = filename
                self.fields.append(field)
            if not include_all:
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
                    fopt, _, _, _ = self.find_best_focus(x, y, freq=freq, axis=axis)
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




    def plot_ideal_focus_field(self, freq=0.1, detail=1.0, axis=MEDIAL, color=None,
                               plot_curvature=False, plot_type=1, show=True, ax=None, skewplane=False, alpha=0.7):
        """
        Plots peak sharpness at each point in field across all focus

        :param freq: Frequency of interest in cy/px (-1 for MTF50)
        :param detail: Alters number of points in plot (relative to 1.0)
        :param axis: SAGGITAL or MERIDIONAL or MEDIAL
        :param plot_type: 0 is 2d contour, 1 is 3d
        :param show: Displays plot if True
        :param ax: Pass existing matplotlib axis to use
        :return: matplotlib axis for reuse
        """
        x_values, y_values = self.fields[0].build_axis_points(24*detail, 16*detail)
        focus_posits = np.ndarray((len(y_values), len(x_values)))
        sharps = focus_posits.copy()
        if plot_curvature:
            colours = focus_posits.copy()
            z_values_low = focus_posits.copy()
            z_values_high = focus_posits.copy()

        tot_locs = len(x_values) * len(y_values)
        locs = 1

        for x_idx, x in enumerate(x_values):
            for y_idx, y in enumerate(y_values):
                print("Finding best focus for location {} / {}".format(locs, tot_locs))
                locs += 1
                peak, sharp, low, high = self.find_best_focus(x, y, freq, axis)

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
                skewplane = interpolate.SmoothBivariateSpline(x_int.flatten(), y_int.flatten(), focus_posits.flatten(), kx=1, ky=1, s=float("inf"))

            for x_idx, x in enumerate(x_values):
                for y_idx, y in enumerate(y_values):
                    sheet = skewplane(x, y)
                    focus_posits[y_idx, x_idx] -= sheet
                    z_values_low[y_idx, x_idx] -= sheet
                    z_values_high[y_idx, x_idx] -= sheet

        low_perf = diffraction_mtf(freq, 24)
        high_perf = diffraction_mtf(freq, 10)

        # print(low_perf, high_perf)
        # print(np.amax(sharps))
        # exit()

        if plot_type == 0:
            fig, ax = plt.subplots()
            if plot_curvature:
                contours = np.arange(int(np.amin(focus_posits)*2)/2.0 - 0.5, np.amax(focus_posits)+0.5, 0.5)
                z_values = focus_posits
            else:
                contours = np.arange(int(low_perf*50)/50.0 -0.02, high_perf+0.02, 0.02)
                z_values = sharps
            colors = []
            linspaced = np.linspace(0.0, 1.0, len(contours))
            for lin, line in zip(linspaced, contours):
                colors.append(plt.cm.viridis(lin))
                # colors.append(colorsys.hls_to_rgb(lin * 0.8, 0.4, 1.0))

            ax.set_ylim(np.amax(y_values), np.amin(y_values))
            CS = ax.contourf(x_values, y_values, z_values, contours, colors=colors)
            CS2 = ax.contour(x_values, y_values, z_values, contours, colors=('black',))
            plt.clabel(CS2, inline=1, fontsize=10)
            plt.title('Simplest default with labels')
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

            NUM_SHEETS = 0
            if plot_curvature:
                if NUM_SHEETS > 0:
                    sheets = []
                    for n in range(NUM_SHEETS):
                        sheets.append(focus_posits * (n/NUM_SHEETS) + z_values_low * (1 - (n/NUM_SHEETS)))

                    for n in range(NUM_SHEETS+1):
                        ratio = n / max(1, NUM_SHEETS)  # Avoid divide by zero
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

                new_color = [color[0], color[1], color[2], 0.5 - (sheet_num ** 2) ** 0.5 * 0.4]
                new_facecolor = [color[0], color[1], color[2], 0.3 - (sheet_num ** 2) ** 0.5 * 0.24]

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
            x = 3000 + height * 3000
            y = 2000 + height * 2000
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

    def find_best_focus(self, x, y, freq, axis=MEDIAL, plot=False, strict=False):
        """
        Get peak SFR at specified location and frequency vs focus, optionally plot.
        This does not call (.show()) on the plot.

        :param x: position in field
        :param freq: frequency of interest (-1 for MTF50)
        :param axis: MERIDIONAL or SAGGITAL or MEDIAL
        :param plot: Draws plot if True
        :return: Tuple (focus position of peak, height of peak)
        """

        if axis == MEDIAL:
            sag_x, sag_y, _, _ = self.find_best_focus(x, y, freq, SAGITTAL)
            mer_x, mer_y, _, _ = self.find_best_focus(x, y, freq, MERIDIONAL)
            mid = (sag_x + mer_x) * 0.5
            prop = mid - math.floor(mid)
            sag_midpeak = self.fields[int(mid)].interpolate_value(x, y, freq, SAGITTAL) * (1-prop) + \
                            self.fields[int(mid + 1)].interpolate_value(x, y, freq, SAGITTAL) * prop
            mer_midpeak = self.fields[int(mid)].interpolate_value(x, y, freq, MERIDIONAL) * (1-prop) + \
                            self.fields[int(mid + 1)].interpolate_value(x, y, freq, MERIDIONAL) * prop
            midpeak = ((sag_midpeak + mer_midpeak)*0.5)

            log.debug(str((mid, midpeak)))
            return mid, midpeak, midpeak, midpeak

        # Get SFR from each field
        y_values = []
        for field in self.fields:
            y_values.append(field.interpolate_value(x, y, freq, axis))


        x_values = np.arange(0, len(y_values), 1)  # Arbitrary focus units
        y_values = np.array(y_values)
        # print(x_values)
        # print(y_values)
        # exit()
        if plot:
            plt.plot(x_values, y_values, '.', color='black')
            # plt.show()
        # Use quadratic spline as fitting curve

        def fit_poly(polyx, polyy, w):
            # fn = interpolate.UnivariateSpline(x_values, y_values, k=2, w=w)
            poly = np.polyfit(polyx, polyy, 2, w=w)
            peak_x = np.roots(np.polyder(poly, 1))[0]
            peak_y = np.polyval(poly, peak_x)
            return peak_x, peak_y, poly

        weighting_power = max(0, 4 - np.exp(-len(x_values) / 20 + 1.4))
        # log.debug("Weighting power {}".format(weighting_power))
        # Plot weighting curve
        # x = np.arange(1, 50)
        # y = 4 - np.exp(-x / 20 + 1.4)
        # plt.plot(x, y)
        # plt.show();exit()

        high_values = (y_values > (np.amax(y_values) * 0.82))
        n_high_values = high_values.sum()
        log.debug("Dataset has {} high y_values".format(n_high_values))
        # exit()

        plot_x = np.linspace(0, max(x_values), 100)
        plot_x_wide = np.linspace(-5, max(x_values) + 5, 100)

        # 1st stage
        weights = (y_values / np.max(y_values)) ** (weighting_power + 2)  # Favour values closer to maximum
        p_peak_x, p_peak_y, poly_draft = fit_poly(x_values, y_values, weights)
        log.debug("1st stage peak is {:.3f} at {:.2f}".format(p_peak_y, p_peak_x))
        if plot:
            fitted_curve_plot_y_1 = np.clip(np.polyval(poly_draft, plot_x), 0.0, float("inf"))
            plt.plot(plot_x, fitted_curve_plot_y_1, color='red')

        # 2nd stage, use only fields close to peak
        closest_field_to_peak = int(np.argmax(y_values))

        if False and n_high_values >= 5:
            trimmed_x_values = x_values[high_values]
            trimmed_y_values = y_values[high_values]
        else:
            trim_low = max(0, closest_field_to_peak - 2)
            trim_high = min(len(y_values), closest_field_to_peak + 3)
            trimmed_x_values = x_values[trim_low: trim_high]
            trimmed_y_values = y_values[trim_low: trim_high]

        if len(trimmed_y_values) < 3 or not 0 < closest_field_to_peak < (len(y_values) - 1):
            if not strict:
                # log.info(x, y)
                # plt.plot(x_values, y_values)
                # plt.plot(trimmed_x_values, trimmed_y_values, color='yellow')
                # plt.show()
                log.warning("Focus peak may not be in range, output clipped")
                return closest_field_to_peak, y_values[closest_field_to_peak],\
                       y_values[closest_field_to_peak], y_values[closest_field_to_peak]
            raise Exception("Not enough data points around peak to analyse")

        trimmed_weights = (trimmed_y_values / np.max(trimmed_y_values)) ** 5  # Favour values closer to maximum

        p_peak_x, p_peak_y, poly = fit_poly(trimmed_x_values, trimmed_y_values, trimmed_weights)

        poly_acceptable = poly.copy()
        if freq < 0:
            derate = 0.75
        else:
            derate = diffraction_mtf(freq)
        poly_acceptable[2] -= p_peak_y * derate
        acceptable_focus_roots = np.roots(poly_acceptable)

        log.debug("2nd stage peak is {:.3f} at {:.2f}".format(p_peak_y, p_peak_x))

        if strict and not 0.0 < p_peak_x < x_values[-1]:
            raise Exception("Focus peak appears to be out of range, strict=True")
        if not -0.5 < p_peak_x < (x_values[-1] + 0.5):
            log.debug(x, y)
            plt.plot(x_values, y_values)
            plt.plot(trimmed_x_values, trimmed_y_values, color='yellow')
            plt.show()
            raise Exception("Focus peak appears to be out of range, strict=False")

        if plot:
            plt.plot(x_values, y_values, color="black")
            plt.plot(trimmed_x_values, trimmed_y_values, color='magenta')
            fitted_curve_plot_y_2 = np.clip(np.polyval(poly, plot_x), 0.0, float("inf"))
            fitted_curve_plot_y_2_acceptable = np.clip(np.polyval(poly_acceptable, plot_x), 0.0, float("inf"))
            plt.plot(plot_x, fitted_curve_plot_y_2, color='orange')
            plt.plot(plot_x, fitted_curve_plot_y_2, '--', color='orange')

        def twogauss(gaussx, a, b, c, peaky):
            const = 0
            a1 = 1 / (1 + peaky)
            a2 = peaky / (1 + peaky)
            c1 = c / 1.5
            c2 = c * 1.5
            wide = a1 * np.exp(-(gaussx - b) ** 2 / (2 * c1 ** 2))
            narrow = a2 * np.exp(-(gaussx - b) ** 2 / (2 * c2 ** 2))
            both = (wide + narrow) * a
            return both * (1-const) + const

        bounds = ((p_peak_y * 0.95, p_peak_x - 0.05, 0.1, -0.1),
                  (p_peak_y * 1.15, p_peak_x + 0.05, 50,   0.1))

        sigmas = (np.max(y_values) / y_values) ** weighting_power

        # plt.plot(x_values, sigmas)
        log.debug(str((x, y)))
        fitted_params, _ = optimize.curve_fit(twogauss, x_values, y_values, bounds=bounds, sigma=sigmas,
                                              p0=(p_peak_y, p_peak_x, 1.0, 0.1))
        log.debug("3rd stage peak is {:.3f} at {:.2f}".format(fitted_params[0], fitted_params[1]))
        log.debug("Gaussian sigma: {:.3f}, peaky {:.3f}".format(*fitted_params[2:]))

        if plot:
            fitted_curve_plot_y_3 = twogauss(plot_x, *fitted_params)
            plt.plot(plot_x, fitted_curve_plot_y_3, color='green')
        g_peak_x = fitted_params[1]
        g_peak_y = fitted_params[0]

        if n_high_values < 3:
            # 3rd Stage, fit gaussians
            # Use sum of two co-incident gaussians as fitting curve

            log.debug("Only {} high values, using guassian fit".format(n_high_values))
            final_peak_x = (p_peak_x + g_peak_x) / 2.0
            final_peak_y = (p_peak_y + g_peak_y) / 2.0
        else:
            final_peak_x = p_peak_x
            final_peak_y = p_peak_y
        return final_peak_x, final_peak_y, acceptable_focus_roots[0], acceptable_focus_roots[1]


    def plot_field_curvature_strip(self, freq, show=True):

        sag = []
        sagl = []
        sagh = []
        mer = []
        merl = []
        merh = []
        x_rng = range(100, 5900, 200)
        for n in x_rng:
            x = n
            y = 4000- n*2/3
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

    def plot_best_sfr_vs_freq_at_point(self, x, y, x_values=None, show=True):
        if x_values is None:
            x_values = np.linspace(0, 0.5, 30)
        y = [self.find_best_focus(x, y, f)[1] for f in x_values]
        plt.plot(x_values, y)
        if show:
            plt.show()
