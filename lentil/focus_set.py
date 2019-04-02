import csv
import math
import os
import logging
from logging import getLogger
from operator import itemgetter
from scipy import optimize, interpolate, stats

from lentil.sfr_point import SFRPoint
from lentil.sfr_field import SFRField, NotEnoughPointsException
from lentil.plot_utils import FieldPlot, Scatter2D, COLOURS
from lentil.constants_utils import *

log = getLogger(__name__)
log.setLevel(logging.INFO)

ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
log.addHandler(ch)

STORE = []

class FocusOb:
    """
    Stores focus data, including position, sharpness (mtf/sfr), fit curves and bounds
    """
    def __init__(self, focuspos=None, sharp=None, interpfn=None, curvefn=None, lowbound=None, highbound=None):
        self.focuspos = focuspos
        self.sharp = sharp
        self.interpfn = interpfn
        self.curvefn = curvefn
        self.lowbound = lowbound
        self.highbound = highbound
        self.focus_data = None
        self.sharp_data = None
        self.x_loc = None
        self.y_loc = None

    @classmethod
    def get_midpoint(cls, a, b):
        new = cls()
        mid_x = (a.focuspos + b.focuspos) * 0.5
        new.focuspos = mid_x

        def interp_merge(in_x):
            return (a.interpfn(in_x) + b.interpfn(in_x)) * 0.5
        def curve_merge(in_x):
            return (a.curvefn(in_x) + b.curvefn(in_x)) * 0.5

        new.sharp = interp_merge(mid_x)
        new.curvefn = curve_merge
        new.interpfn = interp_merge
        new.x_loc = (a.x_loc + b.x_loc) / 2
        new.y_loc = (a.y_loc + b.y_loc) / 2
        return new


class FitError(Exception):
    def __init__(self, error, fitpos: FocusOb):
        super().__init__(error)
        self.fitpos = fitpos


class FocusSet:
    """
    A range of fields with stepped focus, in order
    """

    def __init__(self, rootpath, rescan=False, include_all=False, use_calibration=True):
        self.fields = []
        self.lens_name = rootpath
        calibration = None
        self.calibration = None
        self.base_calibration = np.ones((64,))
        try:
            if len(use_calibration) == 32:
                calibration = use_calibration
                self.base_calibration = calibration
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
                    self.base_calibration = calibration
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
                exif = EXIF(pathnames[0])
                self.exif = exif
                for pathname in pathnames:
                    try:
                        self.fields.append(SFRField(pathname=pathname, calibration=calibration, exif=exif))
                    except NotEnoughPointsException:
                        pass
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
                    elif entry.is_file() and entry.name.endswith("sfr"):
                        print("Found {}".format(entry.name))
                        fullpathname = entry.path
                        stubname = entry.name
                    else:
                        continue
                    filenames.append((entrynumber, fullpathname, stubname))

                    filenames.sort()

            if len(filenames) is 0:
                raise ValueError("No fields found!")

            exif = EXIF(filenames[0][1])
            self.exif = exif
            for entrynumber, pathname, filename in filenames:
                print("Opening file {}".format(pathname))
                try:
                    field = SFRField(pathname=pathname, calibration=calibration, exif=exif)
                except NotEnoughPointsException:
                    continue
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
                    fopt = self.find_best_focus(x, y, freq=freq, axis=axis).focuspos
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

    def plot_ideal_focus_field(self, freq=AUC, detail=1.0, axis=MEDIAL, plot_curvature=True,
                               plot_type=PROJECTION3D, show=True, ax=None, skewplane=False,
                               alpha=0.7, title=None):
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
        :param alpha: plot plane transparency
        :param title: title for plot
        :return: matplotlib axis for reuse, skewplane for reuse
        """

        num_sheets = 0  # Number of acceptable focus sheets each side of peak focus
        if title is None:
            title = self.lens_name

        gridit, focus_posits, x_values, y_values = self.get_grids(detail)
        sharps = focus_posits.copy()
        if plot_curvature:
            colours = focus_posits.copy()
            z_values_low = focus_posits.copy()
            z_values_high = focus_posits.copy()

        tot_locs = len(focus_posits.flatten())
        locs = 1

        for x_idx, y_idx, x, y in gridit:
            print("Finding best focus for location {} / {}".format(locs, tot_locs))
            locs += 1
            bestfocus = self.find_best_focus(x, y, freq, axis)

            sharps[y_idx, x_idx] = bestfocus.sharp
            if plot_curvature:
                focus_posits[y_idx, x_idx] = bestfocus.focuspos
                z_values_low[y_idx, x_idx] = bestfocus.lowbound
                z_values_high[y_idx, x_idx] = bestfocus.highbound

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

        if plot_type == CONTOUR2D:
            plot = FieldPlot()
            plot.xticks = x_values
            plot.yticks = y_values
            plot.set_diffraction_limits(freq)

            if plot_curvature:
                # contours = np.arange(int(np.amin(focus_posits)*2)/2.0 - 0.5, np.amax(focus_posits)+0.5, 0.5)
                plot.zdata = focus_posits
            else:
                plot.zdata = sharps
            plot.yreverse = True
            plot.xlabel = "Image position x"
            plot.ylabel = "Image position y"
            plot.title = title
            ax = plot.contour2d(ax, show=show)
        else:
            plot = FieldPlot()

            plot.set_diffraction_limits(freq, graphaxis="w")

            plot.xticks = x_values
            plot.yticks = y_values
            plot.set_diffraction_limits(freq, graphaxis='w')
            plot.yreverse = True
            plot.xlabel = "Image position x"
            plot.ylabel = "Image position y"
            plot.zlabel = "Best focus"
            plot.title = title
            plot.zdata = focus_posits
            plot.wdata = sharps
            ax = plot.projection3d(ax)
        return ax, skewplane

    def plot_field_curvature_strip_contour(self, freq=DEFAULT_FREQ, axis=MERIDIONAL, theta=THETA_TOP_RIGHT, radius=1.0):
        maxradius = IMAGE_DIAGONAL / 2 * radius
        heights = np.linspace(-1, 1, 41)
        plot_focuses = np.linspace(0, len(self.fields), 40)
        field_focus_locs = np.arange(0, len(self.fields))
        sfrs = np.ndarray((len(plot_focuses), len(heights)))
        for hn, height in enumerate(heights):
            x = np.cos(theta) * height * maxradius + IMAGE_WIDTH/2
            y = np.sin(theta) * height * maxradius + IMAGE_HEIGHT/2
            interpfn = self.get_interpolation_fn_at_point(x, y, freq, axis=axis).interpfn
            height_sfr_interpolated = interpfn(plot_focuses)
            sfrs[:, hn] = height_sfr_interpolated

        plot = FieldPlot()
        plot.set_diffraction_limits(freq=freq)
        plot.xticks = heights
        plot.yticks = plot_focuses
        plot.zdata = sfrs
        plot.xlabel = "Image height"
        plot.ylabel = "Focus position"
        plot.title = "Edge SFR vs focus position for varying height from centre"
        plot.contour2d()

    def get_interpolation_fn_at_point(self, x, y, freq=DEFAULT_FREQ, axis=MEDIAL, limit=None, order=2):
        y_values = []
        if limit is None:
            lowlim = 0
            highlim = len(self.fields)
            fields = self.fields
        else:
            lowlim = max(0, limit[0])
            highlim = min(len(self.fields), max(limit[1], lowlim + 3))
            fields = self.fields[lowlim: highlim]

        if axis == MEDIAL:
            for field in fields:
                s = field.interpolate_value(x, y, freq, SAGITTAL)
                m = field.interpolate_value(x, y, freq, MERIDIONAL)
                y_values.append((m + s) * 0.5)
        else:
            for n, field in enumerate(fields):
                y_values.append(field.interpolate_value(x, y, freq, axis))
        y_values = np.array(y_values)
        x_values = np.arange(lowlim, highlim, 1)  # Arbitrary focus units
        interpfn = interpolate.InterpolatedUnivariateSpline(x_values, y_values, k=order)
        pos = FocusOb(interpfn=interpfn)
        pos.focus_data = x_values
        pos.sharp_data = y_values
        return pos

    def find_best_focus(self, x, y, freq=DEFAULT_FREQ, axis=MEDIAL, plot=False, show=True, strict=False):
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
            best_s = self.find_best_focus(x, y, freq, SAGITTAL, plot, show, strict)
            best_m = self.find_best_focus(x, y, freq, MERIDIONAL, plot, show, strict)
            mid = FocusOb.get_midpoint(best_s, best_m)
            if 0 and 0.6<calc_image_height(x, y)<0.7:
                print("     ", best_s.x_loc, best_s.y_loc, best_s.focuspos)
                print("     ", best_m.x_loc, best_m.y_loc, best_m.focuspos)
                print("     ", mid.x_loc, mid.y_loc, mid.focuspos)
            # STORE.append(mid.focuspos)
            # print(sum(STORE) / len(STORE))
            return mid

        pos = self.get_interpolation_fn_at_point(x, y, freq, axis)
        x_values = pos.focus_data
        y_values = pos.sharp_data
        interp_fn = pos.interpfn

        # Initial fit guesses
        highest_data_x = np.argmax(y_values)
        highest_data_y = y_values[highest_data_x]
        highest_within_tolerance = y_values > highest_data_y * 0.95
        filtered_x_values = (x_values * highest_within_tolerance)[highest_within_tolerance]
        mean_peak_x = filtered_x_values.mean()

        # Define optimisation bounds
        # bounds = ((highest_data_y * 0.95, mean_peak_x - 0.9, 0.8, -0.3),
        #           (highest_data_y * 1.15, mean_peak_x + 0.9, 50.0, 1.3))
        bounds = ((highest_data_y * 0.98, mean_peak_x - 0.9, 0.7,),
                  (highest_data_y * 1.15, mean_peak_x + 0.9, 50.0,))

        offsets = np.arange(len(y_values)) - mean_peak_x  # x-index vs peak estimate
        weights_a = np.clip(1.2 - np.abs(offsets) / 11, 0.1, 1.0) ** 2  # Weight small offsets higher
        norm_y_values = y_values / y_values.max()  # Normalise to 1.0
        weights_b = np.clip(norm_y_values - 0.4, 0.0001, 1.0)  # Weight higher points higher, ignore below 0.4
        weights = weights_a * weights_b  # Merge
        weights = weights / weights.max()  # Normalise

        log.debug("Fit weightings {}".format(weights))
        # print(weights)
        sigmas = 1. / weights

        fitted_params, _ = optimize.curve_fit(fastgauss, x_values, y_values,
                                              bounds=bounds, sigma=sigmas, ftol=0.0001, xtol=0.001,
                                              p0=(highest_data_y, mean_peak_x, 2.0,))

        log.debug("Gaussian fit peak is {:.3f} at {:.2f}".format(fitted_params[0], fitted_params[1]))
        log.debug("Gaussian sigma: {:.3f}".format(fitted_params[2]))
        # log.debug("Gaussian peaky: {:.3f}".format(fitted_params[3]))

        gaussfit_peak_x = fitted_params[1]
        gaussfit_peak_y = fitted_params[0]

        def curvefn(xvals):
            return fastgauss(xvals, *fitted_params)

        fit_y = curvefn(x_values)
        errorweights = np.clip((y_values - y_values.max() * 0.8), 0.000001, 1.0)**1
        mean_abs_error = np.average(np.abs(fit_y - y_values), weights=errorweights)
        mean_abs_error_rel = mean_abs_error / highest_data_y

        log.debug("RMS fit error (normalised 1.0): {:.3f}".format(mean_abs_error_rel))

        if mean_abs_error_rel > 0.12:
            errorstr = "Very high fit error: {:.3f}".format(mean_abs_error_rel)
            log.warning(errorstr)
            print(x, y, freq, axis)
            # plt.plot(x_values, y_values)
            # plt.show()
            pos = FocusOb(gaussfit_peak_x, gaussfit_peak_y, interp_fn, curvefn)
            raise FitError(errorstr, fitpos=pos)
        elif mean_abs_error_rel > 0.06:
            errorstr = "High fit error: {:.3f}".format(mean_abs_error_rel)
            log.warning(errorstr)
            # print(x, y, freq, axis)
            # plt.plot(x_values, y_values)
            # plt.show()
            if strict:
                errorstr = "Strict mode, fit aborted".format(mean_abs_error_rel)
                log.warning(errorstr)
                pos = FocusOb(gaussfit_peak_x, gaussfit_peak_y, interp_fn, curvefn)
                raise FitError(errorstr, fitpos=pos)

        if plot or 0 and  0.6<calc_image_height(x, y)<0.7:
            # print(x,y,freq, axis, gaussfit_peak_x)
            # Plot original data
            plt.plot(x_values, y_values, '.', marker='s', color='forestgreen', label="Original data points", zorder=11)
            plt.plot(x_values, y_values, '-', color='forestgreen', alpha=0.3, label="Original data line", zorder=-1)

            # Plot fit curve
            x_plot = np.linspace(0, x_values.max(), 100)
            y_gaussplot = curvefn(x_plot)
            plt.plot(x_plot, y_gaussplot, color='red', label="Gaussian curve fit", zorder=14)
            # plt.plot(x_values, errorweights / errorweights.max() * y_values.max(), '--', color='gray', label="Sanity checking weighting")

            # Plot interpolation curve
            y_interpplot = interp_fn(x_plot)
            # plt.plot(x_plot, y_interpplot, color='seagreen', label="Interpolated quadratic spline fit", zorder=3)

            # Plot weights
            plt.plot(x_values, weights * gaussfit_peak_y, '--', color='royalblue', label="Curve fit weighting", zorder=1)
            plt.xlabel("Field/image number (focus position)")
            plt.ylabel("Spacial frequency response")
            plt.title("SFR vs focus position")
            plt.legend()
            if show:
                plt.show()
        ob = FocusOb(gaussfit_peak_x, gaussfit_peak_y, interp_fn, curvefn)
        ob.x_loc = x
        ob.y_loc = y
        return ob

    def interpolate_value(self, x, y, focus, freq=AUC, axis=MEDIAL, posh=False):
        if int(focus) == 0:
            limit_low = 0
            limit_high = int(focus) + 3
        elif int(focus+1) >= len(self.fields):
            limit_low = int(focus) - 2
            limit_high = int(focus) + 2
        else:
            limit_low = int(focus) - 1
            limit_high = int(focus) + 2
        if posh:

            point_pos = self.find_best_focus(x, y, freq, axis=axis)
            return point_pos.curvefn(focus)

        else:
            point_pos = self.get_interpolation_fn_at_point(x, y, freq, axis=axis,
                                                           limit=(limit_low, limit_high))
            return point_pos.interpfn(focus)

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

    def plot_best_sfr_vs_freq_at_point(self, x, y, axis=MEDIAL, x_values=None, secondline_fn=None, show=True):
        if x_values is None:
            x_values = RAW_SFR_FREQUENCIES[:32]
        y = [self.find_best_focus(x, y, f, axis).sharp for f in x_values]
        plt.plot(x_values, y)
        plt.ylim(0,1)
        if secondline_fn:
            plt.plot(x_values, secondline_fn(x_values))
        if show:
            plt.show()

    def plot_sfr_vs_freq_at_point_for_each_field(self, x, y, axis=MEDIAL, waterfall=False):
        fig = plt.figure()
        if waterfall:
            ax = fig.add_subplot(111, projection='3d')
        else:
            ax = fig.add_subplot(111)
        freqs = np.concatenate((RAW_SFR_FREQUENCIES[:32:1], [AUC]))
        for nfreq, freq in enumerate(freqs):
            print("Running frequency {:.2f}...".format(freq))
            responses = []
            for fn, field in enumerate(self.fields):
                res = field.interpolate_value(x, y, freq, axis)
                if res > 0.01:
                    responses.append(res)
                else:
                    responses.append(0.01)

            if waterfall:
                if freq == AUC:
                    colour = 'black'
                else:
                    colour = 'black'
                    colour = plt.cm.brg(nfreq / len(freqs))
                plt.plot([nfreq / 65] * len(self.fields), np.arange(len(self.fields)),
                         np.log(responses) / (np.log(10) / 20.0),
                         label="Field {}".format(fn), color=colour, alpha=0.8)
            else:
                if freq == AUC:
                    colour = 'black'
                else:
                    colour = 'black'
                    colour = plt.cm.jet(nfreq / len(freqs))
                if freq == AUC:
                    label = "Mean / Area under curve"
                else:
                    label = "{:.2f} cy/px".format(freq)
                plt.plot(np.arange(len(self.fields)), np.log(responses) / (np.log(10) / 20.0),
                         label=label, color=colour, alpha=1.0 if freq==AUC else 0.9)
        if waterfall:
            ax.set_xlabel("Spacial Frequency (cy/px")
            ax.set_ylabel("Focus position")
            ax.set_zlabel("SFR (dB - log scale)")
            ax.set_zlim(-40, 0)
        else:
            ax.set_xlabel("Focus Position")
            ax.set_ylabel("SFR (dB (log scale))")
            ax.set_ylim(-40,0)
            ax.legend()

        ax.set_title("SFR vs Frequency for {}".format(self.exif.summary))
        # ax.legend()
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
        if axis == BOTH_AXES:
            best = -np.inf, "", 0,0, 0
            for axis in [SAGITTAL, MERIDIONAL]:
                print("Testing axis {}".format(axis))
                ob = self.find_sharpest_location(freq, axis)
                x = ob.x_loc
                y = ob.y_loc
                focuspos = ob.focuspos
                if ob.sharp > best[0]:
                    best = ob.sharp, axis, x, y, focuspos
            axis = best[1]
            focuspos = best[4]
            x = best[2]
            y = best[3]
            print("Found best point {:.3f} on {} axis at ({:.0f}, {:.0f}".format(best[0], axis, x, y))
        else:
            ob = self.find_sharpest_location(freq, axis)
            x = ob.x_loc
            y = ob.y_loc
            focuspos = ob.focuspos

        # Build sfr at that point
        data_sfr = []
        for f in RAW_SFR_FREQUENCIES[:]:
            data_sfr.append(self.interpolate_value(x, y, focuspos, f, axis))

        data_sfr = np.array(data_sfr)
        print("Acutance {:.3f}".format(calc_acutance(data_sfr)))
        if plot:
            plt.plot(RAW_SFR_FREQUENCIES[:], data_sfr)
            plt.ylim(0.0, data_sfr.max())
            if show:
                plt.show()
        return SFRPoint(rawdata=data_sfr)

    def find_sharpest_raw_point(self):
        best = -np.inf, None
        for field in self.fields:
            for point in field.points:
                sharp = point.get_freq(AUC)
                if sharp > best[0]:
                    best = sharp, point
        return best[1]

    def find_sharpest_raw_points_avg_sfr(self, n=5, skip=5):
        best = -np.inf, None
        all = []
        for field in self.fields:
            for point in field.points:
                sharp = point.get_freq(AUC)
                all.append((sharp, point))
        all.sort(reverse=True, key=itemgetter(0))
        best = all[skip: skip + n]
        print(best)
        sum_ = sum([tup[1].raw_sfr_data for tup in best])
        return sum_ / n

    def find_sharpest_location(self, freq=AUC, axis=MEDIAL, detail=1.5):
        gridit, numparr, x_values, y_values = self.get_grids(detail=detail)
        heights = numparr.copy()
        focusposs = numparr.copy()
        # axes = numparr.copy()
        searchradius = 0.20
        lastsearchradius = 0.0
        while searchradius < 1.0:
            for nx, ny, x, y in gridit:
                imgheight = calc_image_height(x, y)
                if lastsearchradius < imgheight < searchradius:
                    focusob = self.find_best_focus(x, y, freq, axis)
                    numparr[ny, nx] = focusob.sharp
                    # axes[ny, nx] = focusob.axis
                    focusposs[ny, nx] = focusob.focuspos
                    heights[ny, nx] = imgheight
            maxcell = np.argmax(numparr)
            best_x_idx = maxcell % len(x_values)
            best_y_idx = int(maxcell / len(x_values))
            winning_height = heights[best_y_idx, best_x_idx]
            if winning_height < (searchradius * 0.7):
                break
            lastsearchradius = searchradius
            searchradius = searchradius / 0.6
            print("Upping search radius to {:.2f}".format(searchradius))

        best_x = x_values[best_x_idx]
        best_y = y_values[best_y_idx]
        bestpos = focusposs[best_y_idx, best_x_idx]
        print("Found best point {:.3f} at ({:.0f}, {:.0f}) (image height {:.2f})"
              "".format(numparr.max(), best_x, best_y, winning_height))
        print("   at focus position {:.2f}".format(bestpos))
        ob = FocusOb(focuspos=bestpos, sharp=numparr.max())
        ob.x_loc = best_x
        ob.y_loc = best_y
        return ob

    def build_calibration(self, fstop, opt_freq=AUC, plot=True, writetofile=False):
        """
        Assume diffraction limited lens to build calibration data

        :param fstop: Taking f-stop
        :param plot: Plot if True
        :param writetofile: Write to calibration.csv file
        :return: Numpy array of correction data
        """
        if self.use_calibration:
            if writetofile:
                pass
                # raise AttributeError("Focusset must be loaded without existing calibration")
            else:
                log.warning("Existing calibration loaded (will compare calibrations)")
        # Get best AUC focus postion

        f_range = RAW_SFR_FREQUENCIES[:]
        # data_sfr = self.get_peak_sfr(freq=opt_freq, axis=BOTH_AXES).raw_sfr_data[:]
        data_sfr = self.find_sharpest_raw_points_avg_sfr()

        if not writetofile:
            data_sfr *= self.base_calibration

        diffraction_sfr = diffraction_mtf(f_range, fstop)  # Ideal sfr

        correction = np.clip(diffraction_sfr / data_sfr, 0, 50.0)

        print("Calibration correction:")
        print(correction)

        if writetofile:
            with open("calibration.csv", 'w') as csvfile:
                csvwriter = csv.writer(csvfile, delimiter=',', quotechar='|')
                csvwriter.writerow(list(f_range))
                csvwriter.writerow(list(correction))
                print("Calibration written!")

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
        self.calibration = self.fields[0].calibration

    def get_grids(self, *args, **kwargs):
        return self.fields[0].get_grids(*args, **kwargs)

    def find_compromise_focus(self, freq=AUC, axis=MEDIAL, detail=1.0, plot_freq=None, weighting_fn=EVEN_WEIGHTED,
                              plot_type=PROJECTION3D, midfield_bias_comp=True, precision=0.1):
        """
        Finds optimial compromise flat-field focus postion

        :param freq: Frequency to use for optimisation
        :param axis: SAGITTAL, MERIDIONAL or MEDIAL
        :param detail: Change number of analysis points (default is 1.0)
        :param plot_freq: Frequency to use for plot of result if different to optimisation frequency
        :param weighting_fn: Pass a function which accepts an image height (0-1) and returns weight (0-1)
        :param plot_type: CONTOUR2D or PROJECTION3D
        :param midfield_bias_comp: Specified whether bias due to large number of mid-field points should be compensated
        :return:
        """

        gridit, numpyarray, x_values, y_values = self.get_grids(detail)
        n_fields = len(self.fields)
        inc = precision
        field_locs = np.arange(0, n_fields, inc)  # Analyse with 0.1 field accuracy

        # Made sfr data array
        sharps = np.repeat(numpyarray[:,:,np.newaxis], len(field_locs), axis=2)

        xm, ym = np.meshgrid(x_values, y_values)
        heights = ((xm - (IMAGE_WIDTH / 2))**2 + (ym - (IMAGE_HEIGHT / 2))**2)**0.5 / ((IMAGE_WIDTH / 2)**2+(IMAGE_HEIGHT / 2)**2)**0.5
        weights = np.ndarray((len(y_values), len(x_values)))

        # Iterate over all locations
        for n_x, n_y, x, y in gridit:
            # Get sharpness data at location
            try:
                interpfn = self.find_best_focus(x, y, freq, axis=axis).interpfn
            except FitError as e:
                # Keep calm and ignore crap fit errors
                interpfn = e.fitpos.interpfn
                # plt.plot(field_locs, interpfn(field_locs))
                # plt.show()

            # Gather data at all sub-points
            pos_sharps = interpfn(field_locs)
            sharps[n_y, n_x] = pos_sharps

            # Build weighting
            img_height = calc_image_height(x, y)
            weights[n_y, n_x] = np.clip(weighting_fn(img_height), 1e-6, 1.0)

        if midfield_bias_comp:
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
            weights = weights / height_density_weight_mods

        weights = np.repeat(weights[:, :, np.newaxis], len(field_locs), axis=2)
        average = np.average(sharps, axis=(0, 1), weights=weights)

        # slice_ = average > (average.max() * 0.9)
        # poly = np.polyfit(field_locs[slice_], average[slice_], 2)
        # polyder = np.polyder(poly, 1)
        # peak_focus_pos = np.roots(polyder)[0]
        peak_focus_pos = np.argmax(average) * inc
        print("Found peak focus at position {:.2f}".format(peak_focus_pos))

        interpfn = interpolate.InterpolatedUnivariateSpline(field_locs, average)

        if not 0 < peak_focus_pos < (n_fields - 1):
            # _fit = np.polyval(poly, field_locs)
            plt.plot(field_locs, average)
            # plt.plot(field_locs, _fit)
            plt.show()

        if plot_freq and plot_freq != freq:
            # Plot frequency different to optimisation frequency

            sharpfield = numpyarray
            for n_x, n_y, x, y in gridit:
                # Get sharpness data at location

                interpfn = self.find_best_focus(x, y, plot_freq, axis=axis).interpfn
                sharpfield[n_y, n_x] = interpfn(peak_focus_pos)
        else:
            sharpfield = sharps[:, :, int(peak_focus_pos/inc + 0.5)]

        # Move on to plotting results
        if plot_type is not None:
            plot = FieldPlot()
            plot.zdata = sharpfield
            plot.xticks = x_values
            plot.yticks = y_values
            plot.yreverse = True
            print(9, self.calibration)
            plot.zmin = diffraction_mtf(freq, LOW_BENCHMARK_FSTOP, calibration=1.0 / self.base_calibration)
            plot.zmax = diffraction_mtf(freq, HIGH_BENCHBARK_FSTOP, calibration=1.0 / self.base_calibration)
            plot.zmin = diffraction_mtf(freq, LOW_BENCHMARK_FSTOP, calibration=None)
            plot.zmax = diffraction_mtf(freq, HIGH_BENCHBARK_FSTOP, calibration=None)
            # print(55, plot.zmin)
            # print(55, plot.zmax)
            plot.title = "Compromise focus flat-field " + self.lens_name
            plot.xlabel = "x image location"
            plot.ylabel = "y image location"
            plot.plot(plot_type)
        return FocusOb(peak_focus_pos, average[int(peak_focus_pos * 10)], interpfn)

    def plot_mtf_vs_image_height(self, analysis_pos=None, freqs=(15 / 250, 45/250, 75/250), detail=0.5, axis=MEDIAL, show=True,
                                 show_diffraction=None, posh=False):
        gridit, numpyarr, x_vals, y_vals = self.get_grids(detail=detail)
        heights = numpyarr.copy()
        arrs = []
        legends = []
        fig = plt.figure()
        if axis == MEDIAL:
            axis = [SAGITTAL, MERIDIONAL]
            axis = [SAGITTAL, MERIDIONAL]
        else:
            axis = [axis]
        for nfreq, freq in enumerate(freqs):
            for loopaxis in axis:
                arr = numpyarr.copy()
                arrs.append(arr)
                for nx, ny, x, y in gridit:
                    heights[ny, nx] = calc_image_height(x, y)
                    if analysis_pos is None:
                        try:
                            ob = self.find_best_focus(x, y, freq, loopaxis)
                            arr[ny, nx] = ob.sharp
                            # if ob.sharp > 0.95:
                            #     print(ob.sharp, x, y, heights[ny, nx], loopaxis)
                        except FitError as e:
                            arr[ny, nx] = np.nan
                    else:
                        arr[ny, nx] = self.interpolate_value(x, y, analysis_pos.focuspos, freq, loopaxis, posh=posh)

                if loopaxis == SAGITTAL:
                    lineformat = '--'
                else:
                    lineformat = '-'
                plot = Scatter2D()
                plot.xdata = heights.flatten()
                plot.ydata = arr.flatten()
                plot.ymin = 0
                plot.ymax = 1.0
                plot.xmin = 0.0
                plot.xmax = 1.0
                plot.xlabel = "Normalised Image Height"
                plot.ylabel = "Modulation Transfer Function"
                plot.title = self.exif.summary
                if show_diffraction:
                    plot.hlines = diffraction_mtf(np.array(freqs), show_diffraction)
                    plot.hlinelabels = "f{} diffraction".format(show_diffraction)
                plot.smoothplot(lineformat=lineformat, show=False, color=COLOURS[[0, 3, 4][nfreq]],
                                label="{:.2f} lp/mm {}".format(freq * 250, loopaxis[0]),
                                marker="^" if loopaxis is SAGITTAL else "s",points_limit=4.0)
                # legends.append("{:.2f} cy/px {}".format(freq, loopaxis))
                # legends.append("{:.2f} cy/px {}".format(freq, loopaxis))
                # legends.append(None)
                # legends.append()
        plt.legend()
        if show:
            plt.show()

    def guess_focus_shift_field(self, detail=1.0, axis=MEDIAL):
        gridit, numarr, x_values, y_values = self.get_grids(detail=detail)
        arrs = []
        for freq in (0.02, 0.3):
            arr = numarr.copy()
            arrs.append(arr)
            for nx, ny, x, y in gridit:
                try:
                    arr[ny, nx] = self.find_best_focus(x, y, freq, axis=axis).focuspos
                except FitError as e:
                    for a in arrs:
                        a[ny, nx] = 0.0
        dif = arrs[1] - arrs[0]
        plot = FieldPlot()
        plot.xticks = x_values
        plot.yticks = y_values
        plot.zdata = dif
        plot.title = "Guessed focus shift"
        plot.zlabel = "Relative focus shift"
        plot.projection3d()

    def plot_best_focus_vs_frequency(self, x, y, axis=MEDIAL):
        freqs = np.logspace(-2, -0.3, 20)
        bests = []
        for freq in freqs:
            try:
                bestpos = self.find_best_focus(x, y, freq, axis=axis).focuspos
            except FitError as e:
                bestpos = float("NaN")
            bests.append(bestpos)
        plot = Scatter2D()
        plot.xdata = freqs
        plot.ydata = bests
        plot.xlog = True
        plot.smoothplot(plot_used_original_data=1)

    def skip_fields_and_check_accuracy(self):
        sharpestpoint = self.find_sharpest_raw_point()
        x = sharpestpoint.x
        y = sharpestpoint.y
        fields = self.fields
        fieldnumbers = list(range(len(fields)))
        skips = [1, 7]
        sharps = []
        sharps = []
        numpoints = []
        count = 1
        print("Inc  Start  Points     SFR   SFR+-  BestFocus  Bestfocus+-")
        for skip in skips:
            sharplst = []
            focusposlst = []
            counts = []
            for start in np.arange(skip):
                usefields = fields[start::skip]
                # Temporarily replace self.fields #naughty
                self.fields = usefields
                focusob = self.find_best_focus(x, y, axis=MERIDIONAL, plot=1)
                sharplst.append(focusob.sharp)
                focusposlst.append(focusob.focuspos)
                counts.append(count)
                count += 1
                text = ""
                if skip == 1 and start == 0:
                    baseline = focusob.sharp, focusob.focuspos
                    text = "** BASELINE ***"
                delta = focusob.sharp - baseline[0]
                percent = delta / baseline[0] * 100
                bestfocus = (focusob.focuspos * skip) + start
                bestfocusdelta = bestfocus - baseline[1]

                print("{:3.0f}  {:5.0f}  {:6.0f}  {:6.3f} {:7.3f} {:10.3f} {:10.3f} {}".format(skip,
                                                                                          start,
                                                                                          len(self.fields),
                                                                                          focusob.sharp,
                                                                                          delta,
                                                                                          bestfocus,
                                                                                          bestfocusdelta,
                                                                                               text))
            # plt.plot(counts, sharplst, '.', color=COLOURS[skip])
            # plt.plot([len(usefields)] * skip, sharplst, '.', color=COLOURS[skip])
            numpoints.append(len(usefields))
        self.fields = fields
        print(count)
        plt.legend(numpoints)
        # plt.xlabel("Testrun number")
        plt.xlabel("Number of images in sequence")
        plt.ylabel("Peak Spacial frequency response")
        plt.title("Peak detection vs number of images in sequence")
        # plt.plot([0, count], [baseline[0], baseline[0]], '--', color='gray')
        plt.plot([3, len(fields)], [baseline[0], baseline[0]], '--', color='gray')
        plt.ylim(baseline[0]-0.1, baseline[0]+0.1)
        plt.show()


"photos@scs.co.uk"
"S 0065-3858491"