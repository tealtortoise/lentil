import csv
import math
import os
import logging
from logging import getLogger

from scipy import optimize, interpolate, stats

from lentil.sfr_point import SFRPoint
from lentil.sfr_field import SFRField
from lentil.plot_utils import FieldPlot
from lentil.constants_utils import *

log = getLogger(__name__)
log.setLevel(logging.INFO)

ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
log.addHandler(ch)


class FocusPos:
    def __init__(self, focuspos=None, sharp=None, interpfn=None, curvefn=None, lowbound=None, highbound=None):
        self.focuspos = focuspos
        self.sharp = sharp
        self.interpfn = interpfn
        self.curvefn = curvefn
        self.lowbound = lowbound
        self.highbound = highbound

    @classmethod
    def get_midpoint(cls, a, b):
        new = cls()
        mid_x = (a.focuspos + b.focuspos) * 0.5
        new.focuspos = mid_x

        def interp_merge(in_x):
            return (a.interpfn(in_x) + b.interpfn(in_x)) * 0.5

        new.sharp = interp_merge(mid_x)
        new.interpfn = interp_merge
        return new


class FitError(Exception):
    def __init__(self, fit_tuple=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fit_tuple = fit_tuple


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
            plot.set_diffraction_limits(freq)
            plot.yreverse = True
            plot.xlabel = "Image position x"
            plot.ylabel = "Image position y"
            plot.zlabel = "Best focus"
            plot.title = title
            plot.zdata = focus_posits
            plot.wdata = sharps
            ax = plot.projection3d(ax)
            """
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
            """
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
            best_s = self.find_best_focus(x, y, freq, SAGITTAL)
            best_m = self.find_best_focus(x, y, freq, MERIDIONAL)
            return FocusPos.get_midpoint(best_s, best_m)

        # Get SFR from each field
        y_values = []
        for field in self.fields:
            y_values.append(field.interpolate_value(x, y, freq, axis))
        y_values = np.array(y_values)
        x_values = np.arange(0, len(y_values), 1)  # Arbitrary focus units

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
        sigmas = 1. / weights

        fitted_params, _ = optimize.curve_fit(fastgauss, x_values, y_values,
                                              bounds=bounds, sigma=sigmas, ftol=0.0001, xtol=0.0001,
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

        interp_fn = interpolate.InterpolatedUnivariateSpline(x_values, y_values, k=2)

        if mean_abs_error_rel > 0.12:
            errorstr = "Very high fit error: {:.3f}".format(mean_abs_error_rel)
            log.warning(errorstr)
            raise FitError(errorstr, fittuple=(gaussfit_peak_x, gaussfit_peak_y, None, None, interp_fn, curvefn))
        elif mean_abs_error_rel > 0.06:
            errorstr = "High fit error: {:.3f}".format(mean_abs_error_rel)
            log.warning(errorstr)
            if strict:
                errorstr = "Strict mode, fit aborted".format(mean_abs_error_rel)
                log.warning(errorstr)
                raise FitError(errorstr, fittuple=(gaussfit_peak_x, gaussfit_peak_y, None, None, interp_fn, curvefn))


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

        return FocusPos(gaussfit_peak_x, gaussfit_peak_y, interp_fn, curvefn)

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

    def plot_best_sfr_vs_freq_at_point(self, x, y, x_values=None, secondline_fn=None, show=True):
        if x_values is None:
            x_values = np.linspace(0, 0.5, 30)
        y = [self.find_best_focus(x, y, f).sharp for f in x_values]
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
        focuspos = self.find_best_focus((IMAGE_WIDTH / 2), (IMAGE_HEIGHT / 2), AUC, axis=axis).focuspos

        # Build sfr at that point
        data_sfr = []
        for f in RAW_SFR_FREQUENCIES[:32]:
            interpfn = self.find_best_focus((IMAGE_WIDTH / 2), (IMAGE_HEIGHT / 2), f, axis=axis).interpfn
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

    def find_compromise_focus(self, freq=AUC, axis=MEDIAL, detail=1.0, plot_freq=None, weighting_fn=EVEN_WEIGHTED):
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
            interpfn = self.find_best_focus(x, y, freq, axis=axis).interpfn

            # Gather data at all sub-points
            pos_sharps = interpfn(field_locs)
            sharps[n_y, n_x] = pos_sharps

            # Build weighting
            img_height = calc_image_height(x, y)
            weights[n_y, n_x] = weighting_fn(img_height)

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
                interpfn = self.find_best_focus(x, y, plot_freq, axis=axis).interpfn
                sharpfield[n_y, n_x] = interpfn(peak_focus_pos)
        else:
            sharpfield = sharps[:, :, int(peak_focus_pos*10 + 0.5)]

        # Move on to plotting results

        plot = FieldPlot()

        plot.zdata = sharpfield
        plot.xticks = x_values
        plot.yticks = y_values
        plot.yreverse = True
        plot.zmin = diffraction_mtf(freq, LOW_BENCHMARK_FSTOP)
        plot.zmax = diffraction_mtf(freq, HIGH_BENCHBARK_FSTOP)
        plot.title = "Compromise focus flat-field"
        plot.xlabel = "x image location"
        plot.ylabel = "y image location"

        plot.contour2d()

    def plot_mtf_vs_image_height(self, detail=1.0):
        pass


"photos@scs.co.uk"
"S 0065-3858491"