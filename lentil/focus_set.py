import csv
import multiprocessing
import os
from logging import getLogger
from operator import itemgetter

import numpy
from scipy import stats

from lentil.sfr_point import FFTPoint
from lentil.sfr_field import SFRField, NotEnoughPointsException
from lentil.plot_utils import FieldPlot, Scatter2D, COLOURS
from lentil.constants_utils import *
# import lentil.wavefront

import prysm

log = getLogger(__name__)
log.setLevel(logging.DEBUG)
log.setLevel(logging.INFO)

ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
log.addHandler(ch)

STORE = []

globalfocusset = None
globaldict = {}


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


class FocusPositionReader(list):
    def __init__(self, rootpath, fallback=True):
        print(rootpath)
        filepath = os.path.join(rootpath, "focus_positions.csv")
        ids = []
        names = []
        values = []
        self.namedict = {}
        self.fileid_dict = {}
        current = True if fallback else False
        try:
            with open(filepath, 'r') as file:
                reader = csv.reader(file, delimiter=" ", quotechar='"')
                for row in reader:
                    if row[0] == '#':
                        continue
                    elif row[0].lower() == "code_version" and not fallback:
                        # If old estimation code used ignore file to trigger re-estimation
                        try:
                            if float(row[1]) < CURRENT_JITTER_CODE_VERSION:
                                raise FileNotFoundError()
                        except (TypeError, IndexError):
                            raise FileNotFoundError()
                        current = True
                        continue
                    elif len(row) == 3:
                        id, name, value = row
                    else:
                        log.warning("Unknown row {}".format(str(row)))
                        continue
                    ids.append(id)
                    names.append(name)
                    values.append(float(value))
                    self.namedict[name] = float(value)
                    self.fileid_dict[id] = float(value)
            if not current:
                raise FileNotFoundError()
        except (FileNotFoundError, ) as e:
            if not fallback:
                raise e
            log.warning("No focus position file found, falling back to counting!")
            files = []
            for entry in os.scandir(rootpath):
                if entry.name.lower().endswith(".sfr"):
                    digits = "".join((char for char in entry.name if char.isdigit()))
                    files.append((int(digits), entry.name))
            files.sort()
            for n, (id, file) in enumerate(files):
                self.namedict[file] = n
                self.fileid_dict[id] = n
                values.append(n)

        super().__init__(values)

    def __getitem__(self, item):
        if type(item) is int:
            try:
                return super().__getitem__(item)
            except IndexError:
                return item
        if item in self.namedict:
            return self.namedict[item]
        return self.fileid_dict.get(item, None)


def read_wavefront_file(path):
    with open(path, 'r') as file:
        reader = csv.reader(file, delimiter=" ", quotechar='"')
        lst = []
        for row in reader:
            row = [cell for cell in row if cell != ""]
            if len(row) == 0:
                continue
            if len(row) == 1:
                if len(row[0]) > 0:
                    if 'wavefront data' in row[0].lower():
                        dct = {}
                        lst.append((row[0], dct))
                    else:
                        continue
            elif len(row) == 2:
                value = row[1]
                dct[row[0]] = tryfloat(value)
            else:
                rowdata = row[1:]
                dct[row[0]] = [tryfloat(_) for _ in rowdata]
    print("Read from path '{}'".format(path))
    return lst

class FocusSet:
    """
    A range of fields with stepped focus, in order
    """

    def __init__(self, rootpath=None, rescan=False, include_all=False, use_calibration=True, load_focus_data=True,
                 load_complex=False):
        global globalfocusset
        globalfocusset = self
        global globalpool
        self.rootpath = rootpath
        fields = []
        self.lens_name = rootpath
        self.focus_scale_label = "Focus position (arbritary units)"
        calibration = None
        self.calibration = None
        self._focus_data = None
        self.focus_scaling_fn = None
        self.wavefront_data = []
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
        if rootpath is None:
            log.warning("Warning, initialising with no field data!")
            return
        # try:
            # Attempt to open lentil_data
            # with open(os.path.join(rootpath, "slfjsadf" if rescan or include_all else "lentil_data.csv"), 'r')\
            #         as csvfile:
            #     print("Found lentildata")
            #     csvreader = csv.reader(csvfile, delimiter=',', quotechar='|')
            #     for row in csvreader:
            #         if row[0] == "Relevant filenames":
            #             stubnames = row[1:]
            #         elif row[0] == "lens_name":
            #             self.lens_name = row[1]
            #     pathnames = [os.path.join(rootpath, stubname) for stubname in stubnames]
            #     exif = EXIF(pathnames[0])
            #     self.exif = exif
            #     for pathname in pathnames:
            #         try:
            #             self.fields.append(SFRField(pathname=pathname, calibration=calibration, exif=exif))
            #         except NotEnoughPointsException:
            #             pass
        # except FileNotFoundError:
        #     print("Did not find lentildata, finding files...")
        with os.scandir(rootpath) as it:
            for entry in it:
                try:
                    entrynumber = int("".join([s for s in entry.name if s.isdigit()]))
                except ValueError:
                    continue

                if entry.is_dir():
                    fullpathname = os.path.join(rootpath, entry.path, SFRFILENAME)
                    sfr_file_exists = os.path.isfile(fullpathname)
                    if not sfr_file_exists:
                        continue
                    stubname = os.path.join(entry.name, SFRFILENAME)
                elif entry.is_file() and entry.name.endswith("sfr"):
                    print("Found {}".format(entry.path))
                    fullpathname = entry.path
                    stubname = entry.name
                else:
                    continue
                filenames.append((entrynumber, fullpathname, stubname))

                filenames.sort()

        if len(filenames) is 0:
            raise ValueError("No fields found! Path '{}'".format(rootpath))

        exif = EXIF(filenames[0][1])
        self.exif = exif
        for entrynumber, pathname, filename in filenames:
            if 1:
                print("Opening file {}".format(pathname))
                try:
                    field = SFRField(pathname=pathname, calibration=calibration, exif=exif, load_complex=load_complex,
                                     filenumber=entrynumber)
                except NotEnoughPointsException:
                    print("Not enough points, skipping!")
                    continue
                field.filenumber = entrynumber
                field.filename = filename
                fields.append(field)
        # if not include_all:
        #     self.remove_duplicated_fields()
        #     self.find_relevant_fields(writepath=rootpath, freq=AUC)

        if load_focus_data:
            self.fields = []
            try:
                focus_data = FocusPositionReader(rootpath, fallback=False)
            except FileNotFoundError:
                focus_data = estimate_focus_jitter(rootpath)
            field_focus_tups = list(zip(focus_data, fields))
            field_focus_tups.sort(key=lambda t: t[0])

            prev_focus = -np.inf
            sorted_included_focus = []
            duplicate_threshold = 0.5
            for focus, field in field_focus_tups:
                if focus > (prev_focus + duplicate_threshold) or include_all:
                    sorted_included_focus.append(focus)
                    self.fields.append(field)
                prev_focus = focus
            self._focus_data = np.array(sorted_included_focus)
        else:
            self.fields = fields

    def find_relevant_fields(self, freq=DEFAULT_FREQ, detail=1, writepath=None):
        min_ = float("inf")
        max_ = float("-inf")
        x_values, y_values = self.fields[0].build_axis_points(24 * detail, 16 * detail)
        totalpoints = len(x_values) * len(y_values) * 2
        done = 0
        for axis in MERIDIONAL, SAGITTAL:
            for x in x_values:
                for y in y_values:
                    if done % 100 == 0:
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
                               alpha=0.7, title=None, fix_zlim=None):
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

        if axis == ALL_THREE_AXES:
            if not plot_curvature or plot_type is not PROJECTION3D:
                raise ValueError("Plot must be field curvature in 3d for ALL THREE AXES")
            ax, skew = self.plot_ideal_focus_field(freq, detail, MERIDIONAL, show=False, skewplane=skewplane,
                                                   alpha=alpha*0.3)
            ax, skew = self.plot_ideal_focus_field(freq, detail, SAGITTAL, show=False, ax=ax, skewplane=skew, alpha=alpha*0.3)
            # if show:
            #     plt.show()
            return self.plot_ideal_focus_field(freq, detail, MEDIAL, show=show, ax=ax, skewplane=skew, alpha=alpha, fix_zlim=fix_zlim)


        num_sheets = 0  # Number of acceptable focus sheets each side of peak focus
        if title is None:
            title = "Ideal focus field " + self.exif.summary

        gridit, focus_posits, x_values, y_values = self.get_grids(detail)
        sharps = focus_posits.copy()
        if plot_curvature:
            colours = focus_posits.copy()
            z_values_low = focus_posits.copy()
            z_values_high = focus_posits.copy()

        tot_locs = len(focus_posits.flatten())
        locs = 1


        multi = 0

        paramlst = []
        for x_idx, y_idx, x, y in gridit:
            if multi > 1:
                paramlst.append((int(x_idx), int(y_idx), float(x), float(y), float(freq), axis, locs, tot_locs))
            else:
                if locs % 50 == 0:
                    print("Finding best focus for location {} / {}".format(locs, tot_locs))
                bestfocus = self.find_best_focus(x, y, freq, axis)

                sharps[y_idx, x_idx] = bestfocus.sharp
                if plot_curvature:
                    focus_posits[y_idx, x_idx] = bestfocus.focuspos
            locs += 1


        if multi > 1:
            global globalfocusset
            globalfocusset = self
            p = multiprocessing.Pool(processes=multi)

            results = p.map(FocusSet.find_best_focus_helper, paramlst)
            p.close()
            p.join()
            # print(results)
            # exit()
            for x_idx, y_idx, sharp, focuspos in results:
                sharps[y_idx, x_idx] = sharp
                if plot_curvature:
                    focus_posits[y_idx, x_idx] = focuspos

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

        if plot_type == CONTOUR2D or plot_type == SMOOTH2D:
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
            ax = plot.plot(plot_type, ax, show=show)
        else:
            plot = FieldPlot()

            plot.set_diffraction_limits(freq, graphaxis="w")

            plot.xticks = x_values
            plot.yticks = y_values
            plot.set_diffraction_limits(freq, graphaxis='w')
            plot.yreverse = True
            plot.xlabel = "Image position x"
            plot.ylabel = "Image position y"
            plot.zlabel = self.focus_scale_label
            plot.title = title
            plot.zdata = focus_posits
            plot.wdata = sharps
            plot.alpha = alpha
            # print(focus_posits.min(), focus_posits.max())
            if fix_zlim is not None:
                plot.zmin = fix_zlim[0]
                plot.zmax = fix_zlim[1]
            else:
                if ax is not None:
                    # print(ax.get_zlim())
                    plot.zmin = min(focus_posits.min(), ax.get_zlim()[0])
                    plot.zmax = max(focus_posits.max(), ax.get_zlim()[1])
                else:
                    plot.zmin = focus_posits.min()
                    plot.zmax = focus_posits.max()
            ax = plot.projection3d(ax, show=show)
        return ax, skewplane

    def plot_field_curvature_strip_contour(self, freq=DEFAULT_FREQ, axis=MERIDIONAL, theta=THETA_TOP_RIGHT, radius=1.0):
        maxradius = IMAGE_DIAGONAL / 2 * radius
        heights = np.linspace(-1, 1, 41)
        plot_focuses = self.focus_data
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
        plot.ylabel = self.focus_scale_label
        plot.ymin = -110
        plot.ymax = 110
        plot.title = "Edge SFR vs focus position for varying height from centre"
        plot.contour2d()

    def get_interpolation_fn_at_point(self, x, y, freq=DEFAULT_FREQ, axis=MEDIAL, limit=None, skip=1, order=2,
                                      multi=False):
        y_values = []
        allidxs = list(range(len(self.fields)))
        if limit is None:
            lowlim = 0
            highlim = len(self.fields)
            fields = self.fields[::skip]
            used_idxs = allidxs[::skip]
        else:
            lowlim = max(0, limit[0])
            highlim = min(len(self.fields), max(limit[1], lowlim + 3))
            fields = self.fields[lowlim: highlim:skip]
            used_idxs = allidxs[lowlim: highlim:skip]

        n = len(used_idxs)

        if axis == MEDIAL:
            for field in fields:
                s = field.interpolate_value(x, y, freq, SAGITTAL)
                m = field.interpolate_value(x, y, freq, MERIDIONAL)
                y_values.append((m + s) * 0.5)
        else:
            if multi:
                y_values = self.pool.starmap(FocusSet.intepolate_value_helper, zip(used_idxs,
                                                                           [float(x)] * n,
                                                                           [float(y)] * n,
                                                                           [float(freq)] * n,
                                                                           [axis] * n))
            else:
                for n, field in enumerate(fields):
                    y_values.append(field.interpolate_value(x, y, freq, axis))
        y_values = np.array(y_values)
        x_ixs = np.arange(lowlim, highlim, skip)  # Arbitrary focus units
        x_values = np.array(self.focus_data)[x_ixs]
        if axis in COMPLEX_AXES:
            realinterpfn = interpolate.InterpolatedUnivariateSpline(x_values, np.real(y_values), k=order)
            imajinterpfn = interpolate.InterpolatedUnivariateSpline(x_values, np.imag(y_values), k=order)

            def interpfn(x_, complex_type=COMPLEX_CARTESIAN):
                return convert_complex((realinterpfn(x_), imajinterpfn(x_)), complex_type)
        else:
            interpfn = interpolate.InterpolatedUnivariateSpline(x_values, y_values, k=order)
        pos = FocusOb(interpfn=interpfn)
        pos.focus_data = x_values
        pos.sharp_data = y_values
        return pos

    def find_focus_spacing(self, plot=True):
        return lentil.wavefront.estimate_wavefront_errors(self)

    def find_best_focus(self, x=None, y=None, freq=DEFAULT_FREQ, axis=MEDIAL, plot=False, show=True, strict=False, fitfn=cauchy,
                        _pos=None, _return_step_data_only=False, _step_est_offset=0.3, _step_estimation_posh=False):
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
        # print(x,y, freq, axis)
        # print("---------")
        # Go recursive if both planes needed
        if _return_step_data_only:
            freq = LOWAVG
        if axis == MEDIAL and _pos is None:
            best_s = self.find_best_focus(x, y, freq, SAGITTAL, plot, show, strict)
            best_m = self.find_best_focus(x, y, freq, MERIDIONAL, plot, show, strict)
            mid = FocusOb.get_midpoint(best_s, best_m)
            if 0 and 0.6 < calc_image_height(x, y) < 0.7:
                print("     ", best_s.x_loc, best_s.y_loc, best_s.focuspos)
                print("     ", best_m.x_loc, best_m.y_loc, best_m.focuspos)
                print("     ", mid.x_loc, mid.y_loc, mid.focuspos)
            # STORE.append(mid.focuspos)
            # print(sum(STORE) / len(STORE))
            return mid

        if _pos is None:
            if x is None or y is None:
                raise ValueError("Co-ordinates must be specified!")
            pos = self.get_interpolation_fn_at_point(x, y, freq, axis)
        else:
            pos = _pos

        x_values = pos.focus_data
        y_values = pos.sharp_data
        interp_fn = pos.interpfn
        # exit()
        # Initial fit guesses
        highest_data_x_idx = np.argmax(y_values)
        highest_data_y = y_values[highest_data_x_idx]

        if highest_data_x_idx > 0:
            x_inc = x_values[highest_data_x_idx] - x_values[highest_data_x_idx-1]
        else:
            x_inc = x_values[highest_data_x_idx+1] - x_values[highest_data_x_idx]

        # y_values = np.cos(np.linspace(-6, 6, len(x_values))) + 1
        absgrad = np.abs(np.gradient(y_values)) / highest_data_y
        gradsum = np.cumsum(absgrad)
        distances_from_peak = np.abs(gradsum - np.mean(gradsum[highest_data_x_idx:highest_data_x_idx+1]))
        shifted_distances = interpolate.InterpolatedUnivariateSpline(x_values, distances_from_peak, k=1)(x_values - x_inc*0.5)
        weights = np.clip(1.0 - shifted_distances * 1.3 , 1e-1, 1.0) ** 5
        mean_peak_x = x_values[highest_data_x_idx]

        # print(mean_peak_x_idx)
        # print(mean_peak_x)
        # Define optimisation bounds
        # bounds = ((highest_data_y * 0.95, mean_peak_x_idx - 0.9, 0.8, -0.3),
        #           (highest_data_y * 1.15, mean_peak_x_idx + 0.9, 50.0, 1.3))
        # bounds = ((highest_data_y * 0.98, mean_peak_x - x_inc * 2, 0.4 * x_inc,),
        #           (highest_data_y * 1.15, mean_peak_x + x_inc * 2, 100.0 * x_inc,))

        bounds = fitfn.bounds(mean_peak_x, highest_data_y, x_inc)

        sigmas = 1. / weights
        initial = fitfn.initial(mean_peak_x, highest_data_y, x_inc)
        fitted_params, _ = optimize.curve_fit(fitfn, x_values, y_values,
                                              bounds=bounds, sigma=sigmas, ftol=1e-5, xtol=1e-5,
                                              p0=initial)
        #
        fit_peak_x = fitted_params[1]
        fit_peak_y = fitted_params[0]

        count = 0

        def prysmfit(defocuss, defocus_offset, defocus_step, aberr, a2=0, plot=False):
            # freqs = np.arange(0, 32, 1) / 64 * 250
            freqs = LOWAVG_NOMBINS / 64 * 250
            # print(defocus_offset, defocus_step, aberr)
            out = []
            for defocus in defocuss:
                pupil = prysm.NollZernike(Z4=(defocus - defocus_offset) * defocus_step * 0.575,#  - aberr / 2,
                                          dia=10, norm=False,
                                          z11=aberr,
                                          z22=a2,
                                          wavelength=0.575,
                                          opd_unit="um",
                                          samples=128)
                # prysm.PSF.from_pupil(pupil, efl=10*self.exif.aperture).plot2d()
                # plt.show()
                m = prysm.MTF.from_pupil(pupil, efl=self.exif.aperture * 10)
                out.append(np.mean(m.exact_tan(freqs)))
            if plot:
                # pupil.plot2d()
                # plt.show()
                # plt.plot(out, label="Prysm Model {:.3f}λ Z4 step size, {:.3f}λ Z11".format(defocus_step, aberr / 0.575))
                plt.plot(out, label="Prysm Model {:.3f}λ Z4 step size, {:.3f}λ Z11, {:.3f}λ Z22".format(defocus_step, aberr / 0.575, a2 / 0.575))
                plt.plot(y_values, label="MTF Mapper data")
                plt.xlabel("Focus position (arbitrary units)")
                plt.ylabel("MTF")
                plt.title("Prysm model vs chart ({:.32}-{:.2f}cy/px AUC) {}".format(LOWAVG_NOMBINS[0] / 64, LOWAVG_NOMBINS[-1] / 64, self.exif.summary))
                plt.ylim(0, 1)
                plt.legend()
                plt.show()
            global count
            count += 1
            # print(count)
            return np.array(out)

        prysm_offset = None

        if _step_estimation_posh and _return_step_data_only:
            fit_peak_y = 0
            #
            initial = [fit_peak_x, 0.3, (1.0 - fitted_params[0]) / 5,(1.0 - fitted_params[0]) / 5, ]#[:3]
            bounds = (min(x_values), 0.001, 0.0, 0), (max(x_values), 0.35, 0.45, 0.45)
            # bounds = bounds[0][:3], bounds[1][:3]
            sigmas = 1.0 / (y_values ** 1)
            prysm_params, _ = optimize.curve_fit(prysmfit, x_values, y_values,
                                                  bounds=bounds, sigma=sigmas, ftol=1e-3, xtol=1e-3,
                                                  p0=initial)
            # print("Prysm fit params", prysm_params)
            if plot:
                prysmfit(x_values, *prysm_params, plot=True)
            prysm_offset = fitted_params[1]

        log.debug("Fn fit peak is {:.3f} at {:.2f}".format(fitted_params[0], fitted_params[1]))
        log.debug("Fn sigma: {:.3f}".format(fitted_params[2]))

        if _return_step_data_only:
            # ---
            # Estimate defocus step size
            # ---
            if _step_estimation_posh:
                est_defocus_rms_wfe_step = prysm_params[1]
                log.info("Using posh estimation model")
            else:
                est_defocus_rms_wfe_step = (4.387 / fitted_params[2] / (fit_peak_y + _step_est_offset)) / self.exif.aperture
            if "_fixed_defocus_step_wfe" in dir(self):
                est_defocus_rms_wfe_step = self._fixed_defocus_step_wfe
            est_defocus_pv_wfe_step = est_defocus_rms_wfe_step * 2 * 3 ** 0.5

            log.info("--- Focus step size estimates ---")
            log.info("    RMS Wavefront defocus error {:8.4f} λ".format(est_defocus_rms_wfe_step))

            longitude_defocus_step_um = est_defocus_pv_wfe_step * self.exif.aperture**2 * 8 * 0.55
            log.info("    Image side focus shift      {:8.3f} µm".format(longitude_defocus_step_um))

            na = 1 / (2.0 * self.exif.aperture)
            theta = np.arcsin(na)
            coc_step = np.tan(theta) * longitude_defocus_step_um * 2

            focal_length_m = self.exif.focal_length * 1e-3

            def get_opposide_dist(dist):
                return 1.0 / (1.0 / focal_length_m - 1.0 / dist)

            lens_angle_of_view = self.exif.angle_of_view
            # print(lens_angle_of_view)
            subject_distance = CHART_DIAGONAL * 0.5 / np.sin(lens_angle_of_view / 2)
            image_distance = get_opposide_dist(subject_distance)

            log.info("    Subject side focus shift    {:8.2f} mm".format((get_opposide_dist(image_distance-longitude_defocus_step_um *1e-6) - get_opposide_dist(image_distance)) * 1e3))
            log.info("    Blur circle  (CoC)          {:8.2f} µm".format(coc_step))

            log.info("Nominal subject distance {:8.2f} mm".format(subject_distance * 1e3))
            log.info("Nominal image distance   {:8.2f} mm".format(image_distance * 1e3))

            # def curvefn(xvals):
            #     image_distances = get_opposide_dist(xvals)
            #     step = longitude_defocus_step_um * 1e-6
            #     return fitfn(image_distances, fit_peak_y, image_distance, fitted_params[2] * step)
            # x_values = get_opposide_dist((x_values - fit_peak_x) * longitude_defocus_step_um * 1e-6 + image_distance)

            return est_defocus_rms_wfe_step, longitude_defocus_step_um, coc_step, image_distance, subject_distance, fit_peak_y, prysm_offset

        # def curvefn(xvals):
        #     return fitfn(xvals, fit_peak_y, 0.0, fitted_params[2] * coc_step)
        def curvefn(xvals):
            return fitfn(xvals, *fitted_params)
        # x_values = (x_values - fit_peak_x) * coc_step

        # fit_peak_x = 0  # X-values have been normalised around zero

        fit_y = curvefn(x_values)
        errorweights = np.clip((y_values - y_values.max() * 0.8), 0.000001, 1.0)**1
        mean_abs_error = np.average(np.abs(fit_y - y_values), weights=errorweights)
        mean_abs_error_rel = mean_abs_error / highest_data_y

        log.debug("RMS fit error (normalised 1.0): {:.3f}".format(mean_abs_error_rel))
        # print(mean_abs_error_rel)
        if mean_abs_error_rel > 0.09:
            errorstr = "Very high fit error: {:.3f}".format(mean_abs_error_rel)
            log.warning(errorstr)
            print("{}, {}, {}, {}".format(x, y, freq, axis))
            if PLOT_ON_FIT_ERROR:
                plt.plot(x_values, y_values)
                plt.show()
            pos = FocusOb(fit_peak_x, fit_peak_y, interp_fn, curvefn)
            raise FitError(errorstr, fitpos=pos)
        elif mean_abs_error_rel > 0.05:
            errorstr = "High fit error: {:.3f}".format(mean_abs_error_rel)
            log.warning(errorstr)
            # print(x, y, freq, axis)
            # plt.plot(x_values, y_values)
            # plt.show()
            if strict:
                errorstr = "Strict mode, fit aborted".format(mean_abs_error_rel)
                log.warning(errorstr)
                pos = FocusOb(fit_peak_x, fit_peak_y, interp_fn, curvefn)
                raise FitError(errorstr, fitpos=pos)

        if plot or 0 and 0.6 < calc_image_height(x, y) < 0.7:
            # print(x,y,freq, axis, fit_peak_x)
            # Plot original data
            plt.plot(x_values, y_values, '.', marker='s', color='forestgreen', label="Original data points", zorder=11)
            plt.plot(x_values, y_values, '-', color='forestgreen', alpha=0.3, label="Original data line", zorder=-1)

            # Plot fit curve
            x_plot = np.linspace(x_values.min(), x_values.max(), 100)
            y_gaussplot = curvefn(x_plot)
            plt.plot(x_plot, y_gaussplot, color='red', label="Gaussian curve fit", zorder=14)
            # plt.plot(x_values, errorweights / errorweights.max() * y_values.max(), '--', color='gray', label="Sanity checking weighting")

            # Plot interpolation curve
            y_interpplot = interp_fn(x_plot)
            # plt.plot(x_plot, y_interpplot, color='seagreen', label="Interpolated quadratic spline fit", zorder=3)

            # Plot weights
            # x_label = "Field/image number (focus position)"
            plt.plot(x_values, weights * fit_peak_y, '--', color='royalblue', label="Curve fit weighting", zorder=1)
            plt.xlabel(self.focus_scale_label)
            plt.ylabel("Spacial frequency response")
            plt.title("SFR vs focus position")
            plt.legend()
            if show:
                plt.show()
        ob = FocusOb(fit_peak_x, fit_peak_y, interp_fn, curvefn)
        ob.x_loc = x
        ob.y_loc = y
        return ob

    def attempt_to_calibrate_focus(self, x=IMAGE_WIDTH/2, y=IMAGE_HEIGHT/2, freq=AUC, plot=False,
                                   unit=FOCUS_SCALE_COC, pixelsize=DEFAULT_PIXEL_SIZE, posh=True):
        tup = self.find_best_focus(x, y, LOWAVG if posh else LOWAVG, MERIDIONAL, plot=plot,
                                   _return_step_data_only=True, _step_estimation_posh=posh)
        if unit is not None:
            est_defocus_rms_wfe_step, longitude_defocus_step_um, coc_step, image_distance, subject_distance, _ = tup

            pos = self.find_best_focus(x, y, freq, MERIDIONAL)
            log.debug("Best focus index {}".format(pos.focuspos))
            old_x = np.arange(0, len(self.fields))
            if unit == FOCUS_SCALE_COC:
                step = coc_step
                self.focus_scale_label = FOCUS_SCALE_COC
            elif unit == FOCUS_SCALE_COC_PIXELS:
                step = coc_step / pixelsize * 1e-6
                self.focus_scale_label = FOCUS_SCALE_COC_PIXELS
            elif unit == FOCUS_SCALE_RMS_WFE:
                step = est_defocus_rms_wfe_step
                self.focus_scale_label = FOCUS_SCALE_RMS_WFE
            elif unit == FOCUS_SCALE_FOCUS_SHIFT:
                step = longitude_defocus_step_um
                self.focus_scale_label = FOCUS_SCALE_FOCUS_SHIFT
            else:
                raise ValueError("Unknown units")
            new_x = (old_x - pos.focuspos) * step
            self._focus_data = new_x

    @property
    def focus_data(self):
        if self._focus_data is None:
            return np.arange(len(self.fields))
        return self._focus_data

    @property
    def scaled_focus_data(self):
        if self.focus_scaling_fn is None:
            return self.focus_data
        return self.focus_scaling_fn(np.array(self.focus_data))

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

    def plot_best_sfr_vs_freq_at_point(self, x, y, axis=MEDIAL, x_values=None, secondline_fn=None, show=True, ax=None, fig=None):
        if ax is None:
            ax = plt.gca()
            fig = ax.figure
        if x_values is None:
            x_values = RAW_SFR_FREQUENCIES[:32]
        y = [self.find_best_focus(x, y, f, axis, plot=True).sharp for f in x_values]
        print(y)
        ax.plot(x_values, y, marker='s')
        ax.set_ylim(0,1)
        if secondline_fn:
            ax.plot(x_values, secondline_fn(x_values))
        if show:
            plt.show()

    def plot_sfr_vs_freq_at_point_for_each_field(self, x, y, axis=MEDIAL, waterfall=False):
        fig = plt.figure()
        if waterfall:
            ax = fig.add_subplot(111, projection='3d')
        else:
            ax = fig.add_subplot(111)
        freqs = np.concatenate((RAW_SFR_FREQUENCIES[:32:2], [AUC]))
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
                         label=label, color=colour, alpha=1.0 if freq == AUC else 0.9)
        if waterfall:
            ax.set_xlabel("Spacial Frequency (cy/px")
            ax.set_ylabel("Focus position")
            ax.set_zlabel("SFR (dB - log scale)")
            ax.set_zlim(-3, 0)
        else:
            ax.set_xlabel("Focus Position")
            ax.set_ylabel("SFR (dB (log scale))")
            ax.set_ylim(-3,0)
            ax.legend()

        ax.set_title("SFR vs Frequency for {}".format(self.exif.summary))
        # ax.legend()
        plt.show()

    def get_peak_sfr(self, x=None, y=None, freq=AUC, axis=BOTH_AXES, plot=False, show=False):
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
            best = -np.inf, "", 0, 0, 0
            for axis in [SAGITTAL, MERIDIONAL]:
                print("Testing axis {}".format(axis))

                if x is None or y is None:
                    ob = self.find_sharpest_location(freq, axis)
                    x = ob.x_loc
                    y = ob.y_loc
                else:
                    ob = self.find_best_focus(x, y, freq, axis)
                focuspos = ob.focuspos
                if ob.sharp > best[0]:
                    best = ob.sharp, axis, x, y, focuspos
            axis = best[1]
            focuspos = best[4]
            x = best[2]
            y = best[3]
            print("Found best point {:.3f} on {} axis at ({:.0f}, {:.0f}".format(best[0], axis, x, y))
        else:
            if x is None or y is None:
                ob = self.find_sharpest_location(freq, axis)
                x = ob.x_loc
                y = ob.y_loc
            else:
                ob = self.find_best_focus(x, y, freq, axis)
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
        return FFTPoint(rawdata=data_sfr)

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

    def find_sharpest_location(self, freq=AUC, axis=MEDIAL, detail=1.4):
        gridit, numparr, x_values, y_values = self.get_grids(detail=detail)
        heights = numparr.copy()
        focusposs = numparr.copy()
        # axes = numparr.copy()
        searchradius = 0.15
        lastsearchradius = 0.0
        while searchradius < 0.5:
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

    def estimate_wavefront_error(self, max_fnumber_error=0.33):
        f_range = RAW_SFR_FREQUENCIES[:30]
        # data_sfr = self.get_peak_sfr(freq=opt_freq, axis=BOTH_AXES).raw_sfr_data[:]
        data_sfr = self.get_peak_sfr(IMAGE_WIDTH/2, IMAGE_HEIGHT/2, axis=BOTH_AXES).mtf[:30]
        data_sfr2 = self.find_sharpest_raw_points_avg_sfr()[:30] * self.fields[0].points[0].calibration[:30]
        # data_sfr = -0.2 + data_sfr * 1.2
        data_mean = np.mean(data_sfr)

        plt.plot(f_range, data_sfr, label="Lens SFR Through interp")
        # plt.plot(f_range, data_sfr2, label="Lens SFR raw point")

        diff = diffraction_mtf(f_range, self.exif.aperture)

        # plt.plot(f_range, diff, label="Diffraction fn")

        print(data_mean)
        print(np.mean(diff))

        def prysmsfr(fin, z11, fstop, plot=False):
            # old_z11 = z11
            # z11 = np.abs(z11)
            pupil = prysm.NollZernike(z22=z11, dia=10, norm=True, wavelength=0.575)
            m = prysm.MTF.from_pupil(pupil, efl=10*fstop)
            modelmtf = m.exact_xy(fin / DEFAULT_PIXEL_SIZE * 1e-3)
            test_mean = np.mean(modelmtf)
            if plot:
                plt.plot(f_range, modelmtf, label="Model SFR WFE Z11 {:.3f}λ f/{:.2f}".format(z11, fstop))
            print(z11, test_mean)
            return modelmtf

        fchange = np.clip(2 ** (max_fnumber_error / 2.0), 1.0001, 10.0)
        params, _ = optimize.curve_fit(prysmsfr, f_range, data_sfr, bounds=([0, self.exif.aperture/fchange], [0.22, self.exif.aperture*fchange]), p0=[0.1, self.exif.aperture])
        wfr, fstop = params
        prysmsfr(f_range, wfr, fstop, True)
        prysmsfr(f_range, 0, fstop, True)
        stops_out = np.log(fstop / self.exif.aperture) / np.log(2)
        print("Est. F number inaccuracy vs exif {:.2f} stops".format(stops_out))
        print("Z11 {:.3f} Fstop {:.2f}".format(wfr, fstop))
        # exit()
        # bounds=bounds, sigma=sigmas, ftol=1e-6, xtol=1e-6,
        #                                       p0=initial)
        # wfr = optimize.bisect(prysmsfr, 0.0, 0.5, xtol=1e-3, rtol=1e-3)
        # prysmsfr(wfr, plot=True)
        print("Wavefront error {:.3f}".format(wfr))
        print("Wavefront error {:.3f} f/2.8 equivalent".format(wfr / 2.8 * self.exif.aperture))
        print("Strehl {:.3f}".format(data_sfr.mean()/diff.mean()))
        plt.legend()
        plt.ylim(0, 1)
        plt.xlim(0, 0.5)
        plt.xlabel("Spacial Frequency (cy/px)")
        plt.ylabel("'Calibrated' MTF")
        plt.title(self.exif.summary)
        plt.show()

    def build_calibration(self, fstop=None, opt_freq=AUC, plot=True, writetofile=False, use_centre=False):
        """
        Assume diffraction limited lens to build calibration data

        :param fstop: Taking f-stop
        :param plot: Plot if True
        :param writetofile: Write to calibration.csv file
        :return: Numpy array of correction data
        """

        if fstop is None:
            fstop = self.exif.aperture
        f_range = RAW_SFR_FREQUENCIES[:40]
        if not use_centre:
            data_sfr = self.get_peak_sfr(freq=opt_freq, axis=BOTH_AXES).raw_sfr_data[:40]
        else:
            data_sfr = self.get_peak_sfr(x=IMAGE_WIDTH/8*7, y=IMAGE_HEIGHT/8*7, freq=opt_freq, axis=BOTH_AXES).raw_sfr_data[:40]
        # data_sfr = self.find_sharpest_raw_points_avg_sfr()[:40]

        if self.use_calibration:
            if writetofile:
                # pass
                raise AttributeError("Focusset must be loaded without existing calibration")
            else:
                log.warning("Existing calibration loaded (will compare calibrations)")
        # Get best AUC focus postion


        # if not writetofile:
        #     data_sfr *= self.base_calibration[:40]

        diffraction_sfr = diffraction_mtf(f_range, fstop/1.00)  # Ideal sfr

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
        return data_sfr, diffraction_sfr, correction

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
            plot.title = "Compromise focus flat-field " + self.exif.summary
            plot.xlabel = "x image location"
            plot.ylabel = "y image location"
            plot.plot(plot_type)
        return FocusOb(peak_focus_pos, average[int(peak_focus_pos * 10)], interpfn)

    def get_mtf_vs_image_height(self, analysis_pos=None, freq=AUC, detail=0.5, axis=MEDIAL, posh=False):
        gridit, numpyarr, x_vals, y_vals = self.get_grids(detail=detail)
        heights = numpyarr.copy()
        arrs = []
        # if axis == MEDIAL:
        #     axis = [SAGITTAL, MERIDIONAL]
        #     axis = [SAGITTAL, MERIDIONAL]
        # else:
        #     axis = [axis]
        fns = []
        arr = numpyarr.copy()
        arrs.append(arr)
        for nx, ny, x, y in gridit:
            heights[ny, nx] = calc_image_height(x, y)
            if analysis_pos is None:
                try:
                    ob = self.find_best_focus(x, y, freq, axis)
                    arr[ny, nx] = ob.sharp
                    # if ob.sharp > 0.95:
                    #     print(ob.sharp, x, y, heights[ny, nx], loopaxis)
                except FitError as e:
                    arr[ny, nx] = np.nan
            else:
                arr[ny, nx] = self.interpolate_value(x, y, analysis_pos.focuspos, freq, axis, posh=posh)

        def fn(hei, width=0.2):
            flatheights = heights.flatten()
            sharps = arr.flatten()
            weights = 1.0001 - np.clip(np.abs(hei - flatheights) / width, 0.0, 1.0)
            return np.average(sharps, weights=weights)
        return fn
        fns.append(fn)
        if len(axis) == 1:
            return fns[0]
        else:
            def combine(hei, width=0.25):
                return (fns[0](hei, width) + fns[1](hei, width)) * 0.5
            return combine



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
                    if show_diffraction is True:
                        show_diffraction = self.exif.aperture
                        print(show_diffraction)
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

    def get_wavefront_data_path(self, seed=None):
        path = os.path.join(self.rootpath, "wavefront_results/")
        return path

    def get_old_wavefront_data_path(self, seed=None):
        path = os.path.join(self.rootpath, "wavefront_data.csv")
        return path

    def read_wavefront_data(self, overwrite, read_old=False, copy_old=True, read_autosave=True, x_loc=None, y_loc=None):
        wfl = read_wavefront_data(self.get_wavefront_data_path(), read_autosave=read_autosave, x_loc=x_loc, y_loc=y_loc)
        if overwrite:
            self.wavefront_data = wfl
        return wfl

    def copy_old_wavefront_data_to_new(self):
        self.read_wavefront_data(overwrite=True, copy_old=False, read_old=True)
        for item in self.wavefront_data:
            save_wafefront_data(self.get_wavefront_data_path(), [item], suffix="copied")

    # @staticmethod
    # def intepolate_value_helper(fieldnumber, x, y, freq, axis):
    #     return float(gfs2.fields[fieldnumber].interpolate_value(x, y, freq, axis))

    # @classmethod
    # def find_best_focus_helper(cls, args):
    #     if args[6] % 50 == 0:
    #         print("Finding best focus for location {} / {} in {}".format(args[6], args[7], args[5]))
    #     ob = globalfocusset.find_best_focus(*args[2:6])
    #
        # return args[0], args[1], float(ob.sharp), float(ob.focuspos)


def clear_numbered_autosaves(path):
    for entry in os.scandir(path):
        if 'autosave' in entry.name.lower() and 'csv' in entry.name.lower():
            digits = [char for char in entry.name if char.isdigit()]
            if len(digits):
                os.remove(entry.path)



def save_focus_jitter(rootpath, focus_values, code_version=0):
    filepath = os.path.join(rootpath, "focus_positions.csv")
    sfr_filenames = []
    with os.scandir(rootpath) as it:
        for entry in it:
            if entry.name.lower().endswith(".sfr"):
                digits = "".join((char for char in entry.name if char.isdigit()))
                sfr_filenames.append((int(digits), entry.name, entry.path))

    sfr_filenames.sort()
    print(sfr_filenames)
    print(focus_values)

    if len(sfr_filenames) != len(focus_values):
        log.warning("Focus value array does not correspond to number of .sfr files in path")

    with open(filepath, 'w') as file:
        writer = csv.writer(file, delimiter=" ", quotechar='"')
        file.writelines(("# ID Filename Estimated_Focus_Position\n",))
        writer.writerow(["Code_version", CURRENT_JITTER_CODE_VERSION])
        for (num, name, path), focus_position in zip(sfr_filenames, focus_values):
            writer.writerow((num, name, focus_position))


def estimate_focus_jitter(path=None, data_in=None, plot=1):
    code_version = 2
    if data_in is not None:
        iter = [(None, None, None)]
        data = data_in
    else:
        focusset = FocusSet(rootpath=path, include_all=True, use_calibration=True, load_focus_data=False)
        xvals = np.linspace(IMAGE_WIDTH / 4, IMAGE_WIDTH * 5/8, 4)[:]
        yvals = np.linspace(IMAGE_HEIGHT / 4, IMAGE_HEIGHT * 5/8, 4)[:]
        axes = [SAGITTAL, MERIDIONAL]
        iter = zip(*(_.flatten() for _ in np.meshgrid(xvals, yvals, axes)))  # Avoid nested loops

    all_errors = []
    freqs = np.arange(2, 19, 2) / 64

    for x, y, axis in iter:
        if data_in is None:
            y = IMAGE_HEIGHT / IMAGE_WIDTH * x
            data = FocusSetData()
            datalst = []
            for freq in freqs:
                pos = focusset.get_interpolation_fn_at_point(x, y, freq, axis)
                datalst.append(pos.sharp_data)

            data.merged_mtf_values = np.array(datalst)
            data.focus_values = pos.focus_data
        nfv = len(data.focus_values)
        if nfv < 8:
            log.warning("Focus jitter estimation may not be reliable with {} samples".format(nfv))
            if nfv < 6:
                raise ValueError("More focus samples needed")

        freq_ix = 0
        nom_focus_values = data.focus_values
        non_inc = nom_focus_values[1] - nom_focus_values[0]

        xmods = []
        grads = []

        lowcut = 0.2

        while freq_ix < (data.merged_mtf_values.shape[0]):
            low_freq_average = data.merged_mtf_values[freq_ix, :]
            if low_freq_average.max() < 0.35:
                break
            print("Using frequency index {}".format(freq_ix))
            freq_ix += 1

            min_modulation = low_freq_average.min()
            print("Min modulation", min_modulation)
            print("Mean modulation", low_freq_average.mean())
            if min_modulation < 0.4:
                maxpoly = 4
            elif min_modulation < 0.8:
                maxpoly = 4
            else:
                maxpoly = 2

            rawpoly = np.polyfit(nom_focus_values, low_freq_average, 4)
            rawpolygrad = np.polyder(rawpoly, 1)
            gradmetric = np.clip(np.abs(np.polyval(rawpolygrad, nom_focus_values)), 0, 0.5 / non_inc)
            valid = np.logical_and(low_freq_average > lowcut, gradmetric > 0.012 / non_inc)
            # print(gradmetric)
            # print(valid)
            valid_x = data.focus_values[valid]
            valid_y = low_freq_average[valid]

            if len(valid_x) < 5:
                continue

            poly_order = min(maxpoly, len(valid_x) - 2)  # Ensure is overdetmined

            # print("Using {} order polynomial to find focus errors with {} samples".format(poly_order, len(valid_x)))

            poly = np.polyfit(nom_focus_values[low_freq_average > lowcut], low_freq_average[low_freq_average > lowcut],
                              poly_order)

            def cost(xmod):
                smoothline = np.polyval(poly, valid_x + xmod)
                # modcost = ((1.0 - cost_gradmetric)**2 * xmod**2).mean() * 10e-5
                modcost = 0
                return ((valid_y - smoothline) ** 2).mean() + modcost

            mul = 3.0
            bounds = [(-non_inc * mul, non_inc * mul)] * len(valid_x)
            opt = optimize.minimize(cost, np.zeros(valid_x.shape), method='L-BFGS-B', bounds=bounds)

            xmod = []
            grad = []
            ix = 0
            # print(valid)
            # print(valid_x)
            # print(opt.x)
            for v, gradmetric_ in zip(valid, gradmetric):
                if v:
                    xmod.append(opt.x[ix])
                    grad.append(gradmetric_)
                    ix += 1
                else:
                    xmod.append(np.nan)
                    grad.append(0)
            xmod = np.array(xmod)
            xmods.append(xmod)
            grads.append(grad)

            # print("Freq focus errors", xmod)

            if plot >= 2 and np.abs(xmod[np.isfinite(xmod)]).sum() > 0:
                plt.plot(nom_focus_values, low_freq_average, label='Raw Data {}'.format(freq_ix))
                # plt.plot(nom_focus_values, validate_freq_average, label="Raw Data (2, validation)")
                plt.plot(nom_focus_values, np.polyval(rawpolygrad, nom_focus_values) * 10 * non_inc,
                         label="Raw Gradient")
                plt.plot(valid_x, np.polyval(poly, valid_x + opt.x) + 0.01, label="Polyfit after optimisation")
                # plt.plot(nom_focus_values, np.polyval(validate_poly, nom_focus_values + xmod)+0.01, label="Data 2 polyfit after optimisation")
                plt.plot(valid_x, np.polyval(poly, valid_x) + 0.01, label="Valid Polyfit")
                plt.plot(nom_focus_values, np.polyval(rawpoly, nom_focus_values) + 0.01, label="Raw Polyfit")
                # plt.plot(nom_focus_values, np.polyval(validate_poly, nom_focus_values), label="Data2 polyfit")
                plt.plot(nom_focus_values, gradmetric * 10 * non_inc, label="Grad valid metric")
                plt.plot(nom_focus_values, xmod, '-', marker='s', label="Estimated focus errors")
                if 'focus_errors' in data.hints:
                    plt.plot(nom_focus_values, data.hints['focus_errors'], '-', marker='v',
                             label="Hint data focus errors")
                plt.legend()
                plt.show()

        # print(xmods)
        # print(grads)
        gradsum = np.sum(np.array(grads), axis=0)
        # xmodmean = np.nansum(np.array(xmods) * np.array(grads), axis=0) / gradsum
        xmodmean = np.nanmean(np.array(xmods), axis=0)

        if np.isfinite(xmodmean).sum() == 0:
            raise ValueError("No data! available")
        # xmodmean[gradsum < 0.] = 0
        # print(xmods)
        # print(xmodmean)
        if np.nansum(xmodmean) != 0:
            xmodmean_zeroed = xmodmean.copy()
            xmodmean_zeroed[~np.isfinite(xmodmean)] = 0
            error_poly = np.polyfit(nom_focus_values, xmodmean_zeroed, 2)
            roots = np.roots(np.polyder(error_poly, 1))
            peakerrors = np.polyval(error_poly, roots)
            polyerrormax = np.abs(peakerrors).max()
            if plot >= 2:
                plt.plot(nom_focus_values, np.polyval(error_poly, nom_focus_values), label="Error polyfit")

            if polyerrormax > 0.06:
                log.warning("Warning errors may not be random")
            elif polyerrormax > 0.2:
                raise ValueError("Error result doesn't appear random, please use more samples")

        if plot >= 2:
            # plt.plot(nom_focus_values, low_freq_average, label='Raw Data')
            # plt.plot(nom_focus_values, validate_freq_average, label="Raw Data (2, validation)")
            # plt.plot(nom_focus_values, np.polyval(rawpolygrad, nom_focus_values) * 10, label="Raw Gradient")
            # plt.plot(valid_x, np.polyval(poly, valid_x + opt.x) + 0.01, label="Polyfit after optimisation")
            # plt.plot(nom_focus_values, np.polyval(validate_poly, nom_focus_values + xmod)+0.01, label="Data 2 polyfit after optimisation")
            # plt.plot(valid_x, np.polyval(poly, valid_x) + 0.01, label="Polyfit")
            # plt.plot(nom_focus_values, np.polyval(validate_poly, nom_focus_values), label="Data2 polyfit")
            # plt.plot(nom_focus_values, gradmetric, label="Grad valid metric")
            plt.plot(nom_focus_values, np.mean(data.merged_mtf_values, axis=0), label="AUC")
            plt.plot(nom_focus_values + xmodmean, np.mean(data.merged_mtf_values + 0.02, axis=0), label="AUCfix")
            plt.plot(nom_focus_values, xmodmean, '-', marker='s', label="Estimated focus errors")
            if 'focus_errors' in data.hints:
                plt.plot(nom_focus_values, data.hints['focus_errors'], '-', marker='v', label="Hint data focus errors")

            plt.legend()
            plt.show()
        all_errors.append(xmodmean)

    for errs in all_errors:
        if plot >= 1 or 1:
            plt.plot(data.focus_values, errs, marker='s')
        for focus, err in zip(data.focus_values, errs):
            print("Offset {:.3f} at position {}".format(err, focus))


    # allxmodmean = np.nanmean(all_errors, axis=0)
    # allxmodmean[~np.isfinite(allxmodmean)] = 0

    # Remove outliers and take mean
    all_errors_ay = np.array(all_errors)
    allxmodmean = np.zeros(all_errors_ay.shape[1])
    for ix in range(len(allxmodmean)):
        errors_at_position = all_errors_ay[np.isfinite(all_errors_ay[:, ix]), ix]
        if len(errors_at_position) >= 3:
            error_high = errors_at_position.max()
            error_low = errors_at_position.min()
            no_outliers = errors_at_position[np.logical_and(errors_at_position > error_low,
                                                            errors_at_position < error_high)]
            if len(no_outliers) > 0:
                1+1
                allxmodmean[ix] = no_outliers.mean()
        elif 0 > len(errors_at_position) >= 2:
            allxmodmean[ix] = errors_at_position.mean()


    print(allxmodmean)
    # plt.show()
    # exit()

    new_focus_values = data.focus_values + allxmodmean
    if plot >= 1:
        plt.plot(data.focus_values, allxmodmean, color='black', marker='v')
        plt.show()
    if data_in is None:
        save_focus_jitter(rootpath=path, focus_values=new_focus_values, code_version=code_version)
        return new_focus_values
    else:
        data_in.focus_values = new_focus_values
        # datain.jittererr = (((data.hints['focus_errors'] - xmodmean))**2).mean()
        # datain.jittererrmax = (np.abs(data.hints['focus_errors'] - xmodmean)).max()
        # datain.hintjit = ((data.hints['focus_errors'])**2).max()
        return data_in


def scan_path(path, make_dir_if_absent=False, find_autosave=False, x_loc=None, y_loc=None):
    if os.path.exists(path):
        autosave_entry = None
        existing_max = -1
        highentry = None
        if x_loc is not None:
            xfindstr = "x{}".format(x_loc)
            yfindstr = "y{}".format(y_loc)
        else:
            xfindstr, yfindstr = "", ""
        for entry in os.scandir(path):
            if entry.is_file:
                if not xfindstr in entry.name or not yfindstr in entry.name:
                    continue
                split = [_ for _ in entry.name.split(".") if len(_) > 1]
                filenumber = -np.inf
                for string in split:
                    if string[0].lower() == "n":
                        digits = string[1:]
                        try:
                            filenumber = int(digits)
                        except ValueError:
                            pass
                if filenumber > existing_max:
                    existing_max = filenumber
                    highentry = entry
                elif "autosave" in entry.name.lower():
                    autosave_entry = entry

        if find_autosave and autosave_entry is not None:
            return autosave_entry, existing_max
        if existing_max >= 0:
            return highentry, existing_max
        return None, -1
    else:
        if make_dir_if_absent:
            os.makedirs(path)
            return None, -1
        else:
            raise FileNotFoundError()


def read_wavefront_data(path=None, focusset_path=None, read_autosave=True, x_loc=None, y_loc=None):
    if path is None:
        path = os.path.join(focusset_path, "wavefront_results")
    entry, number = scan_path(path, make_dir_if_absent=True, find_autosave=read_autosave, x_loc=x_loc, y_loc=y_loc)
    if entry is not None:
        lst = read_wavefront_file(entry.path)
        return lst

    return [("", {})]


def save_wafefront_data(path, wf_data, suffix="", quiet=False):
    # path = os.path.join(rootpath, "wavefront_results/")
    _, existing_max = scan_path(path, make_dir_if_absent=True)

    if "autosave" in suffix:
        save_filename = "wavefront_data.{}.csv".format(suffix)
    elif suffix:
        save_filename = "wavefront_data.{}.n{}.csv".format(suffix, existing_max + 1)
    else:
        save_filename = "wavefront_data.n{}.csv".format(existing_max + 1)

    save_filepath = os.path.join(path, save_filename)

    with open(save_filepath, 'w') as file:
        writer = csv.writer(file, delimiter=" ", quotechar='"')
        for item in wf_data:
            writer.writerow((item[0],))
            for key, value in item[1].items():
                if type(value) is list or type(value) is tuple:
                    writer.writerow([key] + list(value))
                else:
                    writer.writerow([key, value])
        writer.writerow([""])
    if not quiet:
        print("Saved to '{}".format(save_filepath))