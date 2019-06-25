import colorsys
import math
import csv

import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from scipy import interpolate, optimize
from scipy import fftpack, signal

from lentil.constants_utils import *
from lentil.plot_utils import FieldPlot
from lentil.sfr_point import FFTPoint

FUNCSTORE = None


class NotEnoughPointsException(Exception):
    pass


class SFRField:
    """
    Represents entire image field of SFRPoints for a single image
    """

    def __init__(self, points=None, pathname=None, calibration=None, smoothing=0, exif=None, load_complex=False, filenumber=-1):
        """

        :param points: Iterable of SFRPoints, order not important
        :param pathname: Path to MTF Mapper edge_sfr_values.txt file to parse
        :param calibration: calibration data array
        """
        self.filenumber = filenumber
        if points is None:
            points = []
            with open(pathname, 'r') as sfrfile:
                csvreader = csv.reader(sfrfile, delimiter=' ', quotechar='|')

                for row in csvreader:
                    try:
                        points.append(FFTPoint(row, calibration=calibration, filenumber=self.filenumber))
                    except ValueError:
                        if load_complex:
                            # Need to keep points in order to align with esf file
                            points.append(None)
            if exif is not None:
                self.exif = exif
            else:
                self.read_exif(pathname)

        if load_complex:
            if type(load_complex) is not str:

                esfpath1 = pathname[:-3] + "esf"
                esfpath2 = os.path.join(os.path.split(pathname)[0], "raw_esf_values.txt")
                for esfpath in [esfpath1, esfpath2]:
                    if not os.path.exists(esfpath):
                        continue
                    with open(esfpath, 'r') as esffile:
                        csvreader = csv.reader(esffile, delimiter=' ', quotechar='|')
                        esfs = []
                        for row in csvreader:
                            try:
                                esfs.append(row)
                            except ValueError:
                                pass
            if len(points) != len(esfs):
                raise Exception("ESF file has different number of points compared to SFR file "
                                "({} vs {})".format(len(esfs), len(points)))
            points = process_esfs(esfs, points)
            self.has_phase = True
        else:
            self.has_phase = False

        self.points = points
        enough_m = len(self.get_subset(MERIDIONAL)) > 20
        enough_s = len(self.get_subset(SAGITTAL)) > 20

        if enough_m and not enough_s:
            raise NotEnoughPointsException("Only {:.0f} sagittal points!".format(len(self.get_subset(SAGITTAL))))
        if enough_s and not enough_m:
            raise NotEnoughPointsException("Only {:.0f} meridional points!".format(len(self.get_subset(MERIDIONAL))))
        if not enough_s and not enough_m:
            raise NotEnoughPointsException("Only {:.0f} points!"
                                           .format(len(self.get_subset(MERIDIONAL)) + len(self.get_subset(SAGITTAL))))

        # Set up cache for numpy point data
        np_axis = {}
        np_axis['np_x'] = None
        np_axis['np_y'] = None
        np_axis['np_sfr'] = None
        np_axis['np_sfr_freq'] = None
        np_axis['np_value'] = None
        self.np_dict_cache = {MEDIAL: np_axis,
                              SAGITTAL: np_axis.copy(),
                              MERIDIONAL: np_axis.copy(),
                              SAGITTAL_REAL: np_axis.copy(),
                              MERIDIONAL_REAL: np_axis.copy(),
                              SAGITTAL_IMAG: np_axis.copy(),
                              MERIDIONAL_IMAG: np_axis.copy()}

        self.smoothing = smoothing
        self.bounds_tuple_cache = {}

    @property
    def saggital_points(self):
        """

        :return: Returns list of all saggital edge points in field
        """
        return [point for point in self.points if point.is_saggital]

    @property
    def meridional_points(self):
        """

        :return: Returns list of all meridional edge points in field
        """
        return [point for point in self.points if not point.is_saggital]

    def get_subset(self, axis):
        """
        Returns list of all points on chosen axis (or both)
        :param axis: constant SAGGITAL or MERIDIONAL or MEDIAL
        :return: list of points
        """
        return [point for point in self.points if point.is_axis(axis)]

    def get_avg_mtf50(self):
        """
        Calculates Unweighted arithmetic mean of MTF50 for all points in cycles/px
        :return: Average MTF50 for all points (in cycles/px)
        """
        return sum((point.mtf50 for point in self.points)) / len(self.points)

    def get_point_range(self, axis=MEDIAL):
        """
        Find extremeties of point locations
        :return: Tuple (x_min, y_min, x_max, y_max)
        """
        if axis in self.bounds_tuple_cache:
            return self.bounds_tuple_cache[axis]
        lst = self.get_subset(axis)
        x_min = min((point.x for point in lst))
        x_max = max((point.x for point in lst))
        y_min = min((point.y for point in lst))
        y_max = max((point.y for point in lst))
        tup = x_min, y_min, x_max, y_max
        self.bounds_tuple_cache[axis] = tup
        return tup

    def build_axis_points(self, x_len=20, y_len=20, axis=MEDIAL):
        x_min, y_min, x_max, y_max = self.get_point_range(axis)
        # x_values = np.arange(x_min, x_max, (x_max - x_min) / x_len)
        # y_values = np.arange(y_min, y_max, (y_max - y_min) / y_len)
        x_values = np.linspace(x_min, x_max, x_len)
        y_values = np.linspace(y_min, y_max, y_len)
        return x_values, y_values

    def get_grids(self, detail=0.3):
        x_values, y_values = self.build_axis_points(24. * detail, 16. * detail)
        mesh = np.meshgrid(np.arange(len(x_values)), np.arange(len(y_values)))
        mesh2 = np.meshgrid(x_values, y_values)
        meshes = [grid.flatten() for grid in (mesh+mesh2)]
        return list(zip(*meshes)), np.zeros((len(y_values), len(x_values))), x_values, y_values

    def get_simple_interpolation_fn(self, axis=MEDIAL):
        """
        Returns a simple spline interpolation callable fitted across the whole field.

        :param axis: Pass constant SAGGITAL, MERIDIONAL, or MEDIAL
        :return: Scipy callable which accepts x and y positions and provides interpolated value.
        """
        lst = []
        for point in self.get_subset(axis):
            lst.append((point.x, point.y, point.mtf50))
        x_lst, y_lst, z_lst = zip(*lst)

        fn = interpolate.SmoothBivariateSpline(x_lst, y_lst, z_lst, kx=2, ky=2, s=float("inf"))
        return fn

    def interpolate_value(self, x, y, freq=DEFAULT_FREQ, axis=MEDIAL, complex_type=COMPLEX_CARTESIAN):
        """
        Provides an interpolated MTF/SFR for chosen point in field and axis. Uses a locally weighted polynomial plane.

        Pass -1 as frequency for MTF50 results in cy/px (not for complex OTF)

        :param x: x location
        :param y: y location
        :param axis: Pass constants SAGGITAL, MERIDIONAL, or MEDIAL
        :param freq: spacial frequency to return, -1 for mtf50
        :return: interpolated cy/px at specified frequency, or mtf50 frequency if -1 passed
        """

        if axis == SAGITTAL_COMPLEX:
            real_out = self.interpolate_value(x, y, freq, SAGITTAL_REAL)
            imaj_out = self.interpolate_value(x, y, freq, SAGITTAL_IMAG)
            return convert_complex((real_out, imaj_out), complex_type)
        elif axis == MERIDIONAL_COMPLEX:
            real_out = self.interpolate_value(x, y, freq, MERIDIONAL_REAL)
            imaj_out = self.interpolate_value(x, y, freq, MERIDIONAL_IMAG)
            return convert_complex((real_out, imaj_out), complex_type)

        if self.np_dict_cache[axis]['np_x'] is None or self.np_dict_cache[axis]['np_sfr_freq'] != freq:
            lst = []
            for point in self.get_subset(axis):
                if axis in COMPLEX_AXES:
                    real, imaj = point.get_complex_freq(freq, complex_type=COMPLEX_CARTESIAN_REAL_TUPLE)
                    if axis in REAL_AXES:
                        lst.append((point.x, point.y, real))
                    elif axis in IMAG_AXES:
                        lst.append((point.x, point.y, imaj))
                else:
                    lst.append((point.x, point.y, point.get_freq(freq)))
            x_arr, y_arr, z_arr = zip(*lst)
            x_arr = np.array(x_arr)
            y_arr = np.array(y_arr)
            z_arr = np.array(z_arr)
            self.np_dict_cache[axis]['np_x'] = x_arr
            self.np_dict_cache[axis]['np_y'] = y_arr
            self.np_dict_cache[axis]['np_sfr_freq'] = freq
            self.np_dict_cache[axis]['np_value'] = z_arr
        x_arr = self.np_dict_cache[axis]['np_x']
        y_arr = self.np_dict_cache[axis]['np_y']
        z_arr = self.np_dict_cache[axis]['np_value']

        # Calculate distance of each edge location to input location on each axis
        x_distances = (x_arr - x)
        y_distances = (y_arr - y)

        distances = np.sqrt(x_distances ** 2 + y_distances ** 2)

        stack = np.vstack((x_arr, y_arr, z_arr, x_distances, y_distances, distances))

        bbox = [0, IMAGE_WIDTH, 0, IMAGE_HEIGHT]

        order = FIELD_SMOOTHING_ORDER  # Spline order

        points_wanted = min(FIELD_SMOOTHING_MIN_POINTS, len(x_arr) - 1)
        max_ratio = FIELD_SMOOTHING_MAX_RATIO

        sortidx = distances.argsort()[:points_wanted*2]

        sortedstack = stack[:, sortidx]

        sorteddist = sortedstack[5]
        min_ratio_radius = sorteddist[4] / max_ratio
        min_point_radius = sorteddist[points_wanted]

        radius = max(min_point_radius, min_ratio_radius)

        nr = sorteddist < radius

        clippedstack = sortedstack[:, nr]
        angles = np.arctan2(clippedstack[3, :], clippedstack[4, :])

        prop_of_radius = (clippedstack[5] - clippedstack[5, 0]) / (radius - clippedstack[5, 0])
        # print(prop_of_radius)
        weights = np.cos(np.clip(prop_of_radius, 1e-6, 1.0)**0.5 * np.pi) + 1.0

        func = interpolate.SmoothBivariateSpline(clippedstack[0], clippedstack[1], clippedstack[2], bbox=bbox,
                                                 w=weights, kx=order, ky=order, s=float("inf"))
        radius_range = min_point_radius / min_ratio_radius
        angle_std = np.std(angles)

        low = 40
        high = 120

        if radius_range < high:
            func_linear = interpolate.SmoothBivariateSpline(clippedstack[0], clippedstack[1], clippedstack[2],
                                                            bbox=bbox, w=weights, kx=1, ky=1, s=float("inf"))
            low_order_ratio = 1.0 - np.clip((angle_std - low) / (high - low), 0.0, 1.0)

            output = func_linear(x, y)[0][0] * low_order_ratio + func(x, y)[0][0] * (1.0 - low_order_ratio)
        else:
            output = func(x, y)[0][0]
            low_order_ratio = 0.0

        if 0:
            print(sorteddist[0], min_point_radius, min_ratio_radius, nr.sum())
        if 0:
            # print(distances)
            # print(sorteddist)
            # print(nr)
            colours = np.array([z_arr, z_arr, z_arr, z_arr]).T

            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            c = plt.cm.jet(z_arr[nr] / z_arr[nr].max())
            ax.scatter(x_arr[nr], y_arr[nr], weights, c=c, marker='.')
            # c = plt.cm.jet(weights[nr] / weights[nr].max())
            # ax.scatter(x_arr[nr], y_arr[nr], z_arr[nr], c=c, marker='.')
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            plt.show()

        if axis in COMPLEX_AXES:
            return output
        else:
            return np.clip(output, 1e-5, np.inf)

    def interpolate_otf_complex(self, x, y, freq, axis, type=COMPLEX_CARTESIAN):
        if not self.has_phase:
            raise ValueError("Field does not have any phase data, only MTF")
        if axis == SAGITTAL:
            realaxis = SAGITTAL_REAL
            imajaxis = SAGITTAL_IMAG
        elif axis == MERIDIONAL:
            realaxis = MERIDIONAL_REAL
            imajaxis = MERIDIONAL_IMAG
        else:
            raise ValueError("Axis {} is not a valid choice".format(axis))
        real = self.interpolate_otf_complex(x, y, freq, realaxis)
        imaj = self.interpolate_otf_complex(x, y, freq, imajaxis)
        return convert_complex((real, imaj), type)


    def plot(self, freq=DEFAULT_FREQ, axis=MEDIAL, plot_type=1, detail=1.0,
             show=True, ax=None, alpha=0.85):
        """
        Plots SFR/MTF values for chosen axis across field

        :param axis: Constant SAGGITAL, MERIDIONAL, or MEDIAL
        :param plot_type: 0 is 3d surface, 1 is 2d contour
        :param detail: Relative detail in plot (1.0 is default)
        :return:
        """
        gridit, z_values, x_values, y_values = self.get_grids(detail=detail)

        # fn = self.get_simple_interpolation_fn(axis)
        for x_idx, y_idx, x, y in gridit:
            if axis == MEDIAL:
                sag = self.interpolate_value(x, y, freq, SAGITTAL)
                mer = self.interpolate_value(x, y, freq, MERIDIONAL)
                z_values[y_idx, x_idx] = (sag + mer) / 2
            else:
                z_values[y_idx, x_idx] = self.interpolate_value(x, y, freq, axis)
                # print(x, y, z_values[y_idx, x_idx])

        max_z = np.amax(z_values) * 1.1
        plot = FieldPlot()
        plot.zmin = diffraction_mtf(freq, LOW_BENCHMARK_FSTOP)
        plot.zmax = diffraction_mtf(freq, HIGH_BENCHBARK_FSTOP)
        plot.xticks = x_values
        plot.yticks = y_values
        plot.zdata = z_values
        plot.yreverse = True
        # plot.set_diffraction_limits(freq=freq)
        plot.title = "Sharpness"
        if plot_type == CONTOUR2D:
            return plot.contour2d(ax=ax, show=show)
        elif plot_type == PROJECTION3D:
            return plot.projection3d(ax=ax, show=show)
        else:
            raise ValueError("Unknown plot type")

    def plot_edge_angle(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        lst = []
        for point in self.points:
            if point.is_meridional:
                continue
            lst.append((point.x, point.y, point.angle,
                        point.get_complex_freq(0.15, complex_type=COMPLEX_CARTESIAN_REAL_TUPLE)[1]))

        x_arr, y_arr, z_arr, imag_arr = zip(*lst)
        x_arr = np.array(x_arr)
        y_arr = np.array(y_arr)
        z_arr = np.array(z_arr)
        imag_arr = np.array(imag_arr)
        colours = np.array(imag_arr).T
        ax.scatter(x_arr, y_arr, z_arr, c=colours, marker='.', cmap=plt.cm.jet)
        plt.show()

    def plot_points(self, freq=0.05, axis=MEDIAL, autoscale=False, add_corners=False):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        if not autoscale:
            if axis in POLAR_AXES:
                ax.set_zlim(-np.pi, np.pi)
            else:
                ax.set_zlim(0, 1)

        if axis == MEDIAL:
            axis = (SAGITTAL, MERIDIONAL)
        else:
            axis = (axis,)
        for axis_ in axis:
            lst = []
            for point in self.get_subset(axis_):
                if axis_ in COMPLEX_AXES:
                    if axis_ in POLAR_AXES:
                        ctype = COMPLEX_POLAR_TUPLE
                    else:
                        ctype = COMPLEX_CARTESIAN_REAL_TUPLE
                    tup = point.get_complex_freq(freq, complex_type=ctype)
                    if axis_ in REAL_AXES:
                        val = tup[0]
                    elif axis_ in COMPLEX_AXES:
                        val = tup[1]
                    else:
                        raise Exception("Unsure what's going on...")
                    lst.append((point.x, point.y, val))
                else:
                    lst.append((point.x, point.y, point.get_freq(freq)))
            x_arr, y_arr, z_arr = zip(*lst)
            x_arr = np.array(x_arr)
            y_arr = np.array(y_arr)
            z_arr = np.array(z_arr)
            colours = np.array([z_arr, z_arr, z_arr, z_arr]).T
            ax.scatter(x_arr, y_arr, z_arr, c='r' if axis_ is SAGITTAL else 'b', marker='.')

        if add_corners:
            for axis_ in axis:
                for corners in [1.0, -1.0]:
                    x_arr = []
                    y_arr = []
                    z_arr = []
                    for h in np.linspace(0, 1, 180):
                        h = h ** 0.5
                        x = IMAGE_WIDTH * h
                        y = IMAGE_HEIGHT / 2.0 + IMAGE_HEIGHT * (h-0.5) * corners
                        z = self.interpolate_value(x, y, freq, axis_)
                        x_arr.append(x)
                        y_arr.append(y)
                        z_arr.append(z)
                    ax.plot(x_arr, y_arr, z_arr, '-',
                            color='r' if axis_ is SAGITTAL else 'b',
                            alpha=0.5)

        plt.show()

    def get_fit_errors(self, freqs=[0.1], by_percent=False, axis=MEDIAL):
        try:
            numfreqs = len(freqs)
        except TypeError:
            freqs = [freqs]
            numfreqs= 1
        points = self.get_subset(axis)
        err = np.ndarray((numfreqs, len(points)))
        fit = np.ndarray((numfreqs, len(points)))
        orig = np.ndarray((numfreqs, len(points)))
        for n_f, freq in enumerate(freqs):
            for n_point, point in enumerate(points):
                fit_val = self.interpolate_value(point.x, point.y, freq, point.axis)
                fit[n_f, n_point] = fit_val

                orig_val = point.get_freq(freq)
                orig[n_f, n_point] = orig_val

                if by_percent:
                    err[n_f, n_point] = (fit_val - orig_val) / orig_val * 100.0
                else:
                    err[n_f, n_point] = fit_val - orig_val
        return err, fit, orig

    def plot_fit_errors_2d(self, freqs=[0.1], by_percent=False, axis=MEDIAL):
        if axis == MEDIAL:
            axis = [SAGITTAL, MERIDIONAL]
        for axis in axis:
            errors, fit, orig = self.get_fit_errors(freqs, axis=axis, by_percent=by_percent)
            abs_mean = np.mean(np.abs(errors), axis=0)  # Mean of all freqs
            x = self.np_dict_cache[axis]['np_x']
            y = self.np_dict_cache[axis]['np_y']
            norm = (abs_mean - abs_mean.min()) / (abs_mean.max() - abs_mean.min())

            dimension = "%" if by_percent else ""
            if axis == SAGITTAL:
                print("=== SAGITTAL === ")
            else:
                print("=== MERIDIONAL === ")
            self._print_error_stats(abs_mean, orig, dimension)


            plt.scatter(x, y, c=abs_mean, cmap=plt.cm.jet)
            plt.colorbar()
            plt.title('Absolute fit errors ({}cy/px, {})'.format(str(freqs), axis))
            plt.xlabel('Image location x')
            plt.ylabel('Image location y')
            plt.show()

    def plot_fit_errors_histogram(self, freqs=[0.1]):
        err, fit, orig = self.get_fit_errors(freqs)
        plt.hist(err.flatten())
        plt.show()

    def summarise_fit_errors(self, freqs=[0.1], by_freq=False, by_percent=False):
        if not by_freq:
            freqs = [freqs]

        for freq in freqs:
            print("Getting errors for frequency(s)    {}:".format(str(freq)))
            errors, fit, orig = self.get_fit_errors(freq, by_percent=by_percent)
            dimension = "%" if by_percent else ""
            self._print_error_stats(errors, orig, dimension)

    def _print_error_stats(self, err, orig, dimension):
            flat = np.abs(err).flatten()
            print("  n                                {:.0f}".format(len(flat)))
            print("  Original data max                {:.4f}".format(orig.max()))
            origrms =(orig**2).mean()**0.5
            print("  Original data RMS                {:.4f}".format(origrms))
            print("  ---")
            print("  Mean absolute error              {:.4f}{}".format(np.mean(np.abs(flat)), dimension))
            error_rms = np.mean(flat**2)**0.5
            print("  Root mean squared (RMS) error    {:.4f}{}".format(error_rms, dimension))
            print("  95th percentile absolute error   {:.4f}{}".format(np.percentile(np.abs(flat), 95), dimension))
            print("  99th percentile absolute error   {:.4f}{}".format(np.percentile(np.abs(flat), 99), dimension))
            print("  99.9th percentile absolute error {:.4f}{}".format(np.percentile(np.abs(flat), 99.9), dimension))
            print("  Maximum absolute error           {:.4f}{}".format(np.amax(np.abs(flat)), dimension))
            print("")
            print("  RMS Error / RMS Original in %    {:.1f} ".format(error_rms / origrms *100))
            print()

    def set_calibration_sharpen(self, amount, radius, stack=False):
        for point in self.points:
            point.set_calibration_sharpen(amount, radius, stack)
        self.calibration = self.points[0].calibration

    def plot_sfr_at_point(self, x, y, axis=MERIDIONAL):
        freqs = RAW_SFR_FREQUENCIES[:32]
        ys = []
        for freq in freqs:
            ys.append(self.interpolate_value(x, y, freq, axis))
        # ys = truncate_at_zero(ys)
        # exit()
        plt.plot(freqs, ys)
        plt.xlabel("Spacial frequency (cy/px)")
        plt.ylabel("SFR")
        try:
            plt.title(self.exif.summary + " at {:.0f}, {:.0f}".format(x, y))
        except ValueError:
            pass
        plt.show()

    def read_exif(self, sfr_path):
        # exifpath = sfr_path[:-3] + ".exif"
        self.exif = EXIF(sfr_pathname=sfr_path)
        return self.exif


def process_esfs(esfs, points):
    x = np.linspace(0, 4, 256)  # cy/px
    # x = RAW_SFR_FREQUENCIES
    goodpoints = []

    # Work out chart centre by looking at angles

    plotx = np.linspace(-1, 1, 4) * 500 + IMAGE_WIDTH / 2
    a_list = []
    b_list = []
    for point in points:
        if point is None:
            continue
        angle = (point.angle) / 180 * np.pi
        angle_tries = np.linspace(0, np.pi * 3 / 4, 4) + angle
        a_s = np.tan(angle_tries)
        b_s = np.ones_like(a_s) * point.y - a_s * point.x
        y_at_mid_x = a_s * IMAGE_WIDTH / 2 + b_s
        best_ix = np.argmin(abs(y_at_mid_x - IMAGE_HEIGHT / 2))
        min_a = a_s[best_ix]
        min_b = b_s[best_ix]
        a_list.append(min_a)
        b_list.append(min_b)
        # plt.plot(plotx, min_a * plotx + min_b)
    # plt.show()
    a_array = np.array(a_list)
    b_array = np.array(b_list)

    filtercache = None

    def cost(params):
        nonlocal filtercache
        x, y = params * 1000
        test_ys = a_array * x + b_array
        offsets = test_ys - y
        std = np.std(offsets)
        median = np.median(offsets)
        if filtercache is None:
            filteridx = (offsets > (median - std * 3) * (offsets < (median + std * 3)))
            filtercache = filteridx
        else:
            filteridx = filtercache
        cost = (offsets[filteridx]**2).mean()
        return cost * 1e-5

    opt = optimize.minimize(cost, (IMAGE_WIDTH / 2 * 1e-3, IMAGE_HEIGHT / 2 * 1e-3))
    # print(opt)
    # exit()
    # opt.x = (IMAGE_WIDTH / 2, IMAGE_HEIGHT / 2)
    chart_centre_x = opt.x[0] * 1000
    chart_centre_y = opt.x[1] * 1000

    xtrunc = np.linspace(0, 4, 128)
    xtrunc[0] = 1
    correction = 1.0 / (np.sin(np.pi * xtrunc/4)/(np.pi * xtrunc/4) * np.sin(np.pi * xtrunc/8)/(np.pi * xtrunc/8))[:32]
    correction[0] = 1

    squares_s = []
    squares_m = []
    for point, esf in zip(points, esfs):
        if point is None:
            continue
        if point.angle < 6:
            continue
        squareid = point.squareid
        if point.is_saggital:
            squares_s.append((squareid, 0, point, esf))
        else:
            squares_m.append((squareid, 1, point, esf))


        # if not squareid >= 0:
        #     continue
        # for point_compare in goodpoints:
        #     if point_compare.squareid != squareid:
        #         continue
        #     if point.is_saggital != point_compare.is_saggital:
        #         cache.get('squareid', []).append(point_compare)
                # continue
            # if point_compare is point:
            #     continue
    # for lst in [squares_s, squares_m]:
    #     last_tup = None
    #     for tup in lst:
    #         if last_tup is None or last_tup[:2] != tup[:2]:
    #             last_tup = tup
    #             continue
    #         squareid, is_meridional, point, esf = tup
    #         _, _, point2, esf2 = last_tup
    if 1:
        for point, esf in zip(points, esfs):
            if point is None:
                continue
            if point.angle < 6:
                continue
            arr = np.array([float(_) for _ in esf if len(_)])
            if sum(arr) == 0:
                continue

            flip = False
            if arr[0] > arr[-1] and 1:
                lsf = (-np.diff(arr))
                flip = True
            else:
                lsf = np.diff(arr)
            if 1:
                if point.y < chart_centre_y and not point.is_saggital:
                    lsf = np.flip(lsf, axis=0)
                    flip = flip is False
                if point.x < chart_centre_x and point.is_saggital:
                    lsf = np.flip(lsf, axis=0)
                    flip = flip is False
                    # flip = False
            point.flipped = flip

            padded_lsf = np.concatenate((lsf, (0,)))

            window = signal.windows.tukey(len(padded_lsf), 0.6)

            fft_real, fft_imag = normalised_centreing_fft(padded_lsf * window, return_type=COMPLEX_CARTESIAN_REAL_TUPLE)


            # plt.plot(correction)
            # plt.show()

            # fft = scipyfftpack.fft(scipyfftpack.fftshift(padded_lsf))
            # fft = scipyfftpack.fft(padded_lsf)
            # fft /= abs(fft[0])
            # fft_real = fft.real
            # fft_imag = fft.imag
            # if point.filenumber == 9 and point.x > 5100 and point.y > 3500:
            #     plt.plot(arr)
            #     plt.show()
            # plt.plot(x[:4]*4, abs(fft)[:4])
            # plt.plot(x[:32] * 2, abs(fft_real + 1j * fft_imag)[:32], label="fft")
            # plt.plot(x[:64], point.raw_sfr_data[:64], label="mtfm")
            # plt.legend()
            # plt.show()
            #
            # exit()

            interpfn = interpolate.InterpolatedUnivariateSpline(x[:32] * 2, fft_real[:32]*correction, k=1)
            real = interpfn(RAW_SFR_FREQUENCIES)
            interpfn = interpolate.InterpolatedUnivariateSpline(x[:32] * 2, fft_imag[:32]*correction, k=1)
            imag = interpfn(RAW_SFR_FREQUENCIES)
            otf = real + 1j * imag
            height = calc_image_height(point.x, point.y)
            # if point.filenumber == 9 and 4600 < point.x < 5000 and 3200 < point.y < 3600:
            neg = (min(arr) - min(arr[0], arr[-1])) / (arr.max() - arr.min())
            if point.filenumber == -1 and 0.6 < height < 0.9:
                if neg < -0.017:
                    print(neg)
                    print(point)
                    plt.plot(RAW_SFR_FREQUENCIES, point.raw_sfr_data)
                    plt.plot(RAW_SFR_FREQUENCIES, abs(otf))
                    plt.plot(RAW_SFR_FREQUENCIES, abs(otf) / point.raw_sfr_data)
                    plt.plot(arr)
                    plt.show()
            point.raw_sfr_data = (real ** 2 + imag ** 2) ** 0.5
            point.raw_otf = otf
            point.has_phase = True
            # plt.plot(RAW_SFR_FREQUENCIES, point.raw_sfr_data)
            # plt.show()
            # point.raw_otf_real = real
            # point.raw_otf_imag = imag
            # point.raw_otf_phase = np.angle(otf)
            # point.otf_phase = np.angle(otf)  # No calibration needed
            # point.raw_esf_data = arr
            # point.raw_lsf_data = padded_lsf
            goodpoints.append(point)
    if point and point.filenumber == -190:
        exit()
    x_arr = np.array([point.x for point in goodpoints])
    y_arr = np.array([point.y for point in goodpoints])
    rad = ((x_arr - IMAGE_WIDTH/2) ** 2 + (y_arr - IMAGE_HEIGHT/2) ** 2) ** 0.5
    ang = (np.arctan2((x_arr - IMAGE_WIDTH/2), (y_arr - IMAGE_HEIGHT/2)) + 9*np.pi/4) % (np.pi*2)

    attempt_to_spot_faulty_flips = False

    if attempt_to_spot_faulty_flips:
        for point in goodpoints:
            # continue
            if point.angle < 10 and 2700 < point.x < 3300 and point.y < 2000:
                p_ang = (np.arctan2(point.x - IMAGE_WIDTH/2, point.y - IMAGE_HEIGHT/2) + 9*np.pi/4) % (np.pi*2)
                p_rad = ((point.x - IMAGE_WIDTH/2) ** 2 + (point.y - IMAGE_HEIGHT/2)**2) ** 0.5
                point_distances = (((rad - p_rad) / IMAGE_DIAGONAL * 60)**2 + (ang - p_ang)**2 * 1) ** 0.5
                point_distances = abs((rad - p_rad) / IMAGE_DIAGONAL * 60)
                too_far_angle = abs(p_ang - ang) > 0.7
                point_distances[too_far_angle] += 1e8
                # point_distances = (rad - p_rad)**2
                sortix = np.argsort(point_distances)
                sortix = sortix[point_distances[sortix] < 1e6]
                y = []
                # plt.ylim(-10.5, 10.5)
                for ix in sortix:
                    print(goodpoints[ix].flipped)
                    if points[ix].is_meridional != point.is_meridional:
                        continue
                    if points[ix].flipped != point.flipped:
                        continue
                    px , py= goodpoints[ix].x, goodpoints[ix].y
                    # print(px, py, np.arctan2(px, py), ((px-IMAGE_WIDTH/2) **2 + (py-IMAGE_HEIGHT/2) **2)**0.5, rad[ix])
                    # print(x_arr[ix], y_arr[ix])
                    y.append(goodpoints[ix].raw_otf.imag[:40].sum())
                avg = np.mean(y[1:])
                std = np.std(y[1:])
                tol = 1.5
                if abs(y[0] - avg) > std*tol:
                    print("glarg")
                    if abs(-y[0] - avg) < std*tol:
                        point.raw_otf = point.raw_otf.real - 1j * point.raw_otf.imag

                        print("Replace!")
                    pass
                # plt.xlim(0, IMAGE_WIDTH)
                # plt.ylim(0, IMAGE_HEIGHT)
                # plt.plot(x_arr[sortix[:40]], y_arr[sortix[:40]], 's')
                # plt.show()
                # plt.plot(y, 's')
                # plt.show()

    join_sides_of_squares = True

    new_points = []
    cache = {}
    if join_sides_of_squares:
        for point in goodpoints:
            squareid = point.squareid
            if squareid < 0:
                continue
            for point_compare in goodpoints:
                if point_compare.squareid != squareid:
                    continue
                if point.is_saggital != point_compare.is_saggital:
                    # cache.get('squareid', []).append(point_compare)
                    continue
                if point_compare is point:
                    continue
                mean_x = (point.x + point_compare.x) / 2
                mean_y = (point.y + point_compare.y) / 2
                mean_mtf = (abs(point.raw_otf) + abs(point_compare.raw_otf)) / 2
                mean_imag = (point.raw_otf.imag + point_compare.raw_otf.imag) / 2
                mean_real = (mean_mtf**2 - mean_imag**2)**0.5

                # mean_angle = (np.angle(point.raw_otf) + np.angle(point_compare.raw_otf)) / 2
                mean_otf = mean_real + 1j * mean_imag
                # print(point_compare)
                # print(point)
                if point.filenumber > 500000:
                    # plt.plot(point.raw_otf.real, label="p1r")
                    # plt.plot(point_compare.raw_otf.real, label="p2r")
                    # plt.plot(point_compare.raw_otf.imag, label="p2i")
                    # plt.plot(point.raw_otf.imag, label="p1i")
                    plt.plot(abs(point.raw_otf), label="p1abs")
                    plt.plot(abs(point_compare.raw_otf), label="p2abs")
                    plt.plot(mean_otf.real, label="mr")
                    plt.plot(mean_otf.imag, label="mi")
                    plt.plot(abs(mean_otf), label="mabs")
                    plt.legend()
                    plt.show()
                point.x = mean_x
                point.y = mean_y
                point.raw_otf = mean_otf
                point.raw_sfr_data = abs(mean_otf)
                point_compare.squareid = np.nan  # Take it out of matching
                # print(point)
                # print()
                new_points.append(point)
                break

        goodpoints = new_points
    return goodpoints

    points = [point for point in points if point is not None]

    radius = 1500
    for angle in np.linspace(0, 2 * np.pi, 20):
        xscan = np.sin(angle) * radius + 3000
        yscan = np.cos(angle) * radius + 2000
        points.sort(key=lambda p: ((p.x - xscan)**2 + (p.y - yscan)**2)**0.5, reverse=False)
        square = points[0].squareid
        squarepoints = [point for point in points if point.squareid == square]
        squarepoints.sort(key=lambda p: 0 if p.is_saggital else 1)
        # print(xscan, yscan, len([0 for point in points if point.squareid == square]))
        squarepoints = points
        for point in squarepoints:
            # if point.squareid != square:
            #     continue
            if not point.is_saggital:
                continue
            if point.x < 2900:
                continue
            if point.x > 3100:
                continue

            print(point)
            plt.plot(point.raw_sfr_data[:32], label="Magnitude")
            # plt.plot(np.abs(res[:16] / zero), label=)
            # plt.plot(np.unwrap(np.angle(res[:16]) / zero))
            plt.plot(point.raw_otf_phase[:32] / np.pi, label="Angle")
            plt.plot(point.raw_otf_real[:32], label='Real')
            plt.plot(point.raw_otf_imag[:32], label="Imag")
            plt.plot(np.linspace(0, 16, 256), point.raw_esf_data[:] / point.raw_esf_data.max(), label="ESF")
            plt.plot(np.linspace(0, 16, 256), point.raw_lsf_data[:] / point.raw_lsf_data.max(), label="LSF")
            plt.legend()
            # plt.plot(point.raw_sfr_data)
            plt.show()