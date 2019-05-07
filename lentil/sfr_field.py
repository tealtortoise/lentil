import colorsys
import math
import csv

import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from scipy import interpolate

from lentil.constants_utils import *
from lentil.plot_utils import FieldPlot
from lentil.sfr_point import SFRPoint

FUNCSTORE = None

class NotEnoughPointsException(Exception):
    pass

class SFRField():
    """
    Represents entire image field of SFRPoints for a single image
    """

    def __init__(self, points=None, pathname=None, calibration=None, smoothing=0, exif=None):
        """

        :param points: Iterable of SFRPoints, order not important
        :param pathname: Path to MTF Mapper edge_sfr_values.txt file to parse
        :param calibration: calibration data array
        """
        if points is None:
            points = []
            with open(pathname, 'r') as sfrfile:
                print(999, pathname)
                csvreader = csv.reader(sfrfile, delimiter=' ', quotechar='|')

                for row in csvreader:
                    try:
                        points.append(SFRPoint(row, calibration=calibration))
                    except ValueError:
                        pass
            if exif is not None:
                self.exif = exif
            else:
                self.read_exif(pathname)
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
        np_axis['np_mtf'] = None
        np_axis2 = np_axis.copy()
        self.np_dict_cache = {SAGITTAL: np_axis, MERIDIONAL: np_axis2}

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

    def interpolate_value(self, x, y, freq=DEFAULT_FREQ, axis=MEDIAL):
        """
        Provides an interpolated MTF/SFR for chosen point in field and axis. Uses a locally weighted polynomial plane.

        Pass -1 as frequency for MTF50 results in cy/px

        :param x: x location
        :param y: y location
        :param axis: Pass constants SAGGITAL, MERIDIONAL, or MEDIAL
        :param freq: spacial frequency to return, -1 for mtf50
        :return: interpolated cy/px at specified frequency, or mtf50 frequency if -1 passed
        """
        if self.np_dict_cache[axis]['np_x'] is None or self.np_dict_cache[axis]['np_sfr_freq'] != freq:
            lst = []
            for point in self.get_subset(axis):
                lst.append((point.x, point.y, point.get_freq(freq)))
            if len(lst) is 0:
                for p in self.points:
                    print(p.radialangle)
            x_arr, y_arr, z_arr = zip(*lst)
            x_arr = np.array(x_arr)
            y_arr = np.array(y_arr)
            z_arr = np.array(z_arr)
            self.np_dict_cache[axis]['np_x'] = x_arr
            self.np_dict_cache[axis]['np_y'] = y_arr
            self.np_dict_cache[axis]['np_mtf'] = z_arr
            self.np_dict_cache[axis]['np_sfr_freq'] = freq
        x_arr = self.np_dict_cache[axis]['np_x']
        y_arr = self.np_dict_cache[axis]['np_y']
        z_arr = self.np_dict_cache[axis]['np_mtf']

        # Calculate distance of each edge location to input location on each axis
        x_distances = (x_arr - x)
        y_distances = (y_arr - y)

        distances = np.sqrt((x_distances) ** 2 + (y_distances) ** 2)

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
        # print("angle std {}".format(angle_std))
        low = 40
        high = 120
        # print(stack.shape,sortedstack.shape, clippedstack.shape, min_ratio_radius, min_point_radius)
        # print(nr)
        if radius_range < high:
            func_linear = interpolate.SmoothBivariateSpline(clippedstack[0], clippedstack[1], clippedstack[2],
                                                            bbox=bbox, w=weights, kx=1, ky=1, s=float("inf"))
            low_order_ratio = 1.0 - np.clip((angle_std - low) / (high - low), 0.0, 1.0)
            # print("Low", low_order_ratio)
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

        return np.clip(output, 1e-5, np.inf)

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

    def plot_points(self, freq=0.05, axis=MEDIAL, autoscale=False, add_corners=False):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        if not autoscale:
            ax.set_zlim(0, 1)

        if axis == MEDIAL:
            axis = (SAGITTAL, MERIDIONAL)
        else:
            axis = (axis,)
        for axis_ in axis:
            lst = []
            for point in self.get_subset(axis_):
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
        plt.title(self.exif.summary + " at {:.0f}, {:.0f}".format(x, y))
        plt.show()