import colorsys
import math
import csv

import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from scipy import interpolate

from lentil.constants_utils import *
from lentil.sfr_point import SFRPoint

SMOOTHING = 0.22

class SFRField():
    """
    Represents entire image field of SFRPoints for a single image
    """

    def __init__(self, points=None, pathname=None):
        """

        :param points: Iterable of SFRPoints, order not important
        """
        if points is None:
            points = []
            with open(pathname, 'r') as sfrfile:
                csvreader = csv.reader(sfrfile, delimiter=' ', quotechar='|')

                for row in csvreader:
                    points.append(SFRPoint(row))
        self.points = points
        np_axis = {}
        np_axis['np_x'] = None
        np_axis['np_y'] = None
        np_axis['np_sfr'] = None
        np_axis['np_sfr_freq'] = None
        np_axis['np_mtf'] = None
        np_axis2 = np_axis.copy()
        self.np_dict = {SAGITTAL: np_axis, MERIDIONAL: np_axis2}

        self.smoothing = SMOOTHING

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
        lst = self.get_subset(axis)
        x_min = min((point.x for point in lst))
        x_max = max((point.x for point in lst))
        y_min = min((point.y for point in lst))
        y_max = max((point.y for point in lst))
        return x_min, y_min, x_max, y_max

    def build_axis_points(self, x_len=20, y_len=20, axis=MEDIAL):
        x_min, y_min, x_max, y_max = self.get_point_range(axis)
        x_values = np.arange(x_min, x_max, (x_max - x_min) / x_len)
        y_values = np.arange(y_min, y_max, (y_max - y_min) / y_len)
        return x_values, y_values

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

    def interpolate_value(self, x, y, freq=0.1, axis=MEDIAL):
        """
        Provides an interpolated MTF/SFR for chosen point in field and axis. Uses a locally weighted polynomial plane.

        Pass -1 as frequency for MTF50 results in cy/px

        :param x: x location
        :param y: y location
        :param axis: Pass constants SAGGITAL, MERIDIONAL, or MEDIAL
        :param freq: spacial frequency to return, -1 for mtf50
        :return: interpolated cy/px at specified frequency, or mtf50 frequency if -1 passed
        """
        if self.np_dict[axis]['np_x'] is None or self.np_dict[axis]['np_sfr_freq'] != freq:
            lst = []
            for point in self.get_subset(axis):
                lst.append((point.x, point.y, point.get_freq(freq)))
            x_arr, y_arr, z_arr = zip(*lst)
            x_arr = np.array(x_arr)
            y_arr = np.array(y_arr)
            z_arr = np.array(z_arr)
            self.np_dict[axis]['np_x'] = x_arr
            self.np_dict[axis]['np_y'] = y_arr
            self.np_dict[axis]['np_mtf'] = z_arr
            self.np_dict[axis]['np_sfr_freq'] = freq
        x_arr = self.np_dict[axis]['np_x']
        y_arr = self.np_dict[axis]['np_y']
        z_arr = self.np_dict[axis]['np_mtf']

        # Calculate distance of each edge location to input location on each axis
        x_distances = (x_arr - x)
        y_distances = (y_arr - y)

        # Determine scatter of these points
        x_distance_rms = np.sqrt((x_distances ** 2).mean()) * self.smoothing
        y_distance_rms = np.sqrt((y_distances ** 2).mean()) * self.smoothing

        # Calculate distance in 2d plane
        distances = np.sqrt(
            ((x_distances) / x_distance_rms) ** 2 + ((y_distances) / y_distance_rms) ** 2)

        # Determine weightings allowing more weight to nearby points
        exp_weights = np.exp(1 - distances) * 0.5
        raised_cos_weights = np.cos(np.clip(distances / 1.4 * math.pi, 0.0, math.pi)) + 1.0
        weights = (exp_weights * 1.6 + raised_cos_weights * 0.4)

        x_min, y_min, x_max, y_max = self.get_point_range(axis)
        bbox = [x_min, x_max, y_min, y_max]

        order = 2  # Spline order

        # Build spline surface
        func = interpolate.SmoothBivariateSpline(x_arr, y_arr, z_arr, bbox=bbox,
                                                 w=weights, kx=order, ky=order, s=float("inf"))
        output = func(x, y)  # Get (buried) interpolated value at point of interest
        return np.clip(output[0][0], 1e-5, 1.0)  # Return scalar

    def plot(self, freq=0.1, axis=MEDIAL, plot_type=1, detail=1.0,
             show=True, ax=None, alpha=0.85):
        """
        Plots SFR/MTF values for chosen axis across field

        :param axis: Constant SAGGITAL, MERIDIONAL, or MEDIAL
        :param plot_type: 0 is 3d surface, 1 is 2d contour
        :param detail: Relative detail in plot (1.0 is default)
        :return:
        """
        x_values, y_values = self.build_axis_points(int(detail * 20), int(detail * 20))
        z_values = np.ndarray((len(y_values), len(x_values)))

        # fn = self.get_simple_interpolation_fn(axis)
        for x_idx, x in enumerate(x_values):
            for y_idx, y in enumerate(y_values):
                z_values[y_idx, x_idx] = self.interpolate_value(x, y, freq, axis)

        max_z = np.amax(z_values) * 1.1

        if plot_type == 0:
            fig, ax = plt.subplots()
            colors = []
            contours = np.arange(0.02, 0.30, 0.01)
            linspaced = np.linspace(0.0, 1.0, len(contours))
            for lin, line in zip(linspaced, contours):
                colors.append(plt.cm.jet(lin))
                # colors.append(colorsys.hls_to_rgb(lin * 0.8, 0.4, 1.0))

            ax.set_ylim(np.amax(y_values), np.amin(y_values))
            CS = ax.contourf(x_values, y_values, z_values, contours, colors=colors)
            CS2 = ax.contour(x_values, y_values, z_values, contours, colors=('black',))
            plt.clabel(CS2, inline=1, fontsize=10)
            plt.title('Simplest default with labels')

            # plt.figure()
            # contours = np.arange(0.1, 0.6, 0.01)
            # colors = []
            # linspaced = np.linspace(0.0, max_z, len(contours))
            # for lin, line in zip(linspaced, contours):
            #     colors.append(colorsys.hls_to_rgb(lin * 0.8, 0.4, 1.0))
            # CS = plt.contour(x_values, y_values, z_values, contours, colors=colors)
            # plt.clabel(CS, inline=1, fontsize=10)
            # plt.title('Simplest default with labels')
        else:
            fig = plt.figure()
            if ax is None:
                ax = fig.add_subplot(111, projection='3d')
                ax.set_zlim(0.0, max_z)
            # ax.zaxis.set_major_locator(LinearLocator(10))
            # ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

            x, y = np.meshgrid(x_values, y_values)
            print(x.flatten().shape)
            print(y.flatten().shape)
            print(z_values.shape)

            cmap = plt.cm.winter  # Base colormap
            my_cmap = cmap(np.arange(cmap.N))  # Read colormap colours
            my_cmap[:, -1] = alpha  # Set colormap alpha
            # print(my_cmap[1,:].shape);exit()
            new_cmap = np.ndarray((256, 4))

            for a in range(256):
                mod = 0.5 - math.cos(a / 256 * math.pi) * 0.5
                new_cmap[a, :] = my_cmap[int(mod * 256), :]

            # my_col = plt.cm.jet(1.0 - z_values)
            # my_col[:, :, -1] = 0.85

            mycmap = ListedColormap(new_cmap)
            norm = matplotlib.colors.Normalize(vmin=0, vmax=max_z)
            surf = ax.plot_surface(x, y, z_values, cmap=mycmap, norm=norm, edgecolors='b',
                                   rstride=1, cstride=1, linewidth=0.5, antialiased=True)
            fig.colorbar(surf, shrink=0.5, aspect=5)
        if show:
            plt.show()
        return ax

    def plot_points(self, freq=0.05, axis=MEDIAL):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
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
            ax.scatter(x_arr, y_arr, z_arr, c='r' if axis is SAGITTAL else 'b', marker='.')
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
            x = self.np_dict[axis]['np_x']
            y = self.np_dict[axis]['np_y']
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
