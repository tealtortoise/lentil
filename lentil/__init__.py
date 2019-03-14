import csv
import math
import os
import colorsys
import numpy as np
from scipy import interpolate, optimize, signal
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib
from mpl_toolkits.mplot3d import Axes3D

DEFAULT_PIXEL_SIZE = 4e-6

SAGGITAL = 1
MERIDIONAL = 2
BOTH_AXES = 3

PATH = '/mnt/mtfm/16-55mm/27mm f2.8/mtfmappertemp_{:.0f}/'
numberrange = range(14, 24)
PATH = '/mnt/mtfm/16-55mm/27mm f5.6/mtfmappertemp_{:.0f}/'
numberrange = range(43, 52)
sfrfilename = 'edge_sfr_values.txt'
PATH = "/mnt/mtfm/23mm f1.4/Results/"

SFR_HEADER = [
    'blockid',
    'edgex',
    'edgey',
    'edgeangle',
    'radialangle'
]

# with rawpy.imread(path+rawfile) as raw:
#     rgb = raw.postprocess(use_auto_wb=True, no_auto_bright=True, gamma=(1, 1))
# imageio.imsave(path+rawfile+'rawpy.png', rgb, )

RAW_SFR_FREQUENCIES = [x / 64 for x in range(64)]  # List of sfr frequencies in cycles/pixel


class SFRPoint:
    """
    Holds all data for one SFR edge analysis point
    """
    def __init__(self, rowdata, pixelsize=None):
        """
        Processes row from csv reader

        :param rowdata: raw from csv reader
        :param pixelsize: pixel size in metres if required
        """
        self.x = float(rowdata[1])
        self.y = float(rowdata[2])
        self.angle = float(rowdata[3])
        self.radialangle = float(rowdata[4])
        self.raw_sfr_data = [float(cell) for cell in rowdata[5:-1]]
        self.pixelsize = pixelsize or DEFAULT_PIXEL_SIZE
        self._interpolate_fn = None
        self._mtf50 = None
        assert len(self.raw_sfr_data) == 64

    def get_freq(self, cy_px=None, lp_mm=None):
        """
        Returns SFR at specified frequency, or MTF50 if '-1' input

        Using linear interpolation

        :param cy_px: frequency of interest in cycles/px (0.0-1.0)
        :param lp_mm: frequency of interest in line pairs / mm (>0.0)
        :return:
        """
        if lp_mm is not None:
            cy_px = lp_mm * self.pixelsize * 1e3
        if cy_px is None:
            raise AttributeError("Must provide frequency in cycles/px or lp/mm")
        if cy_px == -1:
            if lp_mm is not None:
                return self.mtf50_lpmm
            else:
                return self.mtf50
        if not 0.0 <= cy_px < 1.0:
            raise AttributeError("Frequency response must be between 0 and twice nyquist")

        return self.interpolate_fn(cy_px)

    @property
    def interpolate_fn(self):
        if self._interpolate_fn is None:
            # Build interpolation function from raw data
            self._interpolate_fn = interpolate.InterpolatedUnivariateSpline(RAW_SFR_FREQUENCIES,
                                                                            self.raw_sfr_data, k=1)
        return self._interpolate_fn

    @property
    def mtf50(self):
        """
        Calculates and stores MTF50

        :return: MTF50 in cycles/px
        """
        def callable(fr):
            return self.interpolate_fn(fr) - 0.5
        if self._mtf50 is None:
            self._mtf50 = optimize.newton(callable, 0.05)
        return self._mtf50

    @property
    def mtf50_lpmm(self):
        """
        :return: MTF50 in line/pairs per mm
        """
        return self.mtf50 / self.pixelsize * 1e-3

    @property
    def is_saggital(self):
        if self.radialangle < 45.0:
            return True
    @property
    def is_meridional(self):
        if self.radialangle > 45.0:
            return True

    def is_axis(self, axis):
        if axis == SAGGITAL:
            return self.is_saggital
        if axis == MERIDIONAL:
            return self.is_meridional
        if axis == BOTH_AXES:
            return True
        raise AttributeError("Unknown axis attribute")

    def plot(self):
        """
        Plot spatial frequency response for point
        :return: None
        """
        x_range = np.arange(0, 1.0, 0.01)
        y_vals = [self.get_freq(x) for x in x_range]
        plt.plot(x_range, y_vals)
        plt.show()

    def __str__(self):
        return "x: {:.0f}, y: {:.0f}, angle: {:.0f}, radial angle: {:.0f}".format(self.x,
                                                                                  self.y,
                                                                                  self.angle,
                                                                                  self.radialangle)


class SFRField():
    """
    Represents entire image field of SFRPoints for a single image
    """
    def __init__(self, points):
        """

        :param points: Iterable of SFRPoints, order not important
        """
        self.points = points
        self.np_x = None
        self.np_y = None
        self.np_sfr = None
        self.np_sfr_freq = None
        self.np_mtf = None
        self.np_axis = None
        self.smoothing = 0.5

    @property
    def saggital_points(self):
        """

        :return: Returns list of all saggital edge points in field
        """
        return [point for point in points if point.is_saggital]

    @property
    def meridional_points(self):
        """

        :return: Returns list of all meridional edge points in field
        """
        return [point for point in points if not point.is_saggital]

    def get_subset(self, axis):
        """
        Returns list of all points on chosen axis (or both)
        :param axis: constant SAGGITAL or MERIDIONAL or BOTH_AXES
        :return: list of points
        """
        return [point for point in self.points if point.is_axis(axis)]

    def get_avg_mtf50(self):
        """
        Calculates Unweighted arithmetic mean of MTF50 for all points in cycles/px
        :return: Average MTF50 for all points (in cycles/px)
        """
        return sum((point.mtf50 for point in self.points)) / len(self.points)

    def get_point_range(self, axis=BOTH_AXES):
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

    def build_axis_points(self, x_len=20, y_len=20, axis=BOTH_AXES):
        x_min, y_min, x_max, y_max = self.get_point_range(axis)
        x_values = np.arange(x_min, x_max, (x_max - x_min) / x_len)
        y_values = np.arange(y_min, y_max, (y_max - y_min) / y_len)
        return x_values, y_values

    def get_simple_interpolation_fn(self, axis=BOTH_AXES):
        """
        Returns a simple spline interpolation callable fitted across the whole field.

        :param axis: Pass constant SAGGITAL, MERIDIONAL, or BOTH_AXES
        :return: Scipy callable which accepts x and y positions and provides interpolated value.
        """
        lst = []
        for point in self.get_subset(axis):
            lst.append((point.x, point.y, point.mtf50))
        x_lst, y_lst, z_lst = zip(*lst)

        fn = interpolate.SmoothBivariateSpline(x_lst, y_lst, z_lst, kx=2, ky=2, s=float("inf"))
        return fn

    def interpolate_value(self, x, y, freq=0.1, axis=BOTH_AXES):
        """
        Provides an interpolated MTF/SFR for chosen point in field and axis. Uses a locally weighted polynomial plane.

        Pass -1 as frequency for MTF50 results in cy/px

        :param x: x location
        :param y: y location
        :param axis: Pass constants SAGGITAL, MERIDIONAL, or BOTH_AXES
        :param freq: spacial frequency to return, -1 for mtf50
        :return: interpolated cy/px at specified frequency, or mtf50 frequency if -1 passed
        """
        if self.np_x is None or self.np_axis != axis:
            lst = []
            for point in self.get_subset(axis):
                lst.append((point.x, point.y, point.get_freq(freq)))
            x_arr, y_arr, z_arr = zip(*lst)
            x_arr = np.array(x_arr)
            y_arr = np.array(y_arr)
            z_arr = np.array(z_arr)
            self.np_x = x_arr
            self.np_y = y_arr
            self.np_mtf = z_arr
            self.np_axis = axis
        x_arr = self.np_x
        y_arr = self.np_y
        z_arr = self.np_mtf

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
        return output[0][0]  # Return scalar

    def plot(self, axis=BOTH_AXES, plot_type=0, detail=1.0):
        """
        Plots SFR/MTF values for chosen axis across field

        :param axis: Constant SAGGITAL, MERIDIONAL, or BOTH_AXES
        :param plot_type: 0 is 3d surface, 1 is 2d contour
        :param detail: Relative detail in plot (1.0 is default)
        :return:
        """
        x_values, y_values = self.build_axis_points(int(detail*20), int(detail*20))
        z_values = np.ndarray((len(y_values), len(x_values)))

        # fn = self.get_simple_interpolation_fn(axis)
        for x_idx, x in enumerate(x_values):
            for y_idx, y in enumerate(y_values):
                z_values[y_idx, x_idx] = self.interpolate_value(x, y, axis)

        if plot_type == 0:
            plt.figure()
            contours = np.arange(0.1, 0.6, 0.01)
            colors = []
            linspaced = np.linspace(0.0, 1.0, len(contours))
            for lin, line in zip(linspaced, contours):
                colors.append(colorsys.hls_to_rgb(lin * 0.8, 0.4, 1.0))
            CS = plt.contour(x_values, y_values, z_values, contours, colors=colors)
            plt.clabel(CS, inline=1, fontsize=10)
            plt.title('Simplest default with labels')
        else:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')

            ax.set_zlim(0.0, 1.0)
            # ax.zaxis.set_major_locator(LinearLocator(10))
            # ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

            x, y = np.meshgrid(x_values, y_values)
            print(x.flatten().shape)
            print(y.flatten().shape)
            print(z_values.shape)

            cmap = plt.cm.winter  # Base colormap
            my_cmap = cmap(np.arange(cmap.N))  # Read colormap colours
            my_cmap[:, -1] = 0.85  # Set colormap alpha
            # print(my_cmap[1,:].shape);exit()
            new_cmap = np.ndarray((256, 4))

            for a in range(256):
                mod = 0.5 - math.cos(a / 256 * math.pi) * 0.5
                new_cmap[a, :] = my_cmap[int(mod*256), :]

            # my_col = plt.cm.jet(1.0 - z_values)
            # my_col[:, :, -1] = 0.85

            mycmap = ListedColormap(new_cmap)
            norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
            surf = ax.plot_surface(x, y, z_values, cmap=mycmap, norm=norm,
                                   rstride=1, cstride=1, linewidth=1, antialiased=True)
            fig.colorbar(surf, shrink=0.5, aspect=5)
        plt.show()


class FocusSet:
    """
    A range of fields with stepped focus, in order
    """
    def __init__(self, filenames):
        self.fields = []
        for filename in filenames:
            print("Opening file {}".format(filename))
            with open(filename, 'r') as sfrfile:
                csvreader = csv.reader(sfrfile, delimiter=' ', quotechar='|')

                points = []
                for row in csvreader:
                    points.append(SFRPoint(row))

            field = SFRField(points)
            self.fields.append(field)

    def plot_sfr_vs_focus(self, x, y, freq=-1, axis=BOTH_AXES, show=False):
        y_values = []
        for field in self.fields:
            y_values.append(field.interpolate_value(x, y, freq, axis))
        x_values = np.arange(0, len(y_values), 1)
        y_values = np.array(y_values)
        plt.plot(x_values, y_values, color='black')
        weights = (y_values / np.max(y_values)) ** 4

        y_values = np.array(y_values)
        plot_x = np.linspace(0, max(x_values), 100)
        r = 0
        if r:
            mean_guess = np.argmax(y_values)
            amplitude_guess = y_values[mean_guess] / 2.0
            sigma_guess = 2.0
            peaky = -amplitude_guess/20.0
            print(amplitude_guess, mean_guess, sigma_guess, peaky)

            def guassianfunc(xVar, a, b, c, a2=None):
                if a2 is None:
                    a2 = a
                comp1 = a * np.exp(-(xVar - b) ** 2 / (4 * c ** 2))
                comp2 = a2 * np.exp(-(xVar - b) ** 2 / (0.4 * c ** 2))
                return (comp1 + comp2) # / (np.sum(a+a2))

            fitted_params, _ = optimize.curve_fit(guassianfunc, x_values, y_values,
                                                  p0=[amplitude_guess, mean_guess, sigma_guess])
            print(fitted_params)

            fitted_curve_plot_y = guassianfunc(plot_x, *fitted_params)
        else:
            fn = interpolate.UnivariateSpline(x_values, y_values, k=2, w=weights)
            fitted_curve_plot_y = np.clip(fn(plot_x), 0.0, float("inf"))

        plt.plot(plot_x, fitted_curve_plot_y, color='red')

        if show:
            plt.show()


imagedirs = os.listdir(PATH)
filenames = []
for dir in imagedirs:
    if dir[:5] == 'mtfma':
        filename = os.path.join(PATH, dir, sfrfilename)
        filenames.append(filename)

focusset = FocusSet(filenames)
axis = SAGGITAL
focusset.plot_sfr_vs_focus(4000, 2000, -1, axis)
focusset.plot_sfr_vs_focus(2000, 2000, -1, axis, show=True)

# field.plot(SAGGITAL, 1, detail=1.5)
# field.plot(MERIDIONAL, 1, detail=1.5)


