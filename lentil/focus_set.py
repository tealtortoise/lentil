import colorsys
import csv
import math

import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from scipy import optimize

from lentil.sfr_point import SFRPoint
from lentil.sfr_field import SFRField
from lentil.constants_utils import BOTH_AXES, diffraction_mtf


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

    def plot_ideal_focus_field(self, freq, axis=BOTH_AXES, plot_type=1):
        x_values, y_values = self.fields[0].build_axis_points(24, 16)
        z_values = np.ndarray((len(y_values), len(x_values)))

        # fn = self.get_simple_interpolation_fn(axis)
        for x_idx, x in enumerate(x_values):
            for y_idx, y in enumerate(y_values):
                _, peak, _, _ = self.find_best_focus(x, y, freq, BOTH_AXES)
                # _, peak_m, _, _ = self.find_best_focus(x, y, freq, MERIDIONAL)
                # _, peak_s, _, _ = self.find_best_focus(x, y, freq, SAGGITAL)
                # z_values[y_idx, x_idx] = (peak_m + peak_s) / 2.0
                z_values[y_idx, x_idx] = peak

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
                new_cmap[a, :] = my_cmap[int(mod * 256), :]

            # my_col = plt.cm.jet(1.0 - z_values)
            # my_col[:, :, -1] = 0.85

            mycmap = ListedColormap(new_cmap)
            norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
            surf = ax.plot_surface(x, y, z_values, cmap=mycmap, norm=norm,
                                   rstride=1, cstride=1, linewidth=1, antialiased=True)
            fig.colorbar(surf, shrink=0.5, aspect=5)
        plt.show()

    def find_best_focus(self, x, y, freq, axis=BOTH_AXES, plot=False, strict=True):
        """
        Get peak SFR at specified location and frequency vs focus, optionally plot.
        This does not call (.show()) on the plot.

        :param x: position in field
        :param y: position in field
        :param freq: frequency of interest (-1 for MTF50)
        :param axis: MERIDIONAL or SAGGITAL or BOTH_AXES
        :param plot: Draws plot if True
        :return: Tuple (focus position of peak, height of peak)
        """
        # Get SFR from each field
        y_values = []
        for field in self.fields:
            y_values.append(field.interpolate_value(x, y, freq, axis))
        x_values = np.arange(0, len(y_values), 1)  # Arbitrary focus units
        y_values = np.array(y_values)
        if plot:
            plt.plot(x_values, y_values)
            # plt.show()
        # Use quadratic spline as fitting curve

        def fit_poly(polyx, polyy, w):
            # fn = interpolate.UnivariateSpline(x_values, y_values, k=2, w=w)
            poly = np.polyfit(polyx, polyy, 2, w=w)
            peak_x = np.roots(np.polyder(poly, 1))[0]
            peak_y = np.polyval(poly, peak_x)
            return peak_x, peak_y, poly

        weighting_power = max(0, 4 - np.exp(-len(x_values) / 20 + 1.4))
        # print("Weighting power {}".format(weighting_power))
        # Plot weighting curve
        # x = np.arange(1, 50)
        # y = 4 - np.exp(-x / 20 + 1.4)
        # plt.plot(x, y)
        # plt.show();exit()

        high_values = (y_values > (np.amax(y_values) * 0.7))
        n_high_values = high_values.sum()
        print("Dataset has {} high y_values".format(n_high_values))
        # exit()

        plot_x = np.linspace(0, max(x_values), 100)
        plot_x_wide = np.linspace(-5, max(x_values) +5, 100)

        # 1st stage
        weights = (y_values / np.max(y_values)) ** (weighting_power + 2)  # Favour values closer to maximum
        p_peak_x, p_peak_y, poly_draft = fit_poly(x_values, y_values, weights)
        print("1st stage peak is {:.3f} at {:.2f}".format(p_peak_y, p_peak_x))
        if plot:
            fitted_curve_plot_y_1 = np.clip(np.polyval(poly_draft, plot_x), 0.0, float("inf"))
            plt.plot(plot_x, fitted_curve_plot_y_1, color='red')

        # 2nd stage, use only fields close to peak
        closest_field_to_peak = int(p_peak_x + 0.5)

        if n_high_values >= 3:
            trimmed_x_values = x_values[high_values]
            trimmed_y_values = y_values[high_values]
        else:
            trim_low = max(0, closest_field_to_peak - 2)
            trim_high = min(len(y_values), closest_field_to_peak + 3)
            trimmed_x_values = x_values[trim_low: trim_high]
            trimmed_y_values = y_values[trim_low: trim_high]

        if len(trimmed_y_values) < 3:
            raise Exception("Not enough data points around peak to analyse")

        trimmed_weights = (trimmed_y_values / np.max(trimmed_y_values)) ** 5  # Favour values closer to maximum

        p_peak_x, p_peak_y, poly = fit_poly(trimmed_x_values, trimmed_y_values, trimmed_weights)

        poly[2] -= p_peak_y * diffraction_mtf(freq)
        acceptable_focus_roots = np.roots(poly)

        print("2nd stage peak is {:.3f} at {:.2f}".format(p_peak_y, p_peak_x))

        if strict and not 0.0 < p_peak_x < x_values[-1]:
            raise Exception("Focus peak appears to be out of range, strict=True")
        if not -0.5 < p_peak_x < (x_values[-1] + 0.5):
            raise Exception("Focus peak appears to be out of range, strict=False")

        if plot:
            fitted_curve_plot_y_2 = np.clip(np.polyval(poly, plot_x), 0.0, float("inf"))
            plt.plot(plot_x, fitted_curve_plot_y_2, color='orange')

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
        print(x, y)
        fitted_params, _ = optimize.curve_fit(twogauss, x_values, y_values, bounds=bounds, sigma=sigmas,
                                              p0=(p_peak_y, p_peak_x, 1.0, 0.1))
        print("3rd stage peak is {:.3f} at {:.2f}".format(fitted_params[0], fitted_params[1]))
        print("Gaussian sigma: {:.3f}, peaky {:.3f}".format(*fitted_params[2:]))

        if plot:
            fitted_curve_plot_y_3 = twogauss(plot_x, *fitted_params)
            plt.plot(plot_x, fitted_curve_plot_y_3, color='green')
        g_peak_x = fitted_params[1]
        g_peak_y = fitted_params[0]

        if n_high_values < 3:
            # 3rd Stage, fit gaussians
            # Use sum of two co-incident gaussians as fitting curve

            print("Only {} high values, using guassian fit".format(n_high_values))
            final_peak_x = (p_peak_x + g_peak_x) / 2.0
            final_peak_y = (p_peak_y + g_peak_y) / 2.0
        else:
            final_peak_x = p_peak_x
            final_peak_y = p_peak_y
        return final_peak_x, final_peak_y, acceptable_focus_roots[0], acceptable_focus_roots[1]