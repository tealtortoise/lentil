import numpy as np
from matplotlib import pyplot as plt
from scipy import interpolate, optimize


import lentil.constants_utils


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
        self.pixelsize = pixelsize or lentil.constants_utils.DEFAULT_PIXEL_SIZE
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
            self._interpolate_fn = interpolate.InterpolatedUnivariateSpline(lentil.constants_utils.RAW_SFR_FREQUENCIES,
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
        if axis == lentil.constants_utils.SAGITTAL:
            return self.is_saggital
        if axis == lentil.constants_utils.MERIDIONAL:
            return self.is_meridional
        if axis == lentil.constants_utils.BOTH_AXES:
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