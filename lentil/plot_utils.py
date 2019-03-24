import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib

from lentil.constants_utils import *

class FieldPlot:
    def __init__(self):
        self.zdata = None
        self.xticks = None
        self.yticks = None
        self.wdata = None  # Mystical fourth dimension

        self.xmin = None
        self.ymin = None
        self.xmax = None
        self.ymax = None
        self.zmin = None
        self.zmax = None
        self.wmin = None
        self.wmax = None
        self.title = "Untitled"
        self.xlabel = "x-axis"
        self.ylabel = "y-axis"
        self.zlabel = "z-axis"

        self.xreverse = False
        self.yreverse = False

        self.contours = None
    
    @property
    def _xmin(self):
        if self.xmin is None:
            return np.min(self.xticks)
        else:
            return self.xmin
    @property
    def _ymin(self):
        if self.ymin is None:
            return np.min(self.yticks)
        else:
            return self.ymin
    @property
    def _xmax(self):
        if self.xmax is None:
            return np.max(self.xticks)
        else:
            return self.xmax
    @property
    def _ymax(self):
        if self.ymax is None:
            return np.max(self.yticks)
        else:
            return self.ymax
    @property
    def _zmax(self):
        if self.zmax is None:
            return np.max(self.zdata)
        else:
            return self.zmax
    @property
    def _zmin(self):
        if self.zmin is None:
            return np.min(self.zdata)
        else:
            return self.zmin
    @property
    def _wmax(self):
        if self.wmax is None:
            return np.max(self.wdata)
        else:
            return self.wmax
    @property
    def _wmin(self):
        if self.wmin is None:
            return np.min(self.wdata)
        else:
            return self.wmin

    @property
    def _contours(self):
        inc = np.clip((self._zmax - self._zmin) / 20, 0.002, 0.05)
        contours = np.arange(int(self._zmin / inc) * inc - inc, self._zmax + inc, inc)
        return contours

    def set_diffraction_limits(self, freq=AUC, low_fstop=LOW_BENCHMARK_FSTOP, high_fstop=HIGH_BENCHBARK_FSTOP, graphaxis='z'):
        low_perf = diffraction_mtf(freq, low_fstop)
        high_perf = diffraction_mtf(freq, high_fstop)
        setattr(self, graphaxis + "min", low_perf)
        setattr(self, graphaxis + "max", high_perf)

    def contour2d(self, ax=None, show=True):
        # Move on to plotting results
        if ax is None:
            fig, ax = plt.subplots()

        inc = np.clip((self._zmax - self._zmin) / 20, 0.002, 0.05)
        contours = np.arange(int(self._zmin / inc) * inc - inc, self._zmax + inc, inc)
        # contours = np.arange(0.0, 1.0, 0.005)
        
        colors = []
        linspaced = np.linspace(0.0, 1.0, len(contours))
        for lin, line in zip(linspaced, contours):
            colors.append(plt.cm.jet(1.0 - lin))
            # colors.append(colorsys.hls_to_rgb(lin * 0.8, 0.4, 1.0))

        if self.xreverse:
            ax.set_xlim(self._xmax, self._xmin)
        else:
            ax.set_xlim(self._xmin, self._xmax)

        if self.yreverse:
            ax.set_ylim(self._ymax, self._ymin)
        else:
            ax.set_ylim(self._ymin, self._ymax)

        CS = ax.contourf(self.xticks, self.yticks, self.zdata, contours, colors=colors, extend='both')
        CS2 = ax.contour(self.xticks, self.yticks, self.zdata, contours, colors=('black',))
        plt.xlabel(self.xlabel)
        plt.ylabel(self.ylabel)
        plt.clabel(CS2, inline=1, fontsize=10)
        plt.title(self.title)
        if show:
            plt.show()
            return None
        return ax

    def projection3d(self, ax=None, show=True):
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            
        if self.xreverse:
            ax.set_xlim(self._xmax, self._xmin)
        else:
            ax.set_xlim(self._xmin, self._xmax)

        if self.yreverse:
            ax.set_ylim(self._ymax, self._ymin)
        else:
            ax.set_ylim(self._ymin, self._ymax)

        x, y = np.meshgrid(self.xticks, self.yticks)

        alpha = 0.6
        # print(self.wdata)
        # print(self._wmin)
        # print(self._wmax)
        # exit()

        cmap = plt.cm.jet  # Base colormap
        my_cmap = cmap(np.arange(cmap.N))  # Read colormap colours
        my_cmap[:, -1] = alpha  # Set colormap alpha
        new_cmap = np.ndarray((256, 4))

        facecolours = plt.cm.jet(np.clip(1.0 - ((self.wdata - self._wmin) / (self._wmax - self._wmin)), 0.0, 1.0))
        edgecolours = 'b'
        facecolours[:, :, 3] = alpha

        for a in range(256):
            mod = 0.5 - math.cos(a / 256 * math.pi) * 0.5
            new_cmap[a, :] = my_cmap[int(mod * 256), :]

        # mycmap = ListedColormap(new_cmap)
        norm = matplotlib.colors.Normalize(vmin=self._wmin, vmax=self._wmax)

        surf = ax.plot_surface(x, y, self.zdata, facecolors=facecolours, norm=norm, edgecolors=edgecolours,
                               rstride=1, cstride=1, linewidth=1, antialiased=True)
        ax.set_xlabel(self.xlabel)
        ax.set_ylabel(self.ylabel)
        ax.set_zlabel(self.zlabel)
        ax.set_title(self.title)
        if show:
            plt.show()
            return None
        return ax
