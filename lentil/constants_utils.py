import csv
import os
import logging
import numpy as np
from scipy import interpolate
import  matplotlib.pyplot as plt
import prysm

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
log.addHandler(ch)

SFRFILENAME = 'edge_sfr_values.txt'


SAGITTAL = "SAGITTAL"
MERIDIONAL = "MERIDIONAL"
MEDIAL = "MEDIAL"
BOTH_AXES = "BOTH"
ALL_THREE_AXES = "ALL THREE AXES"

PLOT_ON_FIT_ERROR = True
PLOT_MTF50_ERROR = True

TRUNCATE_MTF_LOBES = False

SFR_HEADER = [
    'blockid',
    'edgex',
    'edgey',
    'edgeangle',
    'radialangle'
]

FIELD_SMOOTHING_MIN_POINTS = 24
FIELD_SMOOTHING_MAX_RATIO = 0.3
FIELD_SMOOTHING_ORDER = 3

LOW_BENCHMARK_FSTOP = 22
HIGH_BENCHBARK_FSTOP = 4
# LOW_BENCHMARK_FSTOP = 32
# HIGH_BENCHBARK_FSTOP = 13

# IMAGE_WIDTH = 8256
IMAGE_WIDTH = 6000
# SENSOR_WIDTH = 0.0357
SENSOR_WIDTH = 0.0236

IMAGE_HEIGHT = IMAGE_WIDTH * 2 / 3
IMAGE_DIAGONAL = (IMAGE_WIDTH**2 + IMAGE_HEIGHT**2)**0.5
DEFAULT_PIXEL_SIZE = SENSOR_WIDTH / IMAGE_WIDTH


THETA_BOTTOM_RIGHT = np.arctan(IMAGE_HEIGHT / IMAGE_WIDTH)
THETA_TOP_RIGHT = np.pi * 2.0 - THETA_BOTTOM_RIGHT

CHART_WIDTH = 18 * 0.0254
# CHART_WIDTH = SENSOR_WIDTH * 33
CHART_DIAGONAL = (CHART_WIDTH ** 2 + (CHART_WIDTH * IMAGE_HEIGHT / IMAGE_WIDTH)**2) ** 0.5

DEFAULT_SENSOR_DIAGONAL = IMAGE_DIAGONAL * DEFAULT_PIXEL_SIZE

# LOWAVG_NOMBINS = np.arange(2, 6)
LOWAVG_NOMBINS = np.arange(3, 12)

ACUTANCE_PRINT_HEIGHT = 0.6
ACUTANCE_VIEWING_DISTANCE = 0.74

CONTOUR2D = 0
PROJECTION3D = 1
SMOOTH2D = 3

DEFAULT_FREQ = -2
MTF50 = -1
AUC = -2
ACUTANCE = -3
LOWAVG = -4

DIFFRACTION_WAVELENGTH = 575e-9

FOCUS_SCALE_COC = "Defocus blur circle diameter (µm)"
FOCUS_SCALE_COC_PIXELS = "Defocus blur circle diameter (pixels)"
FOCUS_SCALE_FOCUS_SHIFT = "Image-side long. focus shift (µm)"
FOCUS_SCALE_SUB_FOCUS_SHIFT = "Subject-side focus shift (mm)"
FOCUS_SCALE_RMS_WFE = "RMS Defocus wavefront error (λ)"


def CENTRE_WEIGHTED(height):
    return (1.0 - height) ** 2

def EDGE_WEIGHTED(height):
    return np.clip(1.1 - np.abs(0.6 - height)*1.4, 0.0001, 1.0) ** 2

def CORNER_WEIGHTED(height):
    return height ** 1

def EVEN_WEIGHTED(height):
    return 1.0

def plot_weighting(weightingfn):
    x = np.linspace(0, 1, 100)
    plt.plot(x, weightingfn(x))
    plt.show()

# plot_weighting(EDGE_WEIGHTED)
# exit()


def EVEN_WEIGHTED(height):
    return 1.0


def diffraction_mtf(freq, fstop=8, calibration=None):
    if type(freq) is int and freq == AUC:
        return diffraction_mtf(np.linspace(0, 0.5-1.0/32, 32), fstop, calibration).mean()
    if type(freq) is int and freq == ACUTANCE:
        # print(22, calibration)
        return calc_acutance(diffraction_mtf(RAW_SFR_FREQUENCIES, fstop, calibration))
    mulfreq = np.clip(freq / DEFAULT_PIXEL_SIZE * DIFFRACTION_WAVELENGTH * fstop, 0, 1)
    if calibration is None:
        calibration_mul = 1.0
    else:
        interpfn = interpolate.InterpolatedUnivariateSpline(RAW_SFR_FREQUENCIES[:],
                                                            np.pad(calibration, (0,64-len(calibration)),
                                                                   'constant',
                                                                   constant_values=0), k=1)
        calibration_mul = np.clip(interpfn(freq), 1e-6, np.inf)
    diff = 2.0 / np.pi * (np.arccos(mulfreq) - mulfreq * (1 - mulfreq ** 2) ** 0.5) * calibration_mul
    return diff * 0.98 + 0.02


def calc_acutance(sfr, print_height=ACUTANCE_PRINT_HEIGHT, viewing_distance=ACUTANCE_VIEWING_DISTANCE):
        if viewing_distance is None:
            viewing_distance = max(0.15, print_height ** 0.5)

        def csf(af):  # Contrast sensitivity function
            return 75 * af ** 0.8 * np.exp(-0.2 * af)

        if len(sfr) < 64:
            sfr = np.pad(sfr, (0, 64 - len(sfr)), 'constant', constant_values=0.0)

        print_cy_per_m = RAW_SFR_FREQUENCIES * 4000 / print_height
        cy_per_rad = print_cy_per_m * viewing_distance  # atan Small angle approximation
        cy_per_degree = cy_per_rad / 180 * np.pi

        specific_csf = csf(cy_per_degree)

        total = (specific_csf * sfr).sum() / specific_csf.sum()
        return total


def gaussian_fourier(c):
    f = RAW_SFR_FREQUENCIES
    gauss = np.exp(-f ** 2 * c ** 2 * 0.5)
    # plt.plot(f, gauss);plt.show()
    return gauss


def pixel_aperture_mtf(freq):
    freq = np.clip(freq, 0.0001, 1.0)
    return np.sin(np.pi*freq) / np.pi / freq


def calc_image_height(x, y):
    """
    Calculate image height (distance from centre) ranging from 0.0 to 1.0

    :param x: x loc(s)
    :param y: y loc(s)
    :return: height(s)
    """
    img_height = (((IMAGE_WIDTH / 2) - x) ** 2 + ((IMAGE_HEIGHT / 2) - y) ** 2) ** 0.5 / IMAGE_DIAGONAL * 2
    return img_height


RAW_SFR_FREQUENCIES = np.array([x / 64 for x in range(64)])  # List of sfr frequencies in cycles/pixel


GOOD = [1., 0.98582051, 0.95216779, 0.91605742, 0.88585631, 0.86172936,
     0.84093781, 0.82116408, 0.80170952, 0.78201686, 0.76154796, 0.73985244,
     0.7166293, 0.69158089, 0.66423885, 0.63510484, 0.60407738, 0.57122645,
     0.53737249, 0.50266147, 0.46764089, 0.43269842, 0.39822897, 0.36466347,
     0.33236667, 0.30161039, 0.27266122, 0.24569197, 0.2208242, 0.19810618,
     0.17752172, 0.15900566, 0.14245044, 0.1277121, 0.11462787, 0.10302666,
     0.09274069, 0.08361389, 0.07550579, 0.06829461, 0.06187432, 0.05615253,
     0.05104666, 0.04648352, 0.04239983, 0.03874731, 0.03549705, 0.03264138,
     0.03019484, 0.0281874, 0.0266599, 0.02565582, 0.02520846, 0.02533362,
     0.02601429, 0.02719823, 0.02879615, 0.03068963, 0.03274225, 0.03481336,
     0.0367723, 0.03850572, 0.03992789, 0.04098472]


def fastgauss(gaussx, a,b,c):
    return a * np.exp(-(gaussx - b) ** 2 / (2 * c ** 2))


def twogauss(gaussx, a, b, c, peaky):
    peaky = peaky * np.clip((c - 0.7) / 2.0, 0.0, 1.0)  # No peaky at low sigma
    a1 = 1 / (1 + peaky)
    a2 = peaky / (1 + peaky)
    c1 = c * 1.8
    c2 = c / 1.4
    wide = a1 * np.exp(-(gaussx - b) ** 2 / (2 * c1 ** 2))
    narrow = a2 * np.exp(-(gaussx - b) ** 2 / (2 * c2 ** 2))
    both = (wide + narrow) * a
    return both

count=0

def cauchy(xin, max, x0, gamma):
    global count
    count += 1
    # print(count)
    # print(xin)
    return max / (1.0 + ((xin - x0) / gamma) ** 2)
def c_init(x, y, inc):
    return y, x, 3.0 * inc
def c_bounds(x, y, inc):
    return ((y * 0.98, x - inc * 2, 0.4 * inc,),
            (y * 1.15, x + inc * 2, 100.0 * inc,))
cauchy.initial = c_init
cauchy.bounds = c_bounds

def psysmfit(defocus, defocus_offset, aberr):
    pupil = NollZernike(Z4=defocus + defocus_offset, dia=10, norm=True, **{zedstr: add}, wavelength=wl,
                        opd_unit="um")
    m = MTF.from_pupil(pupil, efl=fl)
    if 0:
        plt.plot(freqs, m.exact_xy(freqs))


# cauchy.bounds = lambda x, y, inc: (highest_data_y * 0.98, mean_peak_x - x_inc * 2, 0.4 * x_inc,), \
#                                     (highest_data_y * 1.15, mean_peak_x + x_inc * 2, 100.0 * x_inc,)


class EXIF:
    def __init__(self, sfr_pathname=None, exif_pathname=None):
        self.exif = {}
        value = ""
        self.aperture = 1.0
        self.focal_length_str = value
        self.lens_model = value
        self.max_aperture = value
        self.distortionexif = value
        self.ca_exif = value

        if exif_pathname is None and sfr_pathname is not None:
            pathsplit = os.path.split(sfr_pathname)

            fnamesplit = pathsplit[1].split(".")
            exiffilename = ".".join(fnamesplit[:2]) + ".exif.csv"
            exif_pathname = os.path.join(pathsplit[0], exiffilename)
            print(exif_pathname)
        if exif_pathname is not None:
            try:
                print("Tring to open {}".format(exif_pathname))
                print(pathsplit)
                with open(exif_pathname, 'r') as file:
                    print("Found EXIF file")
                    reader = csv.reader(file, delimiter=',', quotechar='|')
                    for row in reader:
                        if row[0] in self.exif:
                            self.exif[row[0]+"_dup"] = row[1]
                        else:
                            self.exif[row[0]] = row[1]

                        tag, value = row[:2]
                        # print(tag, value)
                        if tag == "Aperture":
                            self.aperture = float(value[:])
                        elif tag == "Focal Length" and "equivalent" not in value:
                            self.focal_length_str = value
                        elif tag == "Lens Model":
                            self.lens_model = value
                        elif tag == "Max Aperture Value":
                            self.max_aperture = value
                        elif tag == "Geometric Distortion Params":
                            self.distortionexif = value
                        elif tag == "Chromatic Aberration Params":
                            self.ca_exif = value
            except FileNotFoundError:
                log.warning("No EXIF found")

    @property
    def summary(self):
        if len(self.exif) is 0:
            return "No EXIF available"
        return "{} at {}, f/{}".format(self.lens_model, self.focal_length, self.aperture)

    @property
    def angle_of_view(self):
        sensor_diagonal_m = IMAGE_DIAGONAL * DEFAULT_PIXEL_SIZE
        focal_length_m = self.focal_length * 1e-3
        lens_angle_of_view = 2 * np.arctan(sensor_diagonal_m / focal_length_m / 2)
        return lens_angle_of_view

    @property
    def focal_length(self):
        return float(self.focal_length_str.split(" ")[0])

    @focal_length.setter
    def focal_length(self, floatin):
        self.focal_length_str = "{:.1f} mm".format(floatin)


def truncate_at_zero(in_sfr):
    # in_sfr = np.array(in_sfr) + 0.0
    # plt.plot(RAW_SFR_FREQUENCIES[:len(in_sfr)], in_sfr)
    sfr = np.concatenate(([1.0], in_sfr, [0.0]))
    l = len(sfr)
    derivative = sfr[1:l] - sfr[:l-1]
    # plt.plot(RAW_SFR_FREQUENCIES[:l-3], derivative[1:l-2], '--')
    # plt.plot(RAW_SFR_FREQUENCIES[:l-3], sfr[1:l-2], '--')
    # plt.hlines([0], 0, 1, linestyles='dotted')
    # derivative_shift = derivative[:32]
    # second_der = derivative_shift - derivative[:32]
    # plt.plot(RAW_SFR_FREQUENCIES[:l-3], derivative[:l-3])
    cuts = np.all((derivative[1:l-1] > 0.002, derivative[:l-2] < 0.002, sfr[1:l-1] < 0.13), axis=0)
    cumsum = np.cumsum(cuts)
    # plt.plot(RAW_SFR_FREQUENCIES[:l-2], cumsum)
    out_sfr = in_sfr * (cumsum == 0) + 1e-6
    # print(sfr[1:l-1] < 0.08)
    # print(cuts)
    # plt.plot(RAW_SFR_FREQUENCIES[:len(in_sfr)], out_sfr-0.01)
    # plt.show()
    return out_sfr


def fallback_results_path(basepath, number):
    for n in range(number, 2, -1):
        path = os.path.join(basepath, "mtfm{}".format(n))
        if os.path.exists(path):
            for entry in os.scandir(path):
                # if entry.is_file:
                return path
    if os.path.exists(basepath):
        return basepath
    raise FileNotFoundError("Can't find results at path {}".format(basepath))


COLOURS = ['red',
           'orangered',
           'darkorange',
           'green',
           'blue',
           'darkviolet',
           'deeppink',
           'black']


class Calibrator:
    def __init__(self):
        self.calibrations = []
        self.averaged = None
        self.used_calibration = False

    def add_focusset(self, focusset):
        self.calibrations.append((focusset.exif, focusset.build_calibration(fstop=None, opt_freq=AUC, plot=False, writetofile=False,use_centre=False)))
        if focusset.use_calibration:
            self.used_calibration = True

    def average_calibrations(self, absolute=False, plot=True, trim=None):
        if len(self.calibrations) == 0:
            raise ValueError("No Calibrations!")
        exifs, tups = zip(*self.calibrations)
        datas, diffs, cals = zip(*tups)
        data_stack = np.vstack(datas)
        diff_stack = np.vstack(diffs)
        if absolute:
            stack = diff_stack - data_stack
            invert = False
        else:
            stack = diff_stack / data_stack
            invert = self.used_calibration

        if trim is None:
            trim = not self.used_calibration
        if invert:
            if absolute:
                stack = - stack
            else:
                stack = 1 / stack
        if trim:
            length = int(len(self.calibrations) * 0.7)
        else:
            length = len(self.calibrations)

        aucs = stack[:, :30].mean(axis=1)
        sortorder = np.argsort(aucs)
        use_order = sortorder[:length]

        sortedstack = stack[use_order, :]

        weights = np.linspace(1.0, 0, len(sortedstack))

        averaged = np.average(sortedstack, axis=0, weights=weights)
        sortedcallist = []
        sortedexif = []
        for arg in use_order:
            sortedcallist.append(self.calibrations[arg])
            sortedexif.append(exifs[arg])

        print("Averaged {} calibrations".format(len(sortedstack)))
        order = 0
        colour = 0

        plt.plot(RAW_SFR_FREQUENCIES[:len(averaged)], averaged, '-', label="Average", color='black')
        for exif, line in zip(sortedexif, sortedstack):
            # if exif.aperture != 11.0:
            #     continue
            color = 'grey'
            print("Line", exif.summary)
            print(line)
            if exif.aperture > 5.5:
                color = 'red'
            if exif.aperture > 7.9:
                color = 'green'
            if exif.aperture > 10.9:
                color = 'blue'
            if exif.aperture > 15.0:
                color = 'magenta'
            print(exif.aperture, color)
            color = (COLOURS*2)[colour]
            if 1 or order:
                plt.plot(RAW_SFR_FREQUENCIES[:len(line)], line, '-', label=exif.summary, alpha=0.6, color=color)
                colour += 1
            else:
                plt.plot(RAW_SFR_FREQUENCIES[:len(line)], line, '-', label=exif.summary, alpha=0.8, color=color)
            order = (order + 1) % 2
        plt.legend()
        if absolute:
            plt.ylim(-0.15, 0.15)
        else:
            plt.ylim(0, 1.3)

        plt.xlabel("Spatial Frequency (cy/px)")
        plt.xlim(0, 0.5)
        if invert:
            plt.title("Lens MTF vs Diffraction MTF for EXIF F/ number")
            if absolute:
                plt.ylabel("MTF Error (Inverted)")
            else:
                plt.ylabel("Relative MTF")
        else:
            plt.title("Gain required for Lens MTF to match expected diffraction MTF from EXIF")
            if absolute:
                plt.ylabel("MTF Error")
            else:
                plt.ylabel("Gain")
        plt.hlines([1.0], 0, 0.5, linestyles='--', alpha=0.5)
        plt.grid()
        plt.show()
        self.averaged = averaged

    def write_calibration(self):
        if self.used_calibration:
            raise ValueError("Existing calibration was used in at least one FocusSet, run without calibration")
        with open("calibration.csv", 'w') as csvfile:
            csvwriter = csv.writer(csvfile, delimiter=',', quotechar='|')
            csvwriter.writerow(list(RAW_SFR_FREQUENCIES[:len(self.averaged)]))
            csvwriter.writerow(list(self.averaged))
            print("Calibration written!")
