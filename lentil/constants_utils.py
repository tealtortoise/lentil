import csv
import os
import numpy
import numpy as np
from scipy import interpolate
import  matplotlib.pyplot as plt

SFRFILENAME = 'edge_sfr_values.txt'

DEFAULT_PIXEL_SIZE = 4e-6
SAGITTAL = "SAGITTAL"
MERIDIONAL = "MERIDIONAL"
MEDIAL = "MEDIAL"
BOTH_AXES = "BOTH"

PLOT_ON_FIT_ERROR = True
PLOT_MTF50_ERROR = True

TRUNCATE_MTF_LOBES = True

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

IMAGE_WIDTH = 6000
IMAGE_HEIGHT = 4000
IMAGE_DIAGONAL = (IMAGE_WIDTH**2 + IMAGE_HEIGHT**2)**0.5
THETA_BOTTOM_RIGHT = np.arctan(IMAGE_HEIGHT / IMAGE_WIDTH)
THETA_TOP_RIGHT = np.pi * 2.0 - THETA_BOTTOM_RIGHT

DEFAULT_SENSOR_DIAGONAL = IMAGE_DIAGONAL * DEFAULT_PIXEL_SIZE

ACUTANCE_PRINT_HEIGHT = 0.6
ACUTANCE_VIEWING_DISTANCE = 0.74

CONTOUR2D = 0
PROJECTION3D = 1
SMOOTH2D = 3

DEFAULT_FREQ = -2
MTF50 = -1
AUC = -2
ACUTANCE = -3

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
    mulfreq = np.clip(freq / 8.0 * fstop, 0, 1)
    if calibration is None:
        calibration_mul = 1.0
    else:
        interpfn = interpolate.InterpolatedUnivariateSpline(RAW_SFR_FREQUENCIES[:],
                                                            np.pad(calibration, (0,64-len(calibration)),
                                                                   'constant',
                                                                   constant_values=0), k=1)
        calibration_mul = np.clip(interpfn(freq), 1e-6, np.inf)
    return 2.0 / np.pi * (np.arccos(mulfreq) - mulfreq * (1 - mulfreq ** 2) ** 0.5) * calibration_mul


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


def cauchy(xin, max, x0, gamma):
    return max / (1.0 + ((xin - x0) / gamma) ** 2)


class EXIF:
    def __init__(self, sfr_pathname=None, exif_pathname=None):
        self.exif = {}
        value = ""
        self.aperture = 1.0
        self.focal_length = value
        self.lens_model = value
        self.max_aperture = value
        self.distortionexif = value
        self.ca_exif = value

        if exif_pathname is None:
            pathsplit = os.path.split(sfr_pathname)

            fnamesplit = pathsplit[1].split(".")
            exiffilename = ".".join(fnamesplit[:2]) + ".exif.csv"
            exif_pathname = os.path.join(pathsplit[0], exiffilename)
            print(exif_pathname)
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
                        self.focal_length = value
                    elif tag == "Lens Model":
                        self.lens_model = value
                    elif tag == "Max Aperture Value":
                        self.max_aperture = value
                    elif tag == "Geometric Distortion Params":
                        self.distortionexif = value
                    elif tag == "Chromatic Aberration Params":
                        self.ca_exif = value
        except FileNotFoundError:
            print("No EXIF found")

    @property
    def summary(self):
        if len(self.exif) is 0:
            return "No EXIF available"
        return "{} at {}, f/{}".format(self.lens_model, self.focal_length, self.aperture)


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
    raise FileNotFoundError("Can't find results at path {}".format(basepath))
