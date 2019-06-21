import csv
import os
import timeit
import logging
import numpy as np
from scipy import fftpack as scipyfftpack
from scipy import interpolate, optimize
import  matplotlib.pyplot as plt

log = logging.getLogger(__name__)
np.seterr(all='raise')
log.setLevel(logging.DEBUG)

ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
log.addHandler(ch)

SFRFILENAME = 'edge_sfr_values.txt'

CURRENT_JITTER_CODE_VERSION = 2

MULTIPROCESSING = 8  # Number of processes to use (1 to disable multiprocessing)

SAGITTAL = "SAGITTAL"
MERIDIONAL = "MERIDIONAL"
MEDIAL = "MEDIAL"
BOTH_AXES = "BOTH"
ALL_THREE_AXES = "ALL THREE AXES"

SAGITTAL_COMPLEX = "SAGITTAL_COMPLEX"
MERIDIONAL_COMPLEX = "MERIDIONAL_COMPLEX"
SAGITTAL_REAL = "SAGITTAL_REAL"
MERIDIONAL_REAL = "MERIDIONAL_REAL"
SAGITTAL_IMAG = "SAGITTAL_IMAJ"
MERIDIONAL_IMAG = "MERIDIONAL_IMAJ"
SAGITTAL_ANGLE = "SAGGITAL_ANGLE"
MERIDIONAL_ANGLE = "MERIDIONAL_ANGLE"

COMPLEX_AXES = [SAGITTAL_COMPLEX, MERIDIONAL_COMPLEX, SAGITTAL_REAL, MERIDIONAL_REAL, SAGITTAL_IMAG, MERIDIONAL_IMAG, SAGITTAL_ANGLE, MERIDIONAL_ANGLE]
REAL_AXES = [SAGITTAL_REAL, MERIDIONAL_REAL]
IMAG_AXES = [SAGITTAL_IMAG, MERIDIONAL_IMAG, MERIDIONAL_ANGLE, SAGITTAL_ANGLE]
SAGITTAL_AXES = [SAGITTAL, SAGITTAL_REAL, SAGITTAL_IMAG, SAGITTAL_COMPLEX, SAGITTAL_ANGLE]
MERIDIONAL_AXES = [MERIDIONAL, MERIDIONAL_REAL, MERIDIONAL_IMAG, MERIDIONAL_COMPLEX, MERIDIONAL_ANGLE]
POLAR_AXES = [SAGITTAL_ANGLE, MERIDIONAL_ANGLE]

COMPLEX_CARTESIAN = 1
COMPLEX_POLAR_TUPLE = 2
COMPLEX_CARTESIAN_REAL_TUPLE = 3

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

FIELD_SMOOTHING_MIN_POINTS = 16
FIELD_SMOOTHING_MAX_RATIO = 0.3
FIELD_SMOOTHING_ORDER = 3

LOW_BENCHMARK_FSTOP = 14
HIGH_BENCHBARK_FSTOP = 2.8
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
    return (1.0 - height) ** 1

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


def tukey(x, alpha):
    tukey_window = np.cos(np.clip((abs(x) - 1.0 + alpha) * np.pi / alpha, 0, np.pi)) + 1
    return tukey_window

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
        self.exif = {"NOTHING HERE FOR SPACE PURPOSES": True}
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
                        # if row[0] in self.exif:
                        #     self.exif[row[0]+"_dup"] = row[1]
                        # else:
                        #     self.exif[row[0]] = row[1]

                        tag, value = row[:2]
                        # print(tag, value)
                        if tag == "Aperture":
                            fl = float(value[:])
                            self.aperture = 1.25 if fl == 1.2 else fl
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
        print("Aperture is {}".format(self.aperture))

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
    def focal_length(self, fl):
        self.focal_length_str = "{} mm".format(fl)

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

NICECOLOURS = ['red',
               'green',
               'blue',
               'darkviolet']

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


with open("photopic.csv", 'r') as photopic_file:
    reader = csv.reader(photopic_file, delimiter=',', quotechar='|')
    waves, mags = zip(*reader)
    photopic_fn = interpolate.InterpolatedUnivariateSpline([float(_) for _ in waves], [float(_) for _ in mags], k=1)

# plotfreqs = np.linspace(400, 700, 50)
# plt.plot(plotfreqs, photopic_fn(plotfreqs))
# plt.show()

def convert_complex(tup, type):
    if type == COMPLEX_CARTESIAN_REAL_TUPLE:
        return tup
    if type == COMPLEX_CARTESIAN:
        return tup[0] + 1j * tup[1]
    if type == COMPLEX_POLAR_TUPLE:
        r, i = tup
        return (r**2 + i**2)**0.5, np.angle(r + 1j * i)

def convert_complex_from_polar(tup, type):
    if type == COMPLEX_CARTESIAN_REAL_TUPLE:
        return tup[0] * np.cos(tup[1]), tup[0] * np.sin(tup[1])
    if type == COMPLEX_CARTESIAN:
        return tup[0] * np.exp(1j * tup[1])
    if type == COMPLEX_POLAR_TUPLE:
        return tup

def tryfloat(inp):
    try:
        return float(inp)
    except ValueError:
        return inp


class FocusSetData:
    def __init__(self):
        self.merged_mtf_values = None
        self.sag_mtf_values = None
        self.mer_mtf_values = None
        self.mtf_means = None
        # self.focus_values = None
        self.max_pos = None
        self.weights = None
        self.exif = None
        self.cauchy_peak_x = None
        self.x_loc = None
        self.y_loc = None
        self.hints = {}
        self.wavefront_data = [("", {})]

    def get_wavefront_data_path(self, seed="less"):
        try:
            return "wavefront_results/Seed{}/f{:.2f}/".format(seed, self.exif.aperture)
        except AttributeError:
            return "wavefront_results/Seed{}/f{:.2f}/".format(seed, 0)

D50 = {
380: 24.875289,
385: 27.563481,
390: 30.251674,
395: 40.040332,
400: 49.828991,
405: 53.442452,
410: 57.055912,
415: 58.804446,
420: 60.552981,
425: 59.410306,
430: 58.267630,
435: 66.782105,
440: 75.296579,
445: 81.505921,
450: 87.715262,
455: 89.377806,
460: 91.040350,
465: 91.389339,
470: 91.738329,
475: 93.587777,
480: 95.437226,
485: 93.832642,
490: 92.228058,
495: 94.083274,
500: 95.938491,
505: 96.364129,
510: 96.789768,
515: 97.020168,
520: 97.250568,
525: 99.719339,
530: 102.188110,
535: 101.500286,
540: 100.812463,
545: 101.578486,
550: 102.344510,
555: 101.172255,
560: 100.000000,
565: 98.856409,
570: 97.712817,
575: 98.290562,
580: 98.868307,
585: 96.143758,
590: 93.419210,
595: 95.490174,
600: 97.561139,
605: 98.335311,
610: 99.109482,
615: 98.982006,
620: 98.854530,
625: 97.185755,
630: 95.516980,
635: 97.061662,
640: 98.606343,
645: 97.006890,
650: 95.407437,
655: 96.649097,
660: 97.890758,
665: 100.274818,
670: 102.658878,
675: 100.722246,
680: 98.785615,
685: 92.936539,
690: 87.087464,
695: 89.179124,
700: 91.270785,
705: 91.925918,
710: 92.581051,
715: 84.591223,
720: 76.601396,
725: 81.418425,
730: 86.235455,
735: 89.262560,
740: 92.289664,
745: 85.138388,
750: 77.987113,
755: 67.745912,
760: 57.504710,
765: 70.080157,
770: 82.655604,
775: 80.341321,
780: 78.027038}

nms, spds = zip(*D50.items())
d50_interpolator = interpolate.InterpolatedUnivariateSpline(np.array(nms) * 1e-3, spds, k=1)

# import cupy
# from cupyx.scipy import fftpack


def get_good_fft_sizes():
    _all = []
    _upto = 2048
    _factors = [2, 3, 5]
    _power_lst = []
    for factor in _factors:
        powers = np.arange(-1, int(np.log(_upto) / np.log(factor) + 1.1))
        _power_lst.append(powers)

    _power_lst = [np.arange(-1, 14), [-1,0,1,2,3,4,5,6], [-1,0,1,2,3]]
    print(_power_lst)

    mesh = np.meshgrid(*_power_lst)
    for powers in zip(*[_.flatten() for _ in mesh]):
        sum = 1
        for power, factor in zip(powers, _factors):
            # print(factor, power)
            if power != -1:
                sum *= factor ** power
        # print(sum)
        _all.append(sum)
        # print()
    unique = np.unique(_all)
    unique = unique[unique <= _upto]
    uniquebig = unique[unique >= 64]
    for _ in range(2):
        unique_worth_it = []
        best_time = np.inf
        for size in np.flip(uniquebig):
            if size % 2 == 1:
                continue
            # arr = cupy.ones((size, size)) * 0.2 + 0.1j
            if size == _upto:
                runtimes = 3
            else:
                runtimes = 2
            for _ in range(runtimes):
                reps = int(100 * (2048 + 256)**2 / (size+256)**2)
                # time = timeit.timeit("ndimage.affine_transform(cupy.abs(fftpack.fft(arr))**2, transform, offset, order=1)", number=reps,
                #                      setup="from cupyx.scipy import fftpack, ndimage; import cupy;"
                #                            "transform = cupy.array([[1.01,0.01],[0.99, -0.01]]);offset=0.01;"
                #                            "arr = cupy.ones(({},{}), dtype='complex128') * 0.2 + 0.1j".format(size, size)) / reps * 1000
                reps = int(2 * (2048 + 256)**2 / (size+256)**2)
                # time = timeit.timeit("ndimage.affine_transform(numpy.abs(fftpack.fft(arr))**2, transform, offset, order=1)", number=reps,
                #                      setup="from scipy import fftpack, ndimage; import numpy;"
                #                            "transform = numpy.array([[1.01,0.01],[0.99, -0.01]]);offset=0.01;"
                #                            "arr = numpy.ones(({},{}), dtype='complex128') * 0.2 + 0.1j".format(size, size)) / reps * 1000
                time = timeit.timeit("cupy.abs(fftpack.fft(arr))**2", number=reps,
                                     setup="from cupyx.scipy import fftpack, ndimage; import cupy;"
                                           "transform = cupy.array([[1.01,0.01],[0.99, -0.01]]);offset=0.01;"
                                           "arr = cupy.ones(({},{}), dtype='complex128') * 0.2 + 0.1j".format(size, size)) / reps * 1000
                # time = timeit.timeit("numpy.abs(fftpack.fft(arr))**2", number=reps,
                #                      setup="from scipy import fftpack, ndimage; import numpy;"
                #                            "transform = numpy.array([[1.01,0.01],[0.99, -0.01]]);offset=0.01;"
                #                            "arr = numpy.ones(({},{}), dtype='complex128') * 0.2 + 0.1j".format(size, size)) / reps * 1000
            print("FFT {}, {}s".format(size, time))
            if time < best_time:
                print("Worth it!")
                best_time = time
                unique_worth_it.append(size)
        print(repr(np.array(unique_worth_it)))
    return uniquebig


# CUDA_GOOD_FFT_SIZES = get_good_fft_sizes()
# exit()
CUDA_GOOD_FFT_SIZES = np.flip(np.array([2048, 2000, 1944, 1920, 1800, 1728, 1620, 1600, 1536, 1500, 1458,
                                1440, 1350, 1296, 1280, 1200, 1152, 1080, 1024, 1000, 972, 960,
                                900, 864, 810, 800, 768, 750, 720, 648, 640, 600, 576,
                                540, 512, 486, 480, 450, 432, 400, 384, 324, 320, 300,
                                288, 270, 256, 216, 160, 144, 128, 96, 64]))
CPU_GOOD_FFT_SIZES = np.flip(np.array([2048, 2000, 1944, 1800, 1728, 1620, 1536, 1500, 1458, 1440, 1350,
       1296, 1200, 1152, 1080, 1024, 1000,  972,  900,  864,  810,  768,
        750,  720,  648,  640,  600,  576,  540,  512,  500,  486,  480,
        450,  432,  400,  384,  360,  324,  320,  300,  288,  270,  256,
        250,  240,  216,  200,  192,  180,  162,  160,  150,  144,  128,
        120,  108,  100,   96,   90,   80,   72,   64]))

# CUDA_GOOD_FFT_SIZES = np.array((768,))
# CPU_GOOD_FFT_SIZES = np.array((256,))

class NoPhaseData(Exception):
    pass
class InvalidFrequency(Exception):
    pass


def _norm_phase_and_magnitude(r, i, x, inc_neg_freqs=False, return_type=COMPLEX_CARTESIAN, plot=False):
    """
    Normalises complex phase in array

    Zero frequency is assumed to be at index 0 (ie. unshifted)

    :param r: Real component
    :param i: Imaginary component (as real float)
    :param x: Frequencies
    :param inc_neg_freqs: Includes second half of FFT with neg. frequencies
    :param return_type: default COMPLEX_CARTESIAN (ie. real + imag * 1j)
    :return: Normalised result
    """

    # def custom_unwrap(pha):



    if not inc_neg_freqs:
        meanlen = len(x)
    else:
        meanlen = int(len(x) / 2)
    mag = (r ** 2 + i ** 2) ** 0.5
    phase = np.unwrap(np.angle(r + i*1j))
    weights = mag[:meanlen] * np.linspace(1,0,meanlen)
    weights = np.zeros(meanlen)
    # weights[1] = 1
    weights = mag[:meanlen] ** 2
    meanphase = np.average(phase[:meanlen], weights=weights)
    mean_x = np.average(x[:meanlen], weights=weights)
    phase_shift = - (meanphase / mean_x) * x
    if plot:
        oldphase = phase.copy()
    phase += phase_shift
    if inc_neg_freqs and 1:
        phase[meanlen:] = -np.flip(phase[:meanlen])

    # new_meanphase = np.average(phase[:meanlen], weights=weights)
    # if new_meanphase < 0:
    #     phase *= -1

    if x[0] == 0:
        mag /= mag[0]
    if plot:
        plotx = x[:meanlen]
        if inc_neg_freqs and 0:
            s = np.fft.fftshift
        else:
            s = lambda _: _
        fig, (a1, a2) = plt.subplots(1,2)
        a1.plot(x, s(r), label='real')
        a1.plot(x, s(i), label='imag')
        # a1.plot(plotx, weights, label='weights')
        a1.plot(x, s(mag), label='mag')
        a2.plot(x, s(oldphase), label='oldphase')
        a2.plot(x, s(phase), label='newphase')
        a2.plot(x, s(phase_shift), label="phaseshift")
        nr, ni = convert_complex_from_polar((mag, phase), COMPLEX_CARTESIAN_REAL_TUPLE)
        a1.plot(x, s(nr), label="new real")
        a1.plot(x, s(ni), label="new imag")
        a1.legend()
        a2.legend()
        plt.show()
    return convert_complex_from_polar((mag, phase), return_type)


def ___test_phase_normalisation():
    a = fastgauss(np.arange(64)**2, 1.0, 32**2, 14**2)
    b = np.flip(fastgauss(np.arange(64)**2, 1.0, 32**2, 14**2))
    a /= a.max()
    b /= b.max()
    a = np.roll(a, -3)
    b = np.roll(b, 7)
    ft = np.fft.fft(np.fft.fftshift(a))
    ft_b = np.fft.fft(np.fft.fftshift(b))
    ftr, fti = normalised_centreing_fft(ft.real, ft.imag, np.arange(64), return_type=COMPLEX_CARTESIAN_REAL_TUPLE, inc_neg_freqs=True, plot=True)
    ftr_b, fti_b = normalised_centreing_fft(ft_b.real, ft_b.imag, np.arange(64), return_type=COMPLEX_CARTESIAN_REAL_TUPLE, inc_neg_freqs=True, plot=True)
    plt.plot(a, '--', color="green", alpha=0.5)
    plt.plot(b, '--', color="gray", alpha=0.5)
    plt.plot(ftr[:16], color="red", alpha=0.5)
    plt.plot(fti[:16], color="purple", alpha=0.5)
    plt.plot(ftr_b[:16], color="orange", alpha=0.5)
    plt.plot(fti_b[:16], color="blue", alpha=0.5)
    # plt.plot(ft.real[:16], '--', color="red", alpha=0.5)
    # plt.plot(ft.imag[:16], '--', color="purple", alpha=0.5)
    newgauss = np.fft.fftshift(np.fft.ifft(ftr + 1j * fti))
    newgauss_b = np.fft.fftshift(np.fft.ifft(ftr_b + 1j * fti_b))
    plt.plot(newgauss.real / newgauss.real.max(), color="green", alpha=0.5)
    plt.plot(newgauss_b.real / newgauss_b.real.max(), color="black", alpha=0.5)
    plt.show()


def normalised_centreing_fft(y, x=None, return_type=COMPLEX_CARTESIAN, engine=np, fftpack=None, plot=False):
    """
    Normalises complex wrapped_phase in array

    Zero frequency is assumed to be at index 0 (ie. unshifted)

    :param x: x-axis
    :param y: input to fft
    :param return_type: default COMPLEX_CARTESIAN (ie. real + imag * 1j)
    :return: Normalised result
    """

    if x is None:
        x = engine.arange(len(y))

    if fftpack is None:
        fftpack = scipyfftpack
    yzers = (y == 0).sum() == len(y)
    if yzers:
        return convert_complex((np.zeros_like(x), np.zeros_like(x)), type=return_type)

    if y.sum() == 0:
        mid = x.mean()
    else:
        mid = (x * y).sum() / y.sum()

    ftr = fftpack.fft(engine.fft.fftshift(y))
    ftr /= abs(ftr[0])
    meanlen = int(len(x) / 2)

    mag = abs(ftr)
    phase = engine.angle(ftr)
    phase_shift = (mid - meanlen) * x
    if plot:
        oldphase = phase.copy()
    phase += phase_shift * engine.pi * 2 / len(x)
    phase[meanlen:] = -engine.flip(phase[:meanlen], axis=0)

    if plot:
        plotx = x[:meanlen]
        if 0:
            s = engine.fft.fftshift
        else:
            s = lambda _: _
        fig, (a1, a2) = plt.subplots(1,2)
        a1.plot(x, s(ftr.real), label='real')
        a1.plot(x, s(ftr.imag), label='imag')
        # a1.plot(plotx, weights, label='weights')
        a1.plot(x, s(mag), label='mag')
        a2.plot(x, s(oldphase), label='oldphase')
        a2.plot(x, s(phase), label='oldwrappedphase')
        # a2.plot(x, s(wrapped_phase), label='newphase')
        a2.plot(x, s(phase_shift), label="phaseshift")
        nr, ni = convert_complex_from_polar((mag, phase), COMPLEX_CARTESIAN_REAL_TUPLE)
        a1.plot(x, s(nr), label="new real")
        a1.plot(x, s(ni), label="new imag")
        a1.legend()
        a2.legend()
        plt.show()
    return convert_complex_from_polar((mag, phase), return_type)


def _test_phase_normalisation():
    a = fastgauss(np.arange(64)**2, 1.0, 32**2, 14**2)
    # a = fastgauss(np.arange(64), 1.0, 32, 5)
    b = fastgauss(np.arange(64)**2, 1.0, 32**2, 14**2)
    # b = np.flip(fastgauss(np.arange(64), 1.0, 32, 5))
    a /= a.max()
    b /= b.max()
    a = np.roll(a, -7)
    b = np.roll(b, 4)
    ftr, fti = normalised_centreing_fft(np.arange(64), a, return_type=COMPLEX_CARTESIAN_REAL_TUPLE, plot=True)
    ftr_b, fti_b = normalised_centreing_fft(np.arange(64), b, return_type=COMPLEX_CARTESIAN_REAL_TUPLE, plot=True)
    plt.plot(a, '--', color="green", alpha=0.5)
    plt.plot(b, '--', color="gray", alpha=0.5)
    plt.plot(ftr[:16], color="red", alpha=0.5)
    plt.plot(fti[:16], color="purple", alpha=0.5)
    plt.plot(ftr_b[:16], color="orange", alpha=0.5)
    plt.plot(fti_b[:16], color="blue", alpha=0.5)
    # plt.plot(ft.real[:16], '--', color="red", alpha=0.5)
    # plt.plot(ft.imag[:16], '--', color="purple", alpha=0.5)
    newgauss = np.fft.fftshift(np.fft.ifft(ftr + 1j * fti))
    newgauss_b = np.fft.fftshift(np.fft.ifft(ftr_b + 1j * fti_b))
    plt.plot(newgauss.real / newgauss.real.max(), color="green", alpha=0.5)
    plt.plot(newgauss_b.real / newgauss_b.real.max(), color="black", alpha=0.5)
    plt.show()


def __test_normalisation2():
    x = np.arange(64)
    # ga = np.roll(fastgauss(np.arange(64)**2, 1.0, 32**2, 14**2), -2)
    # gb = np.roll(fastgauss(np.arange(64)**2, 1.0, 32**2, 14**2), 3)
    ga = np.roll(fastgauss(np.arange(64), 1.0, 32, 3.5), -18)
    gb = np.roll(fastgauss(np.arange(64), 1.0, 32, 3.5), 2)

    print((x*ga).sum() / ga.sum())
    print((x*gb).sum() / gb.sum())

    f, (ax1, ax2) = plt.subplots(1,2)
    ax1.plot(x, ga, label="ga")
    ax1.plot(x, gb, label="gb")

    ffta = np.fft.fft(np.fft.fftshift(ga))
    fftb = np.fft.fft(np.fft.fftshift(gb))

    ffta /= abs(ffta[0])
    fftb /= abs(fftb[0])

    v_ffta = abs(ffta) > 1e-10
    v_fftb = abs(fftb) > 1e-10

    ff = lambda _: _
    aa = ff(np.angle(ffta))
    aa[~v_ffta] = np.nan
    ab = ff(np.angle(fftb))
    ab[~v_fftb] = np.nan
    ax2.plot(x, aa, label="angle a")
    ax2.plot(x, ab, label="angle b")
    ax2.plot(x, np.zeros_like(x))
    ax1.plot(x, np.abs(ffta), label="mag a")
    ax1.plot(x, np.abs(fftb), label="mag b")
    ax1.plot(x, np.imag(ffta), label="i a")
    ax1.plot(x, np.imag(fftb), label="i b")
    ax1.legend()
    ax2.legend()
    plt.show()

    # _test_phase_normalisation()
    exit()