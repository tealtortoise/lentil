#!/usr/bin/python3
import subprocess
import os
import shutil
import sys
import csv
import numpy as np
import matplotlib.pyplot as plt
import lentil

IMAGE_WIDTH = 6000
IMAGE_HEIGHT = 4000
IMAGE_DIAGONAL = (IMAGE_WIDTH**2 + IMAGE_HEIGHT**2)**0.5

CORRECT_DISTORTION = True

def check_if_distortion_corrections_needed(exifstring):
    split = exifstring.split(' ')[1:]
    com_errors = np.array([float(txt) for txt in split[9:17]])
    return sum(np.abs(com_errors)) > 0.1


def distortioncalc(exifstring, castring):
    # GEOEXAMPLE = "400.5555556 0.3535211268 0.5 0.6126760563 0.7070422535 0.7908450704 0.8661971831 0.9352112676 1 1.06056338 -1.172515869 -2.354660034 -3.458114624 -4.434677124 -5.274719238 -5.918792725 -6.328369141 -6.426345825 -6.42634582"
    # CAEXAMPLE = "400.5555556 0.3535211268 0.5 0.6126760563 0.7070422535 0.7908450704 0.8661971831 0.9352112676 1 1.06056338 -3.051757812e-05 -9.155273438e-05 -0.0001525878906 -0.0002136230469 -0.0002746582031 -0.0003356933594 -0.0004577636719 -0.0006103515625 -0.0006103515625 0.0007019042969 0.0007934570312 0.0008850097656 0.0009460449219 0.001037597656 0.001129150391 0.001220703125 0.001434326172 0.001434326172 400.5555556"

    # exifstring = GEOEXAMPLE
    # castring = CAEXAMPLE
    max_height = (3**2+2**2)**0.5 / 2.0
    usepoints = 8
    split = castring.split(' ')[1:]
    rb_heights = np.array([float(txt) for txt in split[:usepoints]]) * (3**2+2**2)**0.5 / 2.0
    red_errors = np.array([float(txt) for txt in split[9:9+usepoints]]) #* -1
    blue_errors = np.array([float(txt) for txt in split[18:18+usepoints]]) #* - 1

    if exifstring is None:
        com_heights = rb_heights
        com_errors = np.zeros(com_heights.shape)
    else:
        split = exifstring.split(' ')[1:]
        com_heights = np.array([float(txt) for txt in split[:usepoints]]) * max_height
        com_errors = np.array([float(txt) for txt in split[9:9+usepoints]])
        print(com_errors)
    bother = sum(np.abs(com_errors)) > 0.1

    green_multipliers = com_errors * 0.01 + 1.0
    red_multipliers = (red_errors * 1 + 1.0) * green_multipliers
    blue_multipliers = (blue_errors * 1 + 1.0) * green_multipliers

    for a, b in zip(com_heights, rb_heights):
        assert a == b

    # plt.plot(com_heights, com_errors, '.')
    # plt.plot(rb_heights, red_errors*100, '.', color='red')
    # plt.plot(rb_heights, blue_errors*100, '.', color='blue')
    # plt.show()
    # exit()
    green_coeff = np.polyfit(com_heights, green_multipliers, 3)
    cosum = sum(green_coeff)
    scaling = 1.0 / cosum
    green_coeff = green_coeff * scaling
    red_coeff = np.polyfit(com_heights, red_multipliers, 3) * scaling
    blue_coeff = np.polyfit(com_heights, blue_multipliers, 3) * scaling

    # plt.plot(com_heights, (red_multipliers / green_multipliers - 1) * 10000, color='red')
    # plt.plot(com_heights, (blue_multipliers / green_multipliers - 1) * 10000, color='blue')
    # plt.show()
    # exit()

    # plt.plot(com_heights, green_multipliers,'.')
    # plt.plot(rb_heights, red_multipliers,'.', color='red')
    # plt.plot(rb_heights, blue_multipliers,'.', color='blue')
    # plt.ylim(green_multipliers[-2]-0.0001, green_multipliers[-2]+0.0001)
    # plt.show()

    alter = np.array((1.0, 1.0, 1.0, min(1.0, 1.0 / green_multipliers[-1])))

    greenlst = ["{:.5f}".format(_) for _ in list(green_coeff*alter)[:]]
    redlst = ["{:.5f}".format(_) for _ in list(red_coeff*alter)[:]]
    bluelst = ["{:.5f}".format(_) for _ in list(blue_coeff*alter)[:]]
    #
    green_geostring = " ".join(greenlst)
    red_geostring = " ".join(redlst)
    blue_geostring = " ".join(bluelst)

    print("Geometric distortion coefficients (RGB):")
    print(red_geostring)
    print(green_geostring)
    print(blue_geostring)
    # exit()
    # plt.plot(heights, multipliers)
    # plt.show()
    return red_geostring, green_geostring, blue_geostring, bother

# distortioncalc("", "")
# exit()
WORKING_DIR = "/home/sam/mtfmapper_temp/"

argv = sys.argv

try:
    startfile = int(argv[1])
except (ValueError, IndexError):
    startfile = -1

for pathname in ["/mnt/mtfm/synthchart.PNG"]:
    filename = os.path.split(pathname)[-1]
    # try:
    #     processpath = os.path.abspath(arg)
    #     filename = os.path.split(processpath)[-1]
    # except FileNotFoundError:
    #     continue
    # print(processpath)
    # print(filename)
    # Process RAF exif
    # exif = subprocess.check_output(["exiftool", processpath])
    # exiflines = [line.decode(encoding='utf-8', errors='strict') for line in exif.splitlines()]
    #
    # aperture = ""
    # focal_length = ""
    # lens_model = ""
    # max_aperture = ""
    # distortionexif = ""
    # ca_exif = []
    #
    # exifpath = os.path.join(WORKING_DIR, filename + ".exif.csv")
    #
    # with open(exifpath, 'w') as file:
    #     writer = csv.writer(file, delimiter=",", quotechar="|")
    #     for line in exiflines:
    #         tag, value = [s.strip() for s in line.split(":", 1)]
    #         writer.writerow([tag, value])
    #         # print(tag, value)
    #         if tag == "Aperture":
    #             aperture = value
    #         elif tag == "Focal Length" and "equivalent" not in value:
    #             focal_length = value
    #         elif tag == "Lens Model":
    #             lens_model = value
    #         elif tag == "Max Aperture Value":
    #             max_aperture = value
    #         elif tag == "Geometric Distortion Params":
    #             distortionexif = value
    #         elif tag == "Chromatic Aberration Params":
    #             ca_exif = value
    # print("Lens Model {}, Aperture {}, Focal Length {}".format(lens_model, aperture, focal_length))
    #
    # # Symlink RAF file to working directory
    # linked_raw_path = os.path.join(WORKING_DIR, filename)
    # try:
    #     os.remove(linked_raw_path)
    # except FileNotFoundError:
    #     pass
    # print("Linking {} -> {}".format(processpath, linked_raw_path))
    # os.symlink(processpath, linked_raw_path)

    # Prepare to process raw
    # uncorrected_image_path = linked_raw_path + ".ppm"
    uncorrected_image_path = os.path.join(WORKING_DIR, filename)
    uncorrected_image_path = pathname
    # try:
    #     os.remove(uncorrected_image_path)
    #     print("Removed exising demosaiced image")
    # except FileNotFoundError:
    #     pass
    # if not os.path.exists(uncorrected_image_path):
    #     print("Calling Libraw dcraw_emu to demosaic...")
    #     output = subprocess.check_output(["dcraw_emu", "-4", "-a", linked_raw_path])
    # else:
    #     print("Demosaic already done")

    filterlist = ["point",
                    "hermite",
                    "cubic",
                    "box",
                    "gaussian",
                    "catrom",
                    "triangle",
                    "quadratic",
                    "mitchell",
                    "lanczos",
                    "hamming",
                    "parzen",
                    "blackman",
                    "kaiser",
                    "welsh",
                    "hanning",
                    "bartlett",
                    "bohman",
                  "sinc",
                  "jinc",
                  "special-3",
                  "special-5",
                  "special-7",
                  ]
    for filter in filterlist+["null"]:
        for translate in np.linspace(0.0, 0.5, 5):

            destination_txt_path = os.path.join(WORKING_DIR, "translate_{:.5f}_{}_results.sfr".format(translate, filter))
            if os.path.exists(destination_txt_path):
                continue
            if filter == "null":
                output_image_path = uncorrected_image_path
            else:
                output_image_path = os.path.join(WORKING_DIR, filename + ".translate.{}_{}.ppm".format(translate, filter))

                print("Processing image translation {:.2f} with filter '{}'".format(translate, filter))
                if filter.startswith('special'): # -define filter:window=Jinc -define filter:lobes=3
                    subprocess.check_output(["convert", uncorrected_image_path, "-define", "filter:window=hanning","-define",
                                             "filter:lobes={}".format(filter.split("-")[1]), "-distort", "SRT",
                                             "0.0 0.0 1.0 0.0 {:.5f} {:.5f}".format(translate, translate), output_image_path])

            print("Analysing file '{}'...".format(output_image_path))
            output = subprocess.check_output(["mtf_mapper", "-q", "--nosmoothing", output_image_path, WORKING_DIR])

            temp_txt_output_path = os.path.join(WORKING_DIR, "edge_sfr_values.txt")


            print("Renaming {} -> {}".format(temp_txt_output_path, destination_txt_path))
            try:
                os.remove(destination_txt_path)
            except FileNotFoundError:
                pass
            shutil.move(temp_txt_output_path, destination_txt_path)

            if output_image_path != uncorrected_image_path:
                os.remove(output_image_path)
            print()


            # output = subprocess.check_output(["convert", uncorrected_image_path, "-depth", "8", "-filter", "lanczos",
            #                                   "-write", "MPR:orig", "+delete",
            #                                   "(", "MPR:orig", "-separate", "-delete", "1,2", "-distort", "barrel", red, ")",
            #                                   "(", "MPR:orig", "-separate", "-delete", "1,2", "-distort", "barrel", green, ")",
            #                                   "(", "MPR:orig", "-separate", "-delete", "1,2", "-distort", "barrel", blue, ")",
            #                                   "-combine", corrected_image_path])

fieldlists = {}
for entry in os.scandir(WORKING_DIR):
    if not entry.name.startswith("translate"):
        continue
    split = entry.name.split("_")
    translate = split[1]
    filter = split[2]
    if filter == "null":
        filter = "_null"
    if filter not in fieldlists:
        fieldlists[filter] = []
    fieldlists[filter].append((float(translate), lentil.SFRField(pathname=entry.path)))
print(fieldlists)
baseline = 1.0
mins = []
filters = []
for filter, lst in sorted(fieldlists.items()):
    # if not filter.startswith("special") and filter != "_null" and filter != "lanczos" and filter != "hanning":
    #     continue
    if filter.startswith("special") or filter.startswith("point"):
        continue
    sums = []
    translates = []
    lst.sort()
    for translate, field in lst:
        translates.append(translate)
        sum_ = sum((point.get_freq(0.5) / baseline for point in field.points))
        sums.append(sum_)
    if filter == "_null":
        baseline = sum_
        sums[-1] = 1.0
        filter = "Null"
    mins.append(min(sums))
    filters.append(filter)

    # plt.plot(translates, sums, "." if filter == "lanczos" else "-", label=filter)
print(filters)
ind = np.arange(len(filters))
plt.bar(ind, mins)
plt.xticks(ind, filters)
plt.title("Imagemagick filter worst case SFR performance vs input image")
plt.ylabel("Relative SFR at 0.5 cy/px")
plt.show()
