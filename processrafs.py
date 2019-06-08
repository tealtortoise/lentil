#!/usr/bin/python3
import subprocess
import os
import shutil
import sys
import csv
import random
import numpy as np
import matplotlib.pyplot as plt



IMAGE_WIDTH = 6000
IMAGE_HEIGHT = 4000
IMAGE_DIAGONAL = (IMAGE_WIDTH**2 + IMAGE_HEIGHT**2)**0.5

CORRECT_DISTORTION = True

def check_if_distortion_corrections_needed(exifstring):
    split = exifstring.split(' ')[1:]
    com_errors = np.array([float(txt) for txt in split[9:17]])
    return sum(np.abs(com_errors)) > 0.1

def check_if_ca_corrections_needed(castring):
    split = castring.split(' ')[1:]
    com_errors = np.array([float(txt) for txt in split[9:27]])
    return sum(np.abs(com_errors)) > 0.000001


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
random_number = random.randint(1,9999)
WORKING_DIR = "/home/sam/mtfmapper_temp{}/".format(random_number)
try:
    os.mkdir(WORKING_DIR)
except FileExistsError:
    pass


argv = sys.argv

try:
    startfile = int(argv[1])
except ValueError:
    startfile = -1

for arg in argv[1:]:
    try:
        processpath = os.path.abspath(arg)
    except FileNotFoundError:
        continue

    process_subdir = os.path.join(processpath, "mtfm3")
    process_subdir_ca = os.path.join(processpath, "mtfm4")
    process_subdir_fullcorr = os.path.join(processpath, "mtfm5")
    print("Processing path {}".format(processpath))

    try:
        os.mkdir(process_subdir)
    except FileExistsError:
        pass
    try:
        os.mkdir(process_subdir_ca)
    except FileExistsError:
        pass
    try:
        os.mkdir(process_subdir_fullcorr)
    except FileExistsError:
        pass

    dirlist = os.scandir(processpath)

    for entry in dirlist:
        if entry.name[-3:].upper() != "RAF":
            continue
        entrynumber = int("".join([s for s in entry.name if s.isdigit()]))
        if entrynumber < startfile:
            continue

        #if entrynumber != 8647:
        #    continue

        # Check for existing results
        new_txtfilepath = os.path.join(process_subdir, "{}.no_corr.sfr".format(entry.name))
        new_txtfilepath_ca = os.path.join(process_subdir_ca, "{}.ca_only.sfr".format(entry.name))
        new_txtfilepath_fullcorr = os.path.join(process_subdir_fullcorr, "{}.ca_and_distortion.sfr".format(entry.name))

        new_esf_filepath = os.path.join(process_subdir, "{}.no_corr.esf".format(entry.name))
        new_esf_filepath_ca = os.path.join(process_subdir_ca, "{}.ca_only.esf".format(entry.name))
        new_esf_filepath_fullcorr = os.path.join(process_subdir_fullcorr, "{}.ca_and_distortion.esf".format(entry.name))

        print("Processing file {}".format(entry.name))

        # Process RAF exif
        exif = subprocess.check_output(["exiftool", entry.path])
        exiflines = [line.decode(encoding='utf-8', errors='strict') for line in exif.splitlines()]

        aperture = ""
        focal_length = ""
        lens_model = ""
        max_aperture = ""
        distortionexif = ""
        ca_exif = []

        exifpath = os.path.join(process_subdir, entry.name + ".exif.csv")
        exifpath_ca = os.path.join(process_subdir_ca, entry.name + ".exif.csv")
        exifpath_fullcorr = os.path.join(process_subdir_fullcorr, entry.name + ".exif.csv")

        if os.path.exists(exifpath):
            os.remove(exifpath)
        with open(exifpath, 'w') as file:
            writer = csv.writer(file, delimiter=",", quotechar="|")
            for line in exiflines:
                tag, value = [s.strip() for s in line.split(":", 1)]
                writer.writerow([tag, value])
                # print(tag, value)
                if tag == "Aperture":
                    aperture = value
                elif tag == "Focal Length" and "equivalent" not in value:
                    focal_length = value
                elif tag == "Lens Model":
                    lens_model = value
                elif tag == "Max Aperture Value":
                    max_aperture = value
                elif tag == "Geometric Distortion Params":
                    distortionexif = value
                elif tag == "Chromatic Aberration Params":
                    ca_exif = value
        print("Lens Model {}, Aperture {}, Focal Length {}".format(lens_model, aperture, focal_length))

        distorted = check_if_distortion_corrections_needed(distortionexif)
        caed = check_if_ca_corrections_needed(ca_exif)
        print(ca_exif)
        if caed:
            print("Lens has LaCA correction in EXIF")
        else:
            print("Lens does not need LaCA correction")
        if distorted:
            print("Lens has distortion correction in EXIF")
        else:
            print("Lens does not need distortion correction")

        if (os.path.exists(new_txtfilepath) and
            (os.path.exists(new_txtfilepath_ca) or not caed) and
            (os.path.exists(new_txtfilepath_fullcorr) or not distorted) and
            os.path.exists(new_esf_filepath) and
            (os.path.exists(new_esf_filepath_ca) or not caed) and
            (os.path.exists(new_esf_filepath_fullcorr) or not distorted)):
            print("{} appears to already exist, skipping processing".format(new_txtfilepath))
            continue
        if caed:
            if os.path.exists(exifpath_ca):
                os.remove(exifpath_ca)
            shutil.copy(exifpath, exifpath_ca)  # Copy exif to results dir
        if distorted:
            if os.path.exists(exifpath_fullcorr):
                os.remove(exifpath_fullcorr)
            shutil.copy(exifpath, exifpath_fullcorr)  # Copy exif to results dir

        # Symlink RAF file to working directory
        linked_raw_path = os.path.join(WORKING_DIR, entry.name)
        try:
            os.remove(linked_raw_path)
        except FileNotFoundError:
            pass
        print("Linking {} -> {}".format(entry.path, linked_raw_path))
        os.symlink(entry.path, linked_raw_path)

        # Prepare to process raw
        uncorrected_image_path = linked_raw_path + ".ppm"
        try:
            os.remove(uncorrected_image_path)
            print("Removed exising demosaiced image")
        except FileNotFoundError:
            pass
        print("Calling Libraw dcraw_emu to demosaic...")
        output = subprocess.check_output(["/home/sam/LibRaw-0.19.2/bin/dcraw_emu", "-4", "-a", linked_raw_path])


        print("Running CA and maybe distortion correction loops...")
        print()
        loops = [0]
        if caed:
            loops.append(1)
        if distorted:
            loops.append(2)
        for n in loops:
            if n == 0:
                print("Loop 1: No corrections")
            elif n == 1:
                print("Loop 2: CA Corrections only")
            if n >= 1:
                red, green, blue, bother = distortioncalc(None, ca_exif)
                if n == 2:
                    red, green, blue, bother = distortioncalc(distortionexif, ca_exif)
                    print("Loop 3: CA and distortion correction")

                corrected_image_path = os.path.join(WORKING_DIR, entry.name + ".corrected.ppm")

                channelpath = os.path.join(WORKING_DIR, entry.name+".channel_%d.pgm")

                redpath = os.path.join(WORKING_DIR, entry.name+".channel_0.pgm")
                greenpath = os.path.join(WORKING_DIR, entry.name+".channel_1.pgm")
                bluepath = os.path.join(WORKING_DIR, entry.name+".channel_2.pgm")

                print("Separating channels...")
                output = subprocess.check_output(["convert", uncorrected_image_path, "-depth", "16",
                                                  "-channel", "RGB", "-separate", channelpath])
                print("Processing red...")
                output = subprocess.check_output(["mogrify", "-filter", "lanczos", "-depth",
                                                  "16", "-distort", "barrel", red, redpath])
                if n == 2:
                    print("Processing green...")
                    output = subprocess.check_output(["mogrify", "-filter", "lanczos",  "-depth",
                                                      "16", "-distort", "barrel", green, greenpath])
                else:
                    print("Skipping green as CA correction only")
                print("Processing blue...")
                output = subprocess.check_output(["mogrify", "-filter", "lanczos",  "-depth",
                                                  "16", "-distort", "barrel", blue, bluepath])
                print("Merging channels...")
                output = subprocess.check_output(["convert", "-depth", "16",
                                                  redpath, greenpath, bluepath, "-combine", corrected_image_path])
                print("Removing temporary channel files...")
                os.remove(redpath)
                os.remove(greenpath)
                os.remove(bluepath)

            if n == 0:
                image_to_analyse = uncorrected_image_path

            elif n >= 1:
                image_to_analyse = corrected_image_path
            else:
                raise Exception()

            print("Running mtf_mapper for loop {}...".format(n+1))
            print("Analysing file '{}'...".format(image_to_analyse))
            output = subprocess.check_output(["mtf_mapper", "-a", "-q", "-l", "--nosmoothing", "-e", image_to_analyse, WORKING_DIR])

            temp_mtf_output_path = os.path.join(WORKING_DIR, "edge_sfr_values.txt")
            temp_esf_output_path = os.path.join(WORKING_DIR, "raw_esf_values.txt")

            if n == 0:
                destination_txt_path = new_txtfilepath
                destination_esf_path = new_esf_filepath
                output = subprocess.check_output(["convert", uncorrected_image_path, uncorrected_image_path + ".jpg"])
            elif n == 1:
                destination_txt_path = new_txtfilepath_ca
                destination_esf_path = new_esf_filepath_ca
                output = subprocess.check_output(["convert", corrected_image_path, corrected_image_path + ".ca_only.jpg"])
            elif n == 2:
                destination_txt_path = new_txtfilepath_fullcorr
                destination_esf_path = new_esf_filepath_fullcorr
                output = subprocess.check_output(["convert", corrected_image_path, corrected_image_path + ".full.jpg"])
            else:
                raise Exception()

            print("Moving {} -> {}".format(temp_mtf_output_path, destination_txt_path))
            print("Moving {} -> {}".format(temp_esf_output_path, destination_esf_path))
            try:
                os.remove(destination_txt_path)
            except FileNotFoundError:
                pass
            try:
                os.remove(destination_esf_path)
            except FileNotFoundError:
                pass
            shutil.move(temp_mtf_output_path, destination_txt_path)
            shutil.move(temp_esf_output_path, destination_esf_path)
            print()

        os.remove(linked_raw_path)
        print("Removing temporary file {}".format(uncorrected_image_path))
        os.remove(uncorrected_image_path)
        if caed or distorted:
            print("Removing temporary file {}".format(corrected_image_path))
            os.remove(corrected_image_path)
        print()


        # output = subprocess.check_output(["convert", uncorrected_image_path, "-depth", "8", "-filter", "lanczos",
        #                                   "-write", "MPR:orig", "+delete",
        #                                   "(", "MPR:orig", "-separate", "-delete", "1,2", "-distort", "barrel", red, ")",
        #                                   "(", "MPR:orig", "-separate", "-delete", "1,2", "-distort", "barrel", green, ")",
        #                                   "(", "MPR:orig", "-separate", "-delete", "1,2", "-distort", "barrel", blue, ")",
        #                                   "-combine", corrected_image_path])

