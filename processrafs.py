#!/usr/bin/python3
import subprocess
import os
import shutil
import sys
import csv
import numpy as np
import matplotlib.pyplot as plt

def distortioncalc(exifstring):
    # EXAMPLE = "400.5555556 0.3535211268 0.5 0.6126760563 0.7070422535 0.7908450704 0.8661971831 0.9352112676 1 1.06056338 -1.172515869 -2.354660034 -3.458114624 -4.434677124 -5.274719238 -5.918792725 -6.328369141 -6.426345825 -6.42634582"

    split = exifstring.split(' ')[1:]
    heights = np.array([float(txt) for txt in split[:9]]) * (3**2+2**2)**0.5 / 2.0
    errors = np.array([float(txt) for txt in split[9:]])
    multipliers = np.array(errors) * 0.01 + 1.0
    poly = np.polyfit(heights, multipliers, 3)
    cosum = sum(poly)
    scaledpoly = poly / cosum
    lst = [str(_) for _ in list(scaledpoly)[:3]]
    geostring = " ".join(lst)
    print(geostring)
    # plt.plot(heights, multipliers)
    # plt.show()
    return geostring

# distortioncalc("")
# exit()
WORKING_DIR = "/home/sam/mtfmapper_temp/"

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
    process_subdir_corr = os.path.join(processpath, "mtfm4")
    print("Processing path {}".format(processpath))

    try:
        os.mkdir(process_subdir)
        os.mkdir(process_subdir_corr)
    except FileExistsError:
        pass

    dirlist = os.scandir(processpath)

    for entry in dirlist:
        if entry.name[-3:].upper() != "RAF":
            continue
        entrynumber = int("".join([s for s in entry.name if s.isdigit()]))
        if entrynumber < startfile:
            continue

        # if entrynumber != 5:
        #     continue
        # Check for existing results
        new_txtfilepath = os.path.join(process_subdir, "{}.sfr".format(entry.name))
        new_txtfilepath_corr = os.path.join(process_subdir_corr, "{}.corr.sfr".format(entry.name))
        if os.path.exists(new_txtfilepath):
            print("{} appears to already exist, skipping processing".format(new_txtfilepath))
            continue

        print("Processing file {}".format(entry.name))

        # Process RAF exif
        exif = subprocess.check_output(["exiftool", entry.path])
        exiflines = [line.decode(encoding='utf-8', errors='strict') for line in exif.splitlines()]
        aperture = ""
        focal_length = ""
        lens_model = ""
        max_aperture = ""
        exifpath = os.path.join(process_subdir, entry.name + ".exif.csv")
        exifpath_corr = os.path.join(process_subdir_corr, entry.name + ".exif.csv")
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
        shutil.copy(exifpath, exifpath_corr)
        # Symlink RAF file to working directory
        linkpathname = os.path.join(WORKING_DIR, entry.name)
        try:
            os.remove(linkpathname)
        except FileNotFoundError:
            pass

        print("Linking {} -> {}".format(entry.path, linkpathname))
        os.symlink(entry.path, linkpathname)

        # Prepare to process raw
        ppmpathname = linkpathname + ".ppm"
        try:
            os.remove(ppmpathname)
        except FileNotFoundError:
            pass
        print("Calling Libraw dcraw_emu to demosaic...")
        output = subprocess.check_output(["dcraw_emu", "-a", "-4", linkpathname])

        print("Running distortion correction...")
        geocoeff = distortioncalc(distortionexif)
        correctedpath = os.path.join(WORKING_DIR, entry.name+".corrected.ppm")
        output = subprocess.check_output(["convert", ppmpathname, "-depth", "16", "-filter", "lanczos",
                                          "-distort", "barrel", geocoeff,
                                          correctedpath])
        """convert image.jpg -write MPR:orig +delete \
   \( MPR:orig -separate -delete 1,2 -affine <red transform>  \) \
   \( MPR:orig -separate -delete 0,2 -affine <green tranform> \) \
   \( MPR:orig -separate -delete 0,1 -affine <blue transform> \) \
   -combine result.jpg """
        # exit()
        print("Running mtf_mapper...")
        output = subprocess.check_output(["mtf_mapper", "-q", "--nosmoothing", "--optimize-distortion", ppmpathname, WORKING_DIR])

        # Move output to output directory
        txtfilepath = os.path.join(WORKING_DIR, "edge_sfr_values.txt")
        print("Moving {} -> {}".format(txtfilepath, new_txtfilepath))
        shutil.move(txtfilepath, new_txtfilepath)

        print("Running mtf_mapper corrected...")
        output = subprocess.check_output(["mtf_mapper", "-q", "--nosmoothing", "--optimize-distortion", correctedpath, WORKING_DIR])

        # Move output to output directory
        txtfilepath = os.path.join(WORKING_DIR, "edge_sfr_values.txt")
        print("Moving {} -> {}".format(txtfilepath, new_txtfilepath_corr))
        shutil.move(txtfilepath, new_txtfilepath_corr)

        os.remove(ppmpathname)
        os.remove(correctedpath)
        os.remove(linkpathname)
        print("Removing temporary file {}".format(ppmpathname))
        print()



