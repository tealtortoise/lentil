import numpy as np
from scipy import optimize, interpolate
import matplotlib.pyplot as plt
try:
    import cupy as cp
except ImportError:
    cp = None

from lentil import constants_utils as lentilconf

from lentilwave import helpers


def build_mask(s: helpers.TestSettings, engine=np, dtype="float64", plot=False, cache=None):
    hashtuple = (s.p['base_fstop'],
                 s.p['fstop'],
                 s.x_loc,
                 s.y_loc,
                 s.phasesamples,
                 s.p.get('a', 0.0),
                 s.p.get('b', 0.0),
                 s.p.get('v_scr', 1.0),
                 s.p.get('v_rad', 1.0),
                 s.p.get('squariness', 0.5),
                 "np" if engine is np else "cp",
                 dtype)

    hash = hashtuple.__hash__()

    if cache is not None:
        for cachehash, mask in cache:
            if cachehash == hash:
                return mask

    smoothfactor = s.phasesamples / 1.5

    me = engine

    aperture_stop_norm_radius = s.p['base_fstop'] / s.p['fstop']

    na = 1 / (2.0 * s.p['base_fstop'])
    onaxis_peripheral_ray_angle = me.arcsin(na, dtype=dtype)

    pupil_radius_mm = me.tan(onaxis_peripheral_ray_angle, dtype=dtype) * s.default_exit_pupil_position_mm

    x_displacement_mm = (s.x_loc - lentilconf.IMAGE_WIDTH / 2) * lentilconf.DEFAULT_PIXEL_SIZE * 1e3
    y_displacement_mm = (s.y_loc - lentilconf.IMAGE_HEIGHT / 2) * lentilconf.DEFAULT_PIXEL_SIZE * 1e3

    # angle = s.p.get('v_angle', 0)
    magnitude = (x_displacement_mm ** 2 + y_displacement_mm ** 2) ** 0.5

    if s.fix_pupil_rotation:
        x_displacement_mm = -magnitude
        y_displacement_mm = 0

    x_displacement_mm_min = -x_displacement_mm - pupil_radius_mm
    x_displacement_mm_max = -x_displacement_mm + pupil_radius_mm
    y_displacement_mm_min = -y_displacement_mm - pupil_radius_mm
    y_displacement_mm_max = -y_displacement_mm + pupil_radius_mm

    x = me.linspace(x_displacement_mm_min, x_displacement_mm_max, s.phasesamples, dtype=dtype)
    y = me.linspace(y_displacement_mm_min, y_displacement_mm_max, s.phasesamples, dtype=dtype)
    gridx, gridy = me.meshgrid(x, y)
    displacement_grid = (gridx**2 + gridy**2) ** 0.5
    squariness = (2**0.5 - displacement_grid / me.maximum(abs(gridx), abs(gridy))) ** 2
    pixel_angle_grid = me.arctan(displacement_grid / s.default_exit_pupil_position_mm *
                                 (1.0 + squariness * s.p.get('squariness', 0.5)), dtype=dtype)

    normarr = me.linspace(-1, 1, s.phasesamples, dtype=dtype)
    gridx, gridy = me.meshgrid(normarr, normarr)
    pupil_norm_radius_grid = (gridx ** 2 + gridy ** 2) ** 0.5

    stopmask = np.clip( (aperture_stop_norm_radius - pupil_norm_radius_grid) * smoothfactor + 0.5, 0, 1)

    a = s.p.get('a', 1.0)
    b = s.p.get('b', 1.0)

    if not s.pixel_vignetting:
        mask = stopmask
    else:
        coeff_4 = -18.73 * a
        corff_6 = 485 * b
        square_grid = 1.0 / (1.0 + (pixel_angle_grid**4 * coeff_4 + pixel_angle_grid ** 6 * corff_6))
        mask = stopmask * square_grid

    if s.lens_vignetting:
        # Lens vignette
        normarr = me.linspace(-1, 1, s.phasesamples, dtype=dtype)
        image_circle_modifier = s.p.get('v_slr', 1.0) * 0.6
        gridx, gridy = me.meshgrid(normarr - x_displacement_mm / lentilconf.SENSOR_WIDTH * 1e-3 * image_circle_modifier,
                                   normarr - y_displacement_mm / lentilconf.SENSOR_WIDTH * 1e-3 * image_circle_modifier)
        vignette_radius_grid = (gridx ** 2 + gridy ** 2) ** 0.5
        vignette_crop_circle_radius = s.p.get('v_rad', 1.0)
        vignette_mask = vignette_radius_grid < vignette_crop_circle_radius
        vignette_mask = np.clip((vignette_crop_circle_radius - vignette_radius_grid) * smoothfactor + 0.5, 0, 1)
        mask *= vignette_mask

        normarr = me.linspace(-1, 1, s.phasesamples, dtype=dtype)
        image_circle_modifier = s.p.get('v_slr', 1.0) * 0.6
        gridx, gridy = me.meshgrid(normarr + x_displacement_mm / lentilconf.SENSOR_WIDTH * 1e-3 * image_circle_modifier,
                                   normarr + y_displacement_mm / lentilconf.SENSOR_WIDTH * 1e-3 * image_circle_modifier)
        vignette_radius_grid = (gridx ** 2 + gridy ** 2) ** 0.5
        vignette_crop_circle_radius = s.p.get('v_rad', 1.0) * 1.0
        vignette_mask = vignette_radius_grid < vignette_crop_circle_radius
        vignette_mask = np.clip((vignette_crop_circle_radius - vignette_radius_grid) * smoothfactor + 0.5, 0, 1)
        mask *= vignette_mask

    if plot or s.id_or_hash == -1:
        if engine is cp:
            print(square_grid)
            plt.imshow(cp.asnumpy(mask))
            plt.colorbar()
            plt.show()
        else:
            print(square_grid)
            plt.imshow(mask)
            plt.colorbar()
            plt.show()
    return mask


def plot_pixel_vignetting_loss():
    fstops = 2.0 ** np.linspace(0.0, 1.5, 6)
    test_fstops = (1, 2**0.16667, 1.222, 2**0.5, 1.4 * 2**0.166667, 2, 2 * 2 ** 0.166667, 2 * 2**0.5)
    benefits_exp = (1.93, 1.90, 1.81, 1.70, 1.5, 0.93, 0.62, 0)

    s = helpers.TestSettings(0, dict(base_fstop=1.0, fstop=2.0 * 2**0.5))
    s.phasesamples = 256
    s.pixel_vignetting = False
    baseline = build_mask(s, np).mean()

    s.pixel_vignetting = True

    if "optimise" and 0:
        def callable(x):
            error = 0
            for testfstop, benefit_exp in zip(test_fstops, benefits_exp):
                s.p = dict(base_fstop=1, fstop=testfstop)
                s.p['a'], s.p['b'] = x
                benefit = np.log2(build_mask(s, np).mean() / baseline)
                print(x[0], x[1], testfstop, benefit)
                error += (benefit - benefit_exp) ** 2
            print()
            return error

        opt = optimize.minimize(callable, np.array((0, 0)))  # ,bounds=((-20,20), (-30, 30)
        a, b = opt.x
    else:
        a, b = 0, 0

    plot_fstops = 2 ** np.linspace(0, 1.5, 6)
    benefits = []
    benefits_stop = []
    for testfstop in plot_fstops:
        s.p = dict(base_fstop=testfstop, fstop=testfstop)
        s.p['a'], s.p['b'] = a, b
        benefit = np.log2(build_mask(s, np, plot=True).mean() / (testfstop ** 2) / baseline)
        benefits.append(benefit)
        s.p = dict(base_fstop=1, fstop=testfstop)
        s.p['a'], s.p['b'] = a, b
        benefit = np.log2(build_mask(s, np, plot=True).mean() / baseline)
        benefits_stop.append(benefit)

    plt.plot(plot_fstops, benefits)
    plt.plot(plot_fstops, benefits_stop)
    plt.plot(test_fstops, benefits_exp)
    plt.plot()
    plt.show()
    exit()


def plot_lens_vignetting_loss(base_fstop=1.4):
    fstops = 2.0 ** np.linspace(0.0, 2, 4)
    for stop in fstops:
        s = helpers.TestSettings(0, dict(base_fstop=base_fstop, fstop=stop * base_fstop))
        s.phasesamples = 128
        s.pixel_vignetting = True
        s.lens_vignetting = True
        s.p['v_mag'] = 0.8
        s.p['v_rad'] = 1.3
        s.p['v_x'] = -0.8
        s.p['v_y'] = -0.8
        baseline = build_mask(s, np).mean()
        heights = np.linspace(0, 1, 16)
        losses = []
        s.x_loc = 0
        s.y_loc = 0
        build_mask(s, np, plot=True)
        for height in heights:
            s.x_loc = 3000 + height * lentilconf.IMAGE_WIDTH / 2
            s.y_loc = 2000 + height * lentilconf.IMAGE_HEIGHT / 2
            # losses.append(np.log2(mask_pupil(s, np).mean() / baseline))
            losses.append(build_mask(s, np).mean() / baseline)
        plt.plot(heights, losses)
    plt.show()
