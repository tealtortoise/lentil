import numpy as np
import matplotlib.pyplot as plt
try:
    import cupy as cp
except ImportError:
    cp = None

from lentil import constants_utils as lentilconf

from lentilwave import helpers


def build_mask(s: helpers.TestSettings, engine=np, dtype="float64", plot=False, cache=None):

    # Lets get a hash to we can cache our mask (if cache object provided)
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

    # Check cache
    if cache is not None:
        for cachehash, mask in cache:
            if cachehash == hash:
                return mask

    # Anti-aliasing adjustment
    smoothfactor = s.phasesamples / 1.5

    me = engine

    aperture_stop_norm_radius = s.p['base_fstop'] / s.p['fstop']

    # Numerical aperture
    na = 1 / (2.0 * s.p['base_fstop'])
    onaxis_peripheral_ray_angle = me.arcsin(na, dtype=dtype)

    pupil_radius_mm = me.tan(onaxis_peripheral_ray_angle, dtype=dtype) * s.default_exit_pupil_position_mm

    x_displacement_mm = (s.x_loc - lentilconf.IMAGE_WIDTH / 2) * lentilconf.DEFAULT_PIXEL_SIZE * 1e3
    y_displacement_mm = (s.y_loc - lentilconf.IMAGE_HEIGHT / 2) * lentilconf.DEFAULT_PIXEL_SIZE * 1e3

    # Since our edges were rotated to the sag/tan axis we don't really want arbitrary angles
    magnitude = (x_displacement_mm ** 2 + y_displacement_mm ** 2) ** 0.5
    if s.fix_pupil_rotation:
        x_displacement_mm = -magnitude
        y_displacement_mm = 0

    x_displacement_mm_min = -x_displacement_mm - pupil_radius_mm
    x_displacement_mm_max = -x_displacement_mm + pupil_radius_mm
    y_displacement_mm_min = -y_displacement_mm - pupil_radius_mm
    y_displacement_mm_max = -y_displacement_mm + pupil_radius_mm

    # Build a grid of lateral displacement from directly perpendicular to image plane
    x = me.linspace(x_displacement_mm_min, x_displacement_mm_max, s.phasesamples, dtype=dtype)
    y = me.linspace(y_displacement_mm_min, y_displacement_mm_max, s.phasesamples, dtype=dtype)
    gridx, gridy = me.meshgrid(x, y)
    displacement_grid = (gridx**2 + gridy**2) ** 0.5

    # Our pixels are square so there it's likely not a perfectly uniform response from all angles.
    squariness = (2**0.5 - displacement_grid / me.maximum(abs(gridx), abs(gridy))) ** 2
    pixel_angle_grid = me.arctan(displacement_grid / s.default_exit_pupil_position_mm *
                                 (1.0 + squariness * s.p.get('squariness', 0.5)), dtype=dtype)

    # Build normalised radius grid for circles
    normarr = me.linspace(-1, 1, s.phasesamples, dtype=dtype)
    gridx, gridy = me.meshgrid(normarr, normarr)
    pupil_norm_radius_grid = (gridx ** 2 + gridy ** 2) ** 0.5

    # Get aperture stop mask (with antialiasing)
    stopmask = np.clip((aperture_stop_norm_radius - pupil_norm_radius_grid) * smoothfactor + 0.5, 0, 1)

    # Get pixel vignetting coefficient scalars
    a = s.p.get('a', 1.0)
    b = s.p.get('b', 1.0)

    if not s.pixel_vignetting:
        mask = stopmask
    else:
        # Get pixel vignetting mask
        coeff_4 = -18.73 * a
        coeff_6 = 485 * b
        square_grid = 1.0 / (1.0 + (pixel_angle_grid**4 * coeff_4 + pixel_angle_grid ** 6 * coeff_6))
        mask = stopmask * square_grid

    if s.lens_vignetting:
        # Get vignetting mask circle 1 (front of lens)
        normarr = me.linspace(-1, 1, s.phasesamples, dtype=dtype)
        image_circle_modifier = s.p.get('v_slr', 1.0) * 0.6
        gridx, gridy = me.meshgrid(normarr - x_displacement_mm / lentilconf.SENSOR_WIDTH * 1e-3 * image_circle_modifier,
                                   normarr - y_displacement_mm / lentilconf.SENSOR_WIDTH * 1e-3 * image_circle_modifier)
        vignette_radius_grid = (gridx ** 2 + gridy ** 2) ** 0.5
        vignette_crop_circle_radius = s.p.get('v_rad', 1.0)
        vignette_mask = np.clip((vignette_crop_circle_radius - vignette_radius_grid) * smoothfactor + 0.5, 0, 1)
        mask *= vignette_mask

        # Get vignetting mask circle 2 (back of lens)
        normarr = me.linspace(-1, 1, s.phasesamples, dtype=dtype)
        image_circle_modifier = s.p.get('v_slr', 1.0) * 0.6
        gridx, gridy = me.meshgrid(normarr + x_displacement_mm / lentilconf.SENSOR_WIDTH * 1e-3 * image_circle_modifier,
                                   normarr + y_displacement_mm / lentilconf.SENSOR_WIDTH * 1e-3 * image_circle_modifier)
        vignette_radius_grid = (gridx ** 2 + gridy ** 2) ** 0.5
        vignette_crop_circle_radius = s.p.get('v_rad', 1.0) * 1.0
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
