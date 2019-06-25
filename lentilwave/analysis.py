import prysm
from lentil.constants_utils import *
from lentil.wavefront_utils import plot_nominal_psf
from lentilwave.encode_decode import encode_parameter_tuple, decode_parameter_tuple, convert_wavefront_dicts_to_p_dicts
from lentilwave import helpers, config
from lentilwave.generation import masks


def plot_wfe_data(focusset):
    ps = focusset.read_wavefront_data(overwrite=True)
    for label, p in ps[-1:]:
        print('Cost!', p['final.cost'])
        popt = {}
        for key, value in p.items():
            if key.startswith("p.opt:"):
                split = key.split(":")
                popt[split[1]] = value
        z = {}
        for key, value in popt.items():
            if key.startswith("z") and key[1].isdigit():
                z[key] = value
        print(popt)
        popt['base_fstop'] = p['fstops'][0]
        popt['base_fstop'] = popt['fstop']
        popt['samples'] = 512
        for df in np.linspace(popt['df_offset']-3, popt['df_offset']+3, 10):
        # for df in [popt['df_offset']]:
# def try_wavefront(defocus, p, mono=False, plot=False, dummy=False, use_cuda=True, index=None,
#                   strehl_estimate=1.0, mtf_mean=0.7, fftsize_override=None, samples_override=None):
            args = [df, popt, False, True, False, False, 0, 1.0, 1.0, 512, 256]

            try_wavefront(*args)
[15.80960235, 11.61036639,  9.82530368,  0.42724607,  0.23420716,
        0.14312454,  0.07228009, -0.54367992, -0.6934329 , -0.28648354,
       -3.86548324]


def plot_chromatic_aberration(focusset):
    z4s = []
    z9s = []
    z4s_smooth = []
    z9s_smooth = []
    photopic = []
    wfd = focusset.read_wavefront_data(overwrite=True)[-1][1]

    p = convert_wavefront_dicts_to_p_dicts(wfd)[0]
    p['base_fstop'] = float(wfd['fstops'][0])

    smooth_waves = np.linspace(0.4, 0.67, 50)

    for modelwavelength in config.MODEL_WVLS:
        z4s.append(helpers.get_z4(p['df_offset'], p, modelwavelength))
        z9s.append(helpers.get_z9(p, modelwavelength))

    for modelwavelength in smooth_waves:
        z4s_smooth.append(helpers.get_z4(p['df_offset'], p, modelwavelength))
        z9s_smooth.append(helpers.get_z9(p, modelwavelength))
        photopic.append(photopic_fn(modelwavelength * 1e3))

    plt.plot(config.MODEL_WVLS, z4s, 's', label="Z4")
    plt.plot(config.MODEL_WVLS, z9s, 's', label="Z9")
    plt.plot(smooth_waves, z4s_smooth, '-', label="Z4")
    plt.plot(smooth_waves, z9s_smooth, '-', label="Z9")
    plt.plot(smooth_waves, photopic, label="Photopic Fn")
    plt.xlabel("Wavelength (um)")
    plt.ylabel("RMS WFE (lambdas)")
    plt.legend()
    plt.show()


def plot_nominal_psfs(focusset, stop_downs=(0, 1, 2, 3), x_loc=None, y_loc=None):
    wfd = focusset.read_wavefront_data(overwrite=True, x_loc=x_loc, y_loc=y_loc)
    ps = convert_wavefront_dicts_to_p_dicts(wfd[-1][1])
    print(wfd[-1][1])
    x, y = wfd[-1][1]['x.loc'], wfd[-1][1]['y.loc']
    print(ps)
    if stop_downs:
        args = []
        for stop in stop_downs:
            new_p = ps[0].copy()
            print(new_p)
            args.append(new_p)
            new_p['fstop'] *= 2 ** (stop / 2)
        plot_nominal_psf(*args, wfdd=wfd[-1][1], x_loc=x, y_loc=y)
    else:
        plot_nominal_psf(ps[0], wfdd=wfd[-1][1], x_loc=x, y_loc=y)


def plot_tstops():
    def summask(base_fstop, fstop=None, pixel_vignetting=True):
        if fstop is None:
            fstop = base_fstop
        s = helpers.TestSettings(dict(base_fstop=base_fstop, fstop=fstop))
        s.phasesamples = 256
        s.pixel_vignetting = pixel_vignetting
        return masks.build_mask(s, engine=np).sum()

    plotstops = 2 ** (np.arange(2/3, 2.1, 1.0/3.0)/2.0)
    baseline = summask(1.0, pixel_vignetting=False)
    t_stops = []
    for stop in plotstops:
        stopsum = summask(stop, pixel_vignetting=True)
        t_stop = stop + np.log2(baseline / stopsum)/2
        t_stops.append(t_stop)
    print(plotstops)
    print(t_stops)
    plt.xlabel("F-stop")
    plt.ylabel("T-stop")
    plt.grid()
    plt.title("F-stop vs T-stop on axis")
    # plt.xscale("log")
    # plt.yscale("log")
    plt.plot(plotstops, t_stops)
    plt.stem(plotstops, t_stops, bottom=min(t_stops))
    # plt.vlines(plotstops[np.array([0,2,8])], min(t_stops), max(t_stops))
    plt.show()



def plot_pixel_vignetting_loss():
    fstops = 2.0 ** np.linspace(0.0, 1.5, 6)
    test_fstops = (1, 2**0.16667, 1.222, 2**0.5, 1.4 * 2**0.166667, 2, 2 * 2 ** 0.166667, 2 * 2**0.5)
    benefits_exp = (1.93, 1.90, 1.81, 1.70, 1.5, 0.93, 0.62, 0)

    testfstop = 2 ** (np.arange(0, 3.1, 1.0/3.0)/2.0)

    s = helpers.TestSettings(dict(base_fstop=1.0, fstop=2.0 * 2**0.5))
    s.phasesamples = 256
    s.pixel_vignetting = False
    baseline = masks.build_mask(s, np).mean()

    s.pixel_vignetting = True

    if "optimise" and 0:
        def callable(x):
            error = 0
            for testfstop, benefit_exp in zip(test_fstops, benefits_exp):
                s.p = dict(base_fstop=1, fstop=testfstop)
                s.p['a'], s.p['b'] = x
                benefit = np.log2(masks.build_mask(s, np).mean() / baseline)
                print(x[0], x[1], testfstop, benefit)
                error += (benefit - benefit_exp) ** 2
            print()
            return error

        opt = optimize.minimize(callable, np.array((0, 0)))  # ,bounds=((-20,20), (-30, 30)
        a, b = opt.x
    else:
        a, b = 1, 1

    plot_fstops = 2 ** np.linspace(0, 1.5, 6)
    benefits = []
    benefits_stop = []
    for testfstop in plot_fstops:
        s.p = dict(base_fstop=testfstop, fstop=testfstop)
        s.p['a'], s.p['b'] = a, b
        benefit = np.log2(masks.build_mask(s, np, plot=False).mean() / (testfstop ** 2) / baseline)
        benefits.append(benefit)
        s.p = dict(base_fstop=1, fstop=testfstop)
        s.p['a'], s.p['b'] = a, b
        benefit = np.log2(masks.build_mask(s, np, plot=True).mean() / baseline)
        benefits_stop.append(benefit)

    plt.plot(plot_fstops, benefits)
    plt.plot(plot_fstops, benefits_stop)
    plt.plot(test_fstops, benefits_exp)
    plt.plot()
    plt.show()
    exit()


def plot_lens_vignetting_loss(base_fstop=1.4):
    fstops = 2.0 ** np.linspace(0.0, 2, 5)
    for stop in fstops:
        s = helpers.TestSettings(dict(base_fstop=base_fstop, fstop=stop * base_fstop))
        s.phasesamples = 128
        s.pixel_vignetting = True
        s.lens_vignetting = True
        # s.x_loc = 5600
        # s.y_loc = 3850
        s.p['v_slr'] = 2
        # s.p['v_mag'] = 0.8
        # s.p['v_rad'] = 1.3
        # s.p['v_x'] = -0.8
        # s.p['v_y'] = -0.8
        baseline = build_mask(s, np).mean()
        heights = np.linspace(0, 1, 16)
        losses = []
        s.x_loc = 0
        s.y_loc = 0
        build_mask(s, np, plot=False)
        for height in heights:
            s.x_loc = 3000 + height * lentilconf.IMAGE_WIDTH / 2
            s.y_loc = 2000 + height * lentilconf.IMAGE_HEIGHT / 2
            # losses.append(np.log2(mask_pupil(s, np).mean() / baseline))
            losses.append(build_mask(s, np).mean() / baseline)
        plt.plot(heights, losses, label=str(stop*base_fstop))
    plt.legend()
    plt.show()

# plot_lens_vignetting_loss(1.25)
# exit()