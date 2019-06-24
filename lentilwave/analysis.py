import prysm
from lentil.constants_utils import *
from lentil.wavefront_utils import plot_nominal_psf
from lentilwave.encode_decode import encode_parameter_tuple, decode_parameter_tuple, convert_wavefront_dicts_to_p_dicts


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

    for modelwavelength in MODEL_WVLS:
        z4s.append(get_z4(p['df_offset'], p, modelwavelength))
        z9s.append(get_z9(p, modelwavelength))

    for modelwavelength in smooth_waves:
        z4s_smooth.append(get_z4(p['df_offset'], p, modelwavelength))
        z9s_smooth.append(get_z9(p, modelwavelength))
        photopic.append(photopic_fn(modelwavelength * 1e3))

    plt.plot(MODEL_WVLS, z4s, 's', label="Z4")
    plt.plot(MODEL_WVLS, z9s, 's', label="Z9")
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
