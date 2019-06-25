import numpy as np

from lentilwave import config


def encode_parameter_tuple(dataset, use_initial=False, use_existing=True,
                           params=(config.OPTIMISE_PARAMS, config.FIXED_PARAMS)):
    passed_options_ordering = []
    initial_guess = []
    optimise_bounds = []

    prepend = "p.initial:" if use_initial else "p.opt:"

    if use_existing:
        try:
            existing_dict = dataset[0].wavefront_data[-1][1]
        except (AttributeError, IndexError):
            existing_dict = {}
    else:
        existing_dict = {}

    if 'fstop' in params[0]:
        fstop_first_params = ['fstop']
        for param in params[0]:
            if param != 'fstop':
                fstop_first_params.append(param)
    else:
        fstop_first_params = params[0]
    fstops = [data.exif.aperture for data in dataset]
    base_fstop = min(fstops)

    for pnum, pname in enumerate(fstop_first_params):
        paramconfigtup = config.PARAMS_OPTIONS[pname]
        config_low , config_initial, config_high, f_lambda, optim_per_focusset, scale = paramconfigtup
        scale *= config.SCALE_EXTRA.get(pname, 1.0)

        if optim_per_focusset == config.OPT_PER_FOCUSSET:
            loops = len(dataset)
        else:
            loops = 1
            passed_options_ordering.append((pname, tuple(range(len(dataset)), ), None))

        for a in range(loops):
            if optim_per_focusset == config.OPT_PER_FOCUSSET:
                passed_options_ordering.append((pname, (a,), None))
                existingkey = prepend + pname + "@{}".format(dataset[a].exif.aperture)
                try:
                    hint_dict = dataset[a].hints
                except (AttributeError, IndexError):
                    hint_dict = {}
            else:
                existingkey = prepend + pname
                hint_dict = {}

            fmul = f_lambda(fstops[a], base_fstop)

            if pname == "df_offset":
                if existingkey in existing_dict:
                    initial = existing_dict[existingkey] / fmul
                    low = initial - 6
                    high = initial + 6
                elif pname in hint_dict:
                    initial = hint_dict[pname] / fmul
                    low = initial - 30
                    high = initial + 30
                else:
                    low = min(dataset[a].focus_values) - 2
                    high = max(dataset[a].focus_values) + 2
                    initial = (low + high) * 0.5

            elif pname == 'df_step':
                    if existingkey in existing_dict:
                        initial = existing_dict[existingkey] / fmul
                    elif pname in hint_dict:
                        r = dataset[a].exif.aperture / base_fstop
                        initial = hint_dict[pname] * base_fstop * r
                    else:
                        initial = config_initial / fmul
                    low = initial / config.DF_STEP_TOLERANCE
                    high = initial * config.DF_STEP_TOLERANCE

            elif pname == 'fstop':
                if existingkey in existing_dict:
                    existfstop = existing_dict[existingkey]
                    initial = existfstop / fmul
                    if optim_per_focusset == config.OPT_SHARED:
                        corr = existfstop / fstops[a]
                        for ix in range(len(fstops)):
                            fstops[ix] *= corr
                    else:
                        fstops[a] = existfstop
                    if a == 0:
                        base_fstop = existfstop
                else:
                    initial = dataset[a].exif.aperture / fmul
                low = config_low
                high = config_high

            else:
                if existingkey in existing_dict:
                    initial = existing_dict[existingkey] / fmul
                elif pname in hint_dict:
                    initial = hint_dict[pname] / fmul
                else:
                    initial = config_initial / fmul
                low = min(config_low, initial)
                high = max(config_high, initial)
            initial_guess.append(initial * scale)
            optimise_bounds.append((low * scale, high * scale))
    return initial_guess, optimise_bounds, passed_options_ordering


def decode_parameter_tuple(tup, passed_options_ordering, dataset, use_existing_fixed=config.USE_EXISTING_PARAMETER_IF_FIXED,
                           params=(config.OPTIMISE_PARAMS, config.FIXED_PARAMS)):
    # print(use_existing_fixed, USE_EXISTING_PARAMETER_IF_FIXED)
    # exit()
    ps = []
    for _ in dataset:
        ps.append({})
    popt = {}
    opt_fstops = [data.exif.aperture for data in dataset]
    nominal_fstops = opt_fstops.copy()
    if 'fstop' in config.OPTIMISE_PARAMS:
        # Avoid large gradients due to large pupil change with fstop
        for tix, ((name, setapplies, fieldapplies), val) in enumerate(zip(passed_options_ordering, tup)):
            config_low , config_initial, config_high, f_lambda, optim_per_focusset, scale = config.PARAMS_OPTIONS[name]
            scale *= config.SCALE_EXTRA.get(name, 1.0)
            if name == 'fstop':
                for a in setapplies:
                    opt_fstops[a] = val / scale * dataset[a].exif.aperture

    base_fstop = min(opt_fstops)

    for tix, ((name, setapplies, fieldapplies), val) in enumerate(zip(passed_options_ordering, tup)):
        config_low , config_initial, config_high, f_lambda, optim_per_focusset, scale = config.PARAMS_OPTIONS[name]
        scale *= config.SCALE_EXTRA.get(name, 1.0)
        for a in setapplies:
            if name == "fstop":
                fmul = f_lambda(nominal_fstops[a], base_fstop)
            else:
                fmul = f_lambda(opt_fstops[a], base_fstop)
            if fieldapplies is None:
                ps[a][name] = val * fmul / scale
                if len(setapplies) == 1:
                    popt["{}{}".format(name, a)] = val * fmul / scale
            else:
                ps[a]["{}.{}".format(name, fieldapplies)] = val * fmul / scale
                popt["{}{}.{}".format(name, a, fieldapplies)] = val * fmul / scale
        if len(setapplies) > 1:
            if name == "fstop":
                popt["FSTOP_CORR"] = val / scale
                for p in ps:
                    p['fstop_corr'] = val / scale
            else:
                popt["{}".format(name)] = val * fmul / scale

    pfix = {}
    for name in params[1]:
        config_low , config_initial, config_high, f_lambda, optim_per_focusset, scale = config.PARAMS_OPTIONS[name]
        for a, data in enumerate(dataset):
            existing_dict = data.wavefront_data[-1][1]
            existingkey = "p.opt:" + name

            if use_existing_fixed and existingkey in existing_dict and optim_per_focusset is not config.LOCK:
                value = existing_dict[existingkey]
                if value != config_initial:
                    ps[a][name] = value
                    pfix["{}.{}".format(name, a)] = value
            else:
                fmul = f_lambda(opt_fstops[a], base_fstop)
                # print(ps[a], name, config_initial, fmul)
                try:
                    if name[0] == 'z' and name[1].isdigit():
                        continue
                except IndexError:
                    pass
                ps[a][name] = config_initial * fmul
                pfix["{}.{}".format(name, a)] = config_initial * fmul
    return ps, popt, pfix


def convert_wavefront_dicts_to_p_dicts(wfdd):
    ps = []
    print(wfdd)
    try:
        nomfstops = wfdd['fstops']
    except KeyError:
        return []
    min_fstop = np.inf
    for stop in nomfstops:
        p = {}
        for key, value in wfdd.items():
            split = key.split(":")
            if split[0] == "p.opt":
                if "@" in split[1]:
                    param, nomf = split[1].split("@")
                    if nomf == str(stop):
                        p[param] = value
                else:
                    p[split[1]] = value

        if p['fstop'] < min_fstop:
            min_fstop = p['fstop']
        ps.append(p)
    for p in ps:
        p['base_fstop'] = min_fstop
    return ps