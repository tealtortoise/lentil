from lentil.constants_utils import *
from lentil.wavefront_test import get_z4, get_z9
from lentil.wavefront_config import MODEL_WVLS

def test_z4():
    tests = np.linspace(-2, 2, 5)
    y = []
    for param in ['loca', 'loca1']:
        for test in tests:
            p = dict(fstop=2, base_fstop=2, df_offset=0, df_step=1)
            p[param] = test
            res = get_z4(0, p, MODEL_WVLS)
            weights = np.array([float(photopic_fn(wv * 1e3) * d50_interpolator(wv)) for wv in MODEL_WVLS])
            print("{} {:.4f} {:.4f}".format(test, np.average(res, weights=weights), np.average(abs(res), weights=weights)))
            plt.plot(res, '-' if param == 'loca' else '--' ,label="{} = {}".format(param, test))
        print()
    plt.legend()
    plt.show()

def test_z9():
    tests = np.linspace(-2, 2, 5)
    y = []
    for param in ['spca', 'spca2']:
        for test in tests:
            p = dict(fstop=2, base_fstop=2, df_offset=0, df_step=1)
            p[param] = test
            res = get_z9(p, MODEL_WVLS)
            weights = np.array([float(photopic_fn(wv * 1e3) * d50_interpolator(wv)) for wv in MODEL_WVLS])
            print("{} {:.4f} {:.4f}".format(test, np.average(res, weights=weights), np.average(abs(res), weights=weights)))
            plt.plot(res, '-' if param == 'spca' else '--', label="{} = {}".format(param, test))
        print()
    plt.legend()
    plt.show()
