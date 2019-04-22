import cProfile
import lentil
import pstats
from lentil.constants_utils import *

BASE_PATH = "/home/sam/nashome/MTFMapper Stuff/"

PATHS = [
    "Bernard/",

    "56mm/f1.2/",
    "56mm/f2.8/",
    "56mm/f5.6/",
    "56mm/f8/",]

focusset = lentil.FocusSet(fallback_results_path(os.path.join(BASE_PATH, PATHS[2]), 3), include_all=1,
                           use_calibration=1)
def runme():
    # focusset.find_best_focus(2000,2000)
    focusset.plot_ideal_focus_field(axis=ALL_THREE_AXES, fix_zlim=(-5,5), show=False)

cProfile.run('runme()', 'profilestats')
p = pstats.Stats('profilestats')
p.strip_dirs().sort_stats('cumulative').print_stats()
p.strip_dirs().sort_stats('time').print_stats()