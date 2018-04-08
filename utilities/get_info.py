import numpy as np
import os
import argparse
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, leastsq
import simplejson

# Made for the electrowetting case, thus the default.

def parse_args():
    parser = argparse.ArgumentParser(description="Analyze contact angle")
    parser.add_argument("folders", nargs='+', type=str, help="Folders")
    parser.add_argument("-k", "--keys", type=str,
                        default="V_top,contact_angle,rad_init,concentration_init_s,concentration_init_d,permittivity,surface_tension",
                        help="Keys to get info from.")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    for base_folder in args.folders:
        folder = os.path.join(base_folder, "Analysis", "contour")

        params = simplejson.load(
            open(os.path.join(
                base_folder, "Settings",
                "parameters_from_tstep_0.dat")))

        keys = args.keys.split(",")
        
        for key in keys:
            if key in params:
                print "{}:\t{}".format(key, params[key])


if __name__ == "__main__":
    main()
