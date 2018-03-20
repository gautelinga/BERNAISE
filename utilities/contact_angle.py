import numpy as np
import os
import argparse
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import simplejson


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze contact angle")
    parser.add_argument("folders", nargs='+', type=str, help="Folders")
    parser.add_argument("--plot", "-p", action="store_true", help="Plot")
    args = parser.parse_args()
    return args


def fitfunc(x, x0, R_x):
    y = R_x**2*(1. - ((x-x0)/R_x)**2)
    return y


def expfit(x, A, B, tau):
    y = A + (B-A)*np.exp(-x/tau)
    return y


def main():
    args = parse_args()

    contact_angle = dict()
    relaxation_time = dict()

    for base_folder in args.folders:
        folder = os.path.join(base_folder, "Analysis", "contour")

        time_data = np.loadtxt(os.path.join(
            base_folder, "Analysis", "time_data.dat"))
        time = time_data[:, 1]

        theta_c = []
        x0_ = []
        R_x_ = []
        fs = sorted(os.listdir(folder))
        for f in fs:
            data = np.loadtxt(os.path.join(folder, f))

            x = data[:, 1]
            y2 = data[:, 0]**2
            y2 = y2[x > 0.1]
            x = x[x > 0.1]
        
            popt, pcov = curve_fit(fitfunc,
                                   x, y2, p0=[0., 1.])

            if args.plot and f == fs[-1]:
                plt.plot(data[:, 1], data[:, 0])
                plt.plot(data[:, 1],
                         np.sqrt(fitfunc(data[:, 1], *popt)),
                         'r-')
                plt.show()

            x0, R_x = popt

            x0_.append(x0)
            R_x_.append(R_x)

        x0_ = np.array(x0_)
        R_x_ = np.array(R_x_)

        theta_c = 0.5+np.arcsin(x0_/R_x_)/np.pi

        if args.plot:
            plt.plot(time, x0_)
            plt.show()

            plt.plot(time, R_x_)
            plt.show()

        popt, pcov = curve_fit(expfit,
                               time, theta_c, p0=[1., 1., 1.])
        plt.plot(time, theta_c)
        plt.plot(time, expfit(time, *popt))
        plt.show()

        np.savetxt(
            os.path.join(base_folder, "Analysis", "contact_angle.dat"),
            np.array(zip(time, theta_c)))

        params = simplejson.load(
            open(os.path.join(
                base_folder, "Settings", "parameters_from_tstep_0.dat")))

        contact_angle[params["V_top"]] = theta_c[-1]
        relaxation_time[params["V_top"]] = popt[2]

    for key in sorted(contact_angle.keys()):
        print key, contact_angle[key], relaxation_time[key]


if __name__ == "__main__":
    main()
