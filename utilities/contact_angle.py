import numpy as np
import os
import argparse
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, leastsq
import simplejson


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze contact angle")
    parser.add_argument("folders", nargs='+', type=str, help="Folders")
    parser.add_argument("--plot", "-p", action="store_true", help="Plot")
    args = parser.parse_args()
    return args


def fitfunc(x, x0, R_x):
    y = np.sqrt(R_x**2*np.abs(1. - ((x-x0)/R_x)**2))
    return y


def calc_R(x, y, xc):
    return np.sqrt((x-xc)**2 + y**2)


def ff(c, x, y):
    Ri = calc_R(x, y, c)
    return Ri - Ri.mean()


def leastsq_circle(x, y):
    x_m = x.mean()
    x0, ierr = leastsq(ff, x_m, args=(x, y))
    Ri = calc_R(x, y, *x0)
    R = Ri.mean()
    res = np.sum((Ri-R)**2)
    return x0[0], R, res


def expfit(x, A, B, tau):
    y = A + (B-A)*np.exp(-x/tau)
    return y


def read_blocks(input_file):
    empty_lines = 0
    blocks = [[]]
    for line in open(input_file):
        # Check for empty/commented lines
        if not line or line.startswith('#') or line.startswith('\n'):
            # If 1st one: new block
            if empty_lines == 0:
                blocks.append([])
            empty_lines += 1
        # Non empty line: add line in current(last) block
        else:
            empty_lines = 0
            line.replace("\n", "")
            blocks[-1].append([float(e) for e in line.split(" ")])
    return blocks


def main():
    args = parse_args()

    contact_angle = dict()
    relaxation_time = dict()

    for base_folder in args.folders:
        folder = os.path.join(base_folder, "Analysis", "contour")

        time_data = np.loadtxt(os.path.join(
            base_folder, "Analysis", "time_data.dat"))
        time = time_data[:, 1]

        theta_c_ = []
        x0_ = []
        R_x_ = []
        fs = sorted(os.listdir(folder))
        for f in fs:
            infile = os.path.join(folder, f)
            data = np.array(read_blocks(infile)[0])

            x = data[:, 1]
            y = data[:, 0]
            y_ = y[x > 0.1]
            x_ = x[x > 0.1]

            x0, R, _ = leastsq_circle(x_, y_)

            theta_c = 0.5*np.pi+np.arcsin(x0/R)

            if args.plot and f == fs[-1]:
                plt.plot(x, y)

                theta_fit = np.linspace(0., theta_c, 180)
                x_fit = x0 + R*np.cos(theta_fit)
                y_fit = R*np.sin(theta_fit)
                plt.plot(x_fit, y_fit, 'r-')
                plt.show()

            x0_.append(x0)
            R_x_.append(R)
            theta_c_.append(theta_c)

        x0_ = np.array(x0_)
        R_x_ = np.array(R_x_)
        theta_c_ = np.array(theta_c_)/np.pi

        if args.plot:
            plt.plot(time, x0_)
            plt.show()

            plt.plot(time, R_x_)
            plt.show()

            plt.plot(time, theta_c_)
            plt.show()
            
        popt, pcov = curve_fit(expfit,
                               time,
                               theta_c_, p0=[1., 1., 1.])

        plt.plot(time, theta_c_)
        plt.plot(time, expfit(time, *popt))
        plt.show()

        np.savetxt(
            os.path.join(base_folder, "Analysis", "contact_angle.dat"),
            np.array(zip(time, theta_c_)))

        params = simplejson.load(
            open(os.path.join(
                base_folder, "Settings", "parameters_from_tstep_0.dat")))

        contact_angle[params["V_top"]] = theta_c_[-1]
        relaxation_time[params["V_top"]] = popt[2]

    for key in sorted(contact_angle.keys()):
        print key, contact_angle[key], relaxation_time[key]


if __name__ == "__main__":
    main()
