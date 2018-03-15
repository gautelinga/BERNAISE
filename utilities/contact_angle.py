import numpy as np
import os
import argparse
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze contact angle")
    parser.add_argument("folder", type=str, help="Folder")
    args = parser.parse_args()
    return args


def fitfunc(x, x0, R_x, R_y):
    y = R_y**2*(1. - ((x-x0)/R_x)**2)
    return y


def main():
    args = parse_args()
    folder = os.path.join(args.folder, "Analysis", "contour")

    time_data = np.loadtxt(os.path.join(
        args.folder, "Analysis", "time_data.dat"))
    time = time_data[:, 1]

    theta_c = []
    x0_ = []
    R_x_ = []
    R_y_ = []
    fs = sorted(os.listdir(folder))
    for f in fs:
        #print f
        data = np.loadtxt(os.path.join(folder, f))

        #x = 0.5*(data[1:, 0]+data[:-1, 0])
        #y = 0.5*(data[1:, 1]+data[:-1, 1])
        #x = data[:, 0]
        #y = data[:, 1]

        x = data[:, 1]
        y2 = data[:, 0]**2
        y2 = y2[x > 0.1]
        x = x[x > 0.1]
        
        popt, pcov = curve_fit(fitfunc,
                               x, y2, p0=[0., 1., 1.])

        if f == fs[-1]:
            plt.plot(data[:, 1], data[:, 0])
            plt.plot(data[:, 1],
                     np.sqrt(fitfunc(data[:, 1], *popt)),
                     'r-')
            plt.show()

        x0, R_x, R_y = popt
        
        #plt.show()
        # plt.plot(-data[:, 0], data[:, 1])
        #dx = x[1:]-x[:-1]
        #dy = y[1:]-y[:-1]

        #phi = np.arctan2(y[1:]+y[:-1], x[1:]+x[:-1])
        #theta = np.arctan2(-dy, dx)
        x0_.append(x0)
        R_x_.append(R_x)
        R_y_.append(R_y)

    #np.savetxt(
    #    os.path.join(args.folder, "Analysis", "contang.dat"),
    #    np.array(zip(time, theta_c)))
    x0_ = np.array(x0_)
    R_x_ = np.array(R_x_)
    R_y_ = np.array(R_y_)
    
    plt.plot(time, x0_)
    plt.show()

    plt.plot(time, (R_y_-R_x_)/(R_x_+R_y_))
    plt.show()

    #print theta_c[-1]


if __name__ == "__main__":
    main()
