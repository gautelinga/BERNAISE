import numpy as np
import argparse
import matplotlib.pyplot as plt
import os


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze contact angle")
    parser.add_argument("folders", nargs='+', type=str, help="Folders")
    parser.add_argument("--plot", "-p", action="store_true", help="Plot")
    parser.add_argument("--logx", action="store_true", help="Log on x-axis")
    parser.add_argument("--logy", action="store_true", help="Log on y-axis")
    args = parser.parse_args()
    return args


def trim_string(fi):
    return fi.replace("#", "").strip()


def main():
    args = parse_args()

    lc_cycle = ['r', 'g', 'b', 'y', 'c', 'm', 'y', 'k']
    ls_cycle = ["-", "--", "-.", ":"]

    fig, ax = plt.subplots()
    for j, base_folder in enumerate(args.folders):
        folder = os.path.join(base_folder, "Analysis")

        filename = os.path.join(folder, "energy_in_time.dat")

        folder_id = folder.split("/")[-2]

        data = np.loadtxt(filename)

        with open(filename, "r") as f:
            fields = [trim_string(fi) for fi in f.readline().split("\t")]

        E_fields = dict()
        E_fields["E_tot"] = np.zeros_like(data[:, 0])
        for i, field in enumerate(fields):
            if field == "Time":
                t = data[:, i]
            elif "E" in field:
                E_fields[field] = data[:, i]
                E_fields["E_tot"] += data[:, i]

        for i, (field, E_data) in enumerate(E_fields.items()):
            label = "{} ({})".format(field, folder_id)
            plt.plot(t, E_data, label=label,
                     color=lc_cycle[i], linestyle=ls_cycle[j])

    if args.logx:
        ax.set_xscale("log")
    if args.logy:
        ax.set_yscale("log")
    plt.legend()
    plt.show()

    
if __name__ == "__main__":
    main()
