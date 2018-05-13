import numpy as np
import argparse
import matplotlib.pyplot as plt
import os
import simplejson


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze contact angle")
    parser.add_argument("folders", nargs='+', type=str, help="Folders")
    parser.add_argument("--plot", "-p", action="store_true", help="Plot")
    parser.add_argument("--logx", action="store_true", help="Log on x-axis")
    parser.add_argument("--logy", action="store_true", help="Log on y-axis")
    parser.add_argument("--latex", action="store_true", help="Use latex to plot.")
    parser.add_argument("--nokey", action="store_true", help="No keys.")
    parser.add_argument("--keys", type=str, help="Keys to plot in latex.")
    args = parser.parse_args()
    return args


def trim_string(fi):
    return fi.replace("#", "").strip()


replace_list = [
    ["E", "F"],
    ["kin", "{\\bf u}"],
    ["tot", "{\\rm t}"],
    ["c_p", "{c_+}"],
    ["c_m", "{c_-}"],
    ["c_n", "{c_{\\rm n}}"]]


def parse_field_name(field):
    field_name = field
    for a, b in replace_list:
        field_name = field_name.replace(a, b)
    return field_name


def main():
    args = parse_args()

    if args.latex:
        from matplotlib import rc
        rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
        ## for Palatino and other serif fonts use:
        #rc('font',**{'family':'serif','serif':['Palatino']})
        rc('text', usetex=True)
    
    lc_cycle = ['r', 'g', 'b', 'y', 'c', 'm', 'y', 'k']
    ls_cycle = ["-", "--", "-.", ":"]

    plot_lines = []

    fig, ax = plt.subplots()
    for j, base_folder in enumerate(args.folders):
        folder = os.path.join(base_folder, "Analysis")

        filename = os.path.join(folder, "energy_in_time.dat")

        params = simplejson.load(
            open(os.path.join(
                base_folder, "Settings",
                "parameters_from_tstep_0.dat")))

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
            field_name = field
            label = "{} ({})".format(field_name, folder_id) if j == 0 else ""
            if args.latex:
                field_name = parse_field_name(field)
                label = "${}$".format(field_name) if j == 0 else ""

            l1, = plt.plot(t, E_data, label=label,
                           color=lc_cycle[i], linestyle=ls_cycle[j])

            if args.latex and i == 0:
                l2, = plt.plot(t[0], E_data[0],
                               color="black", linestyle=ls_cycle[j])

                label_string = "{}".format(folder_id)
                if args.keys:
                    keys = args.keys.split(",")
                    parts = []
                    for key in keys:
                        kkey = key.split("=")
                        if len(kkey) > 1:
                            parts.append("$" + kkey[0] + "={" + kkey[1] + "}$")
                        else:
                            parts.append("$\\rm {" + key + "}$")
                    label_string = ", ".join(parts)
                plot_lines.append((l2, label_string.format(**params)))

    if args.logx:
        ax.set_xscale("log")
    if args.logy:
        ax.set_yscale("log")
    if not args.nokey:
        if args.latex:
            legend1 = plt.legend([a for a, b in plot_lines],
                                 [b for a, b in plot_lines],
                                 loc=6)
        plt.legend()
        if args.latex:
            plt.gca().add_artist(legend1)

    plt.xlabel("$\\textrm{Time} \\ t$")
    plt.ylabel("$\\textrm{Energy}$")
    plt.show()


if __name__ == "__main__":
    main()
