from problems import *
import dolfin as df
__author__ = "Gaute Linga"

info_cyan("Welcome to the simple problem!")

parameters.update(
    solver="basic",
    folder="results_simple",
    restart_folder=False,
    tstep=0,
    dt=0.01
)


def mesh(Lx=1, Ly=5, h=1./16, **namespace):
    return df.RectangleMesh(df.Point(0., 0.), df.Point(Lx, Ly),
                            int(Lx/h), int(Ly/h))
