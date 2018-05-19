import pytest
import subprocess
import re
import math

number = "([0-9]+.[0-9]+e[+-][0-9]+)"
tol = 1e-5
# "basicnewton"


@pytest.mark.parametrize("solver",  ["basic"])
@pytest.mark.parametrize("num_proc", [1, 2])
def test_simple(solver, num_proc):
    cmd = ("cd ..; mpiexec -n {} python sauce.py solver={} "
           "problem=simple T=0.1 grid_spacing=0.1 "
           "testing=True")
    d = subprocess.check_output(cmd.format(num_proc, solver), shell=True)
    match = re.search("Velocity norm = " + number, str(d))
    err = match.groups()

    ref = 1.901026e-03
    assert(abs(eval(err[0])-ref) < tol)


@pytest.mark.parametrize("solver", ["basic"])
@pytest.mark.parametrize("num_proc", [1, 2])
def test_taylorgreen(solver, num_proc):
    cmd = ("cd ..; mpiexec -n {} python sauce.py solver={} "
           "problem=taylorgreen T=0.002 testing=True N=20")
    d = subprocess.check_output(cmd.format(num_proc, solver), shell=True)
    match = re.search("Final error norms: u = " + number +
                      " phi = " + number +
                      " c_p = " + number +
                      " c_m = " + number +
                      " V = " + number, str(d))
    err = match.groups()

    for e in err:
        assert eval(e) < 1e-1


if __name__ == "__main__":
    #test_simple("basic", 1)
    test_taylorgreen("basic", 1)
