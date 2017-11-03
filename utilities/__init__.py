import os
import sys
import glob
# Find path to the BERNAISE root folder
bernaise_path = "/" + os.path.join(*os.path.realpath(__file__).split("/")[:-2])
# ...and append it to sys.path to get functionality from BERNAISE
sys.path.append(bernaise_path)
from common import info, info_cyan


def get_help(methods, methods_folder, caller=__file__):
    info("Usage:\n   python " + os.path.basename(caller) +
         " method=... [optional arguments]\n")
    info_cyan("{:<18} {}".format(
        "Method", "Optional arguments (=default value)"))
    for method in methods:
        m = __import__("{}.{}".format(methods_folder, method))
        func = m.__dict__[method].method
        opt_args_str = ""
        argcount = func.__code__.co_argcount
        if argcount > 1:
            opt_args = zip(func.__code__.co_varnames,
                           func.__defaults__)

            opt_args_str = ", ".join(["=".join([str(item)
                                                for item in pair])
                                      for pair in opt_args])
        info("{method:<18} {opt_args_str}".format(
            method=method, opt_args_str=opt_args_str))
    exit()


def get_methods(methods_folder):
    methods = []
    for f in glob.glob(os.path.join(methods_folder, "*.py")):
        name = f.split(methods_folder)[1][1:].split(".py")[0]
        if name[0] != "_":
            methods.append((name, f))
    return dict(methods)
