import sys
import json
from dolfin import MPI, mpi_comm_world
import os


RED = "\033[1;37;31m{s}\033[0m"
BLUE = "\033[1;37;34m{s}\033[0m"
GREEN = "\033[1;37;32m{s}\033[0m"
YELLOW = "\033[1;37;33m{s}\033[0m"
CYAN = "\033[1;37;36m{s}\033[0m"
NORMAL = "{s}"
ON_RED = "\033[41m{s}\033[0m"


# Stolen from Oasis
def convert(data):
    if isinstance(data, dict):
        return {convert(key): convert(value)
                for key, value in data.iteritems()}
    elif isinstance(data, list):
        return [convert(element) for element in data]
    elif isinstance(data, unicode):
        return data.encode('utf-8')
    else:
        return data


def str2list(string):
    if string[0] == "[" and string[-1] == "]":
        li = string[1:-1].split(",")
        for i in range(len(li)):
            li[i] = str2list(li[i])
        return li
    else:
        return parseval(string)


def parseval(value):
    try:
        value = json.loads(value)
    except ValueError:
        # json understands true/false, not True/False
        if value in ["True", "False"]:
            value = eval(value)
        elif "True" in value or "False" in value:
            value = eval(value)

    if isinstance(value, dict):
        value = convert(value)
    elif isinstance(value, list):
        value = convert(value)
    return value


def parse_command_line():
    cmd_kwargs = dict()
    for s in sys.argv[1:]:
        if s.count('=') == 1:
            key, value = s.split('=', 1)
        elif s in ["-h", "--help", "help"]:
            key, value = "help", "true"
        else:
            raise TypeError("Only kwargs separated with '=' allowed.")

        value = parseval(value)
        if isinstance(value, str):
            value = str2list(value)
        
        cmd_kwargs[key] = value
    return cmd_kwargs


def info_style(message, check=True, style=NORMAL):
    if MPI.rank(mpi_comm_world()) == 0 and check:
        print style.format(s=message)


def info_red(message, check=True):
    info_style(message, check, RED)


def info_blue(message, check=True):
    info_style(message, check, BLUE)


def info_yellow(message, check=True):
    info_style(message, check, YELLOW)


def info_green(message, check=True):
    info_style(message, check, GREEN)


def info_cyan(message, check=True):
    info_style(message, check, CYAN)


def info(message, check=True):
    info_style(message, check)


def info_on_red(message, check=True):
    info_style(message, check, ON_RED)


def info_split_style(msg_1, msg_2, style_1=BLUE, style_2=NORMAL, check=True):
    if MPI.rank(mpi_comm_world()) == 0 and check:
        print style_1.format(s=msg_1) + " " + style_2.format(s=msg_2)


def info_split(msg_1, msg_2, check=True):
    info_split_style(msg_1, msg_2)


def print_dir(folder):
    for path in os.listdir(folder):
        filename, ext = os.path.splitext(path)
        if ext == ".py" and filename[0].isalpha():
            info("   " + filename)


def help_menu():
    info_yellow("BERNAISE (Binary ElectRohydrodyNAmIc SolvEr)")
    info_red("You called for help! And here are your options:\n")

    info_cyan("Usage:")
    info("   python sauce.py problem=[...] solver=[...] ...")

    info_cyan("\nImplemented problems:")

    print_dir("problems")

    info_cyan("\nImplemented solvers:")

    print_dir("solvers")

    rank = mpi_comm_world().rank

    info("\n...or were you looking for the recipe "
         "for Bearnaise sauce? [y/N] ")
    try:
        q = raw_input("").lower()
    except:
        pass

    if rank == 0:
        if q in ["y", "yes"]:
            with open("common/recipe.txt") as f:
                lines = f.read().splitlines()

            for line in lines:
                info(line)
