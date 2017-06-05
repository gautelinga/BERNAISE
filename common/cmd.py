import sys
import json
from dolfin import MPI, mpi_comm_world


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


def parse_command_line():
    cmd_kwargs = dict()
    for s in sys.argv[1:]:
        if s.count('=') == 1:
            key, value = s.split('=', 1)
        elif s in ["-h", "--help", "help"]:
            key, value = "help", "true"
        else:
            raise TypeError("Only kwargs separated with '=' allowed.")
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

def help_menu():
    info_yellow("BERNAISE (Binary ElectRohydrodyNAmIc SolvEr)")
    info_red("You called for help! And here are your options:")
    info("")
    info("  Bearnaise Sauce") 
    info("  (Sauce bearnaise a la Paul Bocuse)")
    info("")
    info("  Serves 4 to 6")
    info("")
    info("  Preparation time: 10 minutes")
    info("  Cooking time: 15 minutes")
    info("")
    info("  3 medium shallots")
    info("  3 tablespoons white wine vinegar")
    info("  2 tablespoons chopped tarragon")
    info("  1 pinch chopped chervil ")
    info("  4 egg yolks") 
    info("  250 g unsalted butter") 
    info("  Salt and freshly crushed pepper")  
    info("")
    info("  This sauce must be made over very low heat, just before serving.\n\
    Making it in a double boiler will ensure good results:\n\
    Use the top of the double boiler as an ordinary pot for the first stage of making the sauce.")
    info("")
    info("  Peel and finely chop the shallots.\n\
    Place the vinegar, shallots, chopped tarragon and chervil (setting aside a little of both for the end),\n\
    a little salt and a pinch of freshly crushed pepper in the top part of  the double boiler.\n\
    Place over high heat and reduce until you have the equivalent of 2 teaspoons left in the pan.\n\
    Remove from the heat and leave to cool completely before finishing the sauce \n\
    (you can speed cooling by holding the pot in a bowl of ice water).")
    info("")
    info("  Whisk the egg yolks, one by one, and 2 tablespoons of cold water into the vinegar mixture.\n\
    Heat a little water in the bottom of the double boiler and set the top part in place.")
    info("")
    info("  Break the butter into small pieces.\n\
    Whisk in a piece of the butter, the continue adding the rest of the butter little by little, whisking constantly.\n\
    The sauce should become foamy at first, then thicken as the butter is added. Keep warm over warm hot-water bath.")
    info("")
    info("  Just before serving, add salt and pepper, if needed and the remaining tarragon and chervil. Serve in a sauceboat.")
    info("")
    info("  Note: this sauce should be much thicker than a Hollandaise sauce; it has more egg yolks.\n\
    It should have the consistency as Dijon mustard.")
    info("")
    info("  Serving suggestions: This is the perfect sauce for any grilled meat.")
