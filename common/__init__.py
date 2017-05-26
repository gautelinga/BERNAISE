from io import *
import sys, json


# Stolen from Oasis
def convert(input):
    if isinstance(input, dict):
        return {convert(key): convert

def parse_command_line():
    commandline_kwards = {}
    for s in sys.argv[1:]:
        if s.count('=') == 1:
            key, value = s.split('=', 1)
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
