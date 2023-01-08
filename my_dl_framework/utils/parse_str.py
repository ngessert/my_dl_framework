"""Helper for parsing a string and extract the original type"""
import ast
from typing import Union, List


def parse_str(orig_str: str) -> Union[str, int, float, bool, List, None]:
    if orig_str == "":
        return ""
    if orig_str == "True":
        return True
    if orig_str == "False":
        return False
    try:
        return int(orig_str)
    except ValueError:
        pass
    try:
        return float(orig_str)
    except ValueError:
        pass
    if orig_str.startswith("["):
        return ast.literal_eval(orig_str)
    return orig_str
