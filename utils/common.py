import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import TextIO

import torch
from packaging import version
from torch import Size, Tensor

Device = str | torch.device

torch_var_opts = (
    {"unbiased": False}
    if version.parse(torch.__version__) < version.parse("2.0")
    else {"correction": 0}
)


class StyledTerminal:
    r"""
    Container class for terminal ANSI escape codes.
    """
    # Foreground color
    BLACK = "\033[30m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"
    DEFAULT = "\033[39m"
    GRAY = "\033[90m"
    BRIGHT_RED = "\033[91m"
    BRIGHT_GREEN = "\033[92m"
    BRIGHT_YELLOW = "\033[93m"
    BRIGHT_BLUE = "\033[94m"
    BRIGHT_MAGENTA = "\033[95m"
    BRIGHT_CYAN = "\033[96m"
    BRIGHT_WHITE = "\033[97m"
    # Background color
    BLACKB = "\033[40m"
    REDB = "\033[41m"
    GREENB = "\033[42m"
    YELLOWB = "\033[43m"
    BLUEB = "\033[44m"
    MAGENTAB = "\033[45m"
    CYANB = "\033[46m"
    WHITEB = "\033[47m"
    DEFAULTB = "\033[49m"
    GRAYB = "\033[100m"
    BRIGHT_REDB = "\033[101m"
    BRIGHT_GREENB = "\033[102m"
    BRIGHT_YELLOWB = "\033[103m"
    BRIGHT_BLUEB = "\033[104m"
    BRIGHT_MAGENTAB = "\033[105m"
    BRIGHT_CYANB = "\033[106m"
    BRIGHT_WHITEB = "\033[107m"
    # Font styles
    BOLD = "\033[1m"
    LIGHT = "\033[2m"
    ITALIC = "\033[3m"
    UNDERLINE = "\033[4m"
    STRIKETHROUGH = "\033[9m"
    DEFAULTF = "\033[10m"
    # Reset
    END = "\033[0m"

    @staticmethod
    def sprint(
        text: str,
        fg: str = DEFAULT,
        bg: str = DEFAULTB,
        *styles: str,
        **kwargs
    ):
        r"""
        Styled-print.

        :param text: The output text string.
        :type text: str
        :param fg: Foreground ANSI code. Defaults to DEFAULT.
        :type fg: str, optional
        :param bg: Background ANSI code. Defaults to DEFAULTB.
        :type bg: str, optional
        :param styles: Font style ANSI codes.
        :type styles: str
        :param kwargs: Standard print kwargs.
        """
        print(f"{fg}{bg}{''.join(styles)}{text}{StyledTerminal.END}", **kwargs)


def relpath(_from: Path, _to: Path) -> str:
    r"""
    Compute the relative path from a staring file/folder to a given
    file/folder. If the starting path refers to a file, its parent
    folder will be used instead.

    :param _from: The absolute path of the starting file/folder.
    :type _from: Path
    :param _to: The absolute path of the target file/folder.
    :type _to: Path
    :return: The relative path.
    :rtype: str
    """
    from_dir = str(_from.parent if _from.is_file() else _from)
    _to = str(_to)
    return os.path.relpath(_to, start=from_dir)


def to_full_list(compact_list: str) -> list[int]:
    r"""
    Compact list parser.

    :param compact_list: The input compact list.
    :type compact_list: str
    :return: The full parsed list version.
    :rtype: list[int]
    """
    # Clean and split
    compact_list = compact_list.replace(" ", "").split(",")
    full_list = []
    for item in compact_list:
        if "-" in item:
            # Range value
            l, r = [int(x) for x in item.split("-")]
            step = 1 if l < r else -1
            full_list += list(range(l, r, step))
        else:
            # Integer value
            full_list += [int(item)]
    return full_list


def wrong_shape_msg(
    tensor: Tensor,
    name: str,
    expected_shape: str | tuple[()] | tuple[int, ...] | Size,
    constr: str = None
) -> str:
    r"""
    Return a formatted message that reports a description of the Tensor
    shape error.

    :param tensor: The input tensor, that has the wrong shape.
    :type tensor: Tensor
    :param name: The codename of the input tensor.
    :type name: str
    :param expected_shape: The expected (correct) shape.
    :type expected_shape: str | tuple[()] | tuple[int] | Size
    :param constr: An optional constraint string about the shape.
        Defaults to None (no constraints).
    :type constr: str, optional
    :return: The formatted error message.
    :rtype: str
    """
    exp_shape = tuple(expected_shape)
    constr = f" ({constr})" if constr is not None else ""
    return (
        f"The expected shape for {name} is {exp_shape}{constr}, but a "
        f"tensor of shape {tuple(tensor.shape)} was given instead."
    )


@contextmanager
def output_redirected(to_file: str = os.devnull):
    r"""
    Low-level stdout and stderr redirection.

    :param to_file: The file to redirect to. Defaults to os.devnull.
    :type to_file: str, optional
    """
    fd_out = sys.stdout.fileno()
    fd_err = sys.stderr.fileno()

    def _redirect(to_out: TextIO, to_err: TextIO):
        nonlocal fd_out, fd_err
        sys.stdout.close()  # implicit flush
        sys.stderr.close()  # implicit flush
        os.dup2(to_out.fileno(), fd_out)  # fd_out writes to 'to' file
        os.dup2(to_err.fileno(), fd_err)  # fd_err writes to 'to' file
        sys.stdout = os.fdopen(fd_out, "w")  # Python writes to fd_out
        sys.stderr = os.fdopen(fd_err, "w")  # Python writes to fd_err

    with (
        os.fdopen(os.dup(fd_out), "w") as old_stdout,
        os.fdopen(os.dup(fd_err), "w") as old_stderr
    ):
        with open(to_file, "w") as file:
            _redirect(to_out=file, to_err=file)
        try:
            yield
        finally:
            _redirect(to_out=old_stdout, to_err=old_stderr)  # restore outputs.
