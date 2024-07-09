"""Functions for reading RMG-formatted files."""


def species(inp: str, out: str | None = None):
    """Extract species information as a dataframe from an RMG species dictionary.

    :param inp: An RMG species dictionary, as a file path or string
    :param out: Optionally, write the output to this file path
    :return: The species dataframe
    """
    print(inp)
    print(out)
