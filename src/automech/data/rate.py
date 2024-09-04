"""Kinetics dataclasses.

Eventually, we will want to add a "Raw" rate type with k(T,P) values stored in an xarray
"""

import abc
import dataclasses
import enum
from collections import defaultdict
from collections.abc import Sequence
from typing import Any

import more_itertools as mit
import numpy
import pyparsing as pp
from pyparsing import pyparsing_common as ppc

MatrixLike = Sequence[Sequence[float]] | numpy.ndarray


class RateType(str, enum.Enum):
    """The type of reaction rate (type of pressure dependence)."""

    CONSTANT = "Constant"
    FALLOFF = "Falloff"
    ACTIVATED = "Activated"
    PLOG = "Plog"
    CHEB = "Chebyshev"


class BlendType(str, enum.Enum):
    """The type of blending function for high and low-pressure rates."""

    LIND = "Lind"
    TROE = "Troe"


@dataclasses.dataclass
class ArrheniusFunction:
    """An Arrhenius or Landau-Teller function (see cantera.Arrhenius).

    :param A: The pre-exponential factor [(m^3/kmol)**o/s]
    :param b: The temperature exponent
    :param E: The activation energy E [J/kmol]
    :param B: The Landau-Teller B-factor (optional)
    :param C: The Landau-Teller C-factor (optional)
    """

    A: float = 1.0
    b: float = 0.0
    E: float = 0.0
    B: float | None = None
    C: float | None = None


ArrheniusFunctionOrData = Sequence[float] | ArrheniusFunction


def arrhenius_function_from_data(
    data: ArrheniusFunctionOrData,
) -> ArrheniusFunction:
    """Build an Arrhenius function object from data.

    :param data: The Arrhenius parameters (A, b, E), optionally followed by the two
        Landau factors (B, C)
    :return: The Arrhenius function object
    """
    if isinstance(data, ArrheniusFunction):
        return ArrheniusFunction(*arrhenius_params(data))

    return ArrheniusFunction(*data)


def arrhenius_params(k: ArrheniusFunction, lt: bool = True) -> tuple[float, ...]:
    """Get the parameters for an Arrhenius or Landau-Teller function.

    :param k: The Arrhenius function object
    :param lt: Include Landau-Teller parameters, if defined?
    :return: The parameters A, b, E, (B*, C*)
    """
    lt = False if k.B is None or k.C is None else lt
    return (k.A, k.b, k.E, k.B, k.C) if lt else (k.A, k.b, k.E)


def arrhenius_string(
    k: ArrheniusFunction,
    lt: bool = True,
    digits: int = 4,
) -> str:
    """Write the parameters for an Arrhenius function to a string.

    :param k: The Arrhenius function object
    :param lt: Include Landau-Teller parameters, if defined?
    :param sci_a: Always put the pre-exponential factor in scientific notation?
    :param sci_b: Always put the temperature exponent in scientific notation?
    :param sci_e: Always put the activation energy in scientific notation?
    :return: The string
    """
    nums = arrhenius_params(k, lt=lt)
    always_sci = [True, False, False, False, False]  # use scientific notation for A
    return write_numbers(nums=nums, always_sci=always_sci, digits=digits)


@dataclasses.dataclass
class BlendingFunction:
    """A blending function for high and low-pressure rates (see cantera.Falloff).

    Types:
        Lind   - coeffs: (None)
        Troe   - coeffs: a, T, T***, T*, (T**)

    :param coeffs: A list of coefficients for the parametrization
    :param type_: The type of parametrization: "Lind", "Troe"
    """

    type_: BlendType = BlendType.LIND
    coeffs: tuple[float, ...] | None = None

    def __post_init__(self):
        """Initialize attributes."""
        self.type_ = BlendType(self.type_)


BlendingFunctionOrData = tuple[str | BlendType, Sequence[float]] | BlendingFunction


def blending_function_from_data(data: BlendingFunctionOrData) -> BlendingFunction:
    """Build a blending function object from data.

    If a blending function is passed in, it will be returned as-is.

    :param data: _description_
    :return: _description_
    """
    if isinstance(data, BlendingFunction):
        coeffs = f_coeffs(data)
        type_ = f_type(data)
        return BlendingFunction(type_=type_, coeffs=coeffs)

    return BlendingFunction(*data)


def f_coeffs(f: BlendingFunction) -> BlendType:
    """Get the coefficients of a blending function.

    :param f: The blending function object
    :return: The blend function coefficients
    """
    return f.coeffs


def f_type(f: BlendingFunction) -> BlendType:
    """Get the blend type of a blending function.

    :param f: The blending function object
    :return: The blend type
    """
    return f.type_


class Rate(abc.ABC):
    """Base class for reaction rates.

    :param k: Primary Arrhenius function
    :param is_rev: Whether this rate describes a reversible reaction
    :param type_: The type of reaction
    """

    # @property
    # @abc.abstractmethod
    # def k(self):
    #     """The primary Arrhenius function."""
    #     pass

    @property
    @abc.abstractmethod
    def type_(self):
        """The type of reaction."""
        pass

    @property
    @abc.abstractmethod
    def is_rev(self):
        """Whether this rate describes a reversible reaction."""
        pass


@dataclasses.dataclass
class SimpleRate(Rate):
    """Simple reaction rate, k(T,P) parametrization (see cantera.ReactionRate).

    Types:
        Constant    - k: The rate coefficient
        Falloff     - k: The high-pressure rate coefficient (M-independent)
                    - k0: The low-pressure rate coefficient
                    - f: The blending function, F(T, P_r)
        Activated   - k: The high-pressure rate coefficient
                    - k0: The low-pressure rate coefficient (M-independent)
                    - f: The blending function, F(T, P_r)

    :param k: The (high-pressure limiting) Arrhenius function for the reaction
    :param k0: The low-pressure limiting Arrhenius function for the reaction
    :param f: Falloff function for blending the high- and low-pressure rate coefficients
    :param is_rev: Is this a reversible reaction?
    :param type_: The type of reaction: "Constant", "Falloff", "Activated"
    """

    k: ArrheniusFunction = dataclasses.field(default_factory=ArrheniusFunction)
    k0: ArrheniusFunction | None = None
    f: BlendingFunction | None = None
    is_rev: bool = True
    type_: RateType = RateType.CONSTANT

    def __post_init__(self):
        """Initialize attributes."""
        self.k = arrhenius_function_from_data(self.k)
        self.k0 = None if self.k0 is None else arrhenius_function_from_data(self.k0)
        self.f = None if self.f is None else blending_function_from_data(self.f)

        self.type_ = RateType.CONSTANT if self.type_ is None else RateType(self.type_)
        assert self.type_ in (RateType.CONSTANT, RateType.FALLOFF, RateType.ACTIVATED)

        if self.type_ == RateType.CONSTANT:
            assert self.f is None, f"f={self.f} requires P-dependent reaction type"
            assert self.k0 is None, f"k={self.k0} requires P-dependent reaction type"

        if self.type_ != RateType.CONSTANT:
            self.f = BlendingFunction() if self.f is None else self.f


@dataclasses.dataclass
class PlogRate(Rate):
    """P-Log reaction rate, k(T,P) parametrization (see cantera.ReactionRate).

    :param ks: Rate coefficients at specific pressures, k_P_, k_P2, ...
    :param ps: An array of pressures, P_, P2, ... [Pa]
    :param k: Optional high-pressure rate
    :param is_rev: Is this a reversible reaction?
    """

    ks: tuple[ArrheniusFunction, ...]
    ps: tuple[float, ...]
    k: ArrheniusFunction | None = None
    is_rev: bool = True
    type_: RateType = RateType.PLOG

    def __post_init__(self):
        """Initialize attributes."""
        self.ks = tuple(map(arrhenius_function_from_data, self.ks))
        self.ps = tuple(map(float, self.ps))
        self.k = None if self.k is None else arrhenius_function_from_data(self.k)
        self.type_ = RateType.PLOG if self.type_ is None else RateType(self.type_)
        assert self.type_ == RateType.PLOG


@dataclasses.dataclass
class ChebRate(Rate):
    """Chebyshev reaction rate, k(T,P) parametrization (see cantera.ReactionRate).

    :param t_limits: The min/max temperature limits [K] for the Chebyshev fit
    :param p_limits: The min/max pressure limits [K] for the Chebyshev fit
    :param coeffs: The Chebyshev expansion coefficients
    :param k: Optional high-pressure rate
    :param is_rev: Is this a reversible reaction?
    """

    t_limits: tuple[float, float]
    p_limits: tuple[float, float]
    coeffs: numpy.ndarray
    k: ArrheniusFunction | None = None
    is_rev: bool = True
    type_: str = RateType.CHEB

    def __post_init__(self):
        """Initialize attributes."""
        self.t_limits = tuple(map(float, self.t_limits))
        self.p_limits = tuple(map(float, self.p_limits))
        self.coeffs = numpy.array(self.coeffs)
        assert numpy.ndim(self.coeffs) == 2, f"Must be 2-dimensional: {self.coeffs}"
        self.k = None if self.k is None else arrhenius_function_from_data(self.k)
        self.type_ = RateType.CHEB if self.type_ is None else RateType(self.type_)
        assert self.type_ == RateType.CHEB


# constructors
def from_data(
    k: ArrheniusFunctionOrData,
    k0: ArrheniusFunctionOrData | None = None,
    f: tuple[str, Sequence[float]] | BlendingFunction | None = None,
    plog_ks: Sequence[ArrheniusFunctionOrData] | None = None,
    plog_ps: Sequence[float] | None = None,
    cheb_t_limits: Sequence[float] | None = None,
    cheb_p_limits: Sequence[float] | None = None,
    cheb_coeffs: MatrixLike | None = None,
    type_: str | RateType | None = None,
    is_rev: bool = True,
) -> Rate:
    """Build a rate object from data.

    :param k: The (high-pressure limiting) Arrhenius function for the reaction
    :param k0: The low-pressure limiting Arrhenius function for the reaction
    :param f: Falloff function for blending the high- and low-pressure rate coefficients
    :param plog_ks: P-Log rate coefficients at specific pressures, k_P_, k_P2, ...
    :param plog_ps: P-Log pressures, P_, P2, ... [Pa]
    :param cheb_t_limits: The min/max temperature limits [K] for the Chebyshev fit
    :param cheb_p_limits: The min/max pressure limits [K] for the Chebyshev fit
    :param cheb_coeffs: The Chebyshev expansion coefficients
    :param type_: The type of reaction: "Constant", "Falloff", "Activated", "Plog"
    :param is_rev: Is this a reversible reaction?
    :return: _description_
    """
    type_ = None if type_ is None else RateType(type_)

    plog_args = (plog_ks, plog_ps)
    if any(arg is not None for arg in plog_args) or type_ == RateType.PLOG:
        return PlogRate(ks=plog_ks, ps=plog_ps, k=k, is_rev=is_rev, type_=type_)

    cheb_args = (cheb_t_limits, cheb_p_limits, cheb_coeffs)
    if any(arg is not None for arg in cheb_args) or type_ == RateType.CHEB:
        return ChebRate(
            t_limits=cheb_t_limits,
            p_limits=cheb_p_limits,
            coeffs=cheb_coeffs,
            k=k,
            is_rev=is_rev,
            type_=type_,
        )

    return SimpleRate(k=k, k0=k0, f=f, is_rev=is_rev, type_=type_)


def from_chemkin_string(rate_str: str, is_rev: bool = True) -> Rate:
    """Read the CHEMKIN rate from a string.

    :param rxn_str: CHEMKIN rate string
    :param is_rev: Is this a reversible reaction?
    :return: The reaction rate object
    """
    # Parse the string
    rate_expr = chemkin_rate_expr()
    res = rate_expr.parseString(rate_str).as_dict()

    # Gather auxiliary data
    aux_dct = defaultdict(list)
    for key, *val in res.get("aux"):
        if key == "PLOG":
            aux_dct[key].append(val)
        else:
            aux_dct[key].extend(val)
    aux_dct = dict(aux_dct)

    # Pre-process rate type data
    type_ = RateType.CONSTANT
    k = res.get("params")
    k0 = None
    if "LOW" in aux_dct:
        k0 = aux_dct.get("LOW")
        type_ = RateType.FALLOFF

    if "HIGH" in aux_dct:
        k0 = k
        k = aux_dct.get("HIGH")
        type_ = RateType.ACTIVATED

    plog_ks = plog_ps = None
    if "PLOG" in aux_dct:
        plog_ks = tuple(c[1:] for c in aux_dct["PLOG"])
        plog_ps = tuple(c[0] for c in aux_dct["PLOG"])
        type_ = RateType.PLOG

    cheb_t_limits = cheb_p_limits = cheb_coeffs = None
    if "CHEB" in aux_dct:
        cheb_t_limits = aux_dct.get("TCHEB", (300, 2500))
        cheb_p_limits = aux_dct.get("PCHEB", (0.001, 100))
        cheb_vals = aux_dct["CHEB"]
        cheb_n, cheb_m = cheb_vals[:2]
        cheb_coeffs = numpy.reshape(cheb_vals[2:], (cheb_n, cheb_m))
        type_ = RateType.CHEB

    # Pre-process blending function data
    f = None
    if "TROE" in aux_dct:
        f = (BlendType.TROE, aux_dct.get("TROE"))

    # Call central constructor
    return from_data(
        k=k,
        k0=k0,
        f=f,
        plog_ks=plog_ks,
        plog_ps=plog_ps,
        cheb_t_limits=cheb_t_limits,
        cheb_p_limits=cheb_p_limits,
        cheb_coeffs=cheb_coeffs,
        type_=type_,
        is_rev=is_rev,
    )


# getters
# # common
def arrhenius_function(rate: Rate) -> ArrheniusFunction | None:
    """Get the primary Arrhenius function for the reaction.

    (The one that appears after the reaction equation in a CHEMKIN file)

    :param rate: The rate object
    :return: The primary Arrhenius function
    """
    if isinstance(rate, SimpleRate) and type_(rate) == RateType.ACTIVATED:
        return rate.k0

    return rate.k


def type_(rate: Rate) -> RateType:
    """Get the type of reaction.

    :param rate: The rate object
    :return: The type of reaction
    """
    return rate.type_


def is_reversible(rate: Rate) -> bool:
    """Whether this rate describes a reversible reaction.

    :param rate: The rate object
    :return: `True` if it does, `False` if it doesn't
    """
    return rate.is_rev


# # simple
def high_p_arrhenius_function(rate: Rate) -> ArrheniusFunction | None:
    """Get the high-pressure limiting Arrhenius function for the reaction.

    :param rate: The rate object
    :return: The Arrhenius function
    """
    if not isinstance(rate, SimpleRate | PlogRate):
        return None

    return rate.k


def low_p_arrhenius_function(rate: Rate) -> ArrheniusFunction | None:
    """Get the low-pressure limiting Arrhenius function for the reaction.

    :param rate: The rate object
    :return: The Arrhenius function
    """
    if not isinstance(rate, SimpleRate):
        return None

    return rate.k0


def blend_function(rate: Rate) -> BlendingFunction | None:
    """Get the function for blending high- and low-pressure rates.

    :param rate: The rate object
    :return: The blend function
    """
    if not isinstance(rate, SimpleRate):
        return None

    return rate.f


# # plog
def plog_arrhenius_functions(rate: Rate) -> tuple[ArrheniusFunction, ...] | None:
    """Arrhenius functions for a P-Log reaction rate.

    :param rate: The rate object
    :return: The Arrhenius functions
    """
    if not isinstance(rate, PlogRate):
        return None

    return rate.ks


def plog_pressures(rate: Rate) -> tuple[float, ...] | None:
    """Pressures for a P-Log reaction rate.

    :param rate: The rate object
    :return: The pressures
    """
    if not isinstance(rate, PlogRate):
        return None

    return rate.ps


# # cheb
def chebyshev_temperature_limits(rate: Rate) -> tuple[float, float] | None:
    """Temperature limits for a Chebyshev reaction rate.

    :param rate: The rate object
    :return: The minimum and maximum temperature
    """
    if not isinstance(rate, ChebRate):
        return None

    return rate.t_limits


def chebyshev_pressure_limits(rate: Rate) -> tuple[float, float] | None:
    """Pressure limits for a Chebyshev reaction rate.

    :param rate: The rate object
    :return: The minimum and maximum pressure
    """
    if not isinstance(rate, ChebRate):
        return None

    return rate.p_limits


def chebyshev_coefficients(rate: Rate) -> numpy.ndarray | None:
    """Coefficients for a Chebyshev reaction rate.

    :param rate: The rate object
    :return: The Chebyshev coefficients
    """
    if not isinstance(rate, ChebRate):
        return None

    return rate.coeffs


# properties
# # common
def has_collider(rate: Rate) -> bool:
    """Whether this rate type involves a collider.

    :param rate: The rate object
    :return: `True` if it does, `False` if it doesn't
    """
    return type_(rate) in (RateType.ACTIVATED, RateType.FALLOFF)


def is_falloff(rate: Rate) -> bool:
    """Whether this is a falloff reaction.

    :param rate: The rate object
    :return: `True` if it is, `False` if it isn't
    """
    return type_(rate) == RateType.FALLOFF


# # simple
def high_p_params(rate: Rate, lt: bool = True) -> tuple[float, ...] | None:
    """Get the high-pressure limiting Arrhenius function for the reaction.

    :param rate: The rate object
    :param lt: Include Landau-Teller parameters* along with basic Arrhenius parameters?
    :return: The Arrhenius parameters A, b, E, (B*, C*)
    """
    k = high_p_arrhenius_function(rate)
    if k is None:
        return None

    return arrhenius_params(k, lt=lt)


def low_p_params(rate: Rate, lt: bool = True) -> tuple[float, ...] | None:
    """Get the low-pressure limiting Arrhenius parameters for the reaction.

    :param rate: The rate object
    :param lt: Include Landau-Teller parameters* along with basic Arrhenius parameters?
    :return: The Arrhenius parameters A, b, E, (B*, C*)
    """
    k0 = low_p_arrhenius_function(rate)
    if k0 is None:
        return None

    return arrhenius_params(k0, lt=lt)


def blend_type(rate: Rate) -> BlendType | None:
    """Get the function type for blending high- and low-pressure rates.

    :param rate: The rate object
    :return: The blend type
    """
    f = blend_function(rate)
    if f is None:
        return None

    return f_type(f)


def blend_coeffs(rate: Rate) -> BlendType | None:
    """Get the coefficients for blending high- and low-pressure rates.

    :param rate: The rate object
    :return: The blend coefficients
    """
    f = blend_function(rate)
    if f is None:
        return None

    return f_coeffs(f)


# # plog
def plog_params(rate: Rate, lt: bool = True) -> tuple[ArrheniusFunction, ...] | None:
    """Arrhenius functions for a P-Log reaction rate.

    :param rate: The rate object
    :param lt: Include Landau-Teller parameters* along with basic Arrhenius parameters?
    :return: The Arrhenius parameters A, b, E, (B*, C*)
    """
    if not isinstance(rate, PlogRate):
        return None

    return tuple(arrhenius_params(k, lt=lt) for k in plog_arrhenius_functions(rate))


def plog_params_dict(
    rate: Rate, lt: bool = True
) -> dict[float, ArrheniusFunction] | None:
    """Get the P-Log Arrhenius parameters, as a dictionary by pressure.

    :param rate: The rate object
    :param lt: Include Landau-Teller parameters* along with basic Arrhenius parameters?
    :return: The Arrhenius parameters A, b, E, (B*, C*)
    """
    if not isinstance(rate, PlogRate):
        return None

    return dict(zip(plog_pressures(rate), plog_params(rate, lt=lt), strict=True))


# I/O
def chemkin_string(rate: Rate, eq_width: int = 0) -> str:
    """Write the CHEMKIN rate to a string.

    :param rate_: The reaction rate object
    :param eq_width: The width of the equation, for alignment purposes
    :return: CHEMKIN rate string
    """
    # Write the top line
    top_k = arrhenius_function(rate)
    top_line = arrhenius_string(top_k)
    lines = [top_line]

    # Calculate the total width of the top line for alignment
    top_width = eq_width + len(top_line) + 1

    # Add any auxiliary lines
    if type_(rate) == RateType.ACTIVATED:
        k_hi = high_p_arrhenius_function(rate)
        lines.append(
            chemkin_aux_line("HIGH", arrhenius_string(k_hi), top_width=top_width)
        )

    if type_(rate) == RateType.FALLOFF:
        k_lo = low_p_arrhenius_function(rate)
        lines.append(
            chemkin_aux_line("LOW", arrhenius_string(k_lo), top_width=top_width)
        )

    if type_(rate) == RateType.PLOG:
        ks = plog_arrhenius_functions(rate)
        ps = plog_pressures(rate)
        plog_lines = [
            chemkin_aux_line(
                "PLOG", [write_number(p), arrhenius_string(k)], top_width=top_width
            )
            for p, k in zip(ps, ks, strict=True)
        ]
        lines.extend(plog_lines)

    if type_(rate) == RateType.CHEB:
        t_limits = chebyshev_temperature_limits(rate)
        p_limits = chebyshev_pressure_limits(rate)
        coeffs = chebyshev_coefficients(rate)
        shape = numpy.shape(coeffs)
        cheb_lines = [
            chemkin_aux_line("TCHEB", write_numbers(t_limits), top_width=top_width),
            chemkin_aux_line("PCHEB", write_numbers(p_limits), top_width=top_width),
            chemkin_aux_line(
                "CHEB", write_numbers(shape, as_int=True), top_width=top_width
            ),
        ] + [
            chemkin_aux_line("CHEB", write_numbers(cs), top_width=top_width)
            for cs in mit.chunked(numpy.ravel(coeffs), 4)
        ]
        lines.extend(cheb_lines)

    if blend_type(rate) == BlendType.TROE:
        coeffs = blend_coeffs(rate)
        lines.append(
            chemkin_aux_line("TROE", write_numbers(coeffs), top_width=top_width)
        )

    return "\n".join(lines)


# Legacy
def to_old_object(rate: Rate) -> Any:
    """Convert a new rate object to an old one.

    :param rate: The rate object
    :return: The old rate object
    """
    from autoreact.params import RxnParams

    rxn_params_obj = None
    if type_(rate) == RateType.CONSTANT:
        rxn_params_obj = RxnParams(
            arr_dct={"arr_tuples": [arrhenius_params(rate, lt=False)]}
        )
    elif type_(rate) == RateType.FALLOFF and blend_type(rate) == BlendType.LIND:
        rxn_params_obj = RxnParams(
            lind_dct={
                "highp_arr": [high_p_params(rate, lt=False)],
                "lowp_arr": [low_p_params(rate, lt=False)],
            }
        )
    elif type_(rate) == RateType.FALLOFF and blend_type(rate) == BlendType.TROE:
        rxn_params_obj = RxnParams(
            troe_dct={
                "highp_arr": [high_p_params(rate, lt=False)],
                "lowp_arr": [low_p_params(rate, lt=False)],
                "troe_params": blend_coeffs(rate),
            }
        )
    elif type_(rate) == RateType.PLOG:
        rxn_params_obj = RxnParams(plog_dct=plog_params_dict(rate, lt=False))

    return rxn_params_obj


# Helpers
def chemkin_aux_line(
    key: str,
    val: str | Sequence[str],
    top_width: int = 55,
    key_width: int = 5,
    indent: int = 4,
) -> str:
    """Format a line of auxiliary CHEMKIN reaction data.

    :param key: The key, e.g. 'PLOG'
    :param val: The value(s), e.g. '5.000  8500' or ['5.000', '8500']
    :param top_width: The width of the top line, for alignment purposes
    :param key_width: The key column width, defaults to 5
    :param indent: The indentation, defaults to 4
    :return: The line
    """
    val_width = top_width - indent - key_width - 2
    val = val if isinstance(val, str) else " ".join(val)
    return " " * indent + f"{key:<{key_width}} /{val:>{val_width}}/"


def write_numbers(
    nums: Sequence[float],
    digits: int = 4,
    always_sci: Sequence[bool] | bool = False,
    as_int: bool = False,
) -> str:
    """Write a sequence of numbers to a formatted string.

    :param nums: The numbers
    :param digits: How many digits to include, defaults to 4
    :param always_sci: Whether to always use scientific notation; if given as a list,
        this can be used to set scientific notation for individual numbers
    :param as_int: Whether to write intgeger values
    :return: The formatted number sequence string
    """
    always_sci = (
        [always_sci] * len(nums) if isinstance(always_sci, bool) else always_sci
    )
    assert len(nums) <= len(always_sci), f"Mismatched lengths:\n{nums}\n{always_sci}"
    num_strs = [
        write_number(n, always_sci=a, digits=digits, as_int=as_int)
        for n, a in zip(nums, always_sci, strict=False)
    ]
    return " ".join(num_strs)


def write_number(
    num: float | int, digits: int = 4, always_sci: bool = False, as_int: bool = False
) -> str:
    """Write a number to a formatted string.

    :param num: The number
    :param digits: How many digitst to include, defaults to 4
    :param always_sci: Whether to always use scientific notation
    :param as_int: Whether to write integer values
    :return: The formatted number string
    """
    # Exact width of scientific notation with 2-digit exponent:
    max_width = digits + 6  # from general formula: digits + 4 + |_log(log(num))_|

    if as_int:
        num = int(num)
        return f"{num:>{max_width}d}"

    exp = int(numpy.floor(numpy.log10(numpy.abs(num)))) if num else 0
    float_width = max(exp + 1, digits + 1) if exp > 0 else numpy.abs(exp) + 1 + digits

    if always_sci or float_width > max_width:
        decimals = digits - 1
        return f"{num:>{max_width}.{decimals}E}"

    decimals = max(0, digits - exp - 1)
    return f"{num:>{max_width}.{decimals}f}"


def chemkin_rate_expr() -> pp.ParseExpression:
    """Get the parse expression for chemkin rates."""
    return RATE_EXPR


def number_list_expr(
    nmin: int | None = None, nmax: int | None = None, delim: str = ""
) -> pp.ParseExpression:
    """Get a parse expression for a list of numbers.

    :param nmin: The minimum list length (defaults to `None`)
    :param nmax: The maximum list length (defaults to `nmin` if `None`)
    :param delim: The delimiter between numbers, defaults to ""
    :return: The parse expression
    """
    nmax = nmin if nmax is None else nmax
    return pp.delimitedList(ppc.number, delim=delim, min=nmin, max=nmax)


SLASH = pp.Suppress(pp.Literal("/"))
ARROW = pp.Literal("=") ^ pp.Literal("=>") ^ pp.Literal("<=>")
FALLOFF = pp.Combine(
    pp.Literal("(") + pp.Literal("+") + pp.Word(pp.alphanums) + pp.Literal(")"),
    adjacent=False,
)
ARRH_PARAMS = number_list_expr(3)
AUX_KEYWORD = pp.Word(pp.alphanums)
AUX_PARAMS = SLASH + number_list_expr() + SLASH
AUX_LINE = pp.Group(AUX_KEYWORD + pp.Optional(AUX_PARAMS))
RATE_EXPR = (
    pp.Suppress(...) + ARRH_PARAMS("params") + pp.Group(pp.ZeroOrMore(AUX_LINE))("aux")
)
