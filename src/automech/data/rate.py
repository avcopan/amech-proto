"""Kinetics dataclasses.

Eventually, we will want to add a "Raw" rate type with k(T,P) values stored in an xarray
"""

import abc
import dataclasses
import enum
from collections.abc import Sequence
from typing import Any

import numpy


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


def arrhenius_params_string(
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

    :param is_rev: Whether this rate describes a reversible reaction
    :param type_: The type of reaction
    """

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


# constructors
def from_data(
    k: ArrheniusFunctionOrData,
    k0: ArrheniusFunctionOrData | None = None,
    f: tuple[str, Sequence[float]] | BlendingFunction | None = None,
    plog_ks: Sequence[ArrheniusFunctionOrData] | None = None,
    plog_ps: Sequence[float] | None = None,
    type_: str | RateType | None = None,
    is_rev: bool = True,
) -> Rate:
    """Build a rate object from data.

    :param k: The (high-pressure limiting) Arrhenius function for the reaction
    :param k0: The low-pressure limiting Arrhenius function for the reaction
    :param f: Falloff function for blending the high- and low-pressure rate coefficients
    :param plog_ks: P-Log rate coefficients at specific pressures, k_P_, k_P2, ...
    :param plog_ps: P-Log pressures, P_, P2, ... [Pa]
    :param type_: The type of reaction: "Constant", "Falloff", "Activated", "Plog"
    :param is_rev: Is this a reversible reaction?
    :return: _description_
    """
    type_ = None if type_ is None else RateType(type_)

    if plog_ks is not None or plog_ps is not None or type_ == RateType.PLOG:
        return PlogRate(ks=plog_ks, ps=plog_ps, k=k, is_rev=is_rev, type_=type_)

    return SimpleRate(k=k, k0=k0, f=f, is_rev=is_rev, type_=type_)


# getters
def arrhenius_function(rate: Rate) -> ArrheniusFunction:
    """Get the primary Arrhenius function for the reaction.

    (The one that appears after the reaction equation in a CHEMKIN file)

    :param rate: The rate object
    :return: The Arrhenius function
    """
    if type_(rate) == RateType.ACTIVATED:
        return low_p_arrhenius_function(rate)

    return high_p_arrhenius_function(rate)


def high_p_arrhenius_function(rate: Rate) -> ArrheniusFunction:
    """Get the high-pressure limiting Arrhenius function for the reaction.

    :param rate: The rate object
    :return: The Arrhenius function
    """
    if not isinstance(rate, SimpleRate | PlogRate):
        return None

    return rate.k


def low_p_arrhenius_function(rate: Rate) -> ArrheniusFunction:
    """Get the low-pressure limiting Arrhenius function for the reaction.

    :param rate: The rate object
    :return: The Arrhenius function
    """
    if not isinstance(rate, SimpleRate):
        return None

    return rate.k0


def blend_function(rate: Rate) -> BlendingFunction:
    """Get the function for blending high- and low-pressure rates.

    :param rate: The rate object
    :return: The blend function
    """
    if not isinstance(rate, SimpleRate):
        return None

    return rate.f


def plog_arrhenius_functions(rate: Rate) -> tuple[ArrheniusFunction, ...]:
    """Arrhenius functions for a P-Log reaction rate.

    :param rate: The rate object
    :return: The Arrhenius functions
    """
    if not isinstance(rate, PlogRate):
        return None

    return rate.ks


def plog_pressures(rate: Rate) -> tuple[float, ...]:
    """Pressures for a P-Log reaction rate.

    :param rate: The rate object
    :return: The pressures
    """
    if not isinstance(rate, PlogRate):
        return None

    return rate.ps


# properties
def high_p_params(rate: Rate, lt: bool = True) -> tuple[float, ...]:
    """Get the high-pressure limiting Arrhenius function for the reaction.

    :param rate: The rate object
    :param lt: Include Landau-Teller parameters* along with basic Arrhenius parameters?
    :return: The Arrhenius parameters A, b, E, (B*, C*)
    """
    return arrhenius_params(high_p_arrhenius_function(rate), lt=lt)


def low_p_params(rate: Rate, lt: bool = True) -> tuple[float, ...]:
    """Get the low-pressure limiting Arrhenius parameters for the reaction.

    :param rate: The rate object
    :param lt: Include Landau-Teller parameters* along with basic Arrhenius parameters?
    :return: The Arrhenius parameters A, b, E, (B*, C*)
    """
    if not isinstance(rate, SimpleRate):
        return None

    return arrhenius_params(low_p_arrhenius_function(rate), lt=lt)


def blend_type(rate: Rate) -> BlendType:
    """Get the function type for blending high- and low-pressure rates.

    :param rate: The rate object
    :return: The blend type
    """
    if not isinstance(rate, SimpleRate):
        return None

    return f_type(blend_function(rate))


def blend_coeffs(rate: Rate) -> BlendType:
    """Get the coefficients for blending high- and low-pressure rates.

    :param rate: The rate object
    :return: The blend coefficients
    """
    if not isinstance(rate, SimpleRate):
        return None

    return f_coeffs(blend_function(rate))


def plog_params(rate: Rate, lt: bool = True) -> tuple[ArrheniusFunction, ...]:
    """Arrhenius functions for a P-Log reaction rate.

    :param rate: The rate object
    :param lt: Include Landau-Teller parameters* along with basic Arrhenius parameters?
    :return: The Arrhenius parameters A, b, E, (B*, C*)
    """
    if not isinstance(rate, PlogRate):
        return None

    return tuple(arrhenius_params(k, lt=lt) for k in plog_arrhenius_functions(rate))


def plog_params_dict(rate: Rate, lt: bool = True) -> dict[float, ArrheniusFunction]:
    """Get the P-Log Arrhenius parameters, as a dictionary by pressure.

    :param rate: The rate object
    :param lt: Include Landau-Teller parameters* along with basic Arrhenius parameters?
    :return: The Arrhenius parameters A, b, E, (B*, C*)
    """
    if not isinstance(rate, PlogRate):
        return None

    return dict(zip(plog_pressures(rate), plog_params(rate, lt=lt), strict=True))


# legacy
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


# draft
@dataclasses.dataclass
class ChebRate(Rate):
    """Chebyshev reaction rate, k(T,P) parametrization (see cantera.ReactionRate).

    :param T0: The minimum temperature [K] for the Chebyshev fit
    :param T_: The maximum temperature [K] for the Chebyshev fit
    :param P0: The minimum pressure [K] for the Chebyshev fit
    :param P_: The maximum pressure [K] for the Chebyshev fit
    :param coeffs: The Chebyshev expansion coefficients
    :param is_rev: Is this a reversible reaction?
    """

    T0: float
    T_: float
    P0: float
    P_: float
    coeffs: tuple[tuple[float, ...], ...]
    is_rev: bool = True
    type_: str = RateType.CHEB

    def __post_init__(self):
        """Initialize attributes."""
        self.type_ = RateType(self.type_)
        assert self.type_ == RateType.CHEB


# Helpers
def write_numbers(
    nums: Sequence[float], digits: int = 4, always_sci: Sequence[bool] | bool = False
) -> str:
    """Write a sequence of numbers to a formatted string.

    :param nums: The numbers
    :param digits: How many digits to include, defaults to 4
    :param always_sci: Whether to always use scientific notation; if given as a list,
        this can be used to set scientific notation for individual numbers
    :return: The formatted number sequence string
    """
    always_sci = (
        [always_sci] * len(nums) if isinstance(always_sci, bool) else always_sci
    )
    assert len(nums) <= len(always_sci), f"Mismatched lengths:\n{nums}\n{always_sci}"
    num_strs = [
        write_number(n, always_sci=a, digits=digits)
        for n, a in zip(nums, always_sci, strict=False)
    ]
    return " ".join(num_strs)


def write_number(num: float | int, digits: int = 4, always_sci: bool = False) -> str:
    """Write a number to a formatted string.

    :param num: The number
    :param digits: How many digitst to include, defaults to 4
    :param always_sci: Whether to always use scientific notation
    :return: The formatted number string
    """
    # Exact width of scientific notation with 2-digit exponent:
    max_width = digits + 6  # from general formula: digits + 4 + |_log(log(num))_|

    exp = int(numpy.floor(numpy.log10(numpy.abs(num))))  # scientific notation exponent
    float_width = max(exp + 1, digits + 1) if exp > 0 else numpy.abs(exp) + 1 + digits

    if always_sci or float_width > max_width:
        decimals = digits - 1
        return f"{num:>{max_width}.{decimals}E}"

    decimals = max(0, digits - exp - 1)
    return f"{num:>{max_width}.{decimals}f}"
