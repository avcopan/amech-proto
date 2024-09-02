"""Kinetics dataclasses.

Eventually, we will want to add a "Raw" rate type with k(T,P) values stored in an xarray
"""

import abc
import dataclasses
import enum
from collections.abc import Sequence
from typing import Any

Params = Sequence[float]


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
    :param B: The Landau-Teller B-factor
    :param C: The Landau-Teller C-factor
    """

    A: float = 1.0
    b: float = 0.0
    E: float = 0.0
    B: float = 0.0
    C: float = 0.0


def arrhenius_params(k: ArrheniusFunction, lt: bool = True) -> tuple[float, ...]:
    """Get the parameters for an Arrhenius or Landau-Teller function.

    :param k: The Arrhenius function object
    :param lt: Include Landau-Teller parameters* along with basic Arrhenius parameters?
    :return: The parameters A, b, E, (B*, C*)
    """
    return (k.A, k.b, k.E, k.B, k.C) if lt else (k.A, k.b, k.E)


@dataclasses.dataclass
class BlendingFunction:
    """A blending function for high and low-pressure rates (see cantera.Falloff).

    Types:
        Lind   - coeffs: (None)
        Troe   - coeffs: a, T, T***, T*, (T**)

    :param coeffs: A list of coefficients for the parametrization
    :param type_: The type of parametrization: "Lind", "Troe"
    """

    coeffs: tuple[float, ...] | None = None
    type_: BlendType = BlendType.LIND

    def __post_init__(self):
        """Initialize attributes."""
        self.type_ = BlendType(self.type_)


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
    def is_rev(self):
        """Whether this rate describes a reversible reaction."""
        pass

    @property
    @abc.abstractmethod
    def type_(self):
        """The type of reaction."""
        pass


def is_reversible(rate: Rate) -> bool:
    """Whether this rate describes a reversible reaction.

    :param rate: The rate object
    :return: `True` if it does, `False` if it doesn't
    """
    return rate.is_rev


def type_(rate: Rate) -> RateType:
    """Get the type of reaction.

    :param rate: The rate object
    :return: The type of reaction
    """
    return rate.type_


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
        self.type_ = RateType(self.type_)
        assert self.type_ in (RateType.CONSTANT, RateType.FALLOFF, RateType.ACTIVATED)

        if self.type_ != RateType.CONSTANT:
            self.f = BlendingFunction() if self.f is None else self.f


def arrhenius_function(rate: SimpleRate) -> ArrheniusFunction:
    """Get the (high-pressure limiting) Arrhenius function for the reaction.

    :param rate: The rate object
    :return: The Arrhenius function
    """
    return rate.k


def params(rate: SimpleRate, lt: bool = True) -> tuple[float, ...]:
    """Get the (high-pressure limiting) Arrhenius parameters for the reaction.

    :param rate: The rate object
    :param lt: Include Landau-Teller parameters* along with basic Arrhenius parameters?
    :return: The Arrhenius parameters A, b, E, (B*, C*)
    """
    return arrhenius_params(arrhenius_function(rate), lt=lt)


def low_p_arrhenius_function(rate: SimpleRate) -> ArrheniusFunction:
    """Get the low-pressure limiting Arrhenius function for the reaction.

    :param rate: The rate object
    :return: The Arrhenius function
    """
    return rate.k0


def low_p_params(rate: SimpleRate, lt: bool = True) -> tuple[float, ...]:
    """Get the low-pressure limiting Arrhenius parameters for the reaction.

    :param rate: The rate object
    :param lt: Include Landau-Teller parameters* along with basic Arrhenius parameters?
    :return: The Arrhenius parameters A, b, E, (B*, C*)
    """
    return arrhenius_params(low_p_arrhenius_function(rate), lt=lt)


def blend_function(rate: SimpleRate) -> BlendingFunction:
    """Get the function for blending high- and low-pressure rates.

    :param rate: The rate object
    :return: The blend function
    """
    return rate.f


def blend_type(rate: SimpleRate) -> BlendType:
    """Get the function type for blending high- and low-pressure rates.

    :param rate: The rate object
    :return: The blend type
    """
    return f_type(blend_function(rate))


def blend_coeffs(rate: SimpleRate) -> BlendType:
    """Get the coefficients for blending high- and low-pressure rates.

    :param rate: The rate object
    :return: The blend coefficients
    """
    return f_coeffs(blend_function(rate))


@dataclasses.dataclass
class PlogRate(Rate):
    """P-Log reaction rate, k(T,P) parametrization (see cantera.ReactionRate).

    :param ks: Rate coefficients at specific pressures, k_P_, k_P2, ...
    :param Ps: An array of pressures, P_, P2, ... [Pa]
    :param k: Optional high-pressure rate
    :param is_rev: Is this a reversible reaction?
    """

    ks: tuple[ArrheniusFunction, ...]
    Ps: tuple[float, ...]
    k: ArrheniusFunction | None = None
    is_rev: bool = True
    type_: RateType = RateType.PLOG

    def __post_init__(self):
        """Initialize attributes."""
        self.type_ = RateType(self.type_)
        assert self.type_ == RateType.PLOG


def plog_arrhenius_functions(rate: PlogRate) -> tuple[ArrheniusFunction, ...]:
    """Arrhenius functions for a P-Log reaction rate.

    :param rate: The rate object
    :return: The Arrhenius functions
    """
    return rate.ks


def plog_params(rate: PlogRate, lt: bool = True) -> tuple[ArrheniusFunction, ...]:
    """Arrhenius functions for a P-Log reaction rate.

    :param rate: The rate object
    :param lt: Include Landau-Teller parameters* along with basic Arrhenius parameters?
    :return: The Arrhenius parameters A, b, E, (B*, C*)
    """
    return tuple(arrhenius_params(k, lt=lt) for k in plog_arrhenius_functions(rate))


def plog_pressures(rate: PlogRate) -> tuple[float, ...]:
    """Pressures for a P-Log reaction rate.

    :param rate: The rate object
    :return: The pressures
    """
    return rate.Ps


def plog_params_dict(rate: PlogRate, lt: bool = True) -> dict[float, ArrheniusFunction]:
    """Get the P-Log Arrhenius parameters, as a dictionary by pressure.

    :param rate: The rate object
    :param lt: Include Landau-Teller parameters* along with basic Arrhenius parameters?
    :return: The Arrhenius parameters A, b, E, (B*, C*)
    """
    return dict(zip(plog_pressures(rate), plog_params(rate, lt=lt), strict=True))


def high_p_arrhenius_function(rate: SimpleRate | PlogRate) -> ArrheniusFunction:
    """Get the high-pressure limiting Arrhenius function for the reaction.

    :param rate: The rate object
    :return: The Arrhenius function
    """
    return rate.k


def high_p_params(rate: SimpleRate | PlogRate, lt: bool = True) -> tuple[float, ...]:
    """Get the high-pressure limiting Arrhenius function for the reaction.

    :param rate: The rate object
    :param lt: Include Landau-Teller parameters* along with basic Arrhenius parameters?
    :return: The Arrhenius parameters A, b, E, (B*, C*)
    """
    return arrhenius_params(high_p_arrhenius_function(rate), lt=lt)


def to_old_object(rate: Rate) -> Any:
    """Convert a new rate object to an old one.

    :param rate: The rate object
    :return: The old rate object
    """
    from autoreact.params import RxnParams

    rxn_params_obj = None
    if type_(rate) == RateType.CONSTANT:
        rxn_params_obj = RxnParams(arr_dct={"arr_tuples": [params(rate, lt=False)]})
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
