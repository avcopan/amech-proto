"""Definition and core functionality of the mechanism data structure."""

from dataclasses import dataclass

from pandera.typing import DataFrame

from .schema import Reactions, Species


@dataclass
class Mechanism:
    """A chemical kinetic mechanism."""

    rxn_df: DataFrame[Reactions]
    spc_df: DataFrame[Species]
