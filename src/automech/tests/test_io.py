from pathlib import Path

import mecha
import pytest

DATA_DIR = Path(__file__).parents[0] / "data"


@pytest.mark.parametrize(
    "inp,out", [("13epoxycyclopentane_species.txt", "13epoxycyclopentane_species.csv")]
)
def test__rmg__read__species(inp, out):
    inp = DATA_DIR / inp
    out = DATA_DIR / out
    df = mecha.io.rmg.read.species(inp=inp, out=out)
    print(df)
