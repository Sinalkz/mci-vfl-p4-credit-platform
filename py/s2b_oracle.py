"""Oracle CLI for S2b: exact F for a bottom model, a head, a mask and a list of rows.

stdin : {"fA": {...}, "head": {...}, "mask": m, "rows": [{"y": .., "xa": [..], "hp": [[..],[..],[..]]}, ...]}
stdout: {"F": .., "preds": [..], "TP": .., "FP": .., "FN": .., "emptyClasses": .., "macroF1": ".."}
"""
from __future__ import annotations

import json
import sys

from s2b_model import evaluate


def main() -> None:
    req = json.load(sys.stdin)
    json.dump(evaluate(req["fA"], req["head"], req["mask"], req["rows"]), sys.stdout)


if __name__ == "__main__":
    main()
