"""Inference: output below-standard probability and label for new records."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from iecq.config import CHECKPOINT_DIR, DATA_PROCESSED
from iecq.models.framework import AssessmentFramework


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT_DIR / "framework.joblib")
    parser.add_argument("--input", type=Path, default=None, help="CSV with the same fields as train.csv")
    parser.add_argument("--json", type=Path, default=None, help="JSON object or list of objects")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    import pandas as pd

    fw = AssessmentFramework.load(args.checkpoint, device=args.device)
    if args.json is not None:
        payload = json.loads(Path(args.json).read_text(encoding="utf-8"))
        df = pd.DataFrame(payload if isinstance(payload, list) else [payload])
    elif args.input is not None:
        df = pd.read_csv(args.input)
    else:
        df = pd.read_csv(DATA_PROCESSED / "test.csv").head(5)

    pred = fw.infer(df)
    keep = [
        c
        for c in pred.columns
        if c
        in {
            "survey_id",
            "university_type",
            "s4_course_level",
            "y",
            "p_lr",
            "p_cart",
            "p_extended_cart",
            "yhat_logistic",
            "yhat_extended_cart",
            "tau",
        }
        or c in df.columns[:3]
    ]
    # de-duplicate while preserving order
    seen = set()
    cols = []
    for c in keep:
        if c not in seen and c in pred.columns:
            seen.add(c)
            cols.append(c)
    view = pred[cols]
    if args.out is None:
        print(view.to_string(index=False))
    else:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        view.to_csv(args.out, index=False, encoding="utf-8-sig")
        print("wrote", args.out)


if __name__ == "__main__":
    main()
