"""Compare per-video J/F of the analyzer (author protocol, post-SAM) with eval.py's sequence_metrics.csv."""
import argparse
import sys

import pandas as pd


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--eval_csv", required=True)
    ap.add_argument("--frame_table", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--tol", type=float, default=0.001)
    args = ap.parse_args(argv)
    ev = pd.read_csv(args.eval_csv, dtype={"Sequence": str}).set_index("Sequence")
    ft = pd.read_csv(args.frame_table, dtype={"dataset": str, "video": str})
    ours = ft[ft["dataset"] == args.dataset].groupby("video")[["J_post_0.1", "F_post_0.1"]].mean()
    rows, ok_all = [], True
    for video, e in ev.iterrows():
        if video not in ours.index:
            rows.append({"video": video, "ok": False, "note": "missing in frame_table"})
            ok_all = False
            continue
        dj = ours.loc[video, "J_post_0.1"] - e["Mean_IoU"]
        df_ = ours.loc[video, "F_post_0.1"] - e["Mean_F_Score"]
        ok = abs(dj) < args.tol and abs(df_) < args.tol
        ok_all &= ok
        rows.append({"video": video, "J_eval": e["Mean_IoU"], "J_ours": ours.loc[video, "J_post_0.1"], "dJ": dj,
                     "F_eval": e["Mean_F_Score"], "F_ours": ours.loc[video, "F_post_0.1"], "dF": df_, "ok": ok})
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda x: f"{x:.5f}"))
    print("PASS" if ok_all else f"FAIL (tol={args.tol})")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
