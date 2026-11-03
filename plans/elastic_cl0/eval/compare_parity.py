#!/usr/bin/env python3
"""Compare reproduced group records (parity.sh) episode by episode with the reference H100 episodes.csv records.

An episode is trajectory-identical when success, policy_steps and queries all match (the reference CSV holds no
action digests; with 10 executed actions per query, equal policy_steps and queries on a fixed initial state and seed
is the strongest check the published records allow)."""
import argparse, csv, json
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--parity-root", required=True); p.add_argument("--reference", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    report = dict(groups={}, episodes=0, identical=0, same_outcome=0)
    for cond in sorted(Path(a.parity_root).glob("elastic__E0__*")):
        schedule = cond.name.split("__")[2]
        ref = {}
        with open(Path(a.reference) / f"E0__{schedule}" / "episodes.csv") as f:
            for r in csv.DictReader(f):
                ref[(r["suite"], int(r["task"]), int(r["initial_state_index"]), int(r["seed"]))] = r
        for g in sorted((cond / "groups").glob("*.json")):
            rec = json.loads(g.read_text()); rows = []
            for e in rec["episodes"]:
                r = ref[(e["suite"], e["task"], e["initial_state_index"], e["seed"])]
                same = (bool(e["success"]) == (r["success"] == "True"), e["policy_steps"] == int(r["policy_steps"]),
                        e["queries"] == int(r["queries"]))
                rows.append(dict(episode=e["episode"], success=(e["success"], r["success"]),
                                 policy_steps=(e["policy_steps"], int(r["policy_steps"])),
                                 queries=(e["queries"], int(r["queries"])), identical=all(same), same_outcome=same[0],
                                 query_ms=(1e3 * e["query_seconds"] / max(e["queries"], 1),
                                           1e3 * float(r["query_seconds"]) / max(int(r["queries"]), 1))))
            n, ident, outc = len(rows), sum(r["identical"] for r in rows), sum(r["same_outcome"] for r in rows)
            key = f"{schedule}/{rec['group']['id']}"
            ref_successes = sum(r["success"][1] == "True" for r in rows)
            report["groups"][key] = dict(
                episodes=n, identical=ident, same_outcome=outc, successes=rec["successes"],
                reference_successes=ref_successes, backend=rec["backend"],
                checkpoint_sha256=rec["checkpoint_sha256"], elapsed_seconds=rec["elapsed_seconds"],
                mismatches=[r for r in rows if not r["identical"]])
            report["episodes"] += n; report["identical"] += ident; report["same_outcome"] += outc
            print(f"{key}: {ident}/{n} trajectory-identical, {outc}/{n} same outcome, "
                  f"successes {rec['successes']} vs reference {ref_successes}, {rec['elapsed_seconds']:.0f}s")
    report["passed"] = report["episodes"] == 150 and report["identical"] == report["episodes"]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True); Path(a.out).write_text(json.dumps(report, indent=1))
    print(f"PARITY {report['identical']}/{report['episodes']} trajectory-identical, "
          f"{report['same_outcome']}/{report['episodes']} same outcome -> {'PASSED' if report['passed'] else 'NOT EXACT'}")


if __name__ == "__main__":
    main()
