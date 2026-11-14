#!/usr/bin/env python3
"""Independent check of v0_ka_sweep/suite_breakdown_20261114 (v0 4/4 at 1 and 2 action loops, by suite, seed and task).

Shares no code with the folder's analyze_ka12.py / make_report.py. It
  1. checks every member of source/ka12_groups.tar.gz against the workstation's sha256sum listing;
  2. links the copy to the existing report (queue_status.json, run summaries, v0_ka_results.csv);
  3. recounts successes per suite, seed and task from the per-episode records and compares them with breakdown.json,
     the CSV files, every table of the folder's README.md, and the tables added to v0_ka_sweep/report.md, all_results.md
     and the top-level README.md;
  4. compares the rows quoted for 3, 4, 5 and 6 action loops with their own records, read from other files than the
     ones the page generator used (matched_baselines/README.md, all_results.md, final/results.json suite/seed lists).
Nothing is extracted or re-evaluated. Writes v0_ka_sweep/suite_breakdown_20261114/independent_check.json.

    python3 evaluation_report/verify_v0_ka_breakdown_20261114.py
"""
import csv, hashlib, io, json, re, statistics, sys, tarfile
from collections import Counter
from pathlib import Path

ER = Path(__file__).resolve().parent
F = ER / "v0_ka_sweep/suite_breakdown_20261114"
ORDER = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]
LABEL = {"libero_spatial": "Spatial", "libero_object": "Object", "libero_goal": "Goal", "libero_10": "Long"}
SHA = "1c659a7c2f6970c3ea5972a8fba4abd9db35857c83a67a053c0ee35555fb602b"
checks = []


def check(name, ok, detail=""):
    checks.append(dict(name=name, ok=bool(ok), detail=str(detail)[:300]))


def pct(k, n):
    return 100.0 * k / n


def close(a, b, tol=0.0051):
    return abs(float(a) - float(b)) <= tol


# ---- 1. archive against the workstation listing
want = {}
for line in (F / "source/workstation_sha256.txt").read_text().splitlines():
    digest, path = line.split(None, 1); want[path.strip()] = digest
blobs = {}
with tarfile.open(F / "source/ka12_groups.tar.gz") as tar:
    for member in tar:
        if member.isreg(): blobs[member.name] = tar.extractfile(member).read()
check("archive members equal the workstation listing", set(blobs) == set(want), f"{len(blobs)} files, {len(want)} listed")
check("every archive member has the workstation's sha256", all(hashlib.sha256(b).hexdigest() == want.get(p) for p, b in blobs.items()))
B = json.loads((F / "breakdown.json").read_text())
check("breakdown.json names this archive", B["archive_sha256"] == hashlib.sha256((F / "source/ka12_groups.tar.gz").read_bytes()).hexdigest() and B["archive_files"] == len(blobs))

# ---- 2. link to the existing report
check("workstation queue_state.json is byte-identical to v0_ka_sweep/queue_status.json", blobs["queue_state.json"] == (ER / "v0_ka_sweep/queue_status.json").read_bytes())
queue = json.loads((ER / "v0_ka_sweep/queue_status.json").read_text())
jobs = {j["ka"]: j for j in queue["jobs"]}
check("the queue record has exactly the two completed runs", sorted(jobs) == [1, 2] and all(j["status"] == "complete" for j in jobs.values()))
with open(ER / "all_results/v0_ka_results.csv") as f:
    old_csv = {int(r["action_loops"]): r for r in csv.DictReader(f)}

# ---- 3. recount
count = {}      # (ka) -> dict of Counters
records = {}
for ka in (1, 2):
    eps = []
    for path, blob in sorted(blobs.items()):
        if path.startswith(f"results/ka{ka}/groups/"): eps.extend(json.loads(blob))
    records[ka] = eps
    ok_id = all(e["checkpoint_sha256"] == SHA and e["action_loops"] == ka and e["video_loops"] == 4 and e["nfe"] == 10 and e["success"] in (True, False) for e in eps)
    cells = Counter((e["suite"], e["seed"], e["task"]) for e in eps)
    check(f"Ka{ka}: 6,000 episode records of this checkpoint at action loops {ka}, video loops 4, NFE 10", len(eps) == 6000 and ok_id)
    check(f"Ka{ka}: grid of 4 suites x 3 seeds x 10 tasks with 50 episodes each", len(cells) == 120 and set(cells.values()) == {50}
          and {c[0] for c in cells} == set(ORDER) and {c[1] for c in cells} == {42, 43, 44} and {c[2] for c in cells} == set(range(10)))
    check(f"Ka{ka}: each task/seed group covers initial states 0..49 once",
          all(sorted(e["initial_state_index"] for e in eps if (e["suite"], e["seed"], e["task"]) == c) == list(range(50)) for c in cells))
    states = {}
    for e in eps: states.setdefault((e["suite"], e["task"], e["initial_state_index"]), set()).add(e["initial_state_sha256"])
    check(f"Ka{ka}: the three seeds use the same initial states", all(len(v) == 1 for v in states.values()) and len(states) == 2000)
    c = dict(total=sum(e["success"] for e in eps),
             suite=Counter(), seed=Counter(), suite_seed=Counter(), task=Counter(), task_seed=Counter())
    for e in eps:
        if e["success"]:
            c["suite"][e["suite"]] += 1; c["seed"][e["seed"]] += 1; c["suite_seed"][(e["suite"], e["seed"])] += 1
            c["task"][(e["suite"], e["task"])] += 1; c["task_seed"][(e["suite"], e["task"], e["seed"])] += 1
    count[ka] = c
    summary = json.loads(blobs[f"results/ka{ka}/summary.json"])
    check(f"Ka{ka}: recount equals the run's summary.json", (summary["successes"], summary["episodes"], summary["action_loops"]) == (c["total"], 6000, ka), c["total"])
    check(f"Ka{ka}: run summary.json equals the summary in the report's queue_status.json", summary == jobs[ka]["summary"])
    check(f"Ka{ka}: recount equals all_results/v0_ka_results.csv", int(old_csv[ka]["successes"]) == c["total"] and int(old_csv[ka]["episodes"]) == 6000
          and old_csv[ka]["checkpoint_sha256"] == SHA)
    # breakdown.json
    r = B["runs"][str(ka)]
    ok = r["successes"] == c["total"] and r["episodes"] == 6000 and close(r["success_rate"], pct(c["total"], 6000), 1e-3) and r["passed"] and not r["issues"]
    for s in ORDER:
        x = r["suites"][s]
        ok &= x["successes"] == c["suite"][s] and x["episodes"] == 1500 and close(x["rate"], pct(c["suite"][s], 1500), 1e-3)
        ok &= all(x["seeds"][str(k)]["successes"] == c["suite_seed"][(s, k)] for k in (42, 43, 44))
        ok &= close(x["seed_sample_std"], statistics.stdev(pct(c["suite_seed"][(s, k)], 500) for k in (42, 43, 44)), 1e-3)
        ok &= all(x["tasks"][str(t)]["successes"] == c["task"][(s, t)] and x["tasks"][str(t)]["episodes"] == 150 for t in range(10))
        ok &= all(x["tasks"][str(t)]["seeds"][str(k)] == c["task_seed"][(s, t, k)] for t in range(10) for k in (42, 43, 44))
    ok &= all(r["seeds"][str(k)]["successes"] == c["seed"][k] for k in (42, 43, 44))
    check(f"Ka{ka}: breakdown.json equals the recount (suites, seeds, tasks)", ok)
    # episodes csv
    with open(F / f"episodes_ka{ka}.csv") as f: rows = list(csv.DictReader(f))
    a = sorted((x["suite"], int(x["task"]), int(x["seed"]), int(x["episode"]), int(x["initial_state_index"]), int(x["success"]), int(x["policy_steps"]), int(x["queries"])) for x in rows)
    b = sorted((e["suite"], e["task"], e["seed"], e["episode"], e["initial_state_index"], int(e["success"]), e["policy_steps"], e["queries"]) for e in eps)
    check(f"Ka{ka}: episodes_ka{ka}.csv equals the 6,000 records", a == b and all(int(x["action_loops"]) == ka for x in rows))

with open(F / "suite_results.csv") as f: rows = list(csv.DictReader(f))
ok = len(rows) == 10
for x in rows:
    ka = int(x["action_loops"]); c = count[ka]
    if x["suite"] == "all":
        ok &= int(x["successes"]) == c["total"] and int(x["episodes"]) == 6000 and all(close(x[f"seed{k}_sr"], pct(c["seed"][k], 2000), 1e-3) for k in (42, 43, 44))
    else:
        ok &= int(x["successes"]) == c["suite"][x["suite"]] and int(x["episodes"]) == 1500 and close(x["sr_percent"], pct(c["suite"][x["suite"]], 1500), 1e-3)
        ok &= all(close(x[f"seed{k}_sr"], pct(c["suite_seed"][(x["suite"], k)], 500), 1e-3) for k in (42, 43, 44))
check("suite_results.csv equals the recount", ok)
with open(F / "per_task.csv") as f: rows = list(csv.DictReader(f))
desc = {(e["suite"], e["task"]): e["description"] for e in records[1]}
check("per_task.csv equals the recount", len(rows) == 80 and all(
    int(x["successes"]) == count[int(x["action_loops"])]["task"][(x["suite"], int(x["task"]))] and int(x["episodes"]) == 150
    and x["description"] == desc[(x["suite"], int(x["task"]))]
    and all(int(x[f"seed{k}"]) == count[int(x["action_loops"])]["task_seed"][(x["suite"], int(x["task"]), k)] for k in (42, 43, 44)) for x in rows))

# ---- README tables of the folder
page = (F / "README.md").read_text().split("\n")


def section(title):
    i = page.index(title); j = next((k for k in range(i + 1, len(page)) if page[k].startswith("## ")), len(page))
    return page[i + 1:j]


def cells_of(line):
    return [x.strip() for x in line.strip().strip("|").split("|")]


rows = [cells_of(l) for l in section("## Result by suite") if re.match(r"\| [12] \|", l)]
ok = len(rows) == 2
for x in rows:
    ka = int(x[0]); c = count[ka]
    for s, cell in zip(ORDER, x[1:5]):
        m = re.fullmatch(r"([\d.]+)% \((\d+)/(\d+)\)", cell)
        ok &= bool(m) and int(m[2]) == c["suite"][s] and int(m[3]) == 1500 and close(m[1], pct(c["suite"][s], 1500))
    m = re.fullmatch(r"\*\*([\d.]+)%\*\* \((\d+)/(\d+)\)", x[5])
    ok &= bool(m) and int(m[2]) == c["total"] and int(m[3]) == 6000 and close(m[1], pct(c["total"], 6000))
check("README: 'Result by suite' table equals the recount", ok, rows)

sec = section("## Result by seed"); rows = [cells_of(l) for l in sec if re.match(r"\| [12] \|", l)]
ok = len(rows) == 10
for x in rows:
    ka = int(x[0]); c = count[ka]; name = x[1].strip("*")
    if name == "Overall":
        vals = [pct(c["seed"][k], 2000) for k in (42, 43, 44)]; mean = pct(c["total"], 6000)
    else:
        s = next(k for k, v in LABEL.items() if v == name); vals = [pct(c["suite_seed"][(s, k)], 500) for k in (42, 43, 44)]; mean = pct(c["suite"][s], 1500)
    ok &= all(close(a.strip("*"), b) for a, b in zip(x[2:5], vals))
    m = re.fullmatch(r"([\d.]+) ± ([\d.]+)", x[5].strip("*"))
    ok &= bool(m) and close(m[1], mean) and close(m[2], statistics.stdev(vals))
check("README: 'Result by seed' table equals the recount", ok)
claim = [l for l in sec if l.startswith("Seed 42 is the highest")]
top = all(count[ka]["suite_seed"][(s, 42)] > max(count[ka]["suite_seed"][(s, 43)], count[ka]["suite_seed"][(s, 44)])
          and count[ka]["suite_seed"][(s, 44)] <= count[ka]["suite_seed"][(s, 43)] for ka in (1, 2) for s in ORDER)
nums = re.findall(r"([\d.]+) / ([\d.]+) / ([\d.]+)%", claim[0]) if claim else []
check("README: the seed-order sentence holds in the recount", len(claim) == 1 and top and len(nums) == 2
      and all(close(a, pct(count[ka]["seed"][k], 2000)) for ka, trio in zip((1, 2), nums) for a, k in zip(trio, (42, 43, 44))))

sec = section("## Result by task"); ok = True; seen = 0; cur = None
for l in sec:
    if l.startswith("### "): cur = next(k for k, v in LABEL.items() if v == l[4:].strip()); continue
    m = re.fullmatch(r"\| (\d) \| (.+) \| (\d+) \(([\d.]+)%\) \| (\d+) \(([\d.]+)%\) \|", l)
    if m:
        t = int(m[1]); seen += 1
        ok &= m[2] == desc[(cur, t)] and int(m[3]) == count[1]["task"][(cur, t)] and int(m[5]) == count[2]["task"][(cur, t)]
        ok &= close(m[4], pct(int(m[3]), 150), 0.051) and close(m[6], pct(int(m[5]), 150), 0.051)
    m = re.fullmatch(r"\| \| \*\*(\w+) total\*\* \| \*\*(\d+)/1500\*\* \| \*\*(\d+)/1500\*\* \|", l)
    if m: ok &= int(m[2]) == count[1]["suite"][cur] and int(m[3]) == count[2]["suite"][cur] and m[1] == LABEL[cur]
check("README: the four task tables equal the recount (40 tasks x 2 runs)", ok and seen == 40, seen)
zl = [l for l in sec if l.startswith("Tasks with no success")]
zero = {ka: [sum(count[ka]["task"][(s, t)] == 0 for t in range(10)) for s in ORDER] for ka in (1, 2)}
got = [int(x) for x in re.findall(r"(\d+)(?: of 10)? in (?:Spatial|Object|Goal|Long)", zl[0])] if zl else []
check("README: the count of tasks with no success equals the recount", got == zero[1] + zero[2], f"{got} vs {zero}")

# ---- 4. the six action-loop counts side by side, each against its own record
ref = {}
for ka in (1, 2):
    c = count[ka]
    ref[ka] = dict(suites=[pct(c["suite"][s], 1500) for s in ORDER], successes=c["total"], rate=pct(c["total"], 6000), seeds=[pct(c["seed"][k], 2000) for k in (42, 43, 44)],
                   suite_successes=[c["suite"][s] for s in ORDER])
final = json.loads((ER / "final/results.json").read_text())
fs = {x["suite"]: x for x in final["suite_results"] if x["model"] == "LoopWAM full 4/4"}
fseed = Counter()
for x in final["seed_results"]:
    if x["model"] == "LoopWAM full 4/4": fseed[x["seed"]] += x["successes"]
ref[4] = dict(suites=[pct(fs[s]["successes"], fs[s]["episodes"]) for s in ORDER], successes=sum(fs[s]["successes"] for s in ORDER),
              rate=pct(sum(fs[s]["successes"] for s in ORDER), 6000), seeds=[pct(fseed[k], 2000) for k in (42, 43, 44)],
              suite_successes=[fs[s]["successes"] for s in ORDER])
mb = (ER / "matched_baselines/README.md").read_text().split("\n"); ar = (ER / "all_results.md").read_text().split("\n")
for ka in (3, 5, 6):
    row = [cells_of(l) for l in mb if l.startswith(f"| LoopWAM v0 Ka{ka} |") and "±" in l]
    tot = [cells_of(l) for l in ar if l.startswith(f"| LoopWAM v0 Ka{ka} |")]
    assert len(row) == 1 and len(tot) == 1
    x = row[0]; succ = int(tot[0][6].split("/")[0])
    ref[ka] = dict(suites=[float(v.split("±")[0]) for v in x[1:5]], successes=succ, rate=float(x[5].strip("*")), seeds=[float(v) for v in x[6:9]], suite_successes=None)
    check(f"Ka{ka}: matched_baselines/README.md and all_results.md agree on the total", close(pct(succ, 6000), ref[ka]["rate"]) and tot[0][6].endswith("/6000"))
align = {1: "4", 2: "3,4", 3: "2,3,4", 4: "1,2,3,4", 5: "1,1,2,3,4", 6: "1,1,1,2,3,4"}
setup = (ER / "v0_ka_sweep/setup_report.md").read_text()
check("the video-loop alignment column equals the setup report's rule", "Ka1=[4], Ka2=[3,4], Ka3=[2,3,4], Ka4=[1,2,3,4], Ka5=[1,1,2,3,4], Ka6=[1,1,1,2,3,4]" in setup)
rows = [cells_of(l) for l in section("## All action-loop counts of this checkpoint in one view") if re.match(r"\| [1-6] \| [\d,]+ \|", l)]
ok = [int(x[0]) for x in rows] == [1, 2, 3, 4, 5, 6]
for x in rows:
    ka = int(x[0]); r = ref[ka]
    ok &= x[1] == align[ka] and all(close(a.rstrip("%"), b) for a, b in zip(x[2:6], r["suites"])) and close(x[6].strip("*").rstrip("%"), r["rate"])
    ok &= x[7] == f"{r['successes']}/6000" and all(close(a, b) for a, b in zip(x[8].split(" / "), r["seeds"]))
check("README: the six-row table equals each row's own record", ok)
with open(F / "loop_sweep_all.csv") as f: rows = list(csv.DictReader(f))
ok = [int(x["action_loops"]) for x in rows] == [1, 2, 3, 4, 5, 6]
for x in rows:
    ka = int(x["action_loops"]); r = ref[ka]; got = [int(x[f"{LABEL[s].lower()}_successes"]) for s in ORDER]
    ok &= int(x["successes"]) == r["successes"] == sum(got) and int(x["episodes"]) == 6000 and close(x["overall_sr"], r["rate"])
    ok &= all(close(x[f"{LABEL[s].lower()}_sr"], v) for s, v in zip(ORDER, r["suites"])) and all(close(x[f"seed{k}_sr"], v) for k, v in zip((42, 43, 44), r["seeds"]))
    ok &= all(close(pct(g, 1500), v) for g, v in zip(got, r["suites"])) and (r["suite_successes"] is None or got == r["suite_successes"])
    ok &= x["video_loop_read_by_each_action_loop"].replace(" ", ",") == align[ka]
check("loop_sweep_all.csv equals each row's own record", ok)
text = "\n".join(section("## All action-loop counts of this checkpoint in one view"))
m = re.search(r"Overall success by action loops 1 → 6: ([\d. /]+)%", text); n = re.search(r"Long is the lowest suite at every loop count: ([\d. /]+)%", text)
check("README: the two summary lines under the six-row table equal the records",
      bool(m and n) and all(close(a, ref[k]["rate"]) for a, k in zip(m[1].split(" / "), range(1, 7)))
      and all(close(a, ref[k]["suites"][3]) for a, k in zip(n[1].split(" / "), range(1, 7))) and all(min(ref[k]["suites"]) == ref[k]["suites"][3] for k in range(1, 7)))
d = re.search(r"5 loops differs by ([-+\d.]+) points overall, 3 loops by ([-+\d.]+), 6 loops by ([-+\d.]+), 2 loops by ([-+\d.]+) and 1 loop by ([-+\d.]+)\.", text)
check("README: the differences from the trained 4 loops equal the records", bool(d) and all(close(d[i], ref[k]["rate"] - ref[4]["rate"], 0.0101) for i, k in zip(range(1, 6), (5, 3, 6, 2, 1))))

# ---- tables added to the shared report files
def ka_rows(path, start, n):
    L = path.read_text().split("\n"); i = L.index(start)
    return [cells_of(l) for l in L[i:i + 14] if re.match(r"\| [12] \|", l)][:n]

rows = ka_rows(ER / "v0_ka_sweep/report.md", "## Per-suite breakdown (added 2026-11-14)", 2)
check("v0_ka_sweep/report.md: added table equals the recount", len(rows) == 2 and all(
    all(close(a.rstrip("%"), b) for a, b in zip(x[1:5], ref[int(x[0])]["suites"])) and close(x[5].rstrip("%"), ref[int(x[0])]["rate"])
    and x[6] == f"{ref[int(x[0])]['successes']}/6000" for x in rows))
rows = ka_rows(ER / "all_results.md", "## Original local v0 action-depth queue", 2)
check("all_results.md: action-depth table equals the recount", len(rows) == 2 and all(
    all(close(a.rstrip("%"), b) for a, b in zip(x[1:5], ref[int(x[0])]["suites"])) and x[5] == f"{ref[int(x[0])]['successes']}/6000"
    and close(x[6].rstrip("%"), ref[int(x[0])]["rate"]) for x in rows))
top_readme = (ER / "README.md").read_text()
bullet = [l for l in top_readme.split("\n") if l.startswith("- Added in the third 2026-11-14 sync:")]
ok = len(bullet) == 1
if ok:
    for ka, word in ((1, "1 loop"), (2, "2 loops")):
        m = re.search(word + r": Spatial ([\d.]+) / Object ([\d.]+) / Goal ([\d.]+) / Long ([\d.]+)%, overall \*\*([\d.]+)%\*\* \(([\d,]+)/6,000\)", bullet[0])
        ok &= bool(m) and all(close(m[i + 1], ref[ka]["suites"][i]) for i in range(4)) and close(m[5], ref[ka]["rate"]) and int(m[6].replace(",", "")) == ref[ka]["successes"]
    m = re.search(r"1 → 6 loops ([\d. /]+)%", bullet[0]); ok &= bool(m) and all(close(a, ref[k]["rate"]) for a, k in zip(m[1].split(" / "), range(1, 7)))
check("README.md (top level): the new entry's numbers equal the records", ok)
links = re.findall(r"\]\(([^)#]+)(?:#[^)]*)?\)", "\n".join(page))
check("README: every relative link of the folder page resolves", all((F / l).exists() for l in links if "independent_check.json" not in l), [l for l in links if not (F / l).exists()])

failures = [c for c in checks if not c["ok"]]
out = dict(report="v0_ka_sweep/suite_breakdown_20261114", script="verify_v0_ka_breakdown_20261114.py", passed=not failures, checks=checks,
           failures=[c["name"] for c in failures],
           recount={str(ka): dict(successes=count[ka]["total"], episodes=6000, suites={s: count[ka]["suite"][s] for s in ORDER},
                                  seeds={str(k): count[ka]["seed"][k] for k in (42, 43, 44)}) for ka in (1, 2)})
(F / "independent_check.json").write_text(json.dumps(out, indent=1) + "\n")
for c in checks: print(("ok   " if c["ok"] else "FAIL ") + c["name"] + ("" if c["ok"] else f"  [{c['detail']}]"))
print(f"{len(checks) - len(failures)}/{len(checks)} checks passed"); print("ALL CHECKS PASSED" if not failures else "FAILED")
sys.exit(1 if failures else 0)
