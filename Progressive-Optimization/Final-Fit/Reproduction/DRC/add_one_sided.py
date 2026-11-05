"""add_one_sided.py -- add stick_M_step's minus arm, the one arm build_drc.py cannot make here.

    python add_one_sided.py [--code <Codebase folder>]      (run once, after build_drc.py drc.json)

At node 31 S0_M_step = 1.0, its upper bound, so the plus arm does not exist and build_drc.py refuses the
direction. This writes drc_params/stick_M_step_minus.json exactly as build_drc.py would (root x exp(-0.3), the
engine's ordering constraints applied), appends its manifest row and its 15 task lines with the seed base each
temperature already uses (common random numbers with every other arm). Its effect is one-sided:
D = ln(GR_minus / GR_0) / delta_applied.
"""
import argparse, csv, json, math, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, *[os.pardir] * 4))   # this repository's root
ap = argparse.ArgumentParser()
ap.add_argument("--code", default=os.environ.get("KMC_CODE") or os.path.join(REPO, "Codebase"))
a = ap.parse_args()
sys.path.insert(0, os.path.join(a.code, "kmc"))
import KMC as K

os.chdir(HERE)
NAME, KEY, TAG = "stick_M_step", "S0_M_step", "minus"
fn = "%s_%s.json" % (NAME, TAG)
if os.path.exists(os.path.join("drc_params", fn)):
    sys.exit("[add_one_sided] %s exists -- run build_drc.py drc.json again first" % fn)
root = json.load(open("drc_params/baseline.json"))
d = float(json.load(open("drc.json"))["delta_efolds"])
v0 = root[KEY]
v = v0 / math.exp(d)
cl, what = K.enforce_constraints({KEY: v}, fixed={k: x for k, x in root.items() if k != KEY})
clipped = KEY in what
v = cl[KEY]
p = dict(root); p[KEY] = v
pushed = {k: x for k, x in cl.items() if k != KEY}
p.update(pushed)
json.dump(p, open(os.path.join("drc_params", fn), "w"), indent=2, sort_keys=True)

cols = next(csv.reader(open("drc_manifest.csv")))
row = dict(direction=NAME, key=KEY, arm=TAG, value_0=repr(v0), value=repr(v), delta_nominal=-d,
           delta_applied=round(math.log(v / v0), 12), clipped="CLIPPED" if clipped else "",
           pushed=";".join("%s=%r" % kv for kv in sorted(pushed.items())))
with open("drc_manifest.csv", "a", newline="") as f:
    csv.DictWriter(f, fieldnames=cols, lineterminator="\n").writerow({c: row.get(c, "") for c in cols})

base = [l.split() for l in open("drc_tasklist.txt") if l.startswith("baseline.json ")]
with open("drc_tasklist.txt", "a", newline="\n") as f:
    for _, T, s in base:
        f.write("%s %s %s\n" % (fn, T, s))
print("[add_one_sided] %s: %s %r -> %r (delta %+.6f)%s; %d task lines appended"
      % (fn, KEY, v0, v, math.log(v / v0), " pushed " + str(pushed) if pushed else "", len(base)))
