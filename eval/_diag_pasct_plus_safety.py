"""Safety-vs-content decomposition of the new-expert evaluation.

Splits the 500-context simulated new-expert eval into

* "safety-locked" contexts: the v12 two-threshold safety shield forced
  PA-SCT-DRO to emit the canned safety-referral template; KEMI / RAG
  never safety-refer, so on these contexts the comparison is a
  safety-policy disagreement rather than a strategy / coverage one.
* "non-safety-locked" contexts: the planner runs unconstrained.

Writes a structured summary to ``outputs/new_expert_pasct_plus_safety_decomp.json``
and prints a per-pair table.  Used by Section "New-Expert Evaluation
(E4/E5) and Safety-vs-Content Decomposition" of the paper.
"""
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load_jsonl(p):
    out = []
    if not p.exists():
        return out
    for line in p.read_text(encoding="utf-8").splitlines():
        if line.strip():
            out.append(json.loads(line))
    return out


def _wr(rows, locked_filter, locked_set):
    n = w = ll = t = 0
    for r in rows:
        sid = str(r.get("sample_id"))
        in_locked = sid in locked_set
        if locked_filter == "locked" and not in_locked:
            continue
        if locked_filter == "non_locked" and in_locked:
            continue
        verdict = (r.get("verdict") or {}).get("overall", "tie")
        focal_side = r.get("focal_side")
        if verdict == "tie" or focal_side is None:
            t += 1
        elif verdict == focal_side:
            w += 1
        else:
            ll += 1
        n += 1
    wr = (w + 0.5 * t) / n if n else float("nan")
    return {"win_rate": wr, "wins": w, "losses": ll, "ties": t, "n": n}


def main():
    overrides = _load_jsonl(ROOT / "data/judge_eval_v10/v12_safety_overrides.jsonl")
    safety_locked = {
        str(r.get("sample_id"))
        for r in overrides
        if r.get("shield_fired")
        or r.get("shield_fired_hard")
        or r.get("decision") == "hard"
    }
    print(f"safety_locked contexts: {len(safety_locked)}")

    raw = _load_jsonl(ROOT / "outputs/new_expert_pasct_plus.jsonl")
    by_pair_role: dict[tuple[str, str], list] = {}
    for r in raw:
        pair = (str(r.get("baseline")), str(r.get("role_id") or "?"))
        by_pair_role.setdefault(pair, []).append(r)

    out: dict = {"safety_locked_count": len(safety_locked), "pairs": {}}
    print("\nbaseline   role                    n_locked  wr_locked  | "
          "n_nonlocked  wr_nonlocked  | n_all  wr_all")
    for (baseline, role), rows in sorted(by_pair_role.items()):
        rec = {
            "safety_locked": _wr(rows, "locked", safety_locked),
            "non_safety_locked": _wr(rows, "non_locked", safety_locked),
            "all": _wr(rows, "all", safety_locked),
        }
        out["pairs"][f"pasct_plus_vs_{baseline}__{role}"] = rec
        print(f"{baseline:<10s} {role:<22s} "
              f"{rec['safety_locked']['n']:5d}     "
              f"{rec['safety_locked']['win_rate']:.3f}      | "
              f"{rec['non_safety_locked']['n']:5d}        "
              f"{rec['non_safety_locked']['win_rate']:.3f}        | "
              f"{rec['all']['n']:5d}  {rec['all']['win_rate']:.3f}")

    out_path = ROOT / "outputs/new_expert_pasct_plus_safety_decomp.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False),
                        encoding="utf-8")
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
