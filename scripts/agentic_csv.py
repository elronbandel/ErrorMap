"""Convert `failed_with_gold.jsonl` (from the agentic-data release) to the
per-benchmark CSVs ErrorMap ingests.

Each task contributes a failure row (score=0, output_text=failed
trajectory) and one shared gold row (score=1, output_text=peer-success
trajectory) — they share `example_id` so `single_error.py`'s
success_outputs lookup links them.

Usage:
    python scripts/agentic_csv.py path/to/failed_with_gold.jsonl out/
"""
import csv, json, sys
from pathlib import Path

TRUNC = 1500

def trunc(s, n=TRUNC):
    return s if len(s) <= n else s[:n] + "…"

def serialize(traj):
    if not traj: return ""
    parts = []
    for e in traj:
        ev, step = e.get("event"), e.get("step")
        if ev == "observation":
            obs = e.get("observation")
            if obs is None: continue
            content = obs.get("result", obs) if isinstance(obs, dict) else obs
            s = content if isinstance(content, str) else json.dumps(content, default=str)
            parts.append(f"[step {step}] OBS: {trunc(s)}")
        elif ev == "action":
            act = e.get("action")
            s = act if isinstance(act, str) else json.dumps(act, default=str)
            parts.append(f"[step {step}] ACT: {trunc(s)}")
    return "\n".join(parts)

def main(jsonl_path, out_dir):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    by_bench, seen_gold = {}, set()
    cols = ["example_id", "model", "input_text", "output_text", "score",
            "correct_answer", "dataset", "agent", "session_id", "run_id", "row_kind"]

    for line in open(jsonl_path):
        rec = json.loads(line)
        bench = rec["benchmark"]
        eid = f"{rec.get('task_id') or ''}|{rec.get('task_key') or ''}"
        ctx = json.dumps(rec.get("context") or {}, default=str)
        input_text = f"TASK:\n{rec.get('task') or ''}\n\nINITIAL CONTEXT:\n{trunc(ctx, 3000)}"
        rows = by_bench.setdefault(bench, [])

        rows.append({
            "example_id": eid, "model": rec["model"], "input_text": input_text,
            "output_text": serialize(rec["failed_trajectory"]), "score": 0,
            "correct_answer": "", "dataset": bench, "agent": rec["agent"],
            "session_id": rec["session_id"], "run_id": rec["run_id"], "row_kind": "failure",
        })

        gold = rec.get("gold")
        if gold and (bench, eid) not in seen_gold:
            seen_gold.add((bench, eid))
            rows.append({
                "example_id": eid, "model": gold["model"], "input_text": input_text,
                "output_text": serialize(gold["trajectory"]), "score": 1,
                "correct_answer": "", "dataset": bench, "agent": gold["agent"],
                "session_id": gold["session_id"], "run_id": gold["run_id"], "row_kind": "gold",
            })

    for bench, rows in by_bench.items():
        path = out / f"{bench}.csv"
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
            w.writeheader(); w.writerows(rows)
        print(f"{bench:<25} {len(rows):>5} rows -> {path}")

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
