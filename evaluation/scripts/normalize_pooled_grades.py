"""
Normalize and populate authoritative grades in evaluation/data/pooled_candidates.csv
- Backups the original to evaluation/data/pooled_candidates.csv.bak
- Ensures `author_consolidated_grade` column exists
- For each row, if `author_consolidated_grade` is empty, copy from in-order: single_assessor_grade, author_consolidated_grade, adjudicated_grade, judge1_grade, judge2_grade
- Normalize values: map any numeric >=3 -> 2; coerce to int 0/1/2
- Append provenance to `notes` when the authoritative value was set or changed
- Preserve other columns
"""
import csv
from pathlib import Path
import shutil

CSV = Path("evaluation/data/pooled_candidates.csv")
BACKUP = CSV.with_suffix(".csv.bak")

if not CSV.exists():
    print("Missing:", CSV)
    raise SystemExit(1)

shutil.copy2(CSV, BACKUP)
print("Backup written to:", BACKUP)

rows = []
with CSV.open("r", encoding="utf-8", newline="") as fh:
    reader = csv.DictReader(fh)
    fieldnames = list(reader.fieldnames or [])
    for row in reader:
        # Ensure notes exists
        notes = (row.get("notes") or "").strip()
        # If author_consolidated_grade present and non-empty, normalize it
        acg = (row.get("author_consolidated_grade") or "").strip()
        if acg == "":
            # candidate sources in preferred order
            candidates = [
                row.get("single_assessor_grade", ""),
                row.get("author_consolidated_grade", ""),
                row.get("adjudicated_grade", ""),
                row.get("judge1_grade", ""),
                row.get("judge2_grade", ""),
            ]
            source = None
            val = None
            for cand in candidates:
                if cand is None:
                    continue
                s = str(cand).strip()
                if s != "":
                    source = cand
                    val = s
                    break
            if val is None or val == "":
                # leave empty
                row["author_consolidated_grade"] = ""
            else:
                try:
                    g = int(float(val))
                except Exception:
                    g = 0
                if g >= 3:
                    g = 2
                g = max(0, min(2, int(g)))
                row["author_consolidated_grade"] = str(g)
                note = f"normalized from {val} -> {g}"
                if notes:
                    notes = f"{notes}; {note}"
                else:
                    notes = note
                row["notes"] = notes
        else:
            # Normalize existing authoritative value
            try:
                g = int(float(acg))
            except Exception:
                g = 0
            if g >= 3:
                g = 2
            g = max(0, min(2, int(g)))
            if str(g) != acg:
                note = f"normalized existing author_consolidated_grade {acg} -> {g}"
                if notes:
                    notes = f"{notes}; {note}"
                else:
                    notes = note
                row["notes"] = notes
            row["author_consolidated_grade"] = str(g)

        rows.append(row)

# ensure column exists
if "author_consolidated_grade" not in fieldnames:
    fieldnames.append("author_consolidated_grade")
if "notes" not in fieldnames:
    fieldnames.append("notes")

with CSV.open("w", encoding="utf-8", newline="") as fh:
    writer = csv.DictWriter(fh, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print("Updated rows:", len(rows))
print("Wrote updated CSV with normalized authoritative grades to:", CSV)
print("If you want I can now run build_ground_truth.py")
