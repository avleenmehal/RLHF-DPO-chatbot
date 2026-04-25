"""One-time migration: import data/preferences.jsonl into the Preference DB table."""

import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from db.database import SessionLocal, Preference, init_db

JSONL_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "preferences.jsonl")


def run():
    init_db()
    db = SessionLocal()
    inserted = skipped = 0
    try:
        with open(JSONL_PATH) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)

                query = row.get("prompt", "").strip()
                chosen = row.get("chosen", "").strip()
                rejected = row.get("rejected", "").strip()

                if not query or not chosen or not rejected:
                    skipped += 1
                    continue

                pref = Preference(
                    user_id=None,
                    session_id=None,
                    query=query,
                    chosen_response=chosen,
                    rejected_response=rejected,
                )
                db.add(pref)
                inserted += 1

        db.commit()
        print(f"Done — inserted {inserted}, skipped {skipped}")
    except FileNotFoundError:
        print(f"File not found: {JSONL_PATH}")
        sys.exit(1)
    finally:
        db.close()


if __name__ == "__main__":
    run()
