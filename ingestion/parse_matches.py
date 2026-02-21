import os
import json
import pandas as pd

RAW_PATH = "E:/League_Project_ML/lol-draft-analyzer-ml/data/raw"
PROCESSED_PATH = "E:/League_Project_ML/lol-draft-analyzer-ml/data/processed"

VALID_ROLES = {
    "TOP": "top",
    "JUNGLE": "jungle",
    "MIDDLE": "mid",
    "BOTTOM": "adc",
    "UTILITY": "support"
}

def parse_match(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)

    # Defensive check
    if "info" not in data or "metadata" not in data:
        return None

    info = data["info"]

    # Only ranked solo
    if info.get("queueId") != 420:
        return None

    # Remove remakes
    if info.get("gameDuration", 0) < 300:
        return None

    draft = {
        "match_id": data["metadata"]["matchId"],
        "patch": info["gameVersion"]
    }

    blue_win = None

    for p in info["participants"]:
        role = p.get("individualPosition")

        if role not in VALID_ROLES:
            return None

        team = "blue" if p["teamId"] == 100 else "red"
        mapped_role = VALID_ROLES[role]

        draft[f"{team}_{mapped_role}"] = p["championName"]

        if p["teamId"] == 100:
            blue_win = p["win"]

    draft["blue_win"] = blue_win

    return draft


def main():
    rows = []

    for file in os.listdir(RAW_PATH):
        if file.endswith(".json"):
            parsed = parse_match(os.path.join(RAW_PATH, file))
            if parsed:
                rows.append(parsed)

    df = pd.DataFrame(rows)

    os.makedirs(PROCESSED_PATH, exist_ok=True)
    df.to_csv(os.path.join(PROCESSED_PATH, "draft_dataset.csv"), index=False)

    print("Dataset created ✅")
    print("Rows:", len(df))
    print(df.head())


if __name__ == "__main__":
    main()
