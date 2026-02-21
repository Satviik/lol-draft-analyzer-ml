import os
import requests
import json
import time
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

API_KEY = os.getenv("RIOT_API_KEY")

HEADERS = {
    "X-Riot-Token": API_KEY
}

# ---------- REGION CONFIG ----------
PLATFORM = "kr"        # na1, euw1, kr, etc.
REGION = "asia"        # americas, europe, asia

# ---------- TIERS ----------
TIERS = ["challenger", "grandmaster", "master"]

# ---------- PATH ----------
RAW_DATA_PATH = "E:/League_Project_ML/lol-draft-analyzer-ml/data/raw"

# ---------- SETTINGS ----------
PLAYERS_TO_FETCH = 100
MATCHES_PER_PLAYER = 70

MATCH_REQUEST_SLEEP = 0.2
ID_REQUEST_SLEEP = 0.45
BATCH_COOLDOWN = 6
MAX_RETRIES = 5


# ---------- SAFE REQUEST ----------
def safe_request(url, params=None, sleep_time=0.15):
    retries = 0

    while retries < MAX_RETRIES:
        response = requests.get(url, headers=HEADERS, params=params)

        if response.status_code == 200:
            time.sleep(sleep_time)
            return response

        elif response.status_code == 429:
            print("Rate limit hit. Sleeping 10 seconds...")
            time.sleep(10)
            retries += 1
            continue

        else:
            print(f"Error {response.status_code}: {response.text}")
            return None

    print("Max retries reached.")
    return None


# ---------- GET PLAYERS ----------
def get_ranked_puuids():
    all_puuids = []

    for tier in TIERS:
        print(f"\nFetching {tier.upper()} players...")

        url = f"https://{PLATFORM}.api.riotgames.com/lol/league/v4/{tier}leagues/by-queue/RANKED_SOLO_5x5"

        response = safe_request(url, sleep_time=0.5)

        if not response:
            continue

        data = response.json()
        tier_puuids = [entry["puuid"] for entry in data["entries"]]

        print(f"{tier.upper()} players found: {len(tier_puuids)}")

        all_puuids.extend(tier_puuids)

    unique_puuids = list(set(all_puuids))
    print(f"\nTotal unique players collected: {len(unique_puuids)}")

    return unique_puuids


# ---------- MATCH IDS ----------
def get_match_ids(puuid):
    url = f"https://{REGION}.api.riotgames.com/lol/match/v5/matches/by-puuid/{puuid}/ids"

    params = {
        "queue": 420,  # Ranked Solo
        "count": MATCHES_PER_PLAYER
    }

    response = safe_request(url, params=params, sleep_time=ID_REQUEST_SLEEP)

    if not response:
        return []

    return response.json()


# ---------- MATCH DATA ----------
def get_match_data(match_id):
    url = f"https://{REGION}.api.riotgames.com/lol/match/v5/matches/{match_id}"

    response = safe_request(url, sleep_time=MATCH_REQUEST_SLEEP)

    if not response:
        return None

    return response.json()


# ---------- SAVE MATCH ----------
def save_match(match_data, match_id):
    os.makedirs(RAW_DATA_PATH, exist_ok=True)

    file_path = os.path.join(RAW_DATA_PATH, f"{match_id}.json")

    if os.path.exists(file_path):
        return False

    with open(file_path, "w") as f:
        json.dump(match_data, f)

    return True


# ---------- MAIN ----------
def main():
    print("Starting ranked data collection...")

    puuids = get_ranked_puuids()

    if not puuids:
        print("No players found.")
        return

    existing_files = set(os.listdir(RAW_DATA_PATH)) if os.path.exists(RAW_DATA_PATH) else set()

    print("\nDownloading ranked matches...\n")

    for puuid in puuids[:PLAYERS_TO_FETCH]:

        match_ids = get_match_ids(puuid)

        for match_id in tqdm(match_ids):

            filename = f"{match_id}.json"

            if filename in existing_files:
                continue

            match_data = get_match_data(match_id)

            if match_data is None:
                continue

            saved = save_match(match_data, match_id)

            if saved:
                existing_files.add(filename)

        print("Cooling down between players...\n")
        time.sleep(BATCH_COOLDOWN)

    print("\nDone ✅")


if __name__ == "__main__":
    main()
