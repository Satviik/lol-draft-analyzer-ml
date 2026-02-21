import requests

DDRAGON_URL = "https://ddragon.leagueoflegends.com/api/versions.json"

def get_latest_patch():
    response = requests.get(DDRAGON_URL)
    versions = response.json()
    return versions[0]  # latest patch


def load_champion_tags():
    patch = get_latest_patch()
    url = f"https://ddragon.leagueoflegends.com/cdn/{patch}/data/en_US/champion.json"

    response = requests.get(url)
    data = response.json()["data"]

    champ_tags = {}

    for champ_name, champ_data in data.items():
        champ_tags[champ_name] = champ_data["tags"]

    return champ_tags
