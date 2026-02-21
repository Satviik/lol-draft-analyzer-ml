import json

# print("Total Champions:", len(champs))
FILE_PATH = "champion_roles.json"

with open(FILE_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

# Case 1 — champions stored as dictionary
if isinstance(data, dict):
    print("Total Champions:", len(data.keys()))

# Case 2 — champions stored as list
elif isinstance(data, list):
    print("Total Champions:", len(data))

else:
    print("Unknown JSON structure")