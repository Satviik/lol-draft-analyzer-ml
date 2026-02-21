import pandas as pd

DATA_PATH = "E:/League_Project_ML/lol-draft-analyzer-ml/data/processed/draft_dataset.csv"
OUTPUT_PATH = "E:/League_Project_ML/lol-draft-analyzer-ml/data/processed/patch_role_stats_16_3.csv"


def main():
    df = pd.read_csv(DATA_PATH)

    # Filter to 16.3
    df = df[df["patch"].str.startswith("16.3")]

    role_columns = {
        "top": ("blue_top", "red_top"),
        "jungle": ("blue_jungle", "red_jungle"),
        "mid": ("blue_mid", "red_mid"),
        "adc": ("blue_adc", "red_adc"),
        "support": ("blue_support", "red_support")
    }

    records = []

    for _, row in df.iterrows():

        blue_win = row["blue_win"]

        for role, (blue_col, red_col) in role_columns.items():
            records.append((row[blue_col], role, blue_win))
            records.append((row[red_col], role, 1 - blue_win))

    role_df = pd.DataFrame(records, columns=["champion","role","win"])

    stats = role_df.groupby(["champion","role"]).agg(
        games=("win","count"),
        wins=("win","sum")
    )

    stats["winrate"] = stats["wins"] / stats["games"]

    stats = stats.reset_index()

    stats.to_csv(OUTPUT_PATH, index=False)

    print("Role patch stats saved.")
    print(stats.sort_values("winrate", ascending=False).head(20))


if __name__ == "__main__":
    main()
