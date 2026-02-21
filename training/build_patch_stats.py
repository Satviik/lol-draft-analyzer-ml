import pandas as pd

DATA_PATH = "E:/League_Project_ML/lol-draft-analyzer-ml/data/processed/draft_dataset.csv"
OUTPUT_PATH = "E:/League_Project_ML/lol-draft-analyzer-ml/data/processed/patch_stats_16_3.csv"


def main():
    df = pd.read_csv(DATA_PATH)

    # Filter to patch 16.3
    df = df[df["patch"].str.startswith("16.3")]

    blue_cols = ["blue_top","blue_jungle","blue_mid","blue_adc","blue_support"]
    red_cols  = ["red_top","red_jungle","red_mid","red_adc","red_support"]

    records = []

    for idx, row in df.iterrows():

        blue_win = row["blue_win"]

        for col in blue_cols:
            records.append((row[col], 1, blue_win))

        for col in red_cols:
            records.append((row[col], 0, 1 - blue_win))

    champ_df = pd.DataFrame(records, columns=["champion","side","win"])

    stats = champ_df.groupby("champion").agg(
        games=("win","count"),
        wins=("win","sum")
    )

    stats["winrate"] = stats["wins"] / stats["games"]

    stats = stats.sort_values("winrate", ascending=False)

    stats.to_csv(OUTPUT_PATH)

    print("Patch stats saved.")
    print(stats.head(20))


if __name__ == "__main__":
    main()
