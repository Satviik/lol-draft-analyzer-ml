from ingestion.pick_recommender import DraftRecommender

recommender = DraftRecommender()

blue = ["Aatrox",
        "MonkeyKing",
        "Katarina",
        "Karma"]
red  = ["Ambessa",
        "Belveth",
        "Lux",
        "Lucian",
        "Nami"]

baseline, recs = recommender.recommend_pick(
    blue_team=blue,
    red_team=red,
    role="adc",
    side="blue"
)

print("Baseline Win Prob:", round(baseline,4))
print("\nTop Recommendations:\n")

for r in recs:
    print(r)
