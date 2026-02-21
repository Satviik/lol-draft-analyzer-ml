import joblib
import pandas as pd
from training.champion_metadata import load_champion_tags

MODEL_PATH = "models/draft_model.pkl"

ROLE_TAG_MAP = {
    "top": {"Tank", "Fighter"},
    "jungle": {"Fighter", "Assassin"},
    "mid": {"Mage", "Assassin"},
    "adc": {"Marksman"},
    "support": {"Support"}
}


class DraftRecommender:

    def __init__(self, model_path=MODEL_PATH):
        self.model = joblib.load(model_path)
        self.champ_tags = load_champion_tags()
        self.template_columns = self.model.get_booster().feature_names

    # -------------------------------------------------------
    # Build feature vector exactly aligned with training
    # -------------------------------------------------------
    def build_feature_vector(self, blue_team, red_team):

        X = pd.DataFrame(0.0, index=[0], columns=self.template_columns)

        # One-hot
        for champ in blue_team:
            col = f"blue_{champ}"
            if col in X.columns:
                X.loc[0, col] = 1

        for champ in red_team:
            col = f"red_{champ}"
            if col in X.columns:
                X.loc[0, col] = 1

        # Composition counts
        tag_list = ["Tank","Mage","Assassin","Marksman","Support"]

        blue_counts = {tag:0 for tag in tag_list}
        red_counts  = {tag:0 for tag in tag_list}

        for champ in blue_team:
            for tag in self.champ_tags.get(champ, []):
                if tag in blue_counts:
                    blue_counts[tag] += 1

        for champ in red_team:
            for tag in self.champ_tags.get(champ, []):
                if tag in red_counts:
                    red_counts[tag] += 1

        # Raw counts + diffs
        for tag in tag_list:

            if f"blue_{tag.lower()}_count" in X.columns:
                X.loc[0, f"blue_{tag.lower()}_count"] = blue_counts[tag]

            if f"red_{tag.lower()}_count" in X.columns:
                X.loc[0, f"red_{tag.lower()}_count"] = red_counts[tag]

            diff_col = f"{tag.lower()}_diff"
            if diff_col in X.columns:
                X.loc[0, diff_col] = blue_counts[tag] - red_counts[tag]

        return X

    # -------------------------------------------------------
    # Predict win probability
    # -------------------------------------------------------
    def predict_win_prob(self, blue_team, red_team):
        X = self.build_feature_vector(blue_team, red_team)
        return self.model.predict_proba(X)[0][1]

    # -------------------------------------------------------
    # Delta-based recommendation
    # -------------------------------------------------------
    def recommend_pick(self, blue_team, red_team, role, side="blue", top_n=5):

        baseline_prob = self.predict_win_prob(blue_team, red_team)

        allowed_tags = ROLE_TAG_MAP.get(role.lower(), set())

        all_champs = list(self.champ_tags.keys())

        candidate_pool = [
            champ for champ in all_champs
            if any(tag in allowed_tags for tag in self.champ_tags.get(champ, []))
        ]

        candidate_pool = [
            champ for champ in candidate_pool
            if champ not in blue_team and champ not in red_team
        ]

        results = []

        for champ in candidate_pool:

            temp_blue = blue_team.copy()
            temp_red = red_team.copy()

            if side == "blue":
                temp_blue.append(champ)
            else:
                temp_red.append(champ)

            new_prob = self.predict_win_prob(temp_blue, temp_red)
            delta = new_prob - baseline_prob

            results.append({
            "champion": champ,
            "new_win_prob": float(round(float(new_prob), 4)),
            "delta": float(round(float(delta), 4))
            })

        results = sorted(results, key=lambda x: x["delta"], reverse=True)

        return baseline_prob, results[:top_n]
