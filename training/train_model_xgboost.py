import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from training.champion_metadata import load_champion_tags
import joblib
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss
from sklearn.model_selection import StratifiedKFold

DATA_PATH = "E:/League_Project_ML/lol-draft-analyzer-ml/data/processed/draft_dataset.csv"
PATCH_STATS_PATH = "E:/League_Project_ML/lol-draft-analyzer-ml/data/processed/patch_stats_16_3.csv"
ROLE_STATS_PATH = "E:/League_Project_ML/lol-draft-analyzer-ml/data/processed/patch_role_stats_16_3.csv"


def load_data():
    return pd.read_csv(DATA_PATH)


def build_feature_matrix(df):

    # -----------------------------------
    # Filter Patch
    # -----------------------------------
    df = df[df["patch"].str.startswith("16.3")].reset_index(drop=True)
    df = df.dropna()

    champ_tags = load_champion_tags()

    # -----------------------------------
    # Load Patch Strength
    # -----------------------------------
    patch_df = pd.read_csv(PATCH_STATS_PATH)
    patch_strength = {
        row["champion"]: row["winrate"] - 0.5
        for _, row in patch_df.iterrows()
    }

    role_df = pd.read_csv(ROLE_STATS_PATH)
    role_strength = {
        (row["champion"], row["role"]): row["winrate"] - 0.5
        for _, row in role_df.iterrows()
    }

    blue_cols = ["blue_top","blue_jungle","blue_mid","blue_adc","blue_support"]
    red_cols  = ["red_top","red_jungle","red_mid","red_adc","red_support"]

    roles = ["top","jungle","mid","adc","support"]

    all_cols = blue_cols + red_cols

    champions = pd.unique(df[all_cols].values.ravel())
    champions = [c for c in champions if isinstance(c, str)]

    blue_features = [f"blue_{champ}" for champ in champions]
    red_features  = [f"red_{champ}" for champ in champions]

    X = pd.DataFrame(0.0, index=df.index, columns=blue_features + red_features)

    # -----------------------------------
    # Strategic Columns
    # -----------------------------------
    strategic_cols = [
        # composition
        "tank_diff","mage_diff","assassin_diff",
        "marksman_diff","support_diff",

        # interaction
        "blue_assassin_vs_mage","red_assassin_vs_mage",
        "blue_protected_hypercarry","red_protected_hypercarry",

        # patch strength
        "patch_strength_diff",
        "role_strength_diff",

        # lane strength diffs
        "top_lane_diff",
        "jungle_lane_diff",
        "mid_lane_diff",
        "adc_lane_diff",
        "support_lane_diff"
    ]

    for col in strategic_cols:
        X[col] = 0.0

    # -----------------------------------
    # Build Row By Row
    # -----------------------------------
    for idx, row in df.iterrows():

        blue_counts = {"Tank":0,"Mage":0,"Assassin":0,"Marksman":0,"Support":0}
        red_counts  = {"Tank":0,"Mage":0,"Assassin":0,"Marksman":0,"Support":0}

        blue_patch_total = 0
        red_patch_total = 0

        blue_role_total = 0
        red_role_total = 0

        # ----------------------
        # Lane-by-lane handling
        # ----------------------
        for role in roles:

            blue_champ = row[f"blue_{role}"]
            red_champ  = row[f"red_{role}"]

            # One-hot
            X.loc[idx, f"blue_{blue_champ}"] = 1
            X.loc[idx, f"red_{red_champ}"] = 1

            # Patch strength
            blue_patch = patch_strength.get(blue_champ, 0)
            red_patch  = patch_strength.get(red_champ, 0)

            blue_patch_total += blue_patch
            red_patch_total  += red_patch

            # Role strength
            blue_role = role_strength.get((blue_champ, role), 0)
            red_role  = role_strength.get((red_champ, role), 0)

            blue_role_total += blue_role
            red_role_total  += red_role

            # Lane diff feature
            X.loc[idx, f"{role}_lane_diff"] = blue_role - red_role

            # Composition counting
            for tag in champ_tags.get(blue_champ, []):
                if tag in blue_counts:
                    blue_counts[tag] += 1

            for tag in champ_tags.get(red_champ, []):
                if tag in red_counts:
                    red_counts[tag] += 1

        # ----------------------
        # Composition Diffs
        # ----------------------
        X.loc[idx, "tank_diff"] = blue_counts["Tank"] - red_counts["Tank"]
        X.loc[idx, "mage_diff"] = blue_counts["Mage"] - red_counts["Mage"]
        X.loc[idx, "assassin_diff"] = blue_counts["Assassin"] - red_counts["Assassin"]
        X.loc[idx, "marksman_diff"] = blue_counts["Marksman"] - red_counts["Marksman"]
        X.loc[idx, "support_diff"] = blue_counts["Support"] - red_counts["Support"]

        # ----------------------
        # Interactions
        # ----------------------
        if blue_counts["Assassin"] > 0 and red_counts["Mage"] > 0:
            X.loc[idx, "blue_assassin_vs_mage"] = 1

        if red_counts["Assassin"] > 0 and blue_counts["Mage"] > 0:
            X.loc[idx, "red_assassin_vs_mage"] = 1

        if blue_counts["Marksman"] >= 1 and (blue_counts["Tank"] + blue_counts["Support"]) >= 1:
            X.loc[idx, "blue_protected_hypercarry"] = 1

        if red_counts["Marksman"] >= 1 and (red_counts["Tank"] + red_counts["Support"]) >= 1:
            X.loc[idx, "red_protected_hypercarry"] = 1

        # ----------------------
        # Patch Strength Diff
        # ----------------------
        X.loc[idx, "patch_strength_diff"] = blue_patch_total - red_patch_total
        X.loc[idx, "role_strength_diff"] = blue_role_total - red_role_total

    y = df["blue_win"].astype(int)

    return X, y


def main():
    df = load_data()
    X, y = build_feature_matrix(df)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    accuracies = []
    aucs = []
    log_losses = []

    fold = 1

    # ================================
    # 5-FOLD CROSS VALIDATION
    # ================================
    for train_idx, test_idx in skf.split(X, y):

        print(f"\n===== FOLD {fold} =====")

        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model = XGBClassifier(
            n_estimators=500,
            max_depth=5,
            learning_rate=0.04,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
            eval_metric='logloss'
        )

        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        probs = model.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, preds)
        auc = roc_auc_score(y_test, probs)
        ll = log_loss(y_test, probs)

        print(f"Accuracy: {acc:.4f}")
        print(f"ROC-AUC: {auc:.4f}")
        print(f"Log Loss: {ll:.4f}")

        accuracies.append(acc)
        aucs.append(auc)
        log_losses.append(ll)

        fold += 1

    print("\n===== CROSS VALIDATION RESULTS =====")
    print(f"Mean Accuracy: {np.mean(accuracies):.4f}")
    print(f"Mean ROC-AUC: {np.mean(aucs):.4f}")
    print(f"Mean Log Loss: {np.mean(log_losses):.4f}")
    print(f"Accuracy Std Dev: {np.std(accuracies):.4f}")

    # ================================
    # TRAIN FINAL MODEL ON FULL DATA
    # ================================
    print("\nTraining final model on full dataset...")

    final_model = XGBClassifier(
        n_estimators=500,
        max_depth=5,
        learning_rate=0.04,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        eval_metric='logloss'
    )

    final_model.fit(X, y)

    os.makedirs("models", exist_ok=True)
    joblib.dump(final_model, "models/draft_model.pkl")

    # ================================
    # FEATURE IMPORTANCE
    # ================================
    feature_importance = pd.DataFrame({
        "feature": X.columns,
        "importance": final_model.feature_importances_
    })

    feature_importance = feature_importance.sort_values(
        by="importance",
        ascending=False
    )

    print("\n===== TOP 20 IMPORTANT FEATURES =====")
    print(feature_importance.head(20))



if __name__ == "__main__":
    main()
