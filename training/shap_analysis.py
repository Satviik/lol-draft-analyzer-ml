import shap
import joblib
import pandas as pd
from training.train_model_xgboost import load_data, build_feature_matrix

MODEL_PATH = "models/draft_model.pkl"

def main():

    print("Loading model...")
    model = joblib.load(MODEL_PATH)

    print("Loading dataset...")
    df = load_data()
    X, y = build_feature_matrix(df)

    print("Initializing SHAP Explainer...")
    explainer = shap.TreeExplainer(model)

    print("Computing SHAP values...")
    shap_values = explainer.shap_values(X)

    print("\nGenerating summary plot...")
    shap.summary_plot(shap_values, X)

if __name__ == "__main__":
    main()
