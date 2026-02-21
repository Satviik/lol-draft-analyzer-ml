import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

DATA_PATH = "E:/League_Project_ML/lol-draft-analyzer-ml/data/processed/draft_dataset.csv"

def load_data():
    df = pd.read_csv(DATA_PATH)
    return df

def build_feature_matrix(df):

    # Drop incomplete drafts
    df = df.dropna()

    champion_columns = [
        "blue_top","blue_jungle","blue_mid","blue_adc","blue_support",
        "red_top","red_jungle","red_mid","red_adc","red_support"
    ]

    champions = pd.unique(df[champion_columns].values.ravel())
    champions = [c for c in champions if isinstance(c, str)]

    X = pd.DataFrame(0, index=df.index, columns=champions)
    X.columns = X.columns.astype(str)

    for idx, row in df.iterrows():
        for col in champion_columns[:5]:
            X.loc[idx, row[col]] += 1

        for col in champion_columns[5:]:
            X.loc[idx, row[col]] -= 1

    y = df["blue_win"].astype(int)

    return X, y


def main():
    df = load_data()
    X, y = build_feature_matrix(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    print("Baseline Accuracy:", acc)

if __name__ == "__main__":
    main()
