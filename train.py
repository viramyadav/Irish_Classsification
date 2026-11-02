
from pathlib import Path
import joblib
import pandas as pd

from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Define project paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATA_PATH = DATA_DIR / "iris.csv"
MODEL_DIR = BASE_DIR / "model"
MODEL_PATH = MODEL_DIR / "model.pkl"


def main():

    # Create data folder if it does not exist
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Automatically generate Iris dataset if missing
    if not DATA_PATH.exists():
        print("iris.csv not found. Creating dataset...")

        iris = load_iris(as_frame=True)
        data = iris.frame

        data = data.rename(columns={
            "sepal length (cm)": "sepal_length",
            "sepal width (cm)": "sepal_width",
            "petal length (cm)": "petal_length",
            "petal width (cm)": "petal_width",
            "target": "species_id"
        })

        data["species"] = data["species_id"].map({
            0: "setosa",
            1: "versicolor",
            2: "virginica"
        })

        data = data.drop(columns=["species_id"])

        data.to_csv(DATA_PATH, index=False)

        print("Dataset created successfully!")

    # Load dataset
    data = pd.read_csv(DATA_PATH)

    print("\nDataset loaded successfully!")
    print(data.head())

    # Display species counts
    print("\nSpecies counts:")
    print(data["species"].value_counts())

    # Define features and target
    X = data.drop(columns=["species"])
    y = data["species"]

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Initialize Random Forest
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    # Train model
    model.fit(X_train, y_train)

    print("\nModel training completed!")

    # Make predictions
    y_pred = model.predict(X_test)

    print("\nPredictions:")
    print(y_pred)

    # Evaluate model
    accuracy = accuracy_score(y_test, y_pred)

    print("\nModel Evaluation:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Accuracy Percentage: {accuracy * 100:.2f}%")

    # Save trained model
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, MODEL_PATH)

    print("\nModel saved successfully!")
    print(f"Location: {MODEL_PATH}")


if __name__ == "__main__":
    main()
