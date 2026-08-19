import os
import sys

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


try:
    sys.stdout.reconfigure(encoding="utf-8")
except AttributeError:
    pass


# ============================================================
# Define paths
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)

DATA_PATH = os.path.join(PROJECT_DIR, "data", "train.csv")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "outputs")


def main():
    # Create outputs folder if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ============================================================
    # Load Dataset
    # ============================================================

    try:
        df = pd.read_csv(DATA_PATH)
        print("[OK] Dataset loaded successfully.")
    except FileNotFoundError:
        print("[ERROR] train.csv not found!")
        print("Expected location:")
        print(DATA_PATH)
        raise SystemExit(1)

    # ============================================================
    # Select Features and Target
    # ============================================================

    X = df[["GrLivArea", "BedroomAbvGr", "FullBath"]]
    y = df["SalePrice"]

    # ============================================================
    # Split Data
    # ============================================================

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    # ============================================================
    # Train Model
    # ============================================================

    model = LinearRegression()
    model.fit(X_train, y_train)

    # ============================================================
    # Predict
    # ============================================================

    y_pred = model.predict(X_test)

    # ============================================================
    # Evaluation
    # ============================================================

    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)

    print("\n==============================")
    print("HOUSE PRICE PREDICTION")
    print("==============================")
    print(f"R^2 Score : {r2:.4f}")
    print(f"MAE      : {mae:.2f}")
    print(f"MSE      : {mse:.2f}")

    # ============================================================
    # Save Results
    # ============================================================

    results_path = os.path.join(OUTPUT_DIR, "prediction_results.txt")

    with open(results_path, "w", encoding="utf-8") as file:
        file.write("HOUSE PRICE PREDICTION RESULTS\n")
        file.write("===============================\n\n")
        file.write(f"R^2 Score : {r2:.4f}\n")
        file.write(f"MAE      : {mae:.2f}\n")
        file.write(f"MSE      : {mse:.2f}\n")

    print("\n[OK] Results saved:")
    print(results_path)

    # ============================================================
    # Scatter Plot
    # ============================================================

    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.7)
    plt.xlabel("Actual House Price")
    plt.ylabel("Predicted House Price")
    plt.title("Actual vs Predicted House Price")

    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], "r--")

    plot_path = os.path.join(OUTPUT_DIR, "house_price_plot.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()

    print("[OK] Plot saved:")
    print(plot_path)

    # ============================================================
    # Example Prediction
    # ============================================================

    sample_house = pd.DataFrame({
        "GrLivArea": [2000],
        "BedroomAbvGr": [3],
        "FullBath": [2],
    })

    predicted_price = model.predict(sample_house)

    print("\nExample Prediction")
    print("---------------------------")
    print(f"Predicted House Price: ${predicted_price[0]:,.2f}")


if __name__ == "__main__":
    main()