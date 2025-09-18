import os
import sys
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, roc_auc_score
)
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTE

# Models
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

# Add project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from exception.execptions import MLProjectException


# ----------------- Evaluation Helpers -----------------

def evaluate_clf(true, predicted):
    """Calculate all key metrics for classifier."""
    acc = accuracy_score(true, predicted)
    f1 = f1_score(true, predicted)
    precision = precision_score(true, predicted)
    recall = recall_score(true, predicted)
    roc_auc = roc_auc_score(true, predicted)
    return acc, f1, precision, recall, roc_auc


def evaluate_models(X, y, models):
    """
    Splits the data, trains models, evaluates performance,
    and returns a report DataFrame.
    """
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Apply SMOTEENN with safe k_neighbors
    smt = SMOTEENN(smote=SMOTE(k_neighbors=1), random_state=42)
    X_train_res, y_train_res = smt.fit_resample(X_train, y_train)

    print("✅ Resampling complete.")
    print("Class distribution after resampling:",
          dict(zip(*np.unique(y_train_res, return_counts=True))))

    models_list, accuracy_list, auc_list = [], [], []

    for model_name, model in models.items():
        print(f"\n🚀 Training {model_name}...")
        model.fit(X_train_res, y_train_res)

        # Predictions
        y_train_pred = model.predict(X_train_res)
        y_test_pred = model.predict(X_test)

        # Training metrics
        train_acc, train_f1, train_prec, train_recall, train_auc = evaluate_clf(y_train_res, y_train_pred)

        # Test metrics
        test_acc, test_f1, test_prec, test_recall, test_auc = evaluate_clf(y_test, y_test_pred)

        print(f"\n📊 {model_name} Performance")
        print("Training Set:")
        print(f"- Accuracy: {train_acc:.4f}")
        print(f"- F1 Score: {train_f1:.4f}")
        print(f"- Precision: {train_prec:.4f}")
        print(f"- Recall: {train_recall:.4f}")
        print(f"- ROC-AUC: {train_auc:.4f}")

        print("Test Set:")
        print(f"- Accuracy: {test_acc:.4f}")
        print(f"- F1 Score: {test_f1:.4f}")
        print(f"- Precision: {test_prec:.4f}")
        print(f"- Recall: {test_recall:.4f}")
        print(f"- ROC-AUC: {test_auc:.4f}")
        print("=" * 40)

        # Save report info
        models_list.append(model_name)
        accuracy_list.append(test_acc)
        auc_list.append(test_auc)

    report = pd.DataFrame({
        "Model Name": models_list,
        "Test Accuracy": accuracy_list,
        "Test ROC-AUC": auc_list
    }).sort_values(by="Test Accuracy", ascending=False)

    return report


# ----------------- Main Training Script -----------------

if __name__ == "__main__":
    try:
        file_path = os.path.join("data", "pre_proceesed_data", "processed_data.csv")
        df = pd.read_csv(file_path)

        # Split features & target
        X = df.drop(columns=["is_safe"])
        y = df["is_safe"]

        # Define models
        models = {
            "Random Forest": RandomForestClassifier(random_state=42),
            "Decision Tree": DecisionTreeClassifier(random_state=42),
            "Gradient Boosting": GradientBoostingClassifier(random_state=42),
            "Logistic Regression": LogisticRegression(max_iter=500, random_state=42),
            "K-Neighbors Classifier": KNeighborsClassifier(),
            "XGBClassifier": XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42),
            "Support Vector Classifier": SVC(probability=True, random_state=42),
        }

        # Evaluate models
        results = evaluate_models(X, y, models)

        print("\n📌 Final Model Comparison:")
        print(results)

        # Save results
        os.makedirs("models", exist_ok=True)
        results.to_csv("models/model_results.csv", index=False)
        print("\n✅ Model results saved to models/model_results.csv")

    except MLProjectException as e:
        print(e)
