import pickle
import pandas as pd

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# --------------------------------
# 1. Load Dataset
# --------------------------------

data = pd.read_csv("penguins.csv")

data.columns = data.columns.str.strip()

print("Dataset Loaded Successfully")

print("\nDataset Preview:")
print(data.head(20))

print("\nDataset Shape:")
print(data.shape)

# --------------------------------
# 2. Prepare Dataset
# --------------------------------

data = data.dropna(subset=["species"])

features = [
    "bill_length_mm",
    "bill_depth_mm",
    "flipper_length_mm",
    "body_mass_g"
]

X = data[features].apply(pd.to_numeric, errors="coerce")

y = data["species"].astype(str).str.strip()

# --------------------------------
# 3. Split Dataset
# --------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# --------------------------------
# 4. Create Decision Tree Model
# --------------------------------

model = make_pipeline(
    SimpleImputer(strategy="median"),
    DecisionTreeClassifier(
        max_depth=1,
        random_state=42
    )
)

# --------------------------------
# 5. Train Model
# --------------------------------

model.fit(X_train, y_train)

print("\nModel Training Completed")

# --------------------------------
# 6. Model Prediction
# --------------------------------

y_pred = model.predict(X_test)

# --------------------------------
# 7. Overall Model Metrics
# --------------------------------

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(
    y_test, y_pred,
    average="weighted",
    zero_division=0
)

recall = recall_score(
    y_test, y_pred,
    average="weighted",
    zero_division=0
)

f1 = f1_score(
    y_test, y_pred,
    average="weighted",
    zero_division=0
)

print("\nOverall Model Metrics:")

print(f"Accuracy:           {accuracy:.3f}")
print(f"Precision (Weighted): {precision:.3f}")
print(f"Recall (Weighted):    {recall:.3f}")
print(f"F1-Score (Weighted):  {f1:.3f}")

# --------------------------------
# 8. Predict Penguin Species
# --------------------------------

print("\nPredicted Class:")

sample = X_test.iloc[[0]]

prediction = model.predict(sample)

print(prediction[0])

# --------------------------------
# 9. Train Model on Full Dataset
# --------------------------------

model.fit(X, y)

# --------------------------------
# 10. Save Model
# --------------------------------

with open("penguin_species_model.pkl", "wb") as file:
    pickle.dump(model, file)

print("\nModel Saved Successfully")
print("Saved as: penguin_species_model.pkl")