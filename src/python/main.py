from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

data_path = Path("/app/data")
train = pd.read_csv(data_path / "train.csv")

print("Training data loaded successfully!")
print("Shape:", train.shape)
print(train.head())

# 14.Explore the data
print("\n--- Data types and non-missing counts ---")
train.info()

print("\n--- Missing values by column ---")
print(train.isna().sum())

print("\n--- Numerical summary ---")
print(train.describe())

print("\n--- Survival rate by sex ---")
print(train.groupby("Sex")["Survived"].mean())

print("\n--- Survival rate by passenger class ---")
print(train.groupby("Pclass")["Survived"].mean())

# Create a new feature
train["FamilySize"] = train["SibSp"] + train["Parch"] + 1

# Select features and target
feature_columns = [
    "Pclass",
    "Sex",
    "Age",
    "Fare",
    "Embarked",
    "FamilySize",
]

X = train[feature_columns].copy()
y = train["Survived"].copy()

print("\n--- Data preparation decisions ---")
print("Added FamilySize = SibSp + Parch + 1.")
print("Selected features:", feature_columns)
print("Excluded PassengerId, Name, Ticket, and Cabin from this baseline model.")
print("Missing values and categorical encoding will be handled in the model pipeline.")

print("\n--- Selected features: first five rows ---")
print(X.head())

print("\nFeature shape:", X.shape)
print("Target shape:", y.shape)


# 15.Prepare the features and train a logistic regression model

numeric_features = ["Age", "Fare", "FamilySize"]
categorical_features = ["Pclass", "Sex", "Embarked"]

# Fill missing numerical values and standardize numerical features
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

# Fill missing categorical values and encode categories
categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore")),
])

# Apply the appropriate processing to each group of columns
preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_features),
    ("categorical", categorical_pipeline, categorical_features),
])

# Combine preprocessing and logistic regression
model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000)),
])

# Train using the features and labels from train.csv
model.fit(X, y)

print("\n--- Logistic regression ---")
print("Preprocessing and model training completed successfully.")
print("Number of training passengers:", len(y))


# 16.Measure accuracy on the training set
train_accuracy = model.score(X, y)

print("\n--- Training accuracy ---")
print(f"Training accuracy: {train_accuracy:.2%}")


# 17.Load the test dataset and make predictions
test = pd.read_csv(data_path / "test.csv")

test["FamilySize"] = test["SibSp"] + test["Parch"] + 1

# Select the same features in the same order
X_test = test[feature_columns].copy()

# Predict using the trained preprocessing and model
test_predictions = model.predict(X_test)

# Combine passenger IDs with predicted survival outcomes
prediction_results = pd.DataFrame({
    "PassengerId": test["PassengerId"],
    "Survived": test_predictions,
})

print("\n--- Test set predictions ---")
print("Number of test passengers:", len(test))
print(prediction_results.head(10))

print("\n--- Predicted outcome counts ---")
print(prediction_results["Survived"].value_counts().sort_index())


# 18.(Since there are no labels in the test set, we will skip this step.)
print("\n--- Test set accuracy ---")
print("Test accuracy is unavailable since test.csv doesn't have Survived labels.")