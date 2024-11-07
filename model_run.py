import os
import pandas as pd
import numpy as np
from sklearn.ensemble import VotingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib

# Paths to feature datasets
features_path = "cell_counting/features"

# Load datasets
train_features = pd.read_csv(os.path.join(features_path, 'train_features.csv'))
val_features = pd.read_csv(os.path.join(features_path, 'val_features.csv'))
test_features = pd.read_csv(os.path.join(features_path, 'test_features.csv'))

# Separate features and labels
def prepare_data(features_df):
    """Prepare features (X) and labels (y) from the dataset."""
    X = features_df.drop(columns=['filename', 'count'])
    y = features_df['count']
    return X, y

X_train, y_train = prepare_data(train_features)
X_val, y_val = prepare_data(val_features)
X_test, y_test = prepare_data(test_features)

# Combine train and validation sets for model training
X_train_val = pd.concat([X_train, X_val])
y_train_val = pd.concat([y_train, y_val])

# Define and Train the Ensemble Model
regressors = [
    ('lr', LinearRegression()),
    ('ridge', Ridge(alpha=1.0)),
    ('dtr', DecisionTreeRegressor(max_depth=5)),
    ('knn', KNeighborsRegressor(n_neighbors=5))
]

voting_regressor = VotingRegressor(estimators=regressors)
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('voting', voting_regressor)
])

# 10-Fold Cross-Validation
kfold = KFold(n_splits=10, shuffle=True, random_state=42)
print("Performing 10-Fold Cross-Validation on Training Set...")
cv_results = cross_val_score(pipeline, X_train_val, y_train_val, cv=kfold, scoring='r2')
print(f"Cross-Validation R² Scores: {cv_results}")
print(f"Mean Cross-Validation R² Score: {cv_results.mean()}")

# Train the Model on the Full Training and Validation Set
print("Training the Ensemble Model on Full Training and Validation Set...")
pipeline.fit(X_train_val, y_train_val)

# Evaluate Model Performance
def evaluate_model(model, X, y, dataset_name):
    """Evaluate the model and print metrics."""
    y_pred = model.predict(X)
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    correct_within_5_percent = np.sum(np.abs(y - y_pred) <= 0.05 * y) / len(y) * 100
    print(f"{dataset_name} MSE: {mse:.3f}")
    print(f"{dataset_name} R² Score: {r2:.3f}")
    print(f"{dataset_name} Accuracy within ±5% of actual count: {correct_within_5_percent:.2f}%")
    return mse, r2, correct_within_5_percent

# Store accuracy for visualization
results = {}

# Evaluate on Training Set
print("\nEvaluating on Training Set:")
results['train'] = evaluate_model(pipeline, X_train, y_train, "Training")

# Evaluate on Validation Set
print("\nEvaluating on Validation Set:")
results['validation'] = evaluate_model(pipeline, X_val, y_val, "Validation")

# Evaluate on Test Set
print("\nEvaluating on Test Set:")
results['test'] = evaluate_model(pipeline, X_test, y_test, "Test")

# Save the trained model
model_save_path = "cell_counting/models"
os.makedirs(model_save_path, exist_ok=True)
joblib.dump(pipeline, os.path.join(model_save_path, 'ensemble_model.pkl'))

# Save results for visualization
results_path = os.path.join(features_path, 'evaluation_results.csv')
results_df = pd.DataFrame(results, index=['MSE', 'R²', 'Accuracy_within_5%'])
results_df.to_csv(results_path)

print("\nModel training and evaluation completed.")
