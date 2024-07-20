import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# Load the dataset
iris_data = pd.read_csv('IRIS.csv')

# Define features and target variable
features = iris_data.drop('species', axis=1)
target = iris_data['species']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=42)

# Standardize the feature values
scaler = StandardScaler()
X_train_standardized = scaler.fit_transform(X_train)
X_test_standardized = scaler.transform(X_test)

# Initialize and train the K-Nearest Neighbors model
knn_model = KNeighborsClassifier(n_neighbors=3)
knn_model.fit(X_train_standardized, y_train)

# Predict the species for the test set
y_predictions = knn_model.predict(X_test_standardized)

# Calculate the model's accuracy
model_accuracy = accuracy_score(y_test, y_predictions)
print(f"Model Accuracy: {model_accuracy:.2f}")

# Example prediction for a new sample
new_sample = pd.DataFrame([[5.1, 3.5, 1.4, 0.2]], columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])
new_sample_standardized = scaler.transform(new_sample)
sample_prediction = knn_model.predict(new_sample_standardized)
print(f"Predicted species for new sample: {sample_prediction[0]}")
