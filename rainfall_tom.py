# Importing required libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, roc_auc_score

# Reading the weather dataset
data = pd.read_csv("weather.csv")

# Defining the relevant feature columns and target
columns = ['MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation', 'Sunshine', 'WindGustSpeed', 
           'Humidity9am', 'Humidity3pm', 'Pressure9am', 'Pressure3pm', 'Cloud9am', 
           'Cloud3pm', 'Temp9am', 'Temp3pm', 'RISK_MM']

# Feature selection and handling missing data using KNN Imputer
x = data[columns]       
imputer = KNNImputer(n_neighbors=5)
x = imputer.fit_transform(x)

y = data['RainTomorrow']  # Target variable

# Feature Engineering: Create a new feature to capture temperature differences
x = pd.DataFrame(x, columns=columns)  # Convert to DataFrame for feature creation
x['Temp_diff'] = x['Temp3pm'] - x['Temp9am']  # Adding temperature difference feature

# Feature scaling
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

# Dimensionality Reduction using PCA
pca = PCA(n_components=8)  # Reduce dimensions to 8 components
x_pca = pca.fit_transform(x_scaled)

# Splitting the data into training and testing sets
xtrain, xtest, ytrain, ytest = train_test_split(x_pca, y, test_size=0.25, random_state=0)

# Model Selection and Hyperparameter Tuning
param_grid = {'n_estimators': [50, 100], 'max_depth': [5, 10, None]}
grid_search = GridSearchCV(estimator=RandomForestClassifier(random_state=42), param_grid=param_grid, cv=3)
grid_search.fit(xtrain, ytrain)
best_model = grid_search.best_estimator_

# Cross-validation
cv_scores = cross_val_score(best_model, x_pca, y, cv=5)
print(f"Cross-validation accuracy: {cv_scores.mean():.4f}")

# Fitting the best model
best_model.fit(xtrain, ytrain)

# Making predictions on the test set
y_pred = best_model.predict(xtest)

# Evaluating the model performance
cm = confusion_matrix(ytest, y_pred)
accuracy = accuracy_score(ytest, y_pred)
print(f"Confusion Matrix:\n{cm}")
print(f"Accuracy: {accuracy:.4f}")
print(f"Classification Report:\n{classification_report(ytest, y_pred)}")
print(f"ROC AUC: {roc_auc_score(ytest, best_model.predict_proba(xtest)[:, 1]):.4f}")

# Visualizing the confusion matrix using a heatmap
class_names = [0, 1]
fig, ax = plt.subplots() 
tick_marks = np.arange(len(class_names)) 
plt.xticks(tick_marks, class_names) 
plt.yticks(tick_marks, class_names) 

sns.heatmap(pd.DataFrame(cm), annot=True, cmap="YlGnBu", fmt='g')
ax.xaxis.set_label_position("top") 
plt.tight_layout() 
plt.title('Confusion Matrix', y=1.1) 
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.show()

# Feature Importance Visualization (for tree-based models)
importances = best_model.feature_importances_

# Create feature names for PCA components
feature_names = [f'PC{i+1}' for i in range(8)]  # 8 PCA components

plt.barh(feature_names, importances)
plt.xlabel('Feature Importance')
plt.title('Feature Importance in Model')
plt.show()
