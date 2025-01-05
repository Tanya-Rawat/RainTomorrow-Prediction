# Rain Tomorrow Prediction
## Overview
This project implements a machine learning model to predict whether it will rain tomorrow using historical weather data. The dataset used is [weather.csv](https://www.kaggle.com/datasets/zaraavagyan/weathercsv), which contains various weather-related features.

## Features
- **Random Forest Classifier** model is used for predicting rainfall, with hyperparameter tuning using GridSearchCV for better performance.
- **Principal Component Analysis (PCA)** is applied for dimensionality reduction, transforming the feature space while retaining important information.
- **KNN Imputer** is used for handling missing data by imputation.
- **Standard Scaling** is performed on the features for better convergence of machine learning algorithms.
- Performance is evaluated using confusion matrix, accuracy score, classification report, and ROC AUC score.
- **Visualization** of the confusion matrix as a heatmap for easier interpretation of model performance.
- **Feature Importance Visualization** using bar charts to understand which features most influence the model's decisions.

## How to Run the Code
To run this project, follow these steps:

1. **Install Python** on your system if you haven't already.
2. **Install required libraries** using:
   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn
3. **Download the dataset (weather.csv)** and place it in the same directory as the Python code.
4. **Run the Code** by executing
   ```bash
   python path/to/your/script.py
   
## Output
### Upon running the code, the model's performance is evaluated using:
1. Accuracy
2. Confusion matrix
3. Classification report
4. ROC AUC score

![image](https://github.com/user-attachments/assets/6e18730f-ba73-4dbb-b51b-646f066e6418)


### Heatmap of Confusion Matrix
### A heatmap visualization of the confusion matrix will also be generated for a more intuitive understanding of the model's performance.
![image](https://github.com/user-attachments/assets/82b8fc99-2a62-4fa3-893f-2836e0af8e3c)

### Feature importance using bar charts.
![image](https://github.com/user-attachments/assets/eb1b93c3-3a17-47d7-96bd-e447ce44541e)

## Technologies Used
- Python
- Pandas
- Sci-kit Learn
- Matplotlib
- Seaborn

### Conclusion : This project demonstrates the use of a Random Forest Classifier with PCA for dimensionality reduction to predict whether it will rain tomorrow based on weather data. The project includes data preprocessing, handling missing values, model selection, hyperparameter tuning, and performance evaluation. Visualizations such as the confusion matrix heatmap and feature importance chart provide additional insights into the model's behavior and effectiveness.
