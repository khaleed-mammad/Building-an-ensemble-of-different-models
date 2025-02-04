# Machine Learning Project: Ensemble Models and Random Forest

## Overview
This project focuses on building and evaluating ensemble models and a Random Forest classifier. The dataset used consists of 581,012 rows and 55 columns, which was split into training, validation, and test sets. The primary goal was to compare the performance of individual models against ensemble methods and to explore the impact of hyperparameter tuning on model accuracy.

## Tasks

### Task 1: Voting Classifier

#### Part 1: Loading Data
- **Dataset**: The dataset was loaded using pandas and split into three parts:
  - **Train**: 371,847 rows
  - **Test**: 92,962 rows
  - **Validation**: 116,203 rows
- **Preprocessing**: Features were scaled using `StandardScaler` to reduce computation cost.

#### Part 2: Modeling
- **Models Trained**:
  - `RandomForestClassifier`
  - `ExtraTreesClassifier`
  - `SGDClassifier`
  - `LinearSVC`
  - `MLPClassifier`
- **Performance**:
  - **RandomForest**: 0.95
  - **ExtraTrees**: 0.95
  - **SGD**: 0.71
  - **LinearSVC**: 0.71
  - **MLP**: 0.86

#### Part 3: Ensembling
- **Hard Voting Classifier**: Achieved an accuracy of 90%. After removing linear models, accuracy improved to 95%.
- **Soft Voting Classifier**: Achieved an accuracy of 94%. Removing `MLPClassifier` increased accuracy to 95.15%.

### Task 2: Random Forest

#### Part 1: Loading Data
- **Dataset**: The dataset was loaded and column names were simplified for easier handling. The data was split into 80% training and 20% test sets.
- **Preprocessing**: Features were scaled using `StandardScaler`.

#### Part 2: Modeling
- **Decision Tree Classifier**: Hyperparameters were tuned using `GridSearchCV`. The best parameters found were:
  - `criterion`: entropy
  - `max_depth`: 5
  - `min_samples_leaf`: 10
  - `min_samples_split`: 2
- **Accuracy**: 86% on the test dataset.
- **Subset Training**: 1200 subsets of 100 samples each were used to train the model. The accuracy on these subsets was generally lower due to the small sample size.
- **Ensemble Prediction**: Predictions from all 1200 trees were aggregated using the mode. The final accuracy was 83%, which is lower than the individual Decision Tree model.

## Conclusion
- **Ensemble Methods**: Voting classifiers, especially soft voting, can significantly improve model performance by combining the strengths of multiple models.
- **Random Forest**: While individual trees may perform well, aggregating predictions from multiple trees trained on small subsets can lead to lower accuracy due to insufficient learning.
