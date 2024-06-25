# Parkinson's Disease Detection
![Parkinson's Disease](https://www.askdrray.com/wp-content/uploads/2017/09/bigstock-195525808.jpg)
This project aims to detect Parkinson's disease using various machine learning models. We have implemented and evaluated four different models: Support Vector Machine (SVM), Random Forest (RF), XGBoost, and a Stacking Classifier. The dataset used for this project is available on [Kaggle](https://www.kaggle.com/datasets/vikasukani/parkinsons-disease-data-set?select=parkinsons.data).

## Dataset

The dataset consists of biomedical voice measurements from people with early-stage Parkinson's disease and healthy individuals. The main attributes include:

- `name`: ASCII subject name and recording number
- `MDVP:Fo(Hz)`: Average vocal fundamental frequency
- `MDVP:Fhi(Hz)`: Maximum vocal fundamental frequency
- `MDVP:Flo(Hz)`: Minimum vocal fundamental frequency
- `MDVP:Jitter(%)`: Measure of variation in fundamental frequency
- `MDVP:Jitter(Abs)`: Measure of variation in fundamental frequency
- `MDVP:RAP`: Measure of variation in fundamental frequency
- `MDVP:PPQ`: Measure of variation in fundamental frequency
- `Jitter:DDP`: Measure of variation in fundamental frequency
- `MDVP:Shimmer`: Measure of variation in amplitude
- `MDVP:Shimmer(dB)`: Measure of variation in amplitude
- `Shimmer:APQ3`: Measure of variation in amplitude
- `Shimmer:APQ5`: Measure of variation in amplitude
- `MDVP:APQ`: Measure of variation in amplitude
- `Shimmer:DDA`: Measure of variation in amplitude
- `NHR`: Ratio of noise to tonal components in the voice
- `HNR`: Ratio of noise to tonal components in the voice
- `RPDE`: Nonlinear dynamical complexity measure
- `DFA`: Signal fractal scaling exponent
- `spread1`: Nonlinear measure of fundamental frequency variation
- `spread2`: Nonlinear measure of fundamental frequency variation
- `D2`: Nonlinear measure of fundamental frequency variation
- `PPE`: Nonlinear measure of fundamental frequency variation
- `status`: Health status of the subject (1 - Parkinson's, 0 - Healthy)

## Data Preprocessing

1. **Handling Missing Data**: Imputation using the mean strategy.
2. **Standardization**: Scaling features to have zero mean and unit variance.
3. **Feature Selection**: Using a Random Forest classifier to select important features.

## Models and Accuracies

1. **Support Vector Machine (SVM)**:
   - Accuracy: 85%
   - Code snippet:
     ```python
     from sklearn.svm import SVC
     svm = SVC(kernel='linear', class_weight='balanced', probability=True)
     svm.fit(X_train, y_train)
     y_pred = svc.predict(X_test)
     accuracy = accuracy_score(y_test, y_pred)
     ```

2. **Random Forest (RF)**:
   - Accuracy: 90%
   - Code snippet:
     ```python
     from sklearn.ensemble import RandomForestClassifier
     rf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
     rf.fit(X_train, y_train)
     y_pred = rf.predict(X_test)
     accuracy = accuracy_score(y_test, y_pred)
     ```

3. **XGBoost**:
   - Accuracy: 90%
   - Code snippet:
     ```python
     from xgboost import XGBClassifier
     xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', scale_pos_weight=len(y_train[y_train == 0]) / len(y_train[y_train == 1]))
     xgb.fit(X_train, y_train)
     y_pred = xgb.predict(X_test)
     accuracy = accuracy_score(y_test, y_pred)
     ```

4. **Stacking Classifier**:
   - Accuracy: 95%
   - Code snippet:
     ```python
     from sklearn.ensemble import StackingClassifier
     from sklearn.linear_model import LogisticRegression
     
     base_models = [
     ('rf', RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)),
     ('xgb', XGBClassifier(use_label_encoder=False, eval_metric='logloss', scale_pos_weight=len(y_train[y_train == 0]) / len(y_train[y_train == 1])))
     ]
     stacking_clf = StackingClassifier(estimators=base_models, final_estimator=meta_model, cv=5)
     stacking_clf.fit(X_train, y_train)
     y_pred = stacking_clf.predict(X_test)
     accuracy = accuracy_score(y_test, y_pred)
     ```

## Conclusion

The Stacking Classifier achieved the highest accuracy of 95%, outperforming individual models like SVM, Random Forest, and XGBoost. This indicates that combining multiple models can leverage their individual strengths and improve overall performance.

## Requirements

- Python 
- pandas
- numpy
- scikit-learn
- xgboost

## Installation

You can install the required packages using pip:

```bash
pip install pandas numpy scikit-learn xgboost 
```
