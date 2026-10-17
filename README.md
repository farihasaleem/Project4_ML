# Breast Cancer Diagnosis Prediction

Binary classification of breast tumors (malignant vs benign) using the Breast Cancer Wisconsin (Diagnostic) dataset. Because a missed cancer is costly, the project prioritizes recall for the malignant class over accuracy.

Learning project on a small public dataset. Not a clinical tool.

# Dataset

## Breast Cancer Wisconsin (Diagnostic) on Kaggle: 569 samples, 30 numeric features, target diagnosis (357 benign, 212 malignant).

# Approach

Cleaned data: dropped id and the empty Unnamed: 32, encoded M = 1, B = 0
EDA: class balance, feature distributions by diagnosis, correlation heatmap
Stratified 80/20 train/test split before scaling to avoid leakage
Pipelines (StandardScaler + model); compared Logistic Regression, SVM, Random Forest and XGBoost with 5-fold stratified cross-validation
Tuned Logistic Regression with GridSearchCV (scoring = recall)
Evaluated once on the held-out test set
Explained predictions with SHAP (computed on scaled features)

# Results

5-fold cross-validation (training set)

Model               	Accuracy	    Precision     	Recall       ROC-AUC
Logistic Regression	  0.974	        0.977	          0.953	       0.996
XGBoost	              0.971	        0.976	          0.947	       0.992
SVM	                  0.971	        0.977         	0.947	       0.995
Random Forest	        0.965	        0.964	          0.941      	 0.989

# Tuning: best C = 1 (the default), CV recall 0.953.
Final test set (114 samples: 72 benign, 42 malignant)

Metric	                  Score
Accuracy                 	0.96
Precision (malignant)	    0.97
Recall (malignant)      	0.93
ROC-AUC	                  0.996
Errors	3 false negatives, 1 false positive

# Key findings
All four models performed within about one percentage point of each other, so the simple, interpretable Logistic Regression was chosen.
SHAP shows that larger and more irregular tumors (for example texture_worst, radius_worst, concave points_mean, symmetry_worst) push predictions toward malignant.

# Limitations
Small dataset: the test set has only 42 malignant cases, so each missed case changes recall by about 2.4 points.
The model missed 3 of 42 cancers (recall 0.93), which is too many for real screening use.
Many features are highly correlated, so SHAP values show how the model behaves, not causal effects.
No external validation.
