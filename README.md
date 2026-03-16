# IITM AI/ML Course Assignment

## Week 7 Mini Project 1

### Problem Statement
Predict employee attrition using classification models on the IBM HR Analytics Employee Attrition dataset.

Dataset: [IBM HR Analytics Attrition Dataset on Kaggle](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset)

## What You Need To Submit

Create a ZIP folder containing these files:

1. `notebook.ipynb` - complete end-to-end analysis and modeling
2. `report.pdf` - executive summary
3. `requirements.txt` - Python packages used
4. `model.pkl` - final trained model
5. `README.md` - run instructions and result interpretation

## Step-By-Step Plan To Complete The Assignment

### Step 1: Download the dataset

1. Open the Kaggle dataset link.
2. Download the dataset CSV file.
3. Place it inside your project folder, for example in `data/`.
4. Confirm the target column is `Attrition`.

Suggested structure:

```text
project/
├── data/
│   └── WA_Fn-UseC_-HR-Employee-Attrition.csv
├── notebook.ipynb
├── report.pdf
├── requirements.txt
├── model.pkl
└── README.md
```

### Step 2: Set up the environment

1. Create a virtual environment.
2. Install the required libraries.
3. Start Jupyter Notebook.

Example commands:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install pandas numpy matplotlib seaborn scikit-learn jupyter joblib
jupyter notebook
```

### Step 3: Create the notebook

In `notebook.ipynb`, divide the work into clear sections:

1. Import libraries
2. Load dataset
3. Exploratory Data Analysis
4. Data preprocessing
5. Model building
6. Model evaluation
7. Feature importance / interpretation
8. Final model saving

### Step 4: Explore the dataset

Your EDA should cover:

1. Dataset shape
2. Column names and data types
3. First 5 rows
4. Summary statistics
5. Missing values check
6. Duplicate rows check
7. Target distribution: attrition vs non-attrition
8. Class imbalance confirmation

Useful visuals:

1. Count plot for `Attrition`
2. Histograms for numeric features
3. Boxplots for outlier inspection
4. Correlation heatmap for numeric columns
5. Attrition comparison by:
   `OverTime`, `JobRole`, `MonthlyIncome`, `Age`, `WorkLifeBalance`, `JobSatisfaction`

### Step 5: Preprocess the data

Complete these preprocessing tasks:

1. Separate features `X` and target `y`
2. Convert `Attrition` from `Yes/No` to `1/0`
3. Remove irrelevant columns if needed
   Example: employee identifier-like columns with no predictive value
4. Encode categorical columns
   Use `OneHotEncoder` or `get_dummies`
5. Scale numeric features
   Especially important for logistic regression and SVM
6. Handle class imbalance
   Use one or more of these:
   - `class_weight="balanced"`
   - oversampling such as SMOTE
   - undersampling
7. Split into train and test sets
   Example: `test_size=0.2`, `stratify=y`, fixed `random_state`

### Step 6: Train multiple classification models

The assignment requires multiple models, with at least one non-linear model.

Train at least these 3 models:

1. Logistic Regression
2. Decision Tree Classifier
3. SVM

You can also add:

1. Random Forest
2. XGBoost or Gradient Boosting
3. KNN

Recommended approach:

1. Build a baseline model first
2. Train all models on the same train/test split
3. If possible, use pipelines for preprocessing + model training
4. Tune a few important hyperparameters using cross-validation

### Step 7: Evaluate all models properly

Do not rely only on accuracy because attrition is imbalanced.

Report these metrics for every model:

1. Accuracy
2. Precision
3. Recall
4. F1-score
5. ROC-AUC
6. Confusion matrix

Important note:

For attrition prediction, recall for the attrition class is often important because missing a likely-to-leave employee can be costly.

Create a comparison table like this in the notebook and report:

```text
Model                  Accuracy   Precision   Recall   F1-score   ROC-AUC
Logistic Regression
Decision Tree
SVM
Random Forest
```

### Step 8: Identify important features

The assignment explicitly asks for the main factors influencing attrition.

Use one or more of these methods:

1. Logistic Regression coefficients
2. Decision Tree / Random Forest feature importance
3. Permutation importance
4. SHAP values if you want a stronger explanation section

Focus on explaining business meaning, for example:

1. Overtime increases attrition risk
2. Low job satisfaction is linked to higher attrition
3. Lower monthly income may be associated with attrition
4. Work-life balance may affect employee retention

### Step 9: Select the final model

Choose the final model based on:

1. Best overall performance on relevant metrics
2. Ability to handle class imbalance
3. Interpretability
4. Stability and business usefulness

State clearly in the notebook why this model was selected.

### Step 10: Save the final trained model

Save the selected model as `model.pkl`.

Example:

```python
import joblib
joblib.dump(final_model, "model.pkl")
```

If preprocessing is part of the workflow, save the entire pipeline instead of only the estimator.

### Step 11: Create `requirements.txt`

Add all packages used in the notebook.

Example:

```text
pandas
numpy
matplotlib
seaborn
scikit-learn
jupyter
joblib
imbalanced-learn
```

### Step 12: Write `report.pdf`

Your report should be short, clean, and business-focused.

Include these sections:

1. Dataset overview
2. Problem statement
3. EDA summary
4. Class imbalance challenge
5. Preprocessing steps
6. Models tried
7. Model comparison table
8. Best model selected
9. Important features influencing attrition
10. Key insights
11. Recommendations for the company
12. Challenges faced

Suggested recommendations:

1. Reduce excessive overtime
2. Improve job satisfaction programs
3. Review compensation and role progression
4. Improve work-life balance initiatives

### Step 13: Check your notebook before submission

Before finalizing:

1. Run all cells from top to bottom
2. Make sure there are no execution errors
3. Ensure charts are visible
4. Ensure the comparison table is present
5. Ensure feature importance is explained
6. Confirm `model.pkl` is generated
7. Confirm `requirements.txt` is complete
8. Export the notebook if needed

### Step 14: Package the final submission

Your ZIP file should contain:

```text
submission.zip
├── notebook.ipynb
├── report.pdf
├── requirements.txt
├── model.pkl
└── README.md
```

## Suggested Execution Order Inside The Notebook

Use this order while building the notebook:

1. Import packages
2. Load CSV
3. Inspect data
4. Perform EDA and visualizations
5. Prepare target and features
6. Encode and scale
7. Handle class imbalance
8. Split train/test data
9. Train 3 or more models
10. Evaluate each model
11. Compare metrics in one table
12. Interpret important features
13. Select best model
14. Save final model
15. Write conclusions

## How To Interpret Results

1. A higher recall for attrition means the model catches more employees likely to leave.
2. A higher precision means attrition predictions are more reliable.
3. F1-score balances precision and recall.
4. ROC-AUC shows how well the model separates attrition and non-attrition classes.
5. The best model is not always the one with the highest accuracy, especially with imbalanced data.

## Minimum Checklist Before Submission

- Dataset downloaded and loaded correctly
- EDA completed with visuals
- Imbalance handled explicitly
- At least 3 classification models trained
- At least 1 non-linear model included
- All major evaluation metrics reported
- Important features explained
- Final model saved as `model.pkl`
- `report.pdf` created
- `requirements.txt` created
- Final ZIP prepared
