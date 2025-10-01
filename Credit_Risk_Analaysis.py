import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, MaxNLocator
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, accuracy_score, 
                            roc_curve, confusion_matrix)
from xgboost import XGBClassifier
import warnings

warnings.filterwarnings('ignore')

# Set style for better visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100


# 1. DATA LOADING AND INITIAL EXPLORATION

# Load data - update this path to your file location
data = pd.read_csv('credit_risk_dataset.csv')  # Fixed typo (was ..csv)

print("Dataset Shape:", data.shape)
print("\nFirst Few Rows:")
print(data.head())

print("\nDataset Info:")
data.info()

print("\nMissing Values:")
print(data.isnull().sum())


# 2. DATA CLEANING

# Handle missing values in person_emp_length
data.dropna(subset=['person_emp_length'], inplace=True)
print(f"\nShape after dropping null employment length: {data.shape}")

# Impute missing loan_int_rate based on loan_grade
mean_loan_int_rate = data.groupby('loan_grade')['loan_int_rate'].mean().reset_index()
print("\nMean Loan Interest Rate by Grade:")
print(mean_loan_int_rate)

data['loan_int_rate'] = data['loan_int_rate'].fillna(
    data.groupby('loan_grade')['loan_int_rate'].transform('mean')
)

print("\nMissing Values After Imputation:")
print(data.isnull().sum())


# 3. FEATURE ENGINEERING

# Create flag for income without employment
zero_emp_length = data[data['person_emp_length'] == 0].shape[0]
print(f"\nNumber of entries with employment_length of 0: {zero_emp_length}")

data['has_income_no_emp'] = np.where(
    (data['person_income'] > 0) & (data['person_emp_length'] == 0), 1, 0
).astype(str)

# Create age groups
data['age_group'] = pd.cut(
    data['person_age'], 
    bins=[0, 15, 25, 45, 65, 90, 150], 
    labels=['0-15', '16-25', '26-45', '46-65', '66-90', '91-150']
).astype(str)

# Create income ranges
data['income_range'] = pd.cut(
    data['person_income'], 
    bins=[0, 50000, 100000, 200000, 400000, 600000, 800000, 1000000, 3000000],
    labels=['0-50000', '50001-100000', '100001-200000', '200001-400000', 
            '400001-600000', '600001-800000', '800001-1000000', '1000001-3000000']
).astype(str)

# 4. OUTLIER REMOVAL

# Remove unrealistic values
print(f"\nBefore outlier removal: {data.shape}")
data = data[data['person_emp_length'] != 123]
data = data[data['age_group'] != '91-150']
print(f"After outlier removal: {data.shape}")

print("\nDescriptive Statistics:")
print(data.describe().T)

# Save cleaned data
data.to_csv('cleaned_credit_risk_dataset.csv', index=False)
print("\nCleaned data saved to 'cleaned_credit_risk_dataset.csv'")


# 5. EXPLORATORY DATA ANALYSIS - CATEGORICAL VARIABLES

df = pd.read_csv('cleaned_credit_risk_dataset.csv')

cat_col = ['person_home_ownership', 'loan_intent', 'loan_grade', 
           'loan_status', 'cb_person_default_on_file', 'age_group', 'income_range']

# Countplots for categorical variables
fig, axes = plt.subplots(3, 2, figsize=(18, 15))
fig.suptitle('Bar Plot for All Categorical Variables in Dataset', fontsize=16)

sns.countplot(ax=axes[0, 0], x=df['person_home_ownership'], 
              color='lightgreen', order=df['person_home_ownership'].value_counts().index)
axes[0, 0].set_title('Home Ownership')

sns.countplot(ax=axes[0, 1], x=df['loan_intent'], 
              color='lightgreen', order=df['loan_intent'].value_counts().head(5).index)
axes[0, 1].set_title('Loan Intent')

sns.countplot(ax=axes[1, 0], x=df['loan_grade'], 
              color='lightgreen', order=df['loan_grade'].value_counts().head(5).index)
axes[1, 0].set_title('Loan Grade')

sns.countplot(ax=axes[1, 1], x=df['cb_person_default_on_file'], 
              color='lightgreen', order=df['cb_person_default_on_file'].value_counts().index)
axes[1, 1].set_title('Previous Default')

sns.countplot(ax=axes[2, 0], x=df['age_group'], 
              color='lightgreen', order=df['age_group'].value_counts().index)
axes[2, 0].set_title('Age Group')
axes[2, 0].tick_params(labelrotation=45)

sns.countplot(ax=axes[2, 1], x=df['income_range'], 
              color='lightgreen', order=df['income_range'].value_counts().index)
axes[2, 1].set_title('Income Range')
axes[2, 1].tick_params(labelrotation=45)

plt.tight_layout()
plt.show()


# 6. EXPLORATORY DATA ANALYSIS - NUMERICAL VARIABLES

num_col = ['person_age', 'person_income', 'person_emp_length', 'loan_amnt', 
           'loan_int_rate', 'loan_percent_income', 'cb_person_cred_hist_length']

for col in num_col:
    print(f"\n{col} - Skewness: {df[col].skew():.2f}")
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Histogram with KDE
    sns.histplot(df[col], kde=True, color='lightgreen', ax=axes[0])
    axes[0].set_ylabel('Frequency')
    axes[0].set_title(f'Histogram for {col} with KDE')
    
    # Boxplot
    sns.boxplot(x=df[col], color='lightgreen', ax=axes[1])
    axes[1].set_title(f'Boxplot for {col}')
    
    plt.tight_layout()
    plt.show()

# 7. BIVARIATE ANALYSIS - CATEGORICAL VS LOAN STATUS

for col in cat_col:
    if col != 'loan_status':
        plt.figure(figsize=(10, 5))
        sns.countplot(x=col, hue='loan_status', data=df, palette=["lightgreen", "green"])
        plt.title(f'{col} by Loan Status')
        plt.xlabel(col)
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.legend(title='Loan Status', labels=['No Default', 'Default'])
        plt.tight_layout()
        plt.show()

# 8. BIVARIATE ANALYSIS - NUMERICAL VS LOAN STATUS

for col in num_col:
    plt.figure(figsize=(10, 5))
    sns.kdeplot(data=df, x=col, hue='loan_status', fill=True, 
                palette=['lightgreen', 'green'], linewidth=2)
    plt.title(f'KDE Plot for {col} by Loan Status')
    plt.xlabel(col)
    plt.ylabel('Density')
    plt.legend(title='Loan Status', labels=['No Default', 'Default'])
    plt.tight_layout()
    plt.show()

# 9. CORRELATION ANALYSIS

# Prepare data for correlation
columns_to_exclude = ['age_group', 'income_range']
df_corr = df.drop(columns=columns_to_exclude)

# One-hot encode categorical variables
df_encoded = pd.get_dummies(
    df_corr, 
    columns=['loan_grade', 'person_home_ownership', 'loan_intent', 'cb_person_default_on_file'], 
    drop_first=True
)

# Correlation matrix
correlation_matrix = df_encoded.corr()
plt.figure(figsize=(16, 12))
sns.heatmap(correlation_matrix, annot=True, cmap="YlGnBu", fmt=".2f", 
            cbar_kws={'label': 'Correlation Coefficient'})
plt.title('Correlation Matrix', fontsize=16)
plt.tight_layout()
plt.show()


# 10. KEY INSIGHTS - LOAN INTEREST RATE THRESHOLD

plt.figure(figsize=(10, 6))
sns.kdeplot(data=df, x='loan_int_rate', hue='loan_status', fill=True, 
            palette=['lightgreen', 'green'])
plt.title("Identifying Risky Applicants with Higher Loan Interest Rate", fontsize=14)
plt.axvline(x=15.4, color='red', linestyle='--', linewidth=2, label='Threshold (15.4%)')
plt.xlabel('Loan Interest Rate')
plt.ylabel('Density')
plt.legend(title='Loan Status', labels=['Threshold', 'No Default', 'Default'])
plt.tight_layout()
plt.show()

# Analyze by threshold
loan_int_above_threshold = df[df['loan_int_rate'] >= 15.4]
loan_int_below_threshold = df[df['loan_int_rate'] < 15.4]

loan_status_above = loan_int_above_threshold['loan_status'].value_counts()
loan_status_below = loan_int_below_threshold['loan_status'].value_counts()

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].pie(loan_status_above, autopct='%1.1f%%', colors=['lightgreen', 'green'], 
            labels=['No Default', 'Default'], startangle=90)
axes[0].set_title('Interest Rate ≥ 15.4%')

axes[1].pie(loan_status_below, autopct='%1.1f%%', colors=['lightgreen', 'green'], 
            labels=['No Default', 'Default'], startangle=90)
axes[1].set_title('Interest Rate < 15.4%')

plt.tight_layout()
plt.show()

# 11. ADDITIONAL VISUALIZATIONS

# Loan grade vs interest rate
plt.figure(figsize=(10, 6))
sns.boxplot(x='loan_grade', y='loan_int_rate', data=df, palette='Greens', 
            order=sorted(df['loan_grade'].unique()))
plt.title('Loan Interest Rate by Grade')
plt.xlabel('Loan Grade')
plt.ylabel('Interest Rate (%)')
plt.tight_layout()
plt.show()

# Scatter plot: interest rate vs loan percent income
plt.figure(figsize=(10, 6))
sns.scatterplot(x='loan_int_rate', y='loan_percent_income', data=df, 
                hue='loan_status', palette=['lightgreen', 'green'], alpha=0.6)
plt.title('Loan Interest Rate vs Loan Percent Income')
plt.xlabel('Interest Rate (%)')
plt.ylabel('Loan as % of Income')
plt.legend(title='Loan Status', labels=['No Default', 'Default'])
plt.tight_layout()
plt.show()

# Default history vs interest rate
plt.figure(figsize=(10, 6))
sns.boxplot(x='cb_person_default_on_file', y='loan_int_rate', data=df, 
            palette=['lightgreen', 'green'])
plt.title('Interest Rate by Previous Default History')
plt.xlabel('Previous Default on File')
plt.ylabel('Interest Rate (%)')
plt.tight_layout()
plt.show()

# 12. LOAN PERCENT INCOME THRESHOLD ANALYSIS

plt.figure(figsize=(10, 6))
sns.kdeplot(data=df, x='loan_percent_income', hue='loan_status', fill=True, 
            palette=['lightgreen', 'green'])
plt.axvline(x=0.31, color='red', linestyle='--', linewidth=2, label='Threshold (0.31)')
plt.title('Loan Percent Income Distribution by Loan Status')
plt.xlabel('Loan as % of Income')
plt.ylabel('Density')
plt.legend(title='Loan Status', labels=['Threshold', 'No Default', 'Default'])
plt.tight_layout()
plt.show()

# Analyze by threshold
loan_percent_above = df[df['loan_percent_income'] >= 0.31]
loan_percent_below = df[df['loan_percent_income'] < 0.31]

status_above = loan_percent_above['loan_status'].value_counts()
status_below = loan_percent_below['loan_status'].value_counts()

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].pie(status_above, autopct='%1.1f%%', colors=['lightgreen', 'green'], 
            labels=['No Default', 'Default'], startangle=90)
axes[0].set_title('Loan % Income ≥ 0.31')

axes[1].pie(status_below, autopct='%1.1f%%', colors=['lightgreen', 'green'], 
            labels=['No Default', 'Default'], startangle=90)
axes[1].set_title('Loan % Income < 0.31')

plt.tight_layout()
plt.show()

# 13. CROSSTAB ANALYSIS

print("\n" + "="*80)
print("DEFAULT RATE BY AGE GROUP AND INCOME RANGE")
print("="*80)
crosstab = pd.crosstab([df['age_group'], df['income_range']], 
                       df['loan_status'], normalize='index')
print(crosstab)

# 14. RENTERS WITH ZERO EMPLOYMENT ANALYSIS

renters_zero_emp = df[(df['person_home_ownership'] == 'RENT') & 
                      (df['person_emp_length'] == 0)]
renters_zero_emp_default = renters_zero_emp[renters_zero_emp['loan_status'] == 1]

print(f"\n{'='*80}")
print(f"RENTERS WITH ZERO EMPLOYMENT LENGTH ANALYSIS")
print(f"{'='*80}")
print(f"Total renters with zero employment: {len(renters_zero_emp)}")
print(f"Number defaulting: {len(renters_zero_emp_default)}")
print(f"Default rate: {len(renters_zero_emp_default)/len(renters_zero_emp)*100:.2f}%")


# 15. COMPOSITE RISK SCORE CREATION

# Create mappings
grade_mapping = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 6}
default_mapping = {'Y': 1, 'N': 0}
ownership_mapping = {'OWN': 1, 'MORTGAGE': 2, 'RENT': 3, 'OTHER': 4}

# Apply mappings
df['loan_grade_numeric'] = df['loan_grade'].map(grade_mapping)
df['prvs_default_numeric'] = df['cb_person_default_on_file'].map(default_mapping)
df['person_home_ownership_numeric'] = df['person_home_ownership'].map(ownership_mapping)

# Scale features
scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(df[[
    'loan_grade_numeric', 'loan_int_rate', 'loan_percent_income', 
    'prvs_default_numeric', 'person_home_ownership_numeric'
]])

df[['scaled_loan_grade_numeric', 'scaled_loan_int_rate', 
    'scaled_loan_percent_income', 'scaled_prvs_default_numeric',
    'scaled_person_home_ownership_numeric']] = scaled_features

# Create composite risk score
df['composite_risk_score'] = (
    (0.30 * df['scaled_loan_percent_income']) +
    (0.25 * df['scaled_loan_int_rate']) +
    (0.20 * df['scaled_loan_grade_numeric']) +
    (0.15 * df['scaled_person_home_ownership_numeric']) +
    (0.10 * df['scaled_prvs_default_numeric'])
)

# Categorize risk
bins = [0, 0.3, 0.5, 0.7, 1.0]
labels = ['Low Risk', 'Medium Risk', 'High Risk', 'Very High Risk']
df['risk_category'] = pd.cut(df['composite_risk_score'], bins=bins, labels=labels)

print("\nFirst 15 rows with risk scores:")
print(df[['loan_status', 'composite_risk_score', 'risk_category']].head(15))

# 16. PREPARE DATA FOR MODELING

# Drop unnecessary features
features_to_drop = [
    'age_group', 'scaled_loan_grade_numeric', 'scaled_loan_int_rate', 
    'scaled_loan_percent_income', 'scaled_prvs_default_numeric',
    'scaled_person_home_ownership_numeric', 'loan_amnt', 'loan_grade_numeric',
    'prvs_default_numeric', 'person_home_ownership_numeric', 
    'has_income_no_emp', 'income_range', 'risk_category'
]

df.drop(columns=features_to_drop, inplace=True)

# Prepare features and target
X = df.drop('loan_status', axis=1)
y = df['loan_status']

# One-hot encode categorical variables
X = pd.get_dummies(X, drop_first=True)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\nTraining set size: {X_train.shape}")
print(f"Test set size: {X_test.shape}")


# 17. MODEL TRAINING AND EVALUATION


def evaluate_model(model, X_train, X_test, y_train, y_test, model_name):
    """Train and evaluate a classification model"""
    print(f"\n{'='*80}")
    print(f"{model_name}")
    print(f"{'='*80}")
    
    # Train model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nModel Accuracy: {accuracy:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # ROC curve and optimal threshold
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    print(f"Optimal Threshold: {optimal_threshold:.4f}")
    
    # Confusion matrix
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    # Feature importance (if available)
    if hasattr(model, 'feature_importances_'):
        feature_importances = model.feature_importances_
        importance_df = pd.DataFrame({
            'Feature': X.columns,
            'Importance': feature_importances
        }).sort_values(by='Importance', ascending=False)
        
        print("\nTop 10 Feature Importances:")
        print(importance_df.head(10))
    
    return model, y_pred, y_prob

# MODEL 1: RANDOM FOREST

rf_model = RandomForestClassifier(
    class_weight='balanced', 
    n_estimators=100, 
    random_state=42
)
rf_model, rf_pred, rf_prob = evaluate_model(
    rf_model, X_train_scaled, X_test_scaled, y_train, y_test, 
    "RANDOM FOREST CLASSIFIER"
)

# MODEL 2: K-NEAREST NEIGHBORS

knn_model = KNeighborsClassifier(n_neighbors=6)
knn_model, knn_pred, knn_prob = evaluate_model(
    knn_model, X_train_scaled, X_test_scaled, y_train, y_test, 
    "K-NEAREST NEIGHBORS CLASSIFIER"
)

# MODEL 3: LOGISTIC REGRESSION

lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model, lr_pred, lr_prob = evaluate_model(
    lr_model, X_train_scaled, X_test_scaled, y_train, y_test, 
    "LOGISTIC REGRESSION"
)

# MODEL 4: DECISION TREE

dt_model = DecisionTreeClassifier(random_state=42)
dt_model, dt_pred, dt_prob = evaluate_model(
    dt_model, X_train_scaled, X_test_scaled, y_train, y_test, 
    "DECISION TREE CLASSIFIER"
)

# MODEL 5: XGBOOST

xgb_model = XGBClassifier(random_state=42, eval_metric='logloss')
xgb_model, xgb_pred, xgb_prob = evaluate_model(
    xgb_model, X_train_scaled, X_test_scaled, y_train, y_test, 
    "XGBOOST CLASSIFIER"
)

# 18. MODEL COMPARISON

print(f"\n{'='*80}")
print("MODEL COMPARISON SUMMARY")
print(f"{'='*80}")

models_summary = {
    'Random Forest': accuracy_score(y_test, rf_pred),
    'KNN': accuracy_score(y_test, knn_pred),
    'Logistic Regression': accuracy_score(y_test, lr_pred),
    'Decision Tree': accuracy_score(y_test, dt_pred),
    'XGBoost': accuracy_score(y_test, xgb_pred)
}

for model_name, acc in sorted(models_summary.items(), key=lambda x: x[1], reverse=True):
    print(f"{model_name:20s}: {acc:.4f}")

print(f"\n{'='*80}")
print("Analysis Complete!")
print(f"{'='*80}")