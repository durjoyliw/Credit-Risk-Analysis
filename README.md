# Credit Risk Analysis: Optimizing Lending Decisions Through Data-Driven Risk Assessment

**Role:** Data Analyst (Risk Management Team)  
**Business Context:** In competitive lending markets, loan defaults directly impact profitability and portfolio health. This analysis identifies high-risk borrower segments and provides actionable recommendations to reduce default rates while maintaining loan accessibility.

---

## Executive Summary

**Bottom Line:** By implementing recommended risk thresholds, we can potentially **reduce default rates by 45%** while maintaining 82% loan approval rates for qualified borrowers.

**Key Finding:** Borrowers with loan-to-income ratios above 31% show a **70% default rate** compared to 25% below this threshold - making this our strongest risk predictor.

---

## Critical Risk Insights

### 🎯 High-Impact Risk Factors Identified

#### **1. Loan-to-Income Ratio (Strongest Predictor)**

<img width="341" alt="loan-percent-income-threshold" src="https://github.com/user-attachments/assets/c34fde4e-403b-4d49-acbf-a4b6679c7fad">

- **70% default rate** when loan payments exceed 31% of income
- **25% default rate** when below 31% threshold
- **Business Impact:** Recommend capping loan amounts at 30% of verified income

#### **2. Interest Rate Risk**

<img width="304" alt="interest-rate-threshold" src="https://github.com/user-attachments/assets/ef8c515b-4e7b-4abe-a51a-8905ae1ccf99">

- Loans with rates **above 15.4% show 61% default rates**
- Loans below 15.4% show only 18% default rates
- **Business Impact:** High-rate loans require enhanced underwriting or collateral requirements

#### **3. Vulnerable Borrower Segments**

**Income and Age Risk Analysis:**

<img width="227" alt="income-age-relationship" src="https://github.com/user-attachments/assets/14fbe0f5-3915-41ef-8938-e16fb8cc3ebd">

- **Renters with no employment history:** 35% default rate (929 of 2,666 borrowers)

<img width="289" alt="renters-zero-employment" src="https://github.com/user-attachments/assets/3cbde2f1-12b9-4b79-acd1-c43dec63ec3a">

- **Income below $50K:** Consistently elevated risk across all age groups
- **Loan Grades D-G:** 2.5x higher default rates than A-C grades

#### **4. Lower-Risk Indicators**

**Default Rates by Key Borrower Characteristics:**

<img width="376" alt="home-ownership-defaults" src="https://github.com/user-attachments/assets/ad3140ab-002e-44b5-8a60-0fbafa75b84b"> <img width="383" alt="loan-intent-defaults" src="https://github.com/user-attachments/assets/ae4cb3ee-292f-4ea9-8b1f-bf34e5ca468a"> <img width="374" alt="loan-grade-defaults" src="https://github.com/user-attachments/assets/59cef8e5-46ca-4d6b-9134-7847fe96f50b">

- **Mortgage holders:** 22% more likely to repay than renters
- **Education loans:** 18% lower default rates than medical loans
- **Employment history 5+ years:** 40% reduction in default likelihood

---

## Recommended Actions for Risk Management Team

### Immediate Implementation (0-30 Days)

1. **Implement 31% Loan-to-Income Cap**  
   - Apply to all new loan applications
   - Expected outcome: Reduce portfolio default rate by 12-15%

2. **Enhanced Screening for High-Risk Segments**  
   - Flag applications from renters with <1 year employment
   - Require additional documentation or co-signers
   - Target segment: ~8% of applications, representing 35% default risk

3. **Interest Rate Threshold Review**  
   - Reassess pricing strategy for loans above 15.4%
   - Consider requiring higher credit scores or down payments for high-rate approvals

### Strategic Initiatives (30-90 Days)

4. **Medical Loan Support Program**  
   - Offer financial counseling for medical loan applicants (28% higher default risk)
   - Partner with healthcare payment platforms for structured plans

5. **Tiered Approval Framework**  
   - **Loan Grades A-C:** Standard approval process
   - **Loan Grades D-F:** Enhanced verification + income stability assessment
   - **Loan Grade G:** Restrict or require secured collateral

6. **Income Verification Enhancement**  
   - Prioritize employment verification for applicants earning <$50K
   - Cross-reference with credit history length (strong correlation found)

---

## Business Metrics Summary

**Dataset Overview:** 32,581 loan applications analyzed  
**Current Portfolio Default Rate:** 21.8%  
**Target Default Rate (Post-Implementation):** 12-14%

### Borrower Profile Analysis

**Age and Employment Distribution:**

<img width="424" alt="borrower-demographics" src="https://github.com/user-attachments/assets/e159da73-47a6-49b2-a48d-1acbaa988c84">

<img width="737" alt="age-distribution" src="https://github.com/user-attachments/assets/b5917c4f-e748-44ea-bf59-90dd38f84a17">
<img width="738" alt="employment-length" src="https://github.com/user-attachments/assets/d2b3103e-730a-472b-a088-312767b0027b">

**Key Portfolio Characteristics:**

<img width="731" alt="home-ownership-distribution" src="https://github.com/user-attachments/assets/4493bc5b-f587-4715-a75f-445e19ae33b0">
<img width="733" alt="loan-intent-distribution" src="https://github.com/user-attachments/assets/dae5386c-4f8c-455e-b2a5-46f5ed159607">
<img width="734" alt="loan-status-distribution" src="https://github.com/user-attachments/assets/aa394f3f-5dd7-4651-a981-3bf3efd5885d">

**Risk Segmentation:**
- **Low Risk (Grades A-C):** 78% of portfolio, 8% default rate
- **Medium Risk (Grades D-E):** 16% of portfolio, 32% default rate  
- **High Risk (Grades F-G):** 6% of portfolio, 54% default rate

### Additional Risk Factor Analysis

**Default Patterns Across Key Segments:**

<img width="374" alt="income-defaults" src="https://github.com/user-attachments/assets/a7ffac94-f2d3-47d3-88f4-0a239d20c79f"> <img width="386" alt="employment-defaults" src="https://github.com/user-attachments/assets/c2449a8f-083e-451d-a17c-25c16e3cb985">

<img width="370" alt="interest-rate-defaults" src="https://github.com/user-attachments/assets/bf6e9dd0-77f3-4dc9-8b52-26ba2e1cd5f3"> <img width="399" alt="loan-percent-income-defaults" src="https://github.com/user-attachments/assets/428d232d-28eb-4f38-8136-a610c0edf8a0">

<img width="373" alt="credit-history-defaults" src="https://github.com/user-attachments/assets/1930ffc0-5275-4292-99e8-7ecd24b8ce0a"> <img width="362" alt="previous-default-history" src="https://github.com/user-attachments/assets/55f57a1e-3e26-42ef-9cc0-7b2adf7eb73f"> <img width="374" alt="loan-amount-defaults" src="https://github.com/user-attachments/assets/26746bbf-de05-4851-93ad-f4f4cb8d8e00">

---

## Model Performance

Built predictive models to automate risk scoring:

### Model Comparison

<img width="274" alt="model-performance-metrics" src="https://github.com/user-attachments/assets/2061d447-6bea-49c2-97b9-b7ae817a60ed">

<img width="257" alt="model-comparison" src="https://github.com/user-attachments/assets/ac1f02f7-68ec-474e-b4f0-d8edeff4f379">

- **XGBoost Model:** 87% accuracy, **91% recall** (catches 91% of actual defaults)
- **Random Forest Model:** 89% accuracy, **94% precision** (minimizes false alarms)

**Recommended Approach:** Use XGBoost for initial screening (high recall) + Random Forest for final approval decisions (high precision)

### Feature Importance Analysis

<img width="230" alt="feature-importance" src="https://github.com/user-attachments/assets/8f2d6b33-33af-4baa-80e5-52c5383fce1e">

**Top Risk Predictors (in order of importance):**
1. Loan-to-income percentage (35% importance)
2. Interest rate (22% importance)  
3. Loan grade (18% importance)
4. Previous default history (12% importance)
5. Home ownership status (8% importance)

### Correlation Analysis

**Key Risk Factor Relationships:**

![correlation-matrix](https://github.com/user-attachments/assets/f50d4d44-231f-4eaf-80ef-5a8d89efdeda)

**Critical Correlations Identified:**
- Loan interest rate correlates with loan grade and previous default history
- Loan-to-income percentage is the strongest predictor of loan status
- Age correlates positively with credit history length
- Weak but notable relationship between renters and zero employment length

---

## Next Steps for Analysis

To refine these recommendations further, I recommend investigating:

1. **Geographic Risk Patterns:** Are certain regions showing elevated default rates?
2. **Seasonal Trends:** Do defaults spike during specific months/quarters?
3. **Loan Purpose Analysis:** Deeper dive into medical vs. education loan performance
4. **Customer Lifetime Value:** Compare default risk against potential long-term profitability

---

## Technical Approach

**Tools Used:** Python (pandas, numpy, matplotlib, seaborn, scikit-learn, XGBoost)  
**Dataset:** [Kaggle Credit Risk Dataset](https://www.kaggle.com/datasets/laotse/credit-risk-dataset) (32,581 records, 12 features)

### Data Cleaning & Preprocessing

**Handling Missing Values:**

- **person_emp_length:** 2.7% missing - dropped due to low percentage

<img width="436" alt="missing-values-emp" src="https://github.com/user-attachments/assets/54c639ff-1d5a-4aca-b75a-59f29e97c458">

- **loan_int_rate:** 9.5% missing - imputed based on loan_grade to preserve data quality

<img width="598" alt="missing-values-interest" src="https://github.com/user-attachments/assets/7530985b-458d-4c7c-a218-e668d8c33486">

**Feature Engineering:**

Created **has_income_no_emp** flag to identify self-employed or new workers:

<img width="516" alt="feature-engineering" src="https://github.com/user-attachments/assets/09c86192-309a-4733-9e8f-0d9fec88c6f5">

Binned age groups and income ranges for segment analysis:

<img width="439" alt="age-binning" src="https://github.com/user-attachments/assets/84853079-017b-4836-9ede-c103f89d9bc6">
<img width="723" alt="income-binning" src="https://github.com/user-attachments/assets/33726387-1c3e-48ff-b2aa-82b17fc752d2">

**Outlier Removal:**

Removed unrealistic entries (age 144 years, employment length 123 years):

<img width="419" alt="outlier-removal" src="https://github.com/user-attachments/assets/f7922d9f-3f5f-4b3b-97c7-176bf05f19eb">

**Clean Data Export:**

<img width="307" alt="clean-data-export" src="https://github.com/user-attachments/assets/8eaa4a27-0ecc-48da-b3a6-366d90cf016b">

### Advanced Feature Engineering

Created composite risk score combining multiple financial factors:

<img width="730" alt="composite-risk-score" src="https://github.com/user-attachments/assets/535b2fbd-f646-4622-9e2d-fe9d5ea7daff">

**Risk Score Components:**
- Loan interest rate
- Loan grade  
- Income percentage
- Home ownership status
- Prior default history

All factors mapped to numerical values and scaled using MinMaxScaler, then categorized into four risk levels: Low, Medium, High, and Very High.

### Methodology

1. **Data Cleaning:** Handled 9.5% missing values, removed outliers, engineered 3 new features
2. **Exploratory Analysis:** Univariate, bivariate, and multivariate analysis to identify risk patterns
3. **Threshold Discovery:** Established 31% loan-to-income and 15.4% interest rate thresholds through statistical analysis
4. **Model Building:** Trained and compared 5 classification algorithms (Logistic Regression, Decision Tree, Random Forest, KNN, XGBoost)
5. **Model Selection:** Chose XGBoost (recall) and Random Forest (precision) based on business needs

**Challenges Overcome:**
- Managed imbalanced dataset (78% non-defaulters)
- Avoided bias when imputing missing interest rates
- Validated correlation insights through multivariate analysis

---

## About This Analysis

This project demonstrates how data analytics can directly inform lending strategy and risk management decisions. By focusing on actionable insights rather than just model accuracy, this analysis provides a framework for making data-driven lending decisions that balance risk mitigation with market competitiveness.

**Key Takeaway:** Feature engineering and business-focused analysis are more valuable than complex algorithms. Understanding *why* borrowers default enables better policy decisions than simply predicting *who* will default.

---

**Dataset Source:** [Kaggle Credit Risk Dataset](https://www.kaggle.com/datasets/laotse/credit-risk-dataset)  
