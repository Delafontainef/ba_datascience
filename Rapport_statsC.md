# Computational Statistics

**Course:** Statistics Computational  
**Authors:** François Delafontaine,  Paul Micheli & Nassim
**Date:** May 26, 2025

---

# Computational Statistics

**Course:** Statistics Computational  
**Authors:** Paul Micheli & François Delafontaine  
**Date:** May 26, 2025

---
 

## 1. Introduction

### 1.1 Project context and goals

The goal of this project is to understand and predict house sale prices by using a **time-series model** that captures how prices change over time. We focus on building a model that includes **seasonality**, **trends**, and **important house features** as explanatory variables, using the SARIMAX framework.

To keep the model clear and efficient, we carefully select a small number of meaningful features through statistical testing.

At the end of the project, we may also compare our time-based model with a more traditional **linear model** that does not use time information, to see if adding the time dimension improves prediction.

### 1.2 Dataset and problem description

We use the **Ames, Iowa house prices dataset** (`train.csv`), which contains 1,460 house sales and 79 features about each property — such as size, quality, number of rooms, and neighborhood.

Our goal is to model the sale price (`SalePrice`) in a way that:

- Handles **non-normality** and **variability** in the data
- Identifies which features have the most influence on price
- Captures **time-related patterns** in the housing market


## 2. Data Description  
- **Overview of `train.csv`**  
  The CSV file includes 80 columns: an `Id` index, the target `SalePrice`, 38 numeric features (areas, counts, years), 27 nominal/categorical variables (neighborhood, exterior materials, garage types), and 14 ordinal ratings (quality and condition scales).

- **List of numerical vs. categorical features**  
  Numeric features: `LotArea`, `GrLivArea`, `TotalBsmtSF`, etc.  
  Ordinal categorical (encoded later): `ExterQual`, `KitchenQual`, `HeatingQC`, `LandSlope`  
  Nominal categorical: `Neighborhood`

- **Handling of missing values and removal of the `Id` column**  
  We load with `na_filter=False` to preserve `"NA"` as a valid category string and verify no true missing data remains. The `Id` column is dropped immediately, as it serves only as a unique identifier and carries no predictive information.

## 3. Normalization

Before modeling, we normalize the dataset to improve interpretability and performance:

- We apply a **logarithmic transformation** to the target variable `SalePrice` in order to reduce skewness and stabilize variance. This also brings the distribution closer to normality, which is beneficial for linear modeling.
  
- We also apply the same log transform to surface-related variables, such as `GrLivArea`, `TotalBsmtSF`, `GarageArea`, etc. All transformed columns are renamed with a `log_` prefix.

- Additionally, we transform several **ordinal categorical variables** into numeric scales:  
  - `LandSlope` → `n_LandSlope` (scale: 1–3)  
  - `ExterQual`, `HeatingQC`, `KitchenQual` → `n_ExterQual`, `n_HeatingQC`, `n_KitchenQual` (scale: 0–5)  
  These mappings preserve the ordinal structure of the quality ratings and allow us to include them directly in regressions.

This preprocessing step ensures that the model will treat surface measures and quality levels as continuous values, while also simplifying further analysis such as ANOVA and SARIMAX.

## 4. Time Series

Although tackled later in the process, time-series analysis is logically the first step when working with temporally ordered data. We want to know whether sale prices exhibit **seasonality**, **trends**, or **stationarity issues**, and whether we can model these aspects using SARIMAX.

### 4.1 Monthly Aggregation

To convert the raw dataset into a usable time series:

- We combine `YrSold` and `MoSold` into a single `date` column.  
- We transform `Neighborhood` into dummy variables (one-hot encoding) to aggregate their presence across months.  
- We split the dataset into two halves (train/test) before aggregation, which preserves around 50–55 monthly points per set.  
- We then aggregate all values **per month** using the mean (for both numerical and one-hot columns).  

The resulting time series (`df_tr`, `df_te`) now contains one average data point per month, which we can use for modeling and forecasting.

### 4.2 ACF, PACF, and Decomposition

To inspect the structure of the series, we use:

- **ACF (Autocorrelation Function)** and **PACF (Partial ACF)** to detect serial correlation at various lags  
- A **differencing plot** to visually check if the data needs to be differenced (stationarized)  
- **Seasonal decomposition** using `seasonal_decompose` with various periods (2, 3, 4, 12)  

From these diagnostics, we note that:

- The time series does not exhibit strong trend or autocorrelation  
- Some weak seasonal patterns are present—especially at period 4  
- We choose a seasonal period of 4 (likely quarterly), based on visual inspection and weak significance of lags in the ACF/PACF  

This step prepares the ground for SARIMAX modeling in a later section.

## 6. Factor Selection

We now have a time-series representation of the dataset and have identified a seasonality of 4 months. The next step is to define the **exogenous variables** — the features that will be used in our SARIMAX and linear models.

We proceed through several stages:

### 6.1 Visual Exploration

We start with scatter plots between potential predictors and the log-transformed sale price (`log_SalePrice`). This gives us a rough sense of the shape and strength of relationships, and helps discard irrelevant or noisy variables.

The visualizations are performed on the **differenced training data** (`df_shtr`), which reflects monthly changes and mitigates non-stationarity.

### 6.2 Univariate Hypothesis Testing

We then apply **univariate linear regressions** (OLS) for each individual factor against `log_SalePrice`. For categorical variables, we use ANOVA; for numerical ones, we use t-tests.

Only factors with **p-value < 0.01** are retained. The top predictors include:

- `OverallQual`
- `GarageCars`
- `n_ExterQual`
- `n_KitchenQual`
- `YearBuilt`, `YearRemodAdd`
- `TotRmsAbvGrd`, `FullBath`, etc.

This step provides a ranked list of potentially useful variables for modeling.

### 6.3 Multiple Factor ANOVA (Additive Model)

The subset of predictors selected through univariate analysis is then tested in a **multiple linear regression model** with **additive effects**, followed by a **Type II ANOVA**. This step allows us to assess the **marginal contribution of each factor**, adjusting for the presence of the others.

Results show that only two variables remain **strongly significant** when accounting for redundancy and shared variance:

- `OverallQual`  
- `GarageCars`

This supports a more parsimonious model by eliminating predictors that are only conditionally relevant.

### 6.4 Multiple Factor ANOVA with Interactions

To explore potential **interactions between continuous variables**, we fit a second model including **all two-way and higher-order interactions** between the top predictors using a formula of the form: log_SalePrice ~ OverallQual * GarageCars * n_ExterQual 


A Type II ANOVA is then applied. This analysis reveals that the interaction:

- `GarageCars:n_ExterQual`

is **statistically significant (p < 0.01)**, even though `n_ExterQual` alone is not. This suggests a synergistic effect: the impact of garage capacity on sale price could depend on the exterior quality of the house.

### 6.5 k-Factorial ANOVA (Discretized Interaction Validation)

To validate this interaction result through a different lens, we also apply a **k-factorial ANOVA**:

- Each numeric variable is binarized (`low`/`high`) using the median
- A full factorial model (`~ A * B * C...`) is fit
- We retain main effects and interactions with **p-value < 0.05**

This alternative approach confirms the same three significant terms:

- Main effect: `OverallQual`
- Main effect: `GarageCars`
- Interaction: `GarageCars:n_ExterQual`

Thus, both continuous and discretized interaction analyses converge to the same conclusion.

---

### ✅ Final Selection

Across all methods — univariate tests, multiple ANOVA (additive), ANOVA with interactions, and k-factorial — the following terms consistently emerge as important:

- `OverallQual`
- `GarageCars`
- `GarageCars:n_ExterQual` (interaction)

These variables are therefore selected as **exogenous regressors** for the SARIMAX model. We additionally create a new feature `Garage*Exter` representing the interaction term, which is added to both training and test datasets.

This selection process ensures a **robust**, **interpretable**, and **parsimonious** model design.






# 1. Introduction  
- Project context and objectives  
- Dataset description and problem statement

# 2. Data Description  
- Overview of `train.csv`  
- List of numerical vs. categorical features  
- Handling of missing values and removal of the `Id` column

# 3. Preprocessing (Normalization & Encoding)  
- Log‐transform of `SalePrice` and area/SF columns  
- Ordinal encoding of quality variables (`ExterQual`, `HeatingQC`, etc.)  
- Column name cleanup

# 4. Exploratory Analysis & Factor Selection  
## 4.1 Level-based Visualizations  
- Scatter plots of numerical features vs. `log_SalePrice`  
- Boxplots for categorical features (e.g. `Neighborhood`)  
## 4.2 Univariate Hypothesis Tests  
- Simple regressions and t-tests / one-way ANOVA  
- Table of F-statistics and p-values  
## 4.3 2-Level Factorial Design (k-factors)  
- Binarization into “low”/“high” around the median  
- Main effects and interaction analysis  
## 4.4 Multiple ANOVA  
- Multivariate OLS formula  
- Type II ANOVA table and interpretation of each factor’s unique contribution

# 5. Linear Regression Model (OLS)  
- Fitting with the top 5–6 selected factors  
- Coefficient estimates, p-values, and diagnostics (R², VIF, residual plots)  
- MSE/RMSE on training and test sets

# 6. Time Series Analysis  
## 6.1 Series Construction  
- Creating a `date` index from `YrSold`/`MoSold`  
- Monthly aggregation and generation of exogenous variables (neighborhood proportions, averages)  
- Chronological train/test split  
## 6.2 Temporal Diagnostics  
- ACF/PACF and differencing  
- Seasonal decomposition

# 7. SARIMAX Modeling  
- Selecting (p,d,q)(P,D,Q,s) parameters  
- Choosing exogenous series via univariate Δ vs. Δ tests  
- Fitting the SARIMAX model and summary output  
- AIC/BIC comparison and forecasting RMSE

# 8. Model Comparison  
- OLS vs. SARIMAX performance (RMSE, bias)  
- Strengths and weaknesses of each approach  
- Recommended use cases for micro (“per-house”) vs. macro (time-series) models

# 9. Conclusion  
- Key findings  
- Study limitations  
- Future work (Ridge/Lasso, cross-validation, additional features)

# 10. Appendices  
- Full code listings  
- Detailed tables  
- Additional plots  
