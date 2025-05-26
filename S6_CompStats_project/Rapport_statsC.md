# Project - House Prices

**Course:** 3IN1045 - Computational Statistics 
**Authors:** Paul Micheli, Nassim Maliki & François Delafontaine  
**Date:** May 26, 2025

---
 

## 1. Introduction

We have been tasked with offering a model based on [Kaggle's House Prices dataset](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques). For training purposes, we are to use methods from:
- hypothesis testing (and confidence intervals)
- ANOVA and k-factorial
- time series

We will briefly describe the dataset (1.1), including some normalization (1.2), then discuss our plan (1.3) that covers the other parts of this report: time-series (2), feature selection (3) and modelling (4).

### 1.1 Dataset

We use the **Ames, Iowa house prices dataset** that contains 1,460 house sales between 2006 and 2010. The dependent variable is *SalePrice*, the price at which houses were sold; 79 independent variables are available -
 such as area, quality, number of rooms or neighborhood. We will refer to those as *features* from now on.
 
Of those 79 features, 38 are numeric (areas, counts, years) and 41 are categorical: 27 nominal (neighborhood, exterior materials, etc.) and 14 ordinal (quality, condition, etc.). 

The 'NA' value seems to be a valid one for some categorical features. Therefore, when loading our data, we did not automatically remove them (`na_filter=False`). We have verified that no actual NaN value exists in the 
training dataset. We also dropped the `Id` column as it will never be used here.

## 1.2. Normalization

We have been warned that `SalePrice` does not have a normal distribution, which causes the kurtosis to deviate from the norm. To avoid this, we applied a **logarithmic transformation**. 
We initially applied that transformation to other numeric, continuous variables before realizing that this turned those features into logarithmic ones (which is not their relation); only `SalePrice` is still modified
 and renamed into `log_SalePrice`.

Additionally, we transform several **ordinal categorical variables** into numeric scales: `LandSlope`, `ExterQual`, `HeatingQC` and `KitchenQual`, each renamed with an `n_` prefix. These mappings preserve the ordinal 
structure of the features while allowing us to include them directly in our model without resorting to dummy variables.

## 1.3. Planning

We handle the task in three steps:
1. Time series,
2. Feature selection and
3. Modelling.

Time series is about checking whether our data is actually a time series, meaning not just a temporally-ordered dataset but also with some auto-regression effect. At this step we are meant to format our dataset into 
a series where there is at least one and only one datapoint per time step (in our case every month). Missing datapoints are simply assumed equal to the previous one while a time step with several datapoints aggregates 
them. With 1460 datapoints and 55 months, we faced the second case. Therefore, we aggregated the datapoints (2.1), then checked for **trend** and **seasonality** (2.2). If none is found, given a continuous numeric variable we 
would turn to a linear regression; if there is some **lag**, we would instead opt for a SARIMAX. This also decides what dataset (aggregated or not) we use for feature selection.

Feature selection is about parsing the available features to select significant ones. Instead of training every possible combination possible we test what features are in relation with `log_SalePrice`. To do so we first 
opted for a visualization (3.1), then confirmed it using **univariate linear regressions** (OLS) (3.2). We then used a **multivariate linear regression** with **additive effects**, a **Type 2 ANOVA** (3.3) and a 
**k-factorial** to further select and also include interactions (3.4). 

Finally, modelling is about training a model (4.1), checking its residuals (4.2) and evaluating it (4.3).

### 2. Time series

Our initial dataset has two columns `YrSold` and `MoSold` that, when combined, allow us to build a single `date` column. Even then, that dataset is still not a time series as each time step has several datapoints. We 
aggregate them (2.1) then check for **trend** and **seasonality** (2.2). 

### 2.1 Aggregation

Aggregation is done by averaging all values for `log_SalePrice` and features alike. This turns discrete numericals into continuous features and cannot be done for categoricals: those must be turned into dummy variables, 
as is done for `Neighborhood`: the value of each neighborhood is then how many houses were from that location as a percentage. 

Before aggregating, we take advantage of the fact that 1460 datapoints will be turned into 55 to generate a separate testing/evaluation dataset. To do so, we split the original dataset in two by randomly assigning the 
datapoints. When aggregating, both split datasets should end up with 50-55 datapoints. 

## 2.2. Trend & Seasonality

To inspect the structure of the series, we use:
- **ACF (Autocorrelation Function)** and **PACF (Partial ACF)** to detect serial correlation at various lags  
- A **differencing plot** to visually check if the data has any pattern over time
- **Seasonal decomposition** using `seasonal_decompose` with various periods (2, 3, 4, 12)

**ACF** and **PACF** showed no significant auto-correlation: only lag-4 was outside the norm, but was not visually decisive. So we resorted to **Seasonal decomposition** on a 12-months period that did reveal a pattern. 
However, it would also show us a pattern at other periods (2, 3, 4, etc.). So, while we would still test for a lag-12 seasonality, given our **ACF/PACF** we assumed that we had a lag-4 seasonality with no trend. As a 
result, our model would be SARIMAX.

## 3. Feature Selection

For a time series, the next step is to define the **exogenous variables** - for a linear regression, this would be the features of our model. We have opted for several steps:
- a visualization (3.1)
- a univariate linear regression (3.2)
- a multivariate linear regression & Type 2 ANOVA (3.3)
- a k-factorial (3.4)

While those may be redundant, we wanted to confirm our results to ensure we were applying those methods correctly.

### 3.1. Visualization

The simplest approach was to get a series of scatterplots, one per feature in relation with `log_SalePrice`. This gives us a rough sense of the shape and strenght of those relations ships and helps discard irrelevant or 
noisy variables. 

We generated those scatterplots on our aggregated **time series** and added a lag-4 shift to it to try and reduce any auto-regression effect in the scatterplots. For a normal linear regression, we would use the 1460 
original datapoints. We won't discuss the result of the visualization as the following points (3.2-3.4) end up doing the selection. 

### 3.2 Univariate Hypothesis Testing

For each feature we applied a **univariate linear regression** (OLS) against `log_SalePrice`. For categorical variables, we use ANOVA; for numerical ones, we use t-tests. Only factors with a 99% confidence (**p-value < 0.01**)
 are retained. The top predictors include:
- `OverallQual`
- `GarageCars`
- `n_ExterQual`
- `n_KitchenQual`
- `YearBuilt`, `YearRemodAdd`
- `TotRmsAbvGrd`, `FullBath`, etc.

This roughly confirms our visualization and also provides a ranked list of features. We know want to take into account their interaction, to verify their significance and add those interactions to our model if the case arises.

### 3.3 Multiple Factor ANOVA (Additive Model)

The subset of predictors selected through univariate analysis is tested in a **multiple linear regression model** with **additive effects**, followed by a **Type II ANOVA**. 
This step allows us to assess the **marginal contribution of each factor**, adjusting for the presence of the others.

Results show that only two variables remain **strongly significant** when accounting for redundancy and shared variance:
- `OverallQual`  
- `GarageCars`

This result is why we wanted to ensure our results were accurate, as most predictors are eliminated, including neigbhorhoods. But before that, we wanted to obtain the interactions. For that purpose,
 we fit a second model including **all two-way and higher-order interactions** 
between the top predictors using a formula of the form: `log_SalePrice ~ OverallQual * GarageCars * n_ExterQual` A Type II ANOVA is then applied. This analysis reveals that the interaction:
- `GarageCars:n_ExterQual`

is **statistically significant (p < 0.01)**, even though `n_ExterQual` alone is not.

## 3.4. k-factorial

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

## 4. Modelling

Given our aggregated time series and its exogenous variables, we now have to train a SARIMAX model (4.1), check its residuals (4.2) and evaluate it (4.3).

### 4.1. Training

We still aren't sure of the parameters of our SARIMAX. Therefore, we opt for testing most combinations of:
- Exogenous variables
- ARIMA parameters (trend)
- SARIMA parameters (seasonality)

We won't list all options exhaustively but iterated over them and selected the best model based on the lowest AIC score. We added warnings to our process, the most important of them being
if all coefficients were significant. The final model was:
- SARIMAX(0,0,0)\*(0,1,1,4) with `OverallQual` and `GarageCars`

With all coefficients significant, an AIC of near -390, no heteroskedasticity, no multicolinearity but a bit of kurtosis: the kurtosis should be at ~3.: ours is at 4.09. 

### 4.2. Residuals

We still have to verify our model's assumptions, namely if the residuals are normally distributed. Our focus will be especially on any remaining auto-regression effect. To do so, we plot:
- **The residuals over time**
- **A histogram of their distribution**
- **A QQ plot of quantiles**
- **The ACF and PACF of residuals**

Everything shows that our residuals are normally distributed. What kurtosis is left (skewing the Jarque-Bera test) should be due to exogenous variables.

## 4.3 Evaluation

All that remains is to evaluate the accuracy of our SARIMAX model. To do so we use the dataset we split before aggregation, predict the sale prices and use a mean square error (for a continuous value) as measurement.
This returns a value of 2.1e-5 or close to zero, suggesting a fairly good model.

## Conclusion

The hardest parts are aggregation for the time series (2.1) and feature selection (3) where categorical variables especially prove hard to handle. We also encountered a bit of difficulty in interpreting our ACF/PACF in (2.1)
as we expected a trend (due to inflation) as well as a sudden change (due to the 2008 economic crisis that, according to Kaggle's participants, manifests itself in 2010 in the data). We did not expect a lag-4, or quaterly 
cycle, nor did we expect the features, with earlier tests suggesting up to 9 significant, to plummet to 2. 

We would also struggle a bit to make use of the model we just built: we can predict data once aggregated but Kaggle's competition expects us to predict values for a raw dataset. This is perfectly doable but does take 
some accustomation. In general, if simulating linear regressions is routine, simulating trends and seasonality does require a good understanding of AR and MA principles. 