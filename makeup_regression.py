# %%



# ============================================================================
# Regression Overview — In-Class Live Coding Example
# Dataset: Facebook Performance Metrics (UCI ML Repo ID 368)
# Topics:
#   1. Kernel Density Plot
#   2. Dummy Variables
#   3. Regression Without an Intercept in sklearn
#   4. Multivariate Regression
#   5. Log and arcsinh Transformations
#   6. Dummy Variable Trap (drop one level as reference)
#   7. Polynomial Features from sklearn
#   8. True vs. Predicted Plot with Train/Test Split
# =============================================================================

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ucimlrepo import fetch_ucirepo
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score

# %%
# --- Load Dataset ---
facebook_metrics = fetch_ucirepo(id=368)

X = facebook_metrics.data.features.copy()
y = facebook_metrics.data.targets.copy()

# %% 
# look at the info
X.info()


# %%
# Combine for easier exploration, using concat to keep features and target together
df = pd.concat([X, y], axis=1)
df.head()


# =============================================================================
# SECTION 1: Kernel Density Plot
# =============================================================================
# A kernel density plot (KDE) is a smoothed version of a histogram.
# It estimates the probability density function of a continuous variable.
# Use it to understand the shape and spread of a distribution before modeling.
# %%
# The raw distribution is heavily right-skewed — a common problem in regression.
df['Total Interactions'].plot.kde(color='blue')
plt.title('Kernel Density Plot of Total Interactions')
plt.show()

# %%

# Let's also look at Page total likes
df['Page total likes'].plot.kde(color='pink')
plt.title('Kernel Density Plot of Page total likes')
plt.show()

# KEY POINT: Skewed distributions can violate regression assumptions.
# We'll address this with log/arcsinh transformations in Section 5.

# =============================================================================
# SECTION 2: Dummy Variables
# =============================================================================
# Categorical variables cannot be fed directly into sklearn regression models.
# We convert them to binary (0/1) indicator columns — called dummy variables.
# pd.get_dummies() does this automatically.

# %%

# Value counts
print(df.value_counts())

# %% 
help(pd.get_dummies)

# %%
# One-hot encode 'Type' and 'Category' (creates new columns for each level), 
# replace in the df, using pandas's get_dummies, four attributes, df, columns to encode, 
# drop_first=True to avoid dummy variable trap, and prefix to add a prefix to the new columns
df = pd.get_dummies(data=df, columns=['Type', 'Category'], drop_first=True, prefix=['Ty', 'Cat'])
                

# =============================================================================
# SECTION 3: Regression WITHOUT an Intercept in sklearn
# =============================================================================
# By default, LinearRegression fits: y = b0 + b1*x1 + b2*x2 + ...
# Setting fit_intercept=False removes b0, forcing the line through the origin:
#   y = b1*x1 + b2*x2 + ...
# This is rarely appropriate unless theory demands it, but it's useful to know.

# %%
df.info()

# %%
# Simple example: predict Total Interactions from Page total likes
# do the one feature we want to use - convert to a 2D array 
X_simple = df['Page total likes'].values.reshape(-1,1)
y_target = df['Total Interactions']

# %%
# With intercept (default), fit.intercept=true/false, (then).fit
model_with = LinearRegression(fit_intercept=True).fit(X_simple,y_target)
# Without intercept
model_without = LinearRegression(fit_intercept=False).fit(X_simple, y_target)

# %%
print(f'With intercept coefficient: {model_with.coef_[0]:.4f}, intercept: {model_with.intercept_:.2f}, R squared: {model_with.score(X_simple, y_target):.4f}')
print(f'Without intercept coefficient: {model_without.coef_[0]:.4f}, R squared: {model_without.score(X_simple, y_target):.4f}')

# %%

# KEY POINT: Unless your domain knowledge justifies it, always keep the intercept.
# Forcing through the origin biases the slope estimate when y != 0 at x=0.

# =============================================================================
# SECTION 4: Multivariate Regression
# =============================================================================
# Simple regression: one predictor     → y = b0 + b1*x1
# Multivariate regression: many predictors → y = b0 + b1*x1 + b2*x2 + ... + bn*xn
#
# Each coefficient tells you the expected change in y for a 1-unit change in
# that predictor, HOLDING ALL OTHER PREDICTORS CONSTANT (ceteris paribus).
# This is the key advantage over simple regression — we can isolate effects.

# %%
# Select a handful of numeric features, that are most correlated with Total Interactions
# correlation matrix

corr_matrix = df.corr()
corr_with_target = corr_matrix['Total Interactions'].abs().sort_values(ascending=False)

# %%
# select some kinda middle of the road features
numeric_features = corr_with_target[5:11].index.tolist()  # Exclude the target variable itself   
# %%
# visualize the correlations with a matrix plot
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix[numeric_features + ['Total Interactions']], annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Matrix')
plt.show()

# now you try pick some real terrible variables and see what happens

# %% 
corr_with_target.head()

# %%
df_mv = df[numeric_features + ['Total Interactions']].dropna()

X_mv = df_mv[numeric_features]
y_mv = df_mv['Total Interactions']

# %%
model_mv = LinearRegression().fit(X_mv, y_mv)


coef_df = pd.DataFrame({'Feature': numeric_features, 'Coefficient': model_mv.coef_})
print(coef_df.to_string(index=False))

print(f"\nIntercept: {model_mv.intercept_:.2f}")
print(f"R²: {model_mv.score(X_mv, y_mv):.4f}")

# KEY POINT: R² tells us the fraction of variance in y explained by the model.
# A low R² here suggests the linear numeric features alone are not very informative.

# =============================================================================
# SECTION 5: Log and arcsinh Transformations of Feature Variables
# =============================================================================
# Why transform?
#   - Compress extreme values, improving linearity
#   - Reduces the influence of outliers
#   - Can improve model fit and residual normality
#
# log(x):    works only for strictly positive values (x > 0)
# arcsinh(x): works for zero and negative values — a generalization of log
#             arcsinh(x) ≈ log(2x) for large x, but handles 0s gracefully

# %%
# Look at the raw distribution of Page total likes in a scatter plot against Total Interactions
plt.scatter(df['Page total likes'], df['Total Interactions'], alpha=0.5, edgecolors='steelblue', facecolors='none')
plt.xlabel('Page Total Likes')

# %%
# now convert to log
df['log_page_likes'] = np.log(df['Page total likes'] + 1)  # add 1 to avoid log(0)
# histogram of log_page_likes
df['log_page_likes'].plot.hist(color='steelblue', alpha=0.7)

# %%
# histogram of arcsinh_page_likes
df['arcsinh_page_likes'] = np.arcsinh(df['Page total likes'])
df['arcsinh_page_likes'].plot.hist(color='coral', alpha=0.7)

# %%
# histogram of original page total likes
df['Page total likes'].plot.hist(color='gray', alpha=0.7)

# %%
# rerun the multivariate regression with the arcsinh transformation
features = ['arcsinh_page_likes', 'Post Month', 'Post Weekday', 'Post Hour', 'Paid']
X_trans = df[features].dropna()
# use the same target variable, but drop rows with missing features
y_trans = df.loc[X_trans.index, 'Total Interactions']
model_trans = LinearRegression().fit(X_trans, y_trans)

# %%
coef_df_trans = pd.DataFrame({'Feature': features, 'Coefficient': model_trans.coef_})
print(coef_df_trans.to_string(index=False))

print(f"\nIntercept: {model_trans.intercept_:.2f}")
print(f"R²: {model_trans.score(X_trans, y_trans):.4f}")

# =============================================================================
# SECTION 6: Polynomial Features from sklearn
# =============================================================================
# Linear regression assumes a straight-line relationship between X and y.
# If the true relationship is curved, we can add polynomial terms:
#   y = b0 + b1*x + b2*x² + b3*x³ + ...
#
# PolynomialFeatures() generates all polynomial and interaction terms
# up to the specified degree automatically.

# %%
from sklearn.preprocessing import PolynomialFeatures

# Use a single feature to illustrate the polynomial expansion clearly
X_poly_base = df[['log_page_likes']].dropna()
y_poly      = df.loc[X_poly_base.index, 'Total Interactions']

# Degree 2: adds x and x²
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X_poly_base)

print("Original shape:", X_poly_base.shape)
print("Polynomial (degree=2) shape:", X_poly.shape)
print("Feature names:", poly.get_feature_names_out())

# %%
# Fit linear, degree-2, and degree-3 models on the same feature
X_train_p, X_test_p, y_train_p, y_test_p = train_test_split(
    X_poly_base, y_poly, test_size=0.2, random_state=42
)

results = {}
for degree in [1, 2, 3]:
    pf  = PolynomialFeatures(degree=degree, include_bias=False)
    Xtr = pf.fit_transform(X_train_p)
    Xte = pf.transform(X_test_p)
    m   = LinearRegression().fit(Xtr, y_train_p)
    r2  = m.score(Xte, y_test_p)
    results[f'degree_{degree}'] = r2
    print(f"Degree {degree}  |  Test R²: {r2:.4f}")

# KEY POINT: Higher degree is not always better — watch for overfitting.
# Use train/test split (or cross-validation) to compare out-of-sample performance.

# =============================================================================
# SECTION 8: True vs. Predicted Values Plot with Train/Test Split
# =============================================================================
# The standard sklearn workflow:
#   1. Split data into train and test sets
#   2. Fit the model on training data only
#   3. Predict on test data
#   4. Evaluate on test data (unseen by the model)
#
# A true vs. predicted scatter plot is a quick visual diagnostic:
#   - Perfect model → points fall on the 45-degree line (y = x)
#   - Systematic deviations → bias or missing non-linearity

# %%
# Use the same features as the multivariate regression listed in the numeric feature list
X_final = df[numeric_features].dropna()
y_final = df.loc[X_final.index, 'Total Interactions']
X_train, X_test, y_train, y_test = train_test_split(
    X_final, y_final, test_size=0.2, random_state=42
)
model_final = LinearRegression().fit(X_train, y_train)
y_pred = model_final.predict(X_test)    
# True vs. Predicted plot
plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('True Values')
plt.ylabel('Predicted Values')
plt.title('True vs. Predicted Values')
plt.show()


# %%
# calculate and print evaluation metrics, include RSME and R²

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse:.2f}")
print(f"Root Mean Squared Error: {rmse:.2f}")
print(f"R² Score: {r2:.4f}")

# range of the target variable in the test set
print(f"Range of Total Interactions in Test Set: {y_test.min()} to {y_test.max()}")

# %%
