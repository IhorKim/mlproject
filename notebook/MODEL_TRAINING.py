import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from catboost import CatBoostRegressor
from xgboost import XGBRegressor

df = pd.read_csv("data/students.csv")

# preparing X and y variables
X = df.drop(columns=["math score"], axis=1)

# exploring data for numerical and categorical features
print("Categories in 'gender' variable: ", end=" ")
print(df['gender'].unique())
print("Categories in 'race_ethnicity' variable: ", end=" ")
print(df['race/ethnicity'].unique())
print("Categories in 'parental level of education' variable: ", end=" ")
print(df['parental level of education'].unique())
print("Categories in 'lunch' variable: ", end=" ")
print(df['lunch'].unique())
print("Categories in 'test preparation course' variable: ", end=" ")
print(df['test preparation course'].unique())

y = df["math score"]

# IMPORTANT! create Column Transformer with 3 types of transformers
num_features = X.select_dtypes(exclude="object").columns
cat_features = X.select_dtypes(include="object").columns

numeric_transformer = StandardScaler()
oh_transformer = OneHotEncoder()

preprocessor = ColumnTransformer(
    [
        ("OneHotEncoder", oh_transformer, cat_features),
        ("StandardScaler", numeric_transformer, num_features),
    ]
)

X = preprocessor.fit_transform(X)
print(X)

# divide dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train.shape, X_test.shape


# create and evaluate function to give all metrics after model training
def evaluate_model(true, predicted):
    mae = mean_absolute_error(true, predicted)
    mse = mean_squared_error(true, predicted)
    rmse = np.sqrt(mse)
    r2_square = r2_score(true, predicted)
    return mae, rmse, r2_square


models = {
    "Linear Regression": LinearRegression(),
    "Lasso": Lasso(),
    "Ridge": Ridge(),
    "K-Neighbors Regressor": KNeighborsRegressor(),
    "Decision Tree": DecisionTreeRegressor(),
    "Random Forest Regressor": RandomForestRegressor(),
    "XGBRegressor": XGBRegressor(),
    "CatBoosting Regressor": CatBoostRegressor(verbose=False),
    "AdaBoost Regressor": AdaBoostRegressor()
}
model_list = []
r2_list = []

for i in range(len(list(models))):
    model = list(models.values())[i]
    model.fit(X_train, y_train)  # train the model

    # make predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # evaluate train and test dataset
    model_train_mae, model_train_rmse, model_train_r2 = evaluate_model(y_train, y_train_pred)
    model_test_mae, model_test_rmse, model_test_r2 = evaluate_model(y_test, y_test_pred)

    print(list(models.keys())[i])
    model_list.append(list(models.keys())[i])

    print("Model performance for Training set")
    print(" Root Mean Squared Error: {:.4f}".format(model_train_rmse))
    print(" Mean Absolute Error: {:.4f}".format(model_train_mae))
    print(" R2 Score: {:.4f}".format(model_train_r2))
    print("\n")

    print("Model performance for Test set")
    print(" Root Mean Squared Error: {:.4f}".format(model_test_rmse))
    print(" Mean Absolute Error: {:.4f}".format(model_test_mae))
    print(" R2 Score: {:.4f}".format(model_test_r2))
    r2_list.append(model_test_r2)
    print("\n")

# show sorted results
r2_score_results = pd.DataFrame(list(zip(model_list, r2_list)), columns=["Model name", "R2_score"]).sort_values(
    by=["R2_score"], ascending=False)
print(r2_score_results)
print('\n')

# build Linear Regression model
lin_model = LinearRegression(fit_intercept=True)
lin_model = lin_model.fit(X_train, y_train)
y_pred = lin_model.predict(X_test)
score = r2_score(y_test, y_pred) * 100
print("Accuracy of the model is %.2f" % score)

# plot y_pred and y_test
plt.scatter(y_test, y_pred)
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.show()

# difference between actual and predicted values
pred_df = pd.DataFrame({"Actual value": y_test, "Predicted value": y_pred, "Difference": y_test - y_pred})
print(pred_df)
