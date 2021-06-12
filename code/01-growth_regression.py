import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import MinMaxScaler, FunctionTransformer, StandardScaler, PolynomialFeatures
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingRegressor

###############################################################################
# Attributes to Keep
# Based on the data that could be found on https://www.msn.com/en-ca/money/markets and also in 2019 data set
###############################################################################
subset_to_keep = [
    "Ticker",
    "Revenue",
    "Revenue Growth",
    "Cost of Revenue",
    "Gross Profit",
    "SG&A Expense",
    # "R&D Expenses",
    "Operating Expenses",
    "Operating Income",
    "Net Income",
    # "Dividend per Share",
    "EPS",
    "EPS Diluted",
    "Cash and cash equivalents",
    "Receivables",
    # "Inventories",
    "Property, Plant & Equipment Net",
    # "Goodwill and Intangible Assets",
    "Total assets",
    "Payables",
    # "Total debt",
    "Total liabilities",
    # "Deferred revenue",
    # "Tax Liabilities",
    # "Other comprehensive income",
    "Retained earnings (deficit)",
    "Total shareholders equity",
    "Depreciation & Amortization",
    # "Stock-based compensation",
    "Operating Cash Flow",
    "Capital Expenditure",
    # "Acquisitions and disposals",
    # "Investment purchases and sales",
    "Investing Cash flow",
    # "Issuance (repayment) of debt",
    # "Issuance (buybacks) of shares",
    # "Dividend payments",
    "Financing Cash Flow",
    # "Effect of forex changes on cash",
    "Net cash flow / Change in cash",
    "Free Cash Flow",
    # "PE ratio",
    # "Debt to Equity",
    # "Interest Coverage",
    # "Dividend Yield",
    # "Payout Ratio",
    "ROE",
    "Sector",
    "Next Year Price Var",
    "Class",
    "Year",
    "Identifier"
]


###############################################################################
# Function to print a list of dataframes
###############################################################################
def print_df_from_list(dataframe, labels, data_type):
    pd.set_option('display.max_columns', 100)  # or 1000
    pd.set_option('display.max_rows', 100)  # or 1000
    # pd.set_option('display.max_colwidth', -1)  # or 199
    for i, (df, label) in enumerate(zip(dataframe, labels)):
        print("Printing results for:", label)
        print(df)
        output = "output/growth/"+label+"_results.csv"
        df.to_csv(output, index=False, header=True)
    return


###############################################################################
# Function to import data and make initial changes
###############################################################################
def read_data():
    file1 = "train/2014_Financial_Data.csv"
    file2 = "train/2015_Financial_Data.csv"
    file3 = "train/2016_Financial_Data.csv"
    file4 = "train/2017_Financial_Data.csv"
    file5 = "train/2018_Financial_Data.csv"
    file6 = "test/2019_Financial_Data.csv"

    data_2014_raw = pd.read_csv(file1)
    data_2014_raw["Year"] = "2014"
    data_2014_raw.rename(columns={"2015 PRICE VAR [%]":"Next Year Price Var"}, inplace=True)
    data_2014_raw["Identifier"] = data_2014_raw["Ticker"] + "-" + data_2014_raw["Year"]

    data_2015_raw = pd.read_csv(file2)
    data_2015_raw["Year"] = "2015"
    data_2015_raw.rename(columns={"2016 PRICE VAR [%]":"Next Year Price Var"}, inplace=True)
    data_2015_raw["Identifier"] = data_2015_raw["Ticker"] + "-" + data_2015_raw["Year"]

    data_2016_raw = pd.read_csv(file3)
    data_2016_raw["Year"] = "2016"
    data_2016_raw.rename(columns={"2017 PRICE VAR [%]":"Next Year Price Var"}, inplace=True)
    data_2016_raw["Identifier"] = data_2016_raw["Ticker"] + "-" + data_2016_raw["Year"]

    data_2017_raw = pd.read_csv(file4)
    data_2017_raw["Year"] = "2017"
    data_2017_raw.rename(columns={"2018 PRICE VAR [%]":"Next Year Price Var"}, inplace=True)
    data_2017_raw["Identifier"] = data_2017_raw["Ticker"] + "-" + data_2017_raw["Year"]

    data_2018_raw = pd.read_csv(file5)
    data_2018_raw["Year"] = "2018"
    data_2018_raw.rename(columns={"2019 PRICE VAR [%]":"Next Year Price Var"}, inplace=True)
    data_2018_raw["Identifier"] = data_2018_raw["Ticker"] + "-" + data_2018_raw["Year"]

    data_2019_raw = pd.read_csv(file6)
    data_2019_raw["Year"] = "2019"
    data_2019_raw.rename(columns={"2019 PRICE VAR [%]":"Next Year Price Var"}, inplace=True)
    data_2019_raw["Identifier"] = data_2019_raw["Ticker"] + "-" + data_2019_raw["Year"]
    data_2019_raw = data_2019_raw.merge(data_2018_raw.loc[:,["Sector", "Ticker"]], on="Ticker", how="left")

    return (data_2014_raw, data_2015_raw, data_2016_raw, data_2017_raw, data_2018_raw, data_2019_raw)


###############################################################################
# Function to train regression models to predict growth rates
# @param data_X - a List of DataFrames with the X data
# @param data_y - a List of DataFrames with the y data
# @param container - a List of DataFrames as a container to hold results
# @param list_of_models - a List of model names (strings)
# @param num_of_splits - a number of times to retrain models
# @param labels - a list of labels for use in naming
# note: the number of rows in each DataFrame container must equal num_of_splits
# note: change parameters before main loop to alter analysis
###############################################################################
def train_regressors(data_X, data_y, list_of_models, num_of_splits, X_test, y_test, labels):
    results_template = pd.DataFrame(np.zeros((num_of_splits, len(list_of_models))), columns=list_of_models)
    container = []
    sentinel = -1

    # Polynomial Regression Parameters
    poly_degree = [1, 2, 4, 8]
    poly_degree_str = []
    for t in poly_degree:
        poly_degree_str.append(str(t))

    # Random Forest Regressor Parameters
    rf_estimators = [500]
    rf_depth = [5, 10]
    rf_min_samples = [10]
    rf_parameters = np.array(np.meshgrid(rf_estimators, rf_depth, rf_min_samples)).reshape(3, len(rf_estimators) * len(rf_depth) * len(rf_min_samples)).T
    # https://stackoverflow.com/questions/12130883/r-expand-grid-function-in-python

    # Neural Network Regressor Parameters
    nn_activiation = "logistic"
    nn_solver = "lbfgs"
    nn_layers = [(1), (5), (10), (5, 1), (10, 5), (10, 5, 1)]
    nn_layers_str = []
    for t in nn_layers:
        nn_layers_str.append(str(t))

    # K Means Regressor Parameters
    km_neighbours = range(1, 502, 10)
    km_neighbours_str = []
    for n in km_neighbours:
        km_neighbours_str.append(str(n))

    # Boosting Regressor
    boost_learning_rate = [1, 10, 100]
    boost_estimators = [1000, 5000]
    boost_depth = [1, 2, 4, 8]
    boost_parameters = np.array(np.meshgrid(boost_estimators, boost_depth, boost_learning_rate)).reshape(3, len(boost_estimators) * len(boost_depth) * len(boost_learning_rate)).T

    # Main Loop to train regressors
    for index, (X, y) in enumerate(zip(data_X, data_y)):

        best_score = {
            "Polynomial": sentinel,
            "Random Forest": sentinel,
            "Neural Network": sentinel,
            "KMeans": sentinel,
            "Linear Regression": sentinel,
            "Boosting": sentinel
        }

        best_model = {
            "Polynomial": sentinel,
            "Random Forest": sentinel,
            "Neural Network": sentinel,
            "KMeans": sentinel,
            "Linear Regression": sentinel,
            "Boosting": sentinel
        }

        print("Using dataset:", index + 1, "of", len(data_X))
        container.append(results_template.copy(deep=True))

        # Temporary Containers
        km_results = pd.DataFrame(np.zeros((len(km_neighbours), 1)), index=km_neighbours_str, columns=["Score"])
        nn_results = pd.DataFrame(np.zeros((len(nn_layers), 1)), index=nn_layers_str, columns=["Score"])
        rf_results = pd.DataFrame(np.zeros((len(rf_parameters), 1)), index=rf_parameters.astype(str), columns=["Score"])
        pr_results = pd.DataFrame(np.zeros((len(poly_degree), 1)), index=poly_degree_str, columns=["Score"])
        gb_results = pd.DataFrame(np.zeros((len(boost_parameters), 1)), index=boost_parameters.astype(str), columns=["Score"])

        for curr_run in range(num_of_splits):
            print("Starting Run", curr_run+1, "of", num_of_splits)
            X_train, X_valid, y_train, y_valid = train_test_split(X, y)

            # Linear Regression
            if "Linear Regression" in list_of_models:
                print("Running Linear Regression")
                model_lr = LinearRegression(fit_intercept=True)
                model_lr.fit(X_train, y_train)
                score_lr = round(model_lr.score(X_valid, y_valid), 4)
                container[index].loc[curr_run, "Linear Regression"] = score_lr
                if (score_lr > best_score["Linear Regression"]):
                    best_model["Linear Regression"] = model_lr

            # Neural Network
            if "Neural Network" in list_of_models:
                print("Running Neural Network")
                for nn_nodes in nn_layers:
                    model_nn = make_pipeline(
                        MinMaxScaler(),
                        MLPRegressor(hidden_layer_sizes=nn_nodes, activation=nn_activiation, solver=nn_solver, max_iter=1000000)
                    )
                    model_nn.fit(X_train, y_train)
                    score_nn = round(model_nn.score(X_valid, y_valid), 4)
                    nn_results.loc[str(nn_nodes), "Score"] = score_nn
                    if (score_nn > best_score["Neural Network"]):
                        best_model["Neural Network"] = model_nn
                container[index].loc[curr_run, "Neural Network"] = max(nn_results["Score"])

            # K Means
            if "KMeans" in list_of_models:
                print("Running KMeans")
                for k in km_neighbours:
                    model_km = KNeighborsRegressor(k)
                    model_km.fit(X_train, y_train)
                    score_km = round(model_km.score(X_valid, y_valid), 4)
                    km_results.loc[str(k), "Score"] = score_km
                    if (score_km > best_score["KMeans"]):
                        best_model["KMeans"] = model_km
                container[index].loc[curr_run, "KMeans"] = max(km_results["Score"])

            # Polynomial
            if "Polynomial" in list_of_models:
                print("Running Polynomial")
                for n in poly_degree:
                    model_pr = make_pipeline(
                        PolynomialFeatures(degree=n, include_bias=True),
                        LinearRegression(fit_intercept=True)
                    )
                    model_pr.fit(X_train, y_train)
                    score_pr = round(model_pr.score(X_valid, y_valid), 4)
                    pr_results.loc[str(n), "Score"] = score_pr
                    if (score_pr > best_score["Polynomial"]):
                        best_model["Polynomial"] = model_pr
                container[index].loc[curr_run, "Polynomial"] = max(pr_results["Score"])

            # Random Forest
            if "Random Forest" in list_of_models:
                print("Running Random Forest")
                for this_rf_parameters in rf_parameters:
                    row_label = "(" + str(this_rf_parameters[0]) + ", " + str(this_rf_parameters[1]) + ", " + str(this_rf_parameters[2]) + ")"
                    model_rf = RandomForestRegressor(n_estimators=this_rf_parameters[0], max_depth=this_rf_parameters[1], max_features="sqrt", min_samples_leaf=this_rf_parameters[2])
                    model_rf.fit(X_train, y_train)
                    score_rf = round(model_rf.score(X_valid, y_valid), 4)
                    rf_results.loc[row_label, "Score"] = score_rf
                    if (score_rf > best_score["Random Forest"]):
                        best_model["Random Forest"] = model_rf
                container[index].loc[curr_run, "Random Forest"] = max(rf_results["Score"])

            # Boosting
            if "Boosting" in list_of_models:
                print("Running Boosting")
                for this_boost_parameters in boost_parameters:
                    row_label = "(" + str(this_boost_parameters[0]) + ", " + str(this_boost_parameters[1]) + ", " + str(this_boost_parameters[2]) + ")"
                    model_gb = GradientBoostingRegressor(n_estimators=this_boost_parameters[0], max_depth=this_boost_parameters[1], learning_rate=(this_boost_parameters[2]/1000))
                    model_gb.fit(X_train, y_train)
                    score_gb = round(model_gb.score(X_valid, y_valid), 4)
                    gb_results.loc[row_label, "Score"] = score_gb
                    if (score_gb > best_score["Boosting"]):
                        best_model["Boosting"] = model_gb
                container[index].loc[curr_run, "Boosting"] = max(rf_results["Score"])

            print("Run", curr_run+1, "Complete")

        num_of_rows = len(X_test[0].index)
        test_data = X_test[0]
        if "Linear Regression" in list_of_models and best_model["Linear Regression"] != sentinel:
            lr_predictions = pd.DataFrame(np.zeros((num_of_rows, 2)), columns=["Ticker", "Predicted Growth"])
            lr_predictions["Ticker"] = test_data.index
            lr_predictions.set_index("Ticker", inplace=True)
            lr_predictions["Predicted Growth"] = best_model["Linear Regression"].predict(test_data)
            output = "output/growth/" + labels[index] + "_Linear_Regression_Predictions.csv"
            lr_predictions.to_csv(output)

        if "KMeans" in list_of_models and best_model["KMeans"] != sentinel:
            km_predictions = pd.DataFrame(np.zeros((num_of_rows, 2)), columns=["Ticker", "Predicted Growth"])
            km_predictions["Ticker"] = test_data.index
            km_predictions.set_index("Ticker", inplace=True)
            km_predictions["Predicted Growth"] = best_model["KMeans"].predict(test_data)
            output = "output/growth/" + labels[index] + "_KMeans_Predictions.csv"
            km_predictions.to_csv(output)

        if "Neural Network" in list_of_models and best_model["Neural Network"] != sentinel:
            nn_predictions = pd.DataFrame(np.zeros((num_of_rows, 2)), columns=["Ticker", "Predicted Growth"])
            nn_predictions["Ticker"] = test_data.index
            nn_predictions.set_index("Ticker", inplace=True)
            nn_predictions["Predicted Growth"] = best_model["Neural Network"].predict(test_data)
            output = "output/growth/" + labels[index] + "_Neural_Network_Predictions.csv"
            nn_predictions.to_csv(output)

        if "Random Forest" in list_of_models and best_model["Random Forest"] != sentinel:
            rf_predictions = pd.DataFrame(np.zeros((num_of_rows, 2)), columns=["Ticker", "Predicted Growth"])
            rf_predictions["Ticker"] = test_data.index
            rf_predictions.set_index("Ticker", inplace=True)
            rf_predictions["Predicted Growth"] = best_model["Random Forest"].predict(test_data)
            output = "output/growth/" + labels[index] + "_Random_Forest_Predictions.csv"
            rf_predictions.to_csv(output)

        if "Boosting" in list_of_models and best_model["Boosting"] != sentinel:
            gb_predictions = pd.DataFrame(np.zeros((num_of_rows, 2)), columns=["Ticker", "Predicted Growth"])
            gb_predictions["Ticker"] = test_data.index
            gb_predictions.set_index("Ticker", inplace=True)
            gb_predictions["Predicted Growth"] = best_model["Boosting"].predict(test_data)
            output = "output/growth/" + labels[index] + "_Boosting_Predictions.csv"
            gb_predictions.to_csv(output)

        if "Polynomial" in list_of_models and best_model["Polynomial"] != sentinel:
            pr_predictions = pd.DataFrame(np.zeros((num_of_rows, 2)), columns=["Ticker", "Predicted Growth"])
            pr_predictions["Ticker"] = test_data.index
            pr_predictions.set_index("Ticker", inplace=True)
            pr_predictions["Predicted Growth"] = best_model["Polynomial"].predict(test_data)
            output = "output/growth/" + labels[index] + "_Polynomial_Predictions.csv"
            pr_predictions.to_csv(output)

        print("Analysis of Dataset", index+1, "Complete")
    print("All Done")
    return container


###############################################################################
# Function to ETL Regular Data
# Returns Lists of [2014, 2015, 2016, 2017] Data as well as those years [Combined] into one
###############################################################################
def ETL_regular_data():
    data_2014_raw, data_2015_raw, data_2016_raw, data_2017_raw, data_2018_raw, data_2019_raw = read_data()

    # Keep only the most common data points and drop all rows with null values
    data_2014_orig = data_2014_raw.loc[:,subset_to_keep].dropna()
    data_2015_orig = data_2015_raw.loc[:,subset_to_keep].dropna()
    data_2016_orig = data_2016_raw.loc[:,subset_to_keep].dropna()
    data_2017_orig = data_2017_raw.loc[:,subset_to_keep].dropna()
    data_2018_orig = data_2018_raw.loc[:,subset_to_keep].dropna()  # 2018 data is not used as it would require 2019 revenue growth rate as Y, which is what we are trying to predict
    data_2019_orig = data_2019_raw.loc[:,subset_to_keep].dropna()

    # The revenue growth in the following year is what to be predicted
    # 2019 data is included but only for testing later
    rev_growth_2014 = data_2014_orig["Ticker"].to_frame().merge(data_2015_orig[["Ticker","Revenue Growth"]], on="Ticker", how="inner")
    rev_growth_2015 = data_2015_orig["Ticker"].to_frame().merge(data_2016_orig[["Ticker","Revenue Growth"]], on="Ticker", how="inner")
    rev_growth_2016 = data_2016_orig["Ticker"].to_frame().merge(data_2017_orig[["Ticker","Revenue Growth"]], on="Ticker", how="inner")
    rev_growth_2017 = data_2017_orig["Ticker"].to_frame().merge(data_2018_orig[["Ticker","Revenue Growth"]], on="Ticker", how="inner")
    rev_growth_2018 = data_2018_orig["Ticker"].to_frame().merge(data_2019_orig[["Ticker","Revenue Growth"]], on="Ticker", how="inner")

    # Keep only rows that have data in the following year and change Sector to categorical
    data_2014 = data_2014_orig.merge(rev_growth_2014, on="Ticker", how="inner")
    data_2015 = data_2015_orig.merge(rev_growth_2015, on="Ticker", how="inner")
    data_2016 = data_2016_orig.merge(rev_growth_2016, on="Ticker", how="inner")
    data_2017 = data_2017_orig.merge(rev_growth_2017, on="Ticker", how="inner")
    data_2018 = data_2018_orig.merge(rev_growth_2018, on="Ticker", how="inner")
    data_list_all = [data_2014, data_2015, data_2016, data_2017, data_2018]
    for d in data_list_all:
        d.rename(columns={"Revenue Growth_y": "Y", "Revenue Growth_x": "Revenue Growth"}, inplace=True)
        d["Sector"] = pd.Categorical(d["Sector"])
        d["Sector"] = d["Sector"].cat.codes

    # Training data per year
    X_2014 = data_2014.loc[:,"Ticker":"Sector"]
    y_2014 = data_2014["Y"]
    X_2015 = data_2015.loc[:,"Ticker":"Sector"]
    y_2015 = data_2015["Y"]
    X_2016 = data_2016.loc[:,"Ticker":"Sector"]
    y_2016 = data_2016["Y"]
    X_2017 = data_2017.loc[:,"Ticker":"Sector"]
    y_2017 = data_2017["Y"]
    X_data_yearly = [X_2014, X_2015, X_2016, X_2017]
    for each_X in X_data_yearly:
        each_X.set_index("Ticker", inplace=True)
    y_data_yearly = [y_2014, y_2015, y_2016, y_2017]

    # Test data
    X_2018 = data_2018.loc[:,"Ticker":"Sector"]
    y_2018 = data_2018["Y"]
    X_2018.set_index("Ticker", inplace=True)
    X_test = [X_2018]
    y_test = [y_2018]

    # Training data combined
    data_combined = pd.concat([data_2014, data_2015, data_2016, data_2017], join="outer")
    X = data_combined.loc[:,"Revenue":"Sector"]
    X["Identifier"] = data_combined["Identifier"]
    X.set_index("Identifier", inplace=True)
    y = data_combined["Y"]

    X_data_combined = [X]
    y_data_combined = [y]

    return (X_data_yearly, y_data_yearly, X_data_combined, y_data_combined, X_test, y_test)


###############################################################################
# Function to ETL Regular Data into Percent Data
# Transforms 2 previous years into percentage difference
# Returns Lists of [2015, 2016, 2017] Data as well as those years [Combined] into one
###############################################################################
def ETL_percent_data():
    data_2014_raw, data_2015_raw, data_2016_raw, data_2017_raw, data_2018_raw, data_2019_raw = read_data()

    # Set infinity equal to null since it is possible that we get divide by 0 error (ex. company was not public during the previous year)
    pd.set_option('use_inf_as_na', True)

    data_2015_temp = data_2015_raw.merge(data_2014_raw, on=["Ticker"], suffixes=("", "_2"))
    data_2016_temp = data_2016_raw.merge(data_2015_raw, on=["Ticker"], suffixes=("", "_2"))
    data_2017_temp = data_2017_raw.merge(data_2016_raw, on=["Ticker"], suffixes=("", "_2"))
    data_2018_temp = data_2018_raw.merge(data_2017_raw, on=["Ticker"], suffixes=("", "_2"))

    data_2015_temp["Revenue"] = (data_2015_temp["Revenue"] / data_2015_temp["Revenue_2"]) - 1
    data_2016_temp["Revenue"] = (data_2016_temp["Revenue"] / data_2016_temp["Revenue_2"]) - 1
    data_2017_temp["Revenue"] = (data_2017_temp["Revenue"] / data_2017_temp["Revenue_2"]) - 1
    data_2018_temp["Revenue"] = (data_2018_temp["Revenue"] / data_2018_temp["Revenue_2"]) - 1  # Test data

    for i in range(3, 222, 1):
        data_2015_temp.iloc[:, i] = (data_2015_temp.iloc[:, i] / data_2015_temp.iloc[:,i+226]) - 1
        data_2016_temp.iloc[:, i] = (data_2016_temp.iloc[:, i] / data_2016_temp.iloc[:,i+226]) - 1
        data_2017_temp.iloc[:, i] = (data_2017_temp.iloc[:, i] / data_2017_temp.iloc[:,i+226]) - 1
        data_2018_temp.iloc[:, i] = (data_2018_temp.iloc[:, i] / data_2018_temp.iloc[:,i+226]) - 1

    data_2015_orig = data_2015_temp.loc[:,subset_to_keep].dropna()
    data_2016_orig = data_2016_temp.loc[:,subset_to_keep].dropna()
    data_2017_orig = data_2017_temp.loc[:,subset_to_keep].dropna()
    data_2018_orig = data_2018_temp.loc[:,subset_to_keep].dropna()  # 2018 data is used for testing only as it would require 2019 revenue growth rate as Y, which is what we are trying to predict
    data_2019_orig = data_2019_raw.loc[:,subset_to_keep].dropna()

    rev_growth_2015 = data_2015_orig["Ticker"].to_frame().merge(data_2016_orig[["Ticker","Revenue Growth"]], on="Ticker", how="inner")
    rev_growth_2016 = data_2016_orig["Ticker"].to_frame().merge(data_2017_orig[["Ticker","Revenue Growth"]], on="Ticker", how="inner")
    rev_growth_2017 = data_2017_orig["Ticker"].to_frame().merge(data_2018_orig[["Ticker","Revenue Growth"]], on="Ticker", how="inner")
    rev_growth_2018 = data_2018_orig["Ticker"].to_frame().merge(data_2019_orig[["Ticker","Revenue Growth"]], on="Ticker", how="inner")

    # Keep only rows that have data in the following year and change Sector to categorical
    data_2015 = data_2015_orig.merge(rev_growth_2015, on="Ticker", how="inner")
    data_2016 = data_2016_orig.merge(rev_growth_2016, on="Ticker", how="inner")
    data_2017 = data_2017_orig.merge(rev_growth_2017, on="Ticker", how="inner")
    data_2018 = data_2018_orig.merge(rev_growth_2018, on="Ticker", how="inner")
    data_list_all = [data_2015, data_2016, data_2017, data_2018]
    for d in data_list_all:
        d.rename(columns={"Revenue Growth_y": "Y", "Revenue Growth_x": "Revenue Growth"}, inplace=True)
        d["Sector"] = pd.Categorical(d["Sector"])
        d["Sector"] = d["Sector"].cat.codes
        d = d.drop(columns=["Revenue Growth"])

    # Training data per year
    X_2015 = data_2015.loc[:,"Ticker":"Sector"]
    y_2015 = data_2015["Y"]
    X_2016 = data_2016.loc[:,"Ticker":"Sector"]
    y_2016 = data_2016["Y"]
    X_2017 = data_2017.loc[:,"Ticker":"Sector"]
    y_2017 = data_2017["Y"]
    X_data_yearly = [X_2015, X_2016, X_2017]
    for each_X in X_data_yearly:
        each_X.set_index("Ticker", inplace=True)
    y_data_yearly = [y_2015, y_2016, y_2017]

    # Test data
    X_2018 = data_2018.loc[:,"Ticker":"Sector"]
    y_2018 = data_2018["Y"]
    X_2018.set_index("Ticker", inplace=True)
    X_test = [X_2018]
    y_test = [y_2018]

    # Training data combined
    data_combined = pd.concat([data_2015, data_2016, data_2017], join="outer")
    X = data_combined.loc[:,"Revenue":"Sector"]
    X["Identifier"] = data_combined["Identifier"]
    X.set_index("Identifier", inplace=True)
    y = data_combined["Y"]

    X_data_combined = [X]
    y_data_combined = [y]

    return (X_data_yearly, y_data_yearly, X_data_combined, y_data_combined, X_test, y_test)


###############################################################################
# Function to ETL Regular Data into Percent Data
# Transforms 2 previous years into percentage difference
# Returns Lists of [2015, 2016, 2017] Data as well as those years [Combined] into one
###############################################################################
def ETL_percent_with_classifier_data(threshold, balanced = False):
    X_data_yearly, y_data_yearly, X_data_combined, y_data_combined, X_test, y_test = ETL_percent_data()

    count_true = 0
    count_false = 0
    for X in X_data_yearly:
        X["Growth Type"] = np.where(X["Revenue Growth"] >= threshold, 1, 0)
        count_true = X[X["Growth Type"] == 1].count()["Growth Type"]
        count_false = X[X["Growth Type"] == 0].count()["Growth Type"]
        if balanced and (count_true < count_false):
            first = X[X["Growth Type"] == 1]
            second = X[X["Growth Type"] == 0].sample(count_true)
            X = pd.concat([first, second], join="inner")
        X["Growth Type"] = pd.Categorical(X["Growth Type"])
        X["Growth Type"] = X["Growth Type"].cat.codes

    for X in X_data_combined:
        X["Growth Type"] = np.where(X["Revenue Growth"] >= threshold, 1, 0)
        count_true = X[X["Growth Type"] == 1].count()["Growth Type"]
        count_false = X[X["Growth Type"] == 0].count()["Growth Type"]
        if balanced and (count_true < count_false):
            first = X[X["Growth Type"] == 1]
            second = X[X["Growth Type"] == 0].sample(count_true)
            X = pd.concat([first, second], join="inner")
        X["Growth Type"] = pd.Categorical(X["Growth Type"])
        X["Growth Type"] = X["Growth Type"].cat.codes

    for X in X_test:
        X["Growth Type"] = np.where(X["Revenue Growth"] >= threshold, 1, 0)
        count_true = X[X["Growth Type"] == 1].count()["Growth Type"]
        count_false = X[X["Growth Type"] == 0].count()["Growth Type"]
        if balanced and (count_true < count_false):
            first = X[X["Growth Type"] == 1]
            second = X[X["Growth Type"] == 0].sample(count_true)
            X = pd.concat([first, second], join="inner")
        X["Growth Type"] = pd.Categorical(X["Growth Type"])
        X["Growth Type"] = X["Growth Type"].cat.codes

    return (X_data_yearly, y_data_yearly, X_data_combined, y_data_combined, X_test, y_test)


###############################################################################
# Analysis
###############################################################################
def main(data_type, model_names, reps, growth = 1):
    # Setup
    num_of_splits = reps
    models_to_run = model_names

    if data_type == "Regular":
        yearly_labels = ["2014_Regular", "2015_Regular", "2016_Regular", "2017_Regular"]
        combined_labels = ["Combined_Regular"]
        X_data_yearly, y_data_yearly, X_data_combined, y_data_combined, X_test, y_test = ETL_regular_data()
        results_from_yearly_data = train_regressors(X_data_yearly, y_data_yearly, models_to_run, num_of_splits, X_test, y_test, yearly_labels)
        print_df_from_list(results_from_yearly_data, yearly_labels, data_type)
        results_from_combined_data = train_regressors(X_data_combined, y_data_combined, models_to_run, num_of_splits, X_test, y_test, combined_labels)
        print_df_from_list(results_from_combined_data, combined_labels, data_type)

    elif data_type == "Percent":
        yearly_labels = ["2015_Percent", "2016_Percent", "2017_Percent"]
        combined_labels = ["Combined_Percent"]
        X_data_yearly, y_data_yearly, X_data_combined, y_data_combined, X_test, y_test = ETL_percent_data()
        results_from_yearly_data = train_regressors(X_data_yearly, y_data_yearly, models_to_run, num_of_splits, X_test, y_test, yearly_labels)
        print_df_from_list(results_from_yearly_data, yearly_labels, data_type)
        results_from_combined_data = train_regressors(X_data_combined, y_data_combined, models_to_run, num_of_splits, X_test, y_test, combined_labels)
        print_df_from_list(results_from_combined_data, combined_labels, data_type)

    elif data_type == "Classifier":
        data_type = "Growth_" + str(int(round(growth*100, 0)))
        yearly_labels = ["2015_"+data_type, "2016_"+data_type, "2017_"+data_type]
        combined_labels = ["Combined_"+data_type]
        X_data_yearly, y_data_yearly, X_data_combined, y_data_combined, X_test, y_test = ETL_percent_with_classifier_data(growth, False)
        results_from_yearly_data = train_regressors(X_data_yearly, y_data_yearly, models_to_run, num_of_splits, X_test, y_test, yearly_labels)
        print_df_from_list(results_from_yearly_data, yearly_labels, data_type)
        results_from_combined_data = train_regressors(X_data_combined, y_data_combined, models_to_run, num_of_splits, X_test, y_test, combined_labels)
        print_df_from_list(results_from_combined_data, combined_labels, data_type)

    elif data_type == "Balanced":
        data_type = "Balanced_" + str(int(round(growth*100, 0)))
        yearly_labels = ["2015_"+data_type, "2016_"+data_type, "2017_"+data_type]
        combined_labels = ["Combined_"+data_type]
        X_data_yearly, y_data_yearly, X_data_combined, y_data_combined, X_test, y_test = ETL_percent_with_classifier_data(growth, True)
        results_from_yearly_data = train_regressors(X_data_yearly, y_data_yearly, models_to_run, num_of_splits, X_test, y_test, yearly_labels)
        print_df_from_list(results_from_yearly_data, yearly_labels, data_type)
        results_from_combined_data = train_regressors(X_data_combined, y_data_combined, models_to_run, num_of_splits, X_test, y_test, combined_labels)
        print_df_from_list(results_from_combined_data, combined_labels, data_type)


    else:
        print("Non existant option selected")

    return

if __name__=='__main__':
    for test in ["Classifier", "Balanced", "Percent", "Regular"]:  # Regular or Percent or Classifier
        data_type = test
        model_names = ["Random Forest"] # ["Linear Regression", "Neural Network", "Random Forest", "KMeans", "Polynomial", "Boosting"]
        reps = 5
        if test == "Classifier":
            for amount in [0.2, 0.3, 0.4]:  # Any percentage
                print("Running Analysis on", data_type, "Data Type")
                main(data_type, model_names, reps, amount)
        elif test == "Balanced":
            for amount in [0.2, 0.3, 0.4]:  # Any percentage
                print("Running Analysis on", data_type, "Data Type")
                main(data_type, model_names, reps, amount)
        else:
            main(data_type, model_names, reps)

