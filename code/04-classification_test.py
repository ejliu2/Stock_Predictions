import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import MinMaxScaler, FunctionTransformer, StandardScaler, PolynomialFeatures
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

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
        output = "output/classification/"+label+" "+data_type+" results.csv"
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
# note: the number of rows in each DataFrame container must equal num_of_splits
# note: change parameters before main loop to alter analysis
###############################################################################
def train_regressors(data_X, data_y, list_of_models, num_of_splits):
    results_template = pd.DataFrame(np.zeros((num_of_splits, len(list_of_models))), columns=list_of_models)
    container = []

    # Logistic Regression Parameters
    lr_c = [0.00001, 0.0001, 0.001, 0.01, 1, 10, 50, 100, 250, 500]
    lr_str = []
    for c in lr_c:
        lr_str.append(str(c))

    # Random Forest Classifier Parameters
    rf_estimators = [100, 500, 1000, 3000]
    rf_min_samples_split = [10, 50, 100, 200]
    rf_parameters = np.array(np.meshgrid(rf_estimators, rf_min_samples_split)).reshape(2, len(rf_estimators) * len(rf_min_samples_split)).T
    # https://stackoverflow.com/questions/12130883/r-expand-grid-function-in-python

    # Neural Network Classifier Parameters
    nn_activiation = "logistic"
    nn_solver = "lbfgs"
    nn_layers = [(1), (5), (10), (5, 2), (12, 10), (10, 5), (10, 5, 1)]
    nn_layers_str = []
    for t in nn_layers:
        nn_layers_str.append(str(t))

    # K Neighbours Classifier Parameters
    k_neighbours = [5, 15, 35, 55, 75, 100, 150, 250]
    k_neighbours_str = []
    for n in k_neighbours:
        k_neighbours_str.append(str(n))

    # Linear SVC Parameters
    svc_c = [0.00001, 0.0001, 0.001, 0.01, 1, 2, 10, 50]
    svc_str = []
    for c in svc_c:
        svc_str.append(str(c))

    # Main Loop to train regressors
    for index, (X, y) in enumerate(zip(data_X, data_y)):
        print("Using dataset:", index + 1, "of", len(data_X))
        container.append(results_template.copy(deep=True))

        # Temporary Containers
        svc_results = pd.DataFrame(np.zeros((len(svc_c), 1)), index=svc_str, columns=["Score"])
        kn_results = pd.DataFrame(np.zeros((len(k_neighbours), 1)), index=k_neighbours_str, columns=["Score"])
        nn_results = pd.DataFrame(np.zeros((len(nn_layers), 1)), index=nn_layers_str, columns=["Score"])
        rf_results = pd.DataFrame(np.zeros((len(rf_parameters), 1)), index=rf_parameters.astype(str), columns=["Score"])
        lr_results = pd.DataFrame(np.zeros((len(lr_c), 1)), index=lr_str, columns=["Score"])

        for curr_run in range(num_of_splits):
            print("Starting Run", curr_run+1, "of", num_of_splits)
            X_train, X_valid, y_train, y_valid = train_test_split(X, y)

            # Naive Bayes
            # if "Naive Bayes" in list_of_models:
            #     print("Running Naive Bayes")
            #     model_nb = GaussianNB()
            #     model_nb.fit(X_train, y_train)
            #     container[index].loc[curr_run, "Naive Bayes"] = round(model_nb.score(X_valid, y_valid), 4)

            # Neural Network
            if "Neural Network" in list_of_models:
                print("Running Neural Network")
                for nn_nodes in nn_layers:
                    model_nn = make_pipeline(
                        MinMaxScaler(),
                        MLPClassifier(hidden_layer_sizes=nn_nodes, activation=nn_activiation, solver=nn_solver, max_iter=1000000)
                    )
                    model_nn.fit(X_train, y_train)
                    nn_results.loc[str(nn_nodes), "Score"] = round(model_nn.score(X_valid, y_valid), 4)
                container[index].loc[curr_run, "Neural Network"] = max(nn_results["Score"])

            # K Neighbours
            if "K Neighbours" in list_of_models:
                print("Running K Neighbours")
                for k in k_neighbours:
                    model_kn = KNeighborsClassifier(k)
                    model_kn.fit(X_train, y_train)
                    kn_results.loc[str(k), "Score"] = round(model_kn.score(X_valid, y_valid), 4)
                container[index].loc[curr_run, "K Neighbours"] = max(kn_results["Score"])

            # Logistic Regression
            # if "Logistic Regression" in list_of_models:
            #     print("Running Logistic Regression")
            #     for c in lr_c:
            #         model_lr = make_pipeline(
            #             StandardScaler(),
            #             LogisticRegression(C=c, solver='lbfgs', max_iter=10000, n_jobs=2)
            #         )
            #         model_lr.fit(X_train, y_train)
            #         lr_results.loc[str(c), "Score"] = round(model_lr.score(X_valid, y_valid), 4)
            #     container[index].loc[curr_run, "Logistic Regression"] = max(lr_results["Score"])

            # Random Forest
            if "Random Forest" in list_of_models:
                print("Running Random Forest")
                for this_rf_parameters in rf_parameters:
                    row_label = "(" + str(this_rf_parameters[0]) + ", " + str(this_rf_parameters[1]) + ")"
                    model_rf = RandomForestClassifier(n_estimators=this_rf_parameters[0], max_depth=None, max_features="sqrt", min_samples_split=this_rf_parameters[1])
                    model_rf.fit(X_train, y_train)
                    rf_results.loc[row_label, "Score"] = round(model_rf.score(X_valid, y_valid), 4)
                print(rf_results)
                container[index].loc[curr_run, "Random Forest"] = max(rf_results["Score"])

            # Linear SVC
            # if "Linear SVC" in list_of_models:
            #     print("Running Linear SVC")
            #     for c in svc_c:
            #         model_svc = make_pipeline(
            #             MinMaxScaler(),
            #             SVC(C=c, cache_size=1000)
            #         )
            #         model_svc.fit(X_train, y_train)
            #         svc_results.loc[str(k), "Score"] = round(model_svc.score(X_valid, y_valid), 4)
            #     container[index].loc[curr_run, "Linear SVC"] = max(svc_results["Score"])

            print("Run", curr_run+1, "Complete")
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
    # data_2014_orig = data_2014_raw.loc[:,subset_to_keep].dropna()
    # data_2015_orig = data_2015_raw.loc[:,subset_to_keep].dropna()
    # data_2016_orig = data_2016_raw.loc[:,subset_to_keep].dropna()
    # data_2017_orig = data_2017_raw.loc[:,subset_to_keep].dropna()
    data_2018_orig = data_2018_raw.loc[:,subset_to_keep].dropna()  # 2018 data is not used as it would require 2019 class as Y, which is what we are trying to predict
    data_2019_orig = data_2019_raw.loc[:,subset_to_keep].dropna()

    # The class in the following year is what to be predicted
    # 2019 data is not included as we do not want to train model with data we want to predict
    # class_2014 = data_2014_orig["Ticker"].to_frame().merge(data_2015_orig[["Ticker","Class"]], on="Ticker", how="inner")
    # class_2015 = data_2015_orig["Ticker"].to_frame().merge(data_2016_orig[["Ticker","Class"]], on="Ticker", how="inner")
    # class_2016 = data_2016_orig["Ticker"].to_frame().merge(data_2017_orig[["Ticker","Class"]], on="Ticker", how="inner")
    # class_2017 = data_2017_orig["Ticker"].to_frame().merge(data_2018_orig[["Ticker","Class"]], on="Ticker", how="inner")
    class_2018 = data_2018_orig["Ticker"].to_frame().merge(data_2019_orig[["Ticker","Class"]], on="Ticker", how="inner")

    # Keep only rows that have data in the following year and change Sector to categorical
    # data_2014 = data_2014_orig.merge(class_2014, on="Ticker", how="inner")
    # data_2015 = data_2015_orig.merge(class_2015, on="Ticker", how="inner")
    # data_2016 = data_2016_orig.merge(class_2016, on="Ticker", how="inner")
    # data_2017 = data_2017_orig.merge(class_2017, on="Ticker", how="inner")
    data_2018 = data_2018_orig.merge(class_2018, on="Ticker", how="inner")
    data_list_all = [data_2018]
    for d in data_list_all:
        d.rename(columns={"Class_y": "Y", "Class_x": "Class"}, inplace=True)
        d["Sector"] = pd.Categorical(d["Sector"])
        d["Sector"] = d["Sector"].cat.codes

    # Training data per year
    # X_2014 = data_2014.loc[:,"Ticker":"Sector"]
    # y_2014 = data_2014["Y"]
    # X_2015 = data_2015.loc[:,"Ticker":"Sector"]
    # y_2015 = data_2015["Y"]
    # X_2016 = data_2016.loc[:,"Ticker":"Sector"]
    # y_2016 = data_2016["Y"]
    # X_2017 = data_2017.loc[:,"Ticker":"Sector"]
    # y_2017 = data_2017["Y"]
    X_2018 = data_2018.loc[:,"Ticker":"Sector"]
    y_2018 = data_2018["Y"]
    X_data_yearly = [X_2018]
    for each_X in X_data_yearly:
        each_X.set_index("Ticker", inplace=True)
    y_data_yearly = [y_2018]

    # Training data combined
    # data_combined = pd.concat([data_2018], join="outer")
    # X = data_combined.loc[:,"Revenue":"Sector"]
    # X["Identifier"] = data_combined["Identifier"]
    # X.set_index("Identifier", inplace=True)
    # y = data_combined["Y"]

    # X_data_combined = [X]
    # y_data_combined = [y]

    # return (X_data_yearly, y_data_yearly, X_data_combined, y_data_combined)
    return (X_data_yearly, y_data_yearly)


###############################################################################
# Function to ETL Regular Data into Percent Data
# Transforms 2 previous years into percentage difference
# Returns Lists of [2015, 2016, 2017] Data as well as those years [Combined] into one
###############################################################################
def ETL_percent_data():
    data_2014_raw, data_2015_raw, data_2016_raw, data_2017_raw, data_2018_raw, data_2019_raw = read_data()

    # Set infinity equal to null since it is possible that we get divide by 0 error (ex. company was not public during the previous year)
    pd.set_option('use_inf_as_na', True)

    # data_2015_raw["Revenue"] = round((data_2015_raw["Revenue"] / data_2014_raw["Revenue"]) - 1, 4)
    # data_2016_raw["Revenue"] = round((data_2016_raw["Revenue"] / data_2015_raw["Revenue"]) - 1, 4)
    data_2017_raw["Revenue"] = round((data_2017_raw["Revenue"] / data_2016_raw["Revenue"]) - 1, 4)
    data_2018_raw["Revenue"] = round((data_2018_raw["Revenue"] / data_2017_raw["Revenue"]) - 1, 4)

    # data_2015_raw.loc[:, "Cost of Revenue":"SG&A Expenses Growth"] = round((data_2015_raw.loc[:,"Cost of Revenue":"SG&A Expenses Growth"] / data_2014_raw.loc[:,"Cost of Revenue":"SG&A Expenses Growth"]) - 1, 4)
    # data_2016_raw.loc[:, "Cost of Revenue":"SG&A Expenses Growth"] = round((data_2016_raw.loc[:,"Cost of Revenue":"SG&A Expenses Growth"] / data_2015_raw.loc[:,"Cost of Revenue":"SG&A Expenses Growth"]) - 1, 4)
    data_2017_raw.loc[:, "Cost of Revenue":"SG&A Expenses Growth"] = round((data_2017_raw.loc[:,"Cost of Revenue":"SG&A Expenses Growth"] / data_2016_raw.loc[:,"Cost of Revenue":"SG&A Expenses Growth"]) - 1, 4)
    data_2018_raw.loc[:, "Cost of Revenue":"SG&A Expenses Growth"] = round((data_2018_raw.loc[:,"Cost of Revenue":"SG&A Expenses Growth"] / data_2017_raw.loc[:,"Cost of Revenue":"SG&A Expenses Growth"]) - 1, 4)

    # data_2015_orig = data_2015_raw.loc[:,subset_to_keep].dropna()
    # data_2016_orig = data_2016_raw.loc[:,subset_to_keep].dropna()
    # data_2017_orig = data_2017_raw.loc[:,subset_to_keep].dropna()
    data_2018_orig = data_2018_raw.loc[:,subset_to_keep].dropna()
    data_2019_orig = data_2019_raw.loc[:,subset_to_keep].dropna()

    # class_2015 = data_2015_orig["Ticker"].to_frame().merge(data_2016_orig[["Ticker","Class"]], on="Ticker", how="inner")
    # class_2016 = data_2016_orig["Ticker"].to_frame().merge(data_2017_orig[["Ticker","Class"]], on="Ticker", how="inner")
    # class_2017 = data_2017_orig["Ticker"].to_frame().merge(data_2018_orig[["Ticker","Class"]], on="Ticker", how="inner")
    class_2018 = data_2018_orig["Ticker"].to_frame().merge(data_2019_orig[["Ticker","Class"]], on="Ticker", how="inner")

    # Keep only rows that have data in the following year and change Sector to categorical
    # data_2015 = data_2015_orig.merge(class_2015, on="Ticker", how="inner")
    # data_2016 = data_2016_orig.merge(class_2016, on="Ticker", how="inner")
    # data_2017 = data_2017_orig.merge(class_2017, on="Ticker", how="inner")
    data_2018 = data_2018_orig.merge(class_2018, on="Ticker", how="inner")
    data_list_all = [data_2018]
    for d in data_list_all:
        d.rename(columns={"Class_y": "Y", "Class_x": "Class"}, inplace=True)
        d["Sector"] = pd.Categorical(d["Sector"])
        d["Sector"] = d["Sector"].cat.codes

    # Training data per year
    # X_2015 = data_2015.loc[:,"Ticker":"Sector"]
    # y_2015 = data_2015["Y"]
    X_2018 = data_2018.loc[:,"Ticker":"Sector"]
    y_2018 = data_2018["Y"]
    # X_2017 = data_2017.loc[:,"Ticker":"Sector"]
    # y_2017 = data_2017["Y"]
    X_data_yearly = [ X_2018]
    for each_X in X_data_yearly:
        each_X.set_index("Ticker", inplace=True)
    y_data_yearly = [y_2018]

    # Training data combined
    # data_combined = pd.concat([data_2016], join="outer")
    # X = data_combined.loc[:,"Revenue":"Sector"]
    # X["Identifier"] = data_combined["Identifier"]
    # X.set_index("Identifier", inplace=True)
    # y = data_combined["Y"]

    # X_data_combined = [X]
    # y_data_combined = [y]

    # return (X_data_yearly, y_data_yearly, X_data_combined, y_data_combined)
    return (X_data_yearly, y_data_yearly)
    


###############################################################################
# Analysis
###############################################################################
def main(data_type, model_names, reps):
    # Setup
    num_of_splits = reps
    models_to_run = model_names

    if data_type == "Regular":
        X_data_yearly, y_data_yearly = ETL_regular_data()
        results_from_yearly_data = train_regressors(X_data_yearly, y_data_yearly, models_to_run, num_of_splits)
        print_df_from_list(results_from_yearly_data, ["2018"], data_type)
        # results_from_combined_data = train_regressors(X_data_combined, y_data_combined, models_to_run, num_of_splits)
        # print_df_from_list(results_from_combined_data, ["Combined"], data_type)

    elif data_type == "Percent":
        X_data_yearly, y_data_yearly = ETL_percent_data()
        results_from_yearly_data = train_regressors(X_data_yearly, y_data_yearly, models_to_run, num_of_splits)
        print_df_from_list(results_from_yearly_data, ["2018"], data_type)
        # results_from_combined_data = train_regressors(X_data_combined, y_data_combined, models_to_run, num_of_splits)
        # print_df_from_list(results_from_combined_data, ["Combined"], data_type)

    else:
        print("Non existant option selected")

    return

if __name__=='__main__':
    data_type = sys.argv[1]  # Regular or Percent
    model_names = ["Naive Bayes", "Neural Network", "Random Forest", "K Neighbours", "Logistic Regression", "Linear SVC"]
    reps = 10

    print("Running Analysis on", data_type, "Data Type")
    main(data_type, model_names, reps)
