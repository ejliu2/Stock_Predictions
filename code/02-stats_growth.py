import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
from scipy import stats
from sklearn.metrics import mean_squared_error
from statsmodels.stats.multicomp import pairwise_tukeyhsd

###############################################################################
# Analysis
###############################################################################
def main():
    model = "Random_Forest"

    growth_20_2015 = pd.read_csv("output/growth/2015_Growth_20_"+model+"_Predictions.csv", header=0, names=["Row", "Predicted Growth"])
    growth_20_2016 = pd.read_csv("output/growth/2016_Growth_20_"+model+"_Predictions.csv", header=0, names=["Row", "Predicted Growth"])
    growth_20_2017 = pd.read_csv("output/growth/2017_Growth_20_"+model+"_Predictions.csv", header=0, names=["Row", "Predicted Growth"])
    growth_20_combined = pd.read_csv("output/growth/Combined_Growth_20_"+model+"_Predictions.csv", header=0, names=["Row", "Predicted Growth"])
    growth_20_2015["Type"] = "growth_20_2015"
    growth_20_2016["Type"] = "growth_20_2016"
    growth_20_2017["Type"] = "growth_20_2017"
    growth_20_combined["Type"] = "growth_20_combined"

    growth_30_2015 = pd.read_csv("output/growth/2015_Growth_30_"+model+"_Predictions.csv", header=0, names=["Row", "Predicted Growth"])
    growth_30_2016 = pd.read_csv("output/growth/2016_Growth_30_"+model+"_Predictions.csv", header=0, names=["Row", "Predicted Growth"])
    growth_30_2017 = pd.read_csv("output/growth/2017_Growth_30_"+model+"_Predictions.csv", header=0, names=["Row", "Predicted Growth"])
    growth_30_combined = pd.read_csv("output/growth/Combined_Growth_30_"+model+"_Predictions.csv", header=0, names=["Row", "Predicted Growth"])
    growth_30_2015["Type"] = "growth_30_2015"
    growth_30_2016["Type"] = "growth_30_2016"
    growth_30_2017["Type"] = "growth_30_2017"
    growth_30_combined["Type"] = "growth_30_combined"

    growth_40_2015 = pd.read_csv("output/growth/2015_Growth_40_"+model+"_Predictions.csv", header=0, names=["Row", "Predicted Growth"])
    growth_40_2016 = pd.read_csv("output/growth/2016_Growth_40_"+model+"_Predictions.csv", header=0, names=["Row", "Predicted Growth"])
    growth_40_2017 = pd.read_csv("output/growth/2017_Growth_40_"+model+"_Predictions.csv", header=0, names=["Row", "Predicted Growth"])
    growth_40_combined = pd.read_csv("output/growth/Combined_Growth_40_"+model+"_Predictions.csv", header=0, names=["Row", "Predicted Growth"])
    growth_40_2015["Type"] = "growth_40_2015"
    growth_40_2016["Type"] = "growth_40_2016"
    growth_40_2017["Type"] = "growth_40_2017"
    growth_40_combined["Type"] = "growth_40_combined"

    balanced_20_2015 = pd.read_csv("output/growth/2015_Balanced_20_"+model+"_Predictions.csv", header=0, names=["Row", "Predicted Growth"])
    balanced_20_2016 = pd.read_csv("output/growth/2016_Balanced_20_"+model+"_Predictions.csv", header=0, names=["Row", "Predicted Growth"])
    balanced_20_2017 = pd.read_csv("output/growth/2017_Balanced_20_"+model+"_Predictions.csv", header=0, names=["Row", "Predicted Growth"])
    balanced_20_combined = pd.read_csv("output/growth/Combined_Balanced_20_"+model+"_Predictions.csv", header=0, names=["Row", "Predicted Growth"])
    balanced_20_2015["Type"] = "balanced_20_2015"
    balanced_20_2016["Type"] = "balanced_20_2016"
    balanced_20_2017["Type"] = "balanced_20_2017"
    balanced_20_combined["Type"] = "balanced_20_combined"

    balanced_30_2015 = pd.read_csv("output/growth/2015_Balanced_30_"+model+"_Predictions.csv", header=0, names=["Row", "Predicted Growth"])
    balanced_30_2016 = pd.read_csv("output/growth/2016_Balanced_30_"+model+"_Predictions.csv", header=0, names=["Row", "Predicted Growth"])
    balanced_30_2017 = pd.read_csv("output/growth/2017_Balanced_30_"+model+"_Predictions.csv", header=0, names=["Row", "Predicted Growth"])
    balanced_30_combined = pd.read_csv("output/growth/Combined_Balanced_30_"+model+"_Predictions.csv", header=0, names=["Row", "Predicted Growth"])
    balanced_30_2015["Type"] = "balanced_30_2015"
    balanced_30_2016["Type"] = "balanced_30_2016"
    balanced_30_2017["Type"] = "balanced_30_2017"
    balanced_30_combined["Type"] = "balanced_30_combined"

    balanced_40_2015 = pd.read_csv("output/growth/2015_Balanced_40_"+model+"_Predictions.csv", header=0, names=["Row", "Predicted Growth"])
    balanced_40_2016 = pd.read_csv("output/growth/2016_Balanced_40_"+model+"_Predictions.csv", header=0, names=["Row", "Predicted Growth"])
    balanced_40_2017 = pd.read_csv("output/growth/2017_Balanced_40_"+model+"_Predictions.csv", header=0, names=["Row", "Predicted Growth"])
    balanced_40_combined = pd.read_csv("output/growth/Combined_Balanced_40_"+model+"_Predictions.csv", header=0, names=["Row", "Predicted Growth"])
    balanced_40_2015["Type"] = "balanced_40_2015"
    balanced_40_2016["Type"] = "balanced_40_2016"
    balanced_40_2017["Type"] = "balanced_40_2017"
    balanced_40_combined["Type"] = "balanced_40_combined"

    percent_2015 = pd.read_csv("output/growth/2015_Percent_"+model+"_Predictions.csv", header=0, names=["Row", "Predicted Growth"])
    percent_2016 = pd.read_csv("output/growth/2016_Percent_"+model+"_Predictions.csv", header=0, names=["Row", "Predicted Growth"])
    percent_2017 = pd.read_csv("output/growth/2017_Percent_"+model+"_Predictions.csv", header=0, names=["Row", "Predicted Growth"])
    percent_combined = pd.read_csv("output/growth/Combined_Percent_"+model+"_Predictions.csv", header=0, names=["Row", "Predicted Growth"])
    percent_2015["Type"] = "percent_2015"
    percent_2016["Type"] = "percent_2016"
    percent_2017["Type"] = "percent_2017"
    percent_combined["Type"] = "percent_combined"

    regular_2015 = pd.read_csv("output/growth/2015_Regular_"+model+"_Predictions.csv", header=0, names=["Row", "Predicted Growth"])
    regular_2016 = pd.read_csv("output/growth/2015_Regular_"+model+"_Predictions.csv", header=0, names=["Row", "Predicted Growth"])
    regular_2017 = pd.read_csv("output/growth/2017_Regular_"+model+"_Predictions.csv", header=0, names=["Row", "Predicted Growth"])
    regular_combined = pd.read_csv("output/growth/Combined_Regular_"+model+"_Predictions.csv", header=0, names=["Row", "Predicted Growth"])
    regular_2015["Type"] = "regular_2015"
    regular_2016["Type"] = "regular_2016"
    regular_2017["Type"] = "regular_2017"
    regular_combined["Type"] = "regular_combined"

    original_data_2019 = pd.read_csv("test/2019_Financial_Data.csv")
    data_2019 = original_data_2019.loc[:,"Ticker":"Revenue Growth"].drop(columns=["Revenue"])
    data_2019.rename(columns={"Ticker": "Row"}, inplace=True)
    original_data_2019 = data_2019[data_2019["Row"].isin(regular_2017["Row"])]
    restricted_data_2019 = data_2019[data_2019["Row"].isin(growth_20_2017["Row"])]

    # Histogram Results on Predicted Growth
    hist_data = [growth_20_2017, growth_30_2017, growth_40_2017, balanced_20_2017, balanced_30_2017, balanced_40_2017, percent_2017, regular_2017]
    hist_name = ["Growth 20% 2017","Growth 30% 2017", "Growth 40% 2017", "Balanced 20% 2017", "Balanced 30% 2017", "Balanced 40% 2017", "Percent 2017", "Regular 2017"]
    for i, (d, name) in enumerate(zip(hist_data, hist_name)):
        plt.subplot(3, 3, i+1)
        plt.title(name)
        n_bins = np.arange(-1, 2.5, 0.1)
        plt.hist(d["Predicted Growth"], bins = n_bins)
        axes = plt.gca()
        axes.set_xlim(-1, 2)
    plt.savefig("output/pics/histograms.png")
    plt.clf()

    # Histograms Results (Square Root) on Predicted Growth
    hist_data = [growth_20_2017, growth_30_2017, growth_40_2017, balanced_20_2017, balanced_30_2017, balanced_40_2017, percent_2017, regular_2017]
    hist_name = ["Growth 20% 2017","Growth 30% 2017", "Growth 40% 2017", "Balanced 20% 2017", "Balanced 30% 2017", "Balanced 40% 2017", "Percent 2017", "Regular 2017"]
    for i, (d, name) in enumerate(zip(hist_data, hist_name)):
        plt.subplot(3, 3, i+1)
        plt.title(name)
        n_bins = np.arange(-1, 2.5, 0.1)
        plt.hist(np.sqrt(d["Predicted Growth"]), bins = n_bins)
        axes = plt.gca()
        axes.set_xlim(-1, 2)
    plt.savefig("output/pics/histograms_sqrt.png")
    plt.clf()

    # Normal Tests
    normal_data = [growth_20_2017, growth_30_2017, growth_40_2017, balanced_20_2017, balanced_30_2017, balanced_40_2017, percent_2017, regular_2017]
    normal_name = ["Growth 20% 2017","Growth 30% 2017", "Growth 40% 2017", "Balanced 20% 2017", "Balanced 30% 2017", "Balanced 40% 2017", "Percent 2017", "Regular 2017"]
    for i, (d, name) in enumerate(zip(normal_data, normal_name)):
        print("Normal test on " + name + ": " + str(stats.normaltest(d["Predicted Growth"]).pvalue))

    # ANOVA
    # print("ANOVA 2015:", stats.f_oneway(
    #     growth_20_2015["Predicted Growth"], growth_30_2015["Predicted Growth"], growth_40_2015["Predicted Growth"],
    #     balanced_20_2015["Predicted Growth"], balanced_30_2015["Predicted Growth"], balanced_40_2015["Predicted Growth"],
    #     percent_2015["Predicted Growth"], regular_2015["Predicted Growth"]
    # ).pvalue)

    # print("ANOVA 2016:", stats.f_oneway(
    #     growth_20_2016["Predicted Growth"], growth_30_2016["Predicted Growth"], growth_40_2016["Predicted Growth"],
    #     balanced_20_2016["Predicted Growth"], balanced_30_2016["Predicted Growth"], balanced_40_2016["Predicted Growth"],
    #     percent_2016["Predicted Growth"], regular_2016["Predicted Growth"]
    # ).pvalue)

    # print("ANOVA 2017:", stats.f_oneway(
    #     growth_20_2017["Predicted Growth"], growth_30_2017["Predicted Growth"], growth_40_2017["Predicted Growth"],
    #     balanced_20_2017["Predicted Growth"], balanced_30_2017["Predicted Growth"], balanced_40_2017["Predicted Growth"],
    #     percent_2017["Predicted Growth"], regular_2017["Predicted Growth"]
    # ).pvalue)

    # print("ANOVA Combined:", stats.f_oneway(
    #     growth_20_combined["Predicted Growth"], growth_30_combined["Predicted Growth"], growth_40_combined["Predicted Growth"],
    #     balanced_20_combined["Predicted Growth"], balanced_30_combined["Predicted Growth"], balanced_40_combined["Predicted Growth"],
    #     percent_combined["Predicted Growth"], regular_combined["Predicted Growth"]
    # ).pvalue)

    # Mannwhitenyu
    myu_data = [growth_20_2017, growth_30_2017, growth_40_2017, balanced_20_2017, balanced_30_2017, balanced_40_2017, percent_2017, regular_2017]
    myu_name = ["Growth 20% 2017","Growth 30% 2017", "Growth 40% 2017", "Balanced 20% 2017", "Balanced 30% 2017", "Balanced 40% 2017", "Percent 2017", "Regular 2017"]
    for i, (d, m) in enumerate(zip(myu_data, myu_name)):
        for j, (e, n) in enumerate(zip(myu_data, myu_name)):
            if i == j:
                continue
            print("Mannwhitenyu Test between: " + m + " and " + n + ": " + str((stats.mannwhitneyu(d["Predicted Growth"], e["Predicted Growth"])).pvalue))

    # MSE
    mse_data = [growth_20_2017, growth_30_2017, growth_40_2017, balanced_20_2017, balanced_30_2017, balanced_40_2017, percent_2017]
    mse_name = ["Growth 20% 2017","Growth 30% 2017", "Growth 40% 2017", "Balanced 20% 2017", "Balanced 30% 2017", "Balanced 40% 2017", "Percent 2017"]
    for i, (d, n) in enumerate(zip(mse_data, mse_name)):
        print("MSE - " + n + ": " + str(mean_squared_error(restricted_data_2019["Revenue Growth"], d["Predicted Growth"])))
    print("MSE - Regular Data: " + str(mean_squared_error(original_data_2019["Revenue Growth"], regular_2017["Predicted Growth"])))

    # HSD - mostly for visualization since data is not normal
    hsd = pd.concat([
        growth_20_2015, growth_30_2015, growth_40_2015, balanced_20_2015, balanced_30_2015, balanced_40_2015, percent_2015, regular_2015,
        growth_20_2016, growth_30_2016, growth_40_2016, balanced_20_2016, balanced_30_2016, balanced_40_2016, percent_2016, regular_2016,
        growth_20_2017, growth_30_2017, growth_40_2017, balanced_20_2017, balanced_30_2017, balanced_40_2017, percent_2017, regular_2017,
        growth_20_combined, growth_30_combined, growth_40_combined, balanced_20_combined, balanced_30_combined, balanced_40_combined, percent_combined, regular_combined
    ], join="inner").drop(columns=["Row"])
    hsd_result = pairwise_tukeyhsd(hsd["Predicted Growth"], hsd["Type"], alpha=0.05)
    hsd_result.plot_simultaneous()
    plt.savefig("output/pics/HSD.png")
    plt.clf()

    plot_20 = plt.scatter(growth_20_2017["Row"], growth_20_2017["Predicted Growth"], c="red", s=10)
    plot_30 = plt.scatter(growth_30_2017["Row"], growth_30_2017["Predicted Growth"], c="orange", s=10)
    plot_40 = plt.scatter(growth_40_2017["Row"], growth_40_2017["Predicted Growth"], c="green", s=10)
    plot_original = plt.scatter(restricted_data_2019["Row"], restricted_data_2019["Revenue Growth"], c="pink", s=10)
    plt.title("Predicted Growth vs Actual Growth of Technology Companies in 2019")
    plt.xlabel("Stock Ticker")
    plt.ylabel("Predicted Growth (in percentages (1=100%))")
    plt.legend(
        (plot_20, plot_30, plot_40, plot_original),
        ("20% Classifier", "30% Classifier", "40% Classifier", "Actual Growth"),
        loc='upper left'
    )
    axes = plt.gca()
    axes.set_ylim(-1, 2)
    plt.savefig("output/pics/Classifier_Data.png")
    plt.clf()

    bplot_20 = plt.scatter(balanced_20_2017["Row"], balanced_20_2017["Predicted Growth"], c="red", s=10)
    bplot_30 = plt.scatter(balanced_30_2017["Row"], balanced_30_2017["Predicted Growth"], c="orange", s=10)
    bplot_40 = plt.scatter(balanced_40_2017["Row"], balanced_40_2017["Predicted Growth"], c="green", s=10)
    plot_original = plt.scatter(restricted_data_2019["Row"], restricted_data_2019["Revenue Growth"], c="pink", s=10)
    plt.title("Predicted Growth vs Actual Growth of Technology Companies in 2019")
    plt.xlabel("Stock Ticker")
    plt.ylabel("Predicted Growth (in percentages (1=100%))")
    plt.legend(
        (bplot_20, bplot_30, bplot_40, plot_original),
        ("20% Balanced Classifier", "30% Balanced Classifier", "40% Balanced Classifier", "Actual Growth"),
        loc='upper left'
    )
    axes = plt.gca()
    axes.set_ylim(-1, 2)
    plt.savefig("output/pics/Balanced_Data.png")
    plt.clf()

    plot_percent = plt.scatter(percent_2017["Row"], percent_2017["Predicted Growth"], c="blue", s=10)
    plot_original = plt.scatter(restricted_data_2019["Row"], restricted_data_2019["Revenue Growth"], c="pink", s=10)
    plt.title("Predicted Growth vs Actual Growth of Technology Companies in 2019")
    plt.xlabel("Stock Ticker")
    plt.ylabel("Predicted Growth (in percentages (1=100%))")
    plt.legend(
        (plot_percent, plot_original),
        ("Percent Data", "Actual Growth"),
        loc='upper left'
    )
    axes = plt.gca()
    axes.set_ylim(-1, 2)
    plt.savefig("output/pics/Percent_Data.png")
    plt.clf()

    plot_regular = plt.scatter(regular_2017["Row"], regular_2017["Predicted Growth"], c="purple", s=10)
    plot_original = plt.scatter(data_2019["Row"], data_2019["Revenue Growth"], c="pink", s=10)
    plt.title("Predicted Growth using Regular Data vs Actual Growth of Technology Companies in 2019")
    plt.xlabel("Stock Ticker")
    plt.ylabel("Predicted Growth (in percentages (1=100%))")
    plt.legend(
        (plot_regular, plot_original),
        ("Regular Data", "Actual Growth"),
        loc='upper left'
    )
    axes = plt.gca()
    axes.set_ylim(-1, 2)
    plt.savefig("output/pics/Regular_Data.png")
    return

if __name__=='__main__':
    main()
