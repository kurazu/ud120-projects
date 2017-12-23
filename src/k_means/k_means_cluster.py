"""
Skeleton code for k-means clustering mini-project.
"""

import matplotlib.pyplot as plt

import os.path
from tools.feature_format import featureFormat, targetFeatureSplit
from tools.loading import load_pickle
import final_project

from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler


def draw(
    pred, features, poi, mark_poi=False, name="image.png",
    f1_name="feature 1", f2_name="feature 2"
):
    """ some plotting code designed to help you visualize your clusters """

    # plot each cluster with a different color--add more colors for
    # drawing more than five clusters
    colors = ["b", "c", "k", "m", "g"]
    for ii, pp in enumerate(pred):
        plt.scatter(features[ii][0], features[ii][1], color=colors[pred[ii]])

    # if you like, place red stars over points that are POIs (just for funsies)
    if mark_poi:
        for ii, pp in enumerate(pred):
            if poi[ii]:
                plt.scatter(
                    features[ii][0], features[ii][1], color="r", marker="*"
                )
    plt.xlabel(f1_name)
    plt.ylabel(f2_name)
    file_path = os.path.join(os.path.dirname(__file__), name)
    plt.savefig(file_path)
    plt.show()


def main():
    # load in the dict of dicts containing all the data
    # on each person in the dataset
    file_path = os.path.join(
        os.path.dirname(final_project.__file__), 'final_project_dataset.pkl'
    )
    data_dict = load_pickle(file_path)
    # there's an outlier--remove it!
    del data_dict["TOTAL"]

    # the input features we want to use
    # can be any key in the person-level dictionary
    # (salary, director_fees, etc.)
    feature_1 = "salary"
    feature_2 = "exercised_stock_options"
    # feature_3 = "total_payments"
    poi = "poi"
    features_list = [poi, feature_1, feature_2]  # , feature_3]
    data = featureFormat(data_dict, features_list)
    poi, finance_features = targetFeatureSplit(data)
    scaler = MinMaxScaler()
    finance_features = scaler.fit_transform(finance_features)
    print(scaler.transform([[2e5, 1e6]]))

    # in the "clustering with 3 features" part of the mini-project,
    # you'll want to change this line to
    # for f1, f2, _ in finance_features:
    # (as it's currently written, the line below assumes 2 features)
    for f1, f2 in finance_features:
        plt.scatter(f1, f2)
        plt.xlabel(feature_1)
        plt.ylabel(feature_2)
    plt.show()

    # cluster here; create predictions of the cluster labels
    # for the data and store them to a list called pred
    clustering = KMeans(n_clusters=2)
    clustering.fit(X=finance_features)
    pred = clustering.predict(finance_features)

    # rename the "name" parameter when you change the number of features
    # so that the figure gets saved to a different file
    try:
        draw(
            pred, finance_features, poi, mark_poi=False,
            name="clusters3.pdf", f1_name=feature_1, f2_name=feature_2
        )
    except NameError:
        print("no predictions object named pred found, no clusters to plot")


if __name__ == '__main__':
    main()
