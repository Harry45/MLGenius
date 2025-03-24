from sklearn import datasets
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from helpers import dill_save

MODEL_PATH = "./"


def training(n_estimators):
    iris = datasets.load_iris()

    data = pd.DataFrame(
        {
            "sepal length": iris.data[:, 0],
            "sepal width": iris.data[:, 1],
            "petal length": iris.data[:, 2],
            "petal width": iris.data[:, 3],
            "species": iris.target,
        }
    )

    X_data = data[["sepal length", "sepal width", "petal length", "petal width"]]
    y_data = data["species"]

    # Split dataset into training set and test set
    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.3)

    # training
    clf = RandomForestClassifier(n_estimators=n_estimators)
    clf.fit(X_train.values, y_train.values)

    # prediction
    y_pred = clf.predict(X_test.values)
    print(f"Accuracy: {metrics.accuracy_score(y_test.values, y_pred) * 100:.2f}")

    # save model
    dill_save(clf, MODEL_PATH, "iris_model")


if __name__ == "__main__":
    training(n_estimators=100)
