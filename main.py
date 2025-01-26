import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from joblib import dump
import os
from abc import ABC, abstractmethod


class MyGridSearchCV(GridSearchCV):
    def __init__(self, estimator, param_grid):
        self.estimator = estimator
        self.param_grid = param_grid
        super().__init__(
            estimator=self.estimator,
            param_grid=self.param_grid,
            scoring=None,
            n_jobs=-1,
            refit=True,
            cv=None,
            verbose=0,
            pre_dispatch='2*n_jobs',
            error_score=np.nan,
            return_train_score=False
        )


param_grid = {"C": [0.5, 1, 1.5]}
model = SVC()
grid_model = MyGridSearchCV(estimator=model, param_grid=param_grid)


class BaseClass(ABC):
    def __init__(self, path):

        if not os.path.exists(path):
            raise Exception()

        if not path.endswith(".csv"):
            raise Exception()

        df = pd.read_csv(path)
        if self.check_if_missing_values(df):
            self.df = df
        else:
            raise Exception()

    @staticmethod
    def check_if_missing_values(df):
        if df.notnull().all().all():
            return True
        return False

    def get_target(self, target):
        if target in self.df.columns:
            return target
        else:
            raise Exception()

    @abstractmethod
    def create_xy(self, target):
        pass

    @abstractmethod
    def prepare_for_ml(self, xy):
        pass

    @abstractmethod
    def create_train_test_split(self, xy):
        pass

    @abstractmethod
    def scale(self, datasets):
        pass

    @abstractmethod
    def fit_and_evaluate_model(self, datasets):
        pass


class Classification(BaseClass):
    def __init__(self, path, perform_scaling, model):
        super().__init__(path)
        self.perform_scaling = perform_scaling
        self.model = model

    def create_xy(self, target):
        x = self.df.drop(target, axis=1)
        y = self.df[target]
        return x, y

    def prepare_for_ml(self, xy):
        x, y = xy

        x_str = x.select_dtypes(include="object")
        x_num = x.select_dtypes(exclude="object")

        match x_str.shape[1] > 0:
            case True:
                x_str = pd.get_dummies(x_str, drop_first=True)
                x = pd.concat([x_num, x_str], axis=1)

        match y.dtype == "object":
            case True:
                values = y.unique()
                values_dict = {v: k for k, v in enumerate(values)}
                y = y.map(arg=values_dict)
        return x, y

    def create_train_test_split(self, xy):
        X, y = xy
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)
        return X_train, X_test, y_train, y_test

    def scale(self, datasets):
        if self.perform_scaling:
            X_train, X_test, y_train, y_test = datasets
            scaler = StandardScaler()
            scaled_X_train = scaler.fit_transform(X_train)
            scaled_X_test = scaler.transform(X_test)
            return scaled_X_train, scaled_X_test, y_train, y_test
        return datasets

    def fit_and_evaluate_model(self, datasets):
        scaled_X_train, scaled_X_test, y_train, y_test = datasets
        fitted_model = self.model.fit(scaled_X_train, y_train)
        y_pred = fitted_model.predict(scaled_X_test)
        report = classification_report(y_true=y_test, y_pred=y_pred, zero_division=0)
        return report

    def run(self, target, filename):
        target = instance.get_target(target)
        xy = self.create_xy(target)
        prepared_xy = self.prepare_for_ml(xy)
        datasets = self.create_train_test_split(prepared_xy)
        datasets_after_scaling = self.scale(datasets)
        dump(self.model, f"models/{filename}.joblib")
        report = self.fit_and_evaluate_model(datasets_after_scaling)
        print(report)


if __name__ == "__main__":
    instance = Classification("data/iris.csv", True, grid_model)
    instance.run("species", "svc")
