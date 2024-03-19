import pickle
import random
import re
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, StandardScaler

random.seed(42)
np.random.seed(42)


class PreprocessingTransformer:
    def __init__(
        self,
        cols_to_change,
        cols_for_replace_outliers,
        cols_for_log1p,
        cols_for_poly,
        median_values=None,
        bounds=None,
        one_hot_transformed=None,
        scale_transformed=None,
        poly_transformed=None,
    ):
        self.cols_to_change = (
            cols_to_change  # Признаки для перевода в числовой тип данных
        )
        self.cols_for_replace_outliers = cols_for_replace_outliers
        self.cols_for_log1p = cols_for_log1p
        self.cols_for_poly = cols_for_poly

        self.median_values = median_values  # Медианы числовых столбцов по Train
        self.bounds = bounds  # Границы интервалов числовых столбцов по Train
        self.one_hot_transformed = (
            one_hot_transformed  # Трансформер ohe категориальных столбцов по Train
        )
        self.scale_transformed = (
            scale_transformed  # Трансформер scale категориальных столбцов по Train
        )
        self.poly_transformed = (
            poly_transformed  # Трансформер-генератор полиномиальных признаков по Train
        )

        self.PATH = Path("weights")

    def transform(self, train, test):
        train, test = self.base_preprocessing(train, test)
        X_train, y_train = self.train_split(train)
        X_train, test = self.feature_engineering_preprocessing(X_train, test)
        return X_train, y_train, test

    def base_preprocessing(self, train, test):
        if train is not None:
            train = self.remove_duplicates(train)
            train = self.preprocess_features(train, self.cols_to_change)
            train["max_torque_rpm"] = self.split_torque(
                train["torque"], output="max_torque_rpm"
            )
            train["torque"] = self.split_torque(train["torque"], output="torque")
            train = self.preprocessing_passes(train)

        if test is not None:
            test = self.preprocess_features(test, self.cols_to_change)
            test["max_torque_rpm"] = self.split_torque(
                test["torque"], output="max_torque_rpm"
            )
            test["torque"] = self.split_torque(test["torque"], output="torque")
            test = self.preprocessing_passes(test, "test")

        return train, test

    def feature_engineering_preprocessing(self, X_train, test):
        if X_train is not None:
            X_train = self.replace_outliers(X_train, self.cols_for_replace_outliers)
            X_train = self.preprocessing_ohe(X_train)
            X_train = self.log1p(X_train, self.cols_for_log1p)
            X_train = self.preprocessing_standard_scaler(X_train)
            X_train = self.add_polynomial_features(X_train, self.cols_for_poly)

        if test is not None:
            test = self.replace_outliers(test, self.cols_for_replace_outliers, "test")
            test = self.preprocessing_ohe(test, "test")
            test = self.log1p(test, self.cols_for_log1p)
            test = self.preprocessing_standard_scaler(test, "test")
            test = self.add_polynomial_features(test, self.cols_for_poly)

        return X_train, test

    def remove_duplicates(self, dataset):
        """
        Функция для удаления дубликатов из датасета
        """
        coll_names_dupl = dataset.drop(["selling_price"], axis=1).columns
        return dataset.drop_duplicates(
            subset=coll_names_dupl, keep="first"
        ).reset_index(drop=True)

    def preprocess_features(self, dataset, cols):
        """
        Функция для преобразования признаков в числовой тип данных
        """
        for col in cols:
            if dataset[col].dtype != "float64":
                dataset[col] = pd.to_numeric(
                    dataset[col].str.split(" ", expand=True)[0]
                )
        return dataset

    def split_torque(self, dataset, output: str = "torque") -> pd.Series:
        """
        Обработка признака `torque`
        """

        def split_torque(torque: Union[str, float]) -> Union[int, float]:
            if isinstance(torque, str):
                torque = torque.lower().replace(",", "").replace("/", "")
                torque = re.findall(r"[\d\.]+", torque)
                torque = [float(val) for val in torque]

                if "kgm" in torque:
                    torque[0] = np.round(torque[0] * 9.80665)

                if len(torque) < 2:
                    torque = (
                        [np.nan] + torque if "kgm" not in torque else torque + [np.nan]
                    )

                return torque[1] if output == "max_torque_rpm" else torque[0]
            else:
                return torque

        return dataset.apply(split_torque).copy()

    def preprocessing_passes(self, dataset, name="train"):
        """
        Замена пропусков в чиcловых признаках медианой
        Медиана определяется для столбцов по train
        """
        filename = "medians_for_passes.pkl"
        numerical = dataset.select_dtypes(include=np.number).columns

        if name == "test":
            self.median_values = self.open_pickle(filename)
        else:
            self.median_values = dataset[numerical].median()
            self.save_in_pickle(self.median_values, filename)

        dataset[numerical] = dataset[numerical].fillna(self.median_values)

        return dataset

    def train_split(self, train):
        """
        Train-test разделение
        """
        if train is not None:
            train = train.sample(frac=1, random_state=42).reset_index(drop=True).copy()

            y_train = train["selling_price"]
            X_train = train.drop(columns=["name", "selling_price"], errors="ignore")
        else:
            y_train = None
            X_train = None

        return X_train, y_train

    def replace_outliers(self, dataset, cols, name="train"):
        """
        Замена выбросов на границах интервалов, определенные квантилями по Train
        """
        filename = "bounds_for_outliers.pkl"

        if name == "test":
            self.bounds = self.open_pickle(filename)
            upper_bounds = self.bounds["upper_bounds"]
            lower_bounds = self.bounds["lower_bounds"]
        else:
            quantile_upper = 0.996
            quantile_lower = 0.004

            upper_bounds = dataset[cols].quantile(quantile_upper)
            lower_bounds = dataset[cols].quantile(quantile_lower)

            self.bounds = {"upper_bounds": upper_bounds, "lower_bounds": lower_bounds}

            self.save_in_pickle(self.bounds, filename)

        for col in cols:
            upper_bound = upper_bounds[col]
            lower_bound = lower_bounds[col]

            dataset[col] = np.clip(dataset[col], lower_bound, upper_bound)

        return dataset

    def preprocessing_ohe(self, dataset, name="train"):
        """
        Преобразование всех категориальных переменных методом One Hot Encoding
        """
        filename = "one_hot_transformed.pkl"

        dataset["seats"] = dataset["seats"].astype("O")
        categorical = dataset.select_dtypes(include="object").columns

        if name == "test":
            ohe = self.open_pickle(filename)
        else:
            ohe = OneHotEncoder(drop="first", sparse_output=False).set_output(
                transform="pandas"
            )
            ohe = ohe.fit(dataset[categorical])

            self.one_hot_transformed = ohe
            self.save_in_pickle(ohe, filename)

        transformed_data = ohe.transform(dataset[categorical])
        dataset = pd.concat([dataset, transformed_data], axis=1)
        dataset.drop(categorical, axis=1, inplace=True)

        return dataset

    def log1p(self, dataset, cols):
        """
        Замена нулевых и отрицательных значений 1.0e-10, а затем вычисление натурального логарифма плюс 1.
        """
        dataset[cols] = np.log1p(np.maximum(dataset[cols], 1.0e-10))
        return dataset

    def preprocessing_standard_scaler(self, dataset, name="train"):
        """
        Сандартизация всех стобцов данных
        """
        filename = "scale_transformed.pkl"

        if name == "test":
            scaler = self.open_pickle(filename)
        else:
            scaler = StandardScaler()
            scaler = scaler.fit(dataset)

            self.scale_transformed = scaler
            self.save_in_pickle(scaler, filename)

        scaled_data = scaler.transform(dataset)
        dataset = pd.DataFrame(scaled_data, columns=dataset.columns)

        return dataset

    def add_polynomial_features(self, dataset, cols, name="train"):
        """
        Генерация полиномиальных признаков
        """
        filename = "poly_transformed.pkl"

        if name == "test":
            poly = self.open_pickle(filename)
        else:
            poly = PolynomialFeatures()
            poly = poly.fit(dataset[cols])

            self.poly_transformed = poly
            self.save_in_pickle(poly, filename)

        poled_data = poly.transform(dataset[cols])

        columns = poly.get_feature_names_out(input_features=cols)
        poled_data_df = pd.DataFrame(poled_data, columns=columns, index=dataset.index)
        dataset = pd.concat([dataset, poled_data_df], axis=1)

        return dataset

    def save_in_pickle(self, file_to_save, filename):
        """
        Сохранение значений столбцов по Train, необходимых для преобразования Test
        """
        with open(self.PATH / filename, "wb") as f:
            pickle.dump(file_to_save, f)

    def open_pickle(self, filename):
        """
        Открытие сохраненных значений столбцов Train, необходимых для преобразования Test
        """
        with open(self.PATH / filename, "rb") as f:
            file_to_read = pickle.load(f)
        return file_to_read
