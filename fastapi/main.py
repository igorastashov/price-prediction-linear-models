import io
import pickle

import numpy as np
import pandas as pd
from ds.pre_processing import PreprocessingTransformer
from ds.ridge import ridge_model
from schemas.schema import Item
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from fastapi import FastAPI, UploadFile
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse, StreamingResponse

description = """
Данная работа была проделана в соответствии с заданием по курсу Машинное обучение
[Магистратура НИУ ВШЭ](https://www.hse.ru/en/ma/mlds/).


Это приложение может:
1. Показать базовую информацию о приложении;
2. Предсказать стоимость авто по ее введенным характеристикам;
3. Предсказать стоимость новых авто по характеристикам, переданным .csv файлом;
4. Переобучить модель на новых данных.
"""

tags = [{"name": "Допустимые операции"}]

app = FastAPI(
    title="Сервис предсказания стоимости авто",
    description=description,
    openapi_tags=tags,
    contact={
        "name": "Асташов И. В.",
        "url": "https://github.com/igorastashov",
    },
)


with open("models/model_ridge.pkl", "rb") as file:
    model = pickle.load(file)


def read_csv(file: UploadFile):
    content = file.file.read()
    buffer = io.BytesIO(content)
    dataset = pd.read_csv(buffer)
    buffer.close()
    return dataset


transform_data = PreprocessingTransformer(
    cols_to_change=["mileage", "engine", "max_power"],
    cols_for_replace_outliers=[
        "year",
        "km_driven",
        "mileage",
        "engine",
        "max_power",
        "torque",
        "max_torque_rpm",
    ],
    cols_for_log1p=[
        "year",
        "km_driven",
        "mileage",
        "engine",
        "max_power",
        "torque",
        "max_torque_rpm",
    ],
    cols_for_poly=[
        "year",
        "km_driven",
        "mileage",
        "engine",
        "max_power",
        "torque",
        "max_torque_rpm",
    ],
)


@app.get("/", tags=["1. Показать базовую информацию о приложении"])
def root() -> dict:
    return {"message": "Привет!!! Это сервис для предсказания цен на авто."}


@app.post(
    "/predict_item",
    tags=["2. Предсказать стоимость авто по ее введенным характеристикам"],
)
def predict_item(item: Item) -> float:
    """
    Пример передаваемых данных:

     - **"year"**: 2007,
     - **"km_driven"**: 38000,
     - **"fuel"**: "Petrol",
     - **"seller_type"**: "Individual",
     - "transmission": "Manual",
     - **"owner"**: "Second Owner",
     - **"mileage"**: "15.3 kmpl",
     - **"engine"**: "1193 CC",
     - **"max_power"**: "65.3 bhp",
     - **"torque"**: "102Nm@ 2600rpm",
     - **"seats"**: 5.0

    :param item: Параметры авто;

    :return: Предсказание стоимости авто.
    """
    try:
        car = pd.DataFrame(jsonable_encoder(item), index=[0])
        _, _, car = transform_data.transform(None, car)
        prediction = model.predict(car)[0]
        return prediction
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.post(
    "/predict_items",
    tags=[
        "3. Предсказать стоимость новых авто по характеристикам, переданным .csv файлом"
    ],
)
def predict_items(file: UploadFile) -> StreamingResponse:
    """
    :param file: Файл .csv с параметрами автомобилей;

    :return: Предсказание стоимости автомобилей.
    """
    try:
        cars = read_csv(file)

        cols_to_keep = [
            "year",
            "km_driven",
            "fuel",
            "seller_type",
            "transmission",
            "owner",
            "mileage",
            "engine",
            "max_power",
            "torque",
            "seats",
        ]
        cars = cars[cols_to_keep]
        _, _, cars = transform_data.transform(None, cars)
        cars["selling_price"] = model.predict(cars)

        stream = io.StringIO()
        cars.to_csv(stream, index=False)
        response = StreamingResponse(iter([stream.getvalue()]), media_type="text/csv")
        response.headers["Content-Disposition"] = "attachment; filename=export.csv"
        return response
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.post("/fit_model", tags=["4. Переобучить модель на новых данных"])
def fit_model(file: UploadFile) -> StreamingResponse:
    """
    :param file: Файл .csv с параметрами автомобилей;

    :return: Переобученная и сохраненная модель.
    """
    try:
        cars = read_csv(file)

        X_train, y_train, _ = transform_data.transform(cars, None)

        model_ridge = ridge_model()
        ridge_params = {"regressor__estimator__alpha": np.linspace(0.5, 1.2, num=100)}
        model_ridge = GridSearchCV(
            model_ridge, param_grid=ridge_params, cv=10, scoring="r2"
        ).fit(X_train, y_train)
        model_ridge = model_ridge.best_estimator_

        y_pred_train = model_ridge.predict(X_train)
        r2_test = r2_score(y_train, y_pred_train)

        with open("models/model_ridge.pkl", "wb") as file:
            pickle.dump(model_ridge, file)

        response_data = {"R квадрат на Train": round(r2_test, 4)}
        return JSONResponse(content=jsonable_encoder(response_data))
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
