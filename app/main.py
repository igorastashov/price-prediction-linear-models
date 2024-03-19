import io
import pickle
from typing import Optional

import numpy as np
import pandas as pd
import uvicorn
from db.db import SessionLocal
from db.schemas import schemas
from db.sessions import sessions
from ds.pre_processing import PreprocessingTransformer
from ds.ridge import ridge_model
from fastapi import Depends, FastAPI, HTTPException, Query, UploadFile
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse, StreamingResponse
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from sqlalchemy.orm import Session

description = """
Данная работа была проделана в соответствии с заданием по курсу Машинное обучение
[Магистратура НИУ ВШЭ](https://www.hse.ru/en/ma/mlds/).


Это приложение может:
1. Показать базовую информацию о приложении;
2. Добавить новый автомобиль;
3. Добавить автомобили из .csv файла;
4. Вывести список автомобилей в соответствии с выбранными характеристиками;
5. Вывести автомобиль по уникальному ключу;
6. Обновить информацию об автомобиле по его уникальному ключу";
7. Предсказать стоимость автомобиля/-ей по выбранному диапазону ключей;
8. Переобучить модель на новых данных.
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


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


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
    "/create_car", response_model=schemas.Car, tags=["2. Добавить новый автомобиль"]
)
def create_car(car: schemas.Car, db: Session = Depends(get_db)):
    """
    Пример передаваемых данных:

    - **"pk"**: 3,
    - **"name"**: "Toyota Land Cruiser 300",
    - **"year"**: 2022,
    - **"km_driven"**: 1000,
    - **"fuel"**: "Petrol",
    - **"seller_type"**: "Individual",
    - "transmission": "Automatic",
    - **"owner"**: "First Owner",
    - **"mileage"**: "15 kmpl",
    - **"engine"**: "4500 CC",
    - **"max_power"**: "415 bhp",
    - **"torque"**: "650Nm@ 3600rpm",
    - **"seats"**: 5.0
    """
    db_car = sessions.get_car(db=db, car_pk=car.pk)
    if db_car is not None:
        raise HTTPException(status_code=409, detail="This pk is exist")
    return sessions.create_car(db=db, car=car)


@app.post("/add_cars_from_csv", tags=["3. Добавить автомобили из .csv файла"])
def add_cars_from_csv(file: UploadFile, db: Session = Depends(get_db)):
    try:
        df = pd.read_csv(file.file)

        df.fillna("", inplace=True)
        cars_data = df.to_dict(orient="records")

        added_cars = []
        last_pk = sessions.get_max_pk(db=db)
        for car_data in cars_data:
            last_pk += 1
            car_data["pk"] = last_pk
            car = schemas.Car(**car_data)
            added_car = sessions.create_car(db=db, car=car)
            added_cars.append(added_car)

        return {"message": f"Добавлено {len(added_cars)} автомобилей в базу данных"}
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.get(
    "/cars",
    response_model=list[schemas.Car],
    tags=["4. Вывести список автомобилей в соответствии с выбранными характеристиками"],
)
def get_cars(
    fuel: schemas.CarFuel = None,
    seller_type: schemas.SellerType = None,
    transmission: schemas.Transmission = None,
    owner: schemas.Owner = None,
    db: Session = Depends(get_db),
):
    cars = sessions.get_cars(
        db=db,
        fuel=fuel,
        seller_type=seller_type,
        transmission=transmission,
        owner=owner,
    )
    return cars


@app.get(
    "/car/{pk}",
    response_model=schemas.Car,
    tags=["5. Вывести автомобиль по уникальному ключу"],
)
def get_car_by_pk(pk: int, db: Session = Depends(get_db)):
    db_car = sessions.get_car(db=db, car_pk=pk)
    if db_car is None:
        raise HTTPException(status_code=404, detail="Car not found")
    return db_car


@app.patch(
    "/car/{pk}",
    response_model=schemas.Car,
    tags=["6. Обновить информацию об автомобиле по его уникальному ключу"],
)
def update_car(pk: int, car: schemas.Car, db: Session = Depends(get_db)):
    db_car = sessions.get_car(db=db, car_pk=pk)
    if db_car is None:
        raise HTTPException(status_code=404, detail="This pk is not exist")
    db_car = sessions.get_car(db=db, car_pk=car.pk)
    if pk != car.pk and db_car is not None:
        raise HTTPException(status_code=409, detail="Pk does not match index")
    return sessions.update_car(db=db, car=car)


@app.get(
    "/predict_by_range/",
    tags=["7. Предсказать стоимость автомобиля/-ей по выбранному диапазону ключей"],
)
def predict_by_range(
    start_pk: Optional[int] = Query(None, description="Start PK value"),
    end_pk: Optional[int] = Query(None, description="End PK value"),
    db: Session = Depends(get_db),
):
    try:
        if start_pk is None or end_pk is None:
            raise HTTPException(
                status_code=400, detail="Both start_pk and end_pk must be provided."
            )

        db_cars = sessions.get_cars_in_range(db=db, start_pk=start_pk, end_pk=end_pk)
        if not db_cars:
            raise HTTPException(
                status_code=404, detail="No cars found in the specified range."
            )

        cars_data = [
            {
                "year": car.year,
                "km_driven": car.km_driven,
                "fuel": car.fuel,
                "seller_type": car.seller_type,
                "transmission": car.transmission,
                "owner": car.owner,
                "mileage": car.mileage,
                "engine": car.engine,
                "max_power": car.max_power,
                "torque": car.torque,
                "seats": car.seats,
            }
            for car in db_cars
        ]

        cars_df = pd.DataFrame(cars_data)
        _, _, cars_df = transform_data.transform(None, cars_df)
        predictions = model.predict(cars_df)

        predict_data_for_save = [
            {"car_pk": car.pk, "predicted_price": prediction}
            for car, prediction in zip(db_cars, predictions)
        ]

        # Сохраняем предсказания в БД
        sessions.save_predictions(db=db, predictions=predict_data_for_save)

        return predict_data_for_save
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    #     # Вывод предсказания в виде csv файла
    #     cars_pk_data = [{
    #         "pk": car.pk,
    #     } for car in db_cars]
    #
    #     predict_df = pd.DataFrame(cars_pk_data)
    #     predict_df['selling_price'] = predictions
    #
    #     stream = io.StringIO()
    #     predict_df.to_csv(stream, index=False)
    #     response = StreamingResponse(iter([stream.getvalue()]), media_type="text/csv")
    #     response.headers["Content-Disposition"] = "attachment; filename=export.csv"
    #     return response
    # except Exception as e:
    #     return JSONResponse(content={"error": str(e)}, status_code=500)


@app.post("/fit_model", tags=["8. Переобучить модель на новых данных"])
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
    uvicorn.run(app, host="0.0.0.0", port=8000)
