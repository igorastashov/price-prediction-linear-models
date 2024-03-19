from sqlalchemy import func
from sqlalchemy.orm import Session

from ..models import models
from ..schemas import schemas


def create_car(db: Session, car: schemas.Car):
    db_car = models.Car(**car.model_dump())
    db.add(db_car)
    db.commit()
    db.refresh(db_car)
    return db_car


def get_car(db: Session, car_pk: int):
    return db.query(models.Car).filter(models.Car.pk == car_pk).first()


def get_cars_in_range(db: Session, start_pk: int, end_pk: int):
    return (
        db.query(models.Car)
        .filter(models.Car.pk >= start_pk, models.Car.pk <= end_pk)
        .all()
    )


def get_cars(
    db: Session,
    fuel: str = None,
    seller_type: str = None,
    transmission: str = None,
    owner: str = None,
):
    query = db.query(models.Car)

    if fuel:
        query = query.filter(models.Car.fuel == fuel)
    if seller_type:
        query = query.filter(models.Car.seller_type == seller_type)
    if transmission:
        query = query.filter(models.Car.transmission == transmission)
    if owner:
        query = query.filter(models.Car.owner == owner)

    return query.all()


def update_car(db: Session, car: schemas.Car):
    db_car = db.query(models.Car).filter(models.Car.pk == car.pk).first()
    db_car.name = car.name
    db_car.fuel = car.fuel
    db_car.seller_type = car.seller_type
    db_car.transmission = car.transmission
    db_car.owner = car.owner
    db_car.mileage = car.mileage
    db_car.engine = car.engine
    db_car.max_power = car.max_power
    db_car.torque = car.torque
    db_car.seats = car.seats
    db_car.pk = car.pk
    db.commit()
    db.refresh(db_car)
    return db_car


def get_max_pk(db: Session):
    return db.query(func.max(models.Car.pk)).scalar() or 0


def save_predictions(db: Session, predictions):
    for prediction in predictions:
        db_prediction = models.PredictionsModel(**prediction)
        db.add(db_prediction)
    db.commit()
