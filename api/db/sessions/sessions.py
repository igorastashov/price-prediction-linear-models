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


# def get_cars(db: Session, fuel: str = None):
#     if fuel:
#         return db.query(models.Car).filter(models.Car.kind == fuel).all()
#     else:
#         return db.query(models.Car).all()

def get_cars(db: Session, fuel: str = None, seller_type: str = None, transmission: str = None, owner: str = None):
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


def create_timestamp(db: Session, timestamp: schemas.Timestamp):
    db_timestamp = models.Timestamp(**timestamp.model_dump())
    db.add(db_timestamp)
    db.commit()
    db.refresh(db_timestamp)
    return db_timestamp


def get_timestamp(db: Session, timestamp_id: int):
    return (
        db.query(models.Timestamp).filter(models.Timestamp.id == timestamp_id).first()
    )