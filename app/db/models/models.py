from sqlalchemy import Column, Float, Integer, String

from ..db import Base


class Car(Base):
    __tablename__ = "cars"

    pk = Column(Integer, primary_key=True, unique=True, autoincrement=True)
    name = Column(String)
    year = Column(Integer)
    km_driven = Column(Integer)
    fuel = Column(String)
    seller_type = Column(String)
    transmission = Column(String)
    owner = Column(String)
    mileage = Column(String)
    engine = Column(String)
    max_power = Column(String)
    torque = Column(String)
    seats = Column(Float)


class PredictionsModel(Base):
    __tablename__ = "predictions"

    car_pk = Column(Integer, primary_key=True)
    predicted_price = Column(Float)
