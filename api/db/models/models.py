from datetime import datetime

from sqlalchemy import Column, DateTime, Integer, String, Float

from ..db import Base


class Car(Base):
    __tablename__ = "cars"

    pk = Column(Integer, primary_key=True, unique=True)
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


class Timestamp(Base):
    __tablename__ = "timestamps"

    id = Column(Integer, primary_key=True, unique=True)
    timestamp = Column(DateTime, default=datetime.now())
