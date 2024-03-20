from enum import Enum

from pydantic import BaseModel


class CarFuel(str, Enum):
    diesel = "Diesel"
    petrol = "Petrol"
    lpg = "LPG"
    cng = "CNG"


class SellerType(str, Enum):
    individual = "Individual"
    dealer = "Dealer"
    trustmark_dealer = "Trustmark Dealer"


class Transmission(str, Enum):
    manual = "Manual"
    automatic = "Automatic"


class Owner(str, Enum):
    first_owner = "First Owner"
    second_owner = "Second Owner"
    third_owner = "Third Owner"
    fourth_and_above_owner = "Fourth & Above Owner"
    test_drive_car = "Test Drive Car"


class Car(BaseModel):
    pk: int
    name: str = None
    year: int = None
    km_driven: int = None
    fuel: CarFuel = None
    seller_type: SellerType = None
    transmission: Transmission = None
    owner: Owner = None
    mileage: str = None
    engine: str = None
    max_power: str = None
    torque: str = None
    seats: float = None

    class Config:
        from_attributes = True


class Predictions(BaseModel):
    car_pk: int
    predicted_price: float

    class Config:
        from_attributes = True
