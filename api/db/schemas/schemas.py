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
    name: str
    fuel: CarFuel
    seller_type: SellerType
    transmission: Transmission
    owner: Owner
    mileage: str
    engine: str
    max_power: str
    torque: str
    seats: float

    class Config:
        from_attributes = True


class Item(BaseModel):
    year: int
    km_driven: int
    fuel: CarFuel
    seller_type: SellerType
    transmission: Transmission
    owner: Owner
    mileage: str
    engine: str
    max_power: str
    torque: str
    seats: float


class Timestamp(BaseModel):
    id: int
    timestamp: int

    class Config:
        from_attributes = True
