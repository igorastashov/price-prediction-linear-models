import os

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker


MY_DB = "db_cars_for_sale"
MY_USER = "astashovivl"
HOST_NAME = "dpg-cns562ol6cac73ask0ug-a.frankfurt-postgres.render.com"

# Load .env file
load_dotenv()

SQLALCHEMY_DB_URL = (
    f"postgresql+psycopg2://{MY_USER}:{os.getenv('MY_PASS')}@{HOST_NAME}/{MY_DB}"
)

cars_data = {
    "pk": list(range(3)),
    "name": ["Renault KWID Climber 1.0 MT BSIV", "Maruti Wagon R LXI", "Hyundai i20 Asta 1.2"],
    "year": [2019, 2013, 2013],
    "km_driven": [35000, 58343, 30000],
    "fuel": ["Petrol", "Petrol", "Petrol"],
    "seller_type": ["Individual", "Trustmark Dealer", "Individual"],
    "transmission": ["Manual", "Manual", "Manual"],
    "owner": ["First Owner", "First Owner", "First Owner"],
    "mileage": ["23.01 kmpl", "21.79 kmpl", "18.5 kmpl"],
    "engine": ["999 CC", "998 CC", "1197 CC"],
    "max_power": ["67 bhp", "67.05 bhp", "82.85 bhp"],
    "torque": ["91Nm@ 4250rpm", "90Nm@ 3500rpm", "113.7Nm@ 4000rpm"],
    "seats": [5.0, 5.0, 5.0],
}

timestamps_data = {"id": [0, 1], "timestamp": [12, 10]}

engine = create_engine(SQLALCHEMY_DB_URL, connect_args={})

# populate database with existing data
for data_name, data in zip(["cars", "timestamps"], [cars_data, timestamps_data]):
    df = pd.DataFrame.from_dict(data)
    df.to_sql(data_name, engine, index=False, if_exists="replace")

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()
