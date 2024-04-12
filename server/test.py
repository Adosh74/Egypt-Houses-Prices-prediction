from pydantic import BaseModel
import numpy as np
import joblib

# Load the trained Random Forest model
model = joblib.load("./random_forest_model.pkl")


class InputData(BaseModel):
    Type: str
    Bathrooms: float
    Area: float
    Furnished: str
    Payment_Option: str
    Delivery_Term: str
    City: str

# test model with random data
data = {
    "Type": "Apartment",
    "Bathrooms": 2,
    "Area": 120,
    "Furnished": "Yes",
    "Payment_Option": "Cash",
    "Delivery_Term": "Immediate",
    "City": "Cairo"
}

out_input = np.array([
    data['Type'],
    data['Bathrooms'],
    data['Area'],
    data['Furnished'],
    data['Payment_Option'],
    data['Delivery_Term'],
    data['City']
]).reshape(1, -1)
out_predict = model.predict(out_input)
print(out_predict)