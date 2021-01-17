import pickle
from fastapi import FastAPI
from sklearn.ensemble import RandomForestClassifier

# Load Prediction Model
with open('room_trained_model.plk','rb') as f:
    model = pickle.load(f)

app = FastAPI()
@app.get("/")
async def main():
    return 'Hello Room ASQ & ALQ'

@app.get("/predict")
async def create_item(Size:int,
                        Price:int,
                        Number_of_Person:int,
                        Can_add_Child:int,
                        Smoking_Room:int,
                        Sum_of_Facilities:int):

    # Test API Link
    # http://127.0.0.1:5000/predict/?alcohol=12.37&malic_acid=1.17&ash=1.92&alcalinity=19.6&magnesium=78&total_phenols=2.11&flavanoids=2.0&nonflavanoid_phenols=0.27&proanthocyanins=1.04&color_intensity=4.68&hue=1.12&OD280_OD315=3.48&proline=510
    ### http://127.0.0.1:5000/predict/?Size=42&Price=64000&Number_of_Person=1&Can_add_Child=0&Smoking_Room=0&Sum_of_Facilities=1

    # Result Class 0

    instant = [[Size,Price,Number_of_Person,Can_add_Child,Smoking_Room,Sum_of_Facilities]]

    result = model.predict(instant)[0]
    return {"Hotel_Class":int(result)}

if __name__ == '__main__':
    #app.run(threaded=True)
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000, debug=True)