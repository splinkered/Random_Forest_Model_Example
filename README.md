This is an app that uses a model trained using the random forest algorithm to use data as given on a dataset in Kaggle of Heart Attack Dataset Zheen hospital in Erbil, Iraq. To predict whether the given input parameters would reap positive or negative result for heart attack.

Dataset link: https://www.kaggle.com/datasets/fatemehmohammadinia/heart-attack-dataset-tarik-a-rashid

I trained the model using the sklearn library in python and pandas in google colab notebook to read the csv file of my dataset.
Deployment done using fastAPI to serve frontend and Render for hosting

Application Link: https://random-forest-model-example.onrender.com/

Model was trained in Google Colab Notebook using the code I wrote below

```
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import pickle

# Here, heart.csv is my Medicaldataset.csv
filename = "heart.csv"
df = pd.read_csv(filename)
print("Dataset Loaded Successfully!")

label_encoder = LabelEncoder()
df["Result_encoded"] = label_encoder.fit_transform(df["Result"])  # 0 means negative and 1 = positive

X = df.drop(["Result", "Result_encoded"], axis=1)
y = df["Result_encoded"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# Train using RandomForest model
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# Generate saved model and label encoder
with open("heart_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)

print("Model and LabelEncoder saved successfully!")

# Example prediction using the trained models
new_patient = {
    "Age": 60,
    "Gender": 1,
    "Heart rate": 88,
    "Systolic blood pressure": 190,
    "Diastolic blood pressure": 90,
    "Blood sugar": 10.0,
    "CK-MB": 3.5,
    "Troponin": 0.045
}

new_df = pd.DataFrame([new_patient])
prediction_num = model.predict(new_df)[0]
prediction_label = label_encoder.inverse_transform([prediction_num])[0]

print("Predicted Result:", prediction_label)

```
