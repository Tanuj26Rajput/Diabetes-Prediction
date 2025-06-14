import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle

raw_df = pd.read_csv("diabetes.csv")

train_val_df, test_df = train_test_split(raw_df, test_size=0.2, random_state=42)
train_df, val_df = train_test_split(train_val_df, test_size=0.25, random_state=42)

input_cols = list(train_df.columns[:-1])
target_col = "Outcome"

train_inputs = train_df[input_cols].copy()
train_target = train_df[target_col].copy()

val_inputs = val_df[input_cols].copy()
val_target = val_df[target_col].copy()

test_inputs = test_df[input_cols].copy()
test_target = test_df[target_col].copy()

numeric_cols = train_inputs.select_dtypes(include=np.number).columns.tolist()

imputer = SimpleImputer(strategy="mean")
imputer.fit(train_inputs[numeric_cols])

train_inputs[numeric_cols] = imputer.transform(train_inputs[numeric_cols])
val_inputs[numeric_cols] = imputer.transform(val_inputs[numeric_cols])
test_inputs[numeric_cols] = imputer.transform(test_inputs[numeric_cols])

scaler = MinMaxScaler()
scaler.fit(train_inputs[numeric_cols])

train_inputs = scaler.transform(train_inputs[numeric_cols])
val_inputs = scaler.transform(val_inputs[numeric_cols])
test_inputs = scaler.transform(test_inputs[numeric_cols])

model = LogisticRegression(solver='liblinear')
model.fit(train_inputs, train_target)

with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("imputer.pkl", "wb") as f:
    pickle.dump(imputer, f)

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)