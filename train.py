from sklearn.linear_model import LogisticRegression #1
import argparse #1
import os
import numpy as np #1
from sklearn.metrics import mean_squared_error
import joblib #1
from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import OneHotEncoder
import pandas as pd
from azureml.core.run import Run#1
from azureml.data.dataset_factory import TabularDatasetFactory
from azureml.core import Dataset

# TODO: Create TabularDataset using TabularDatasetFactory
# Data is located at:
# "https://automlsamplenotebookdata.blob.core.windows.net/automl-sample-notebook-data/bankmarketing_train.csv"
run = Run.get_context() #1

def clean_data(data):#1

    # Clean and one hot encode data
    # x_df = data.to_pandas_dataframe().dropna()
    # jobs = pd.get_dummies(x_df.job, prefix="job")
    # x_df.drop("job", inplace=True, axis=1)
    # x_df = x_df.join(jobs)
    # x_df["marital"] = x_df.marital.apply(lambda s: 1 if s == "married" else 0)
    # x_df["default"] = x_df.default.apply(lambda s: 1 if s == "yes" else 0)
    # x_df["housing"] = x_df.housing.apply(lambda s: 1 if s == "yes" else 0)
    # x_df["loan"] = x_df.loan.apply(lambda s: 1 if s == "yes" else 0)
    # contact = pd.get_dummies(x_df.contact, prefix="contact")
    # x_df.drop("contact", inplace=True, axis=1)
    # x_df = x_df.join(contact)
    # education = pd.get_dummies(x_df.education, prefix="education")
    # x_df.drop("education", inplace=True, axis=1)
    # x_df = x_df.join(education)
    # x_df["month"] = x_df.month.map(months)
    # x_df["day_of_week"] = x_df.day_of_week.map(weekdays)
    # x_df["poutcome"] = x_df.poutcome.apply(lambda s: 1 if s == "success" else 0)
    
    x_df = data.to_pandas_dataframe()
    
    y_df = x_df.pop("DEATH_EVENT")

    return x_df,y_df

data = pd.read_csv("heart_failure_clinical_records_dataset.csv")

ds = Dataset.Tabular.from_delimited_files(data)

x, y = clean_data(ds) #1

# TODO: Split data into train and test sets.
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=403,shuffle=True) # 1 

run = Run.get_context() #1

def main():
    # Add arguments to script
    parser = argparse.ArgumentParser() #1

    parser.add_argument('-f') #1
    
    parser.add_argument('--C', type=float, default=1.0, help="Inverse of regularization strength. Smaller values cause stronger regularization") #1
    parser.add_argument('--max_iter', type=int, default=100, help="Maximum number of iterations to converge") #1

    primary_metric_name="Accuracy" #1
    args = parser.parse_args() #1

    run.log("Regularization Strength:", np.float(args.C)) #1
    run.log("Max iterations:", np.int(args.max_iter)) #1 

    model = LogisticRegression(C=args.C, max_iter=args.max_iter).fit(x_train, y_train) #1
    joblib.dump(model,'outputs/model.joblib') #1
    accuracy = model.score(x_test, y_test) #1
    
    run.log("Accuracy", np.float(accuracy)) #1
if __name__ == '__main__': #1
    main() #1
