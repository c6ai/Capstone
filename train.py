
from azureml.core.run import Run
run = Run.get_context()
try:
    run  = Run.get_context()
    ws = run.experiment.workspace # !exp
except:
    ws = Workspace.from_config()

import argparse, joblib, os
import numpy as np
import pandas as pd
from azureml.core import Workspace, Dataset, Run
from azureml.core.run import Run
from azureml.core.model import Model  # for Model Deserialization Opt#1
from azureml.core.dataset import Dataset
from azureml.data.dataset_factory import TabularDatasetFactory
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

dataset = Dataset.get_by_name(ws, name='IDS2017Infilteration') # !data
ds = dataset.to_pandas_dataframe()

def clean_data(data):
    # del data['Flow Bytes/s'] # df
    # del data[' Flow Packets/s'] # df
    csv_data = "https://workspace1st4305015718.blob.core.windows.net/public/IDS2017-Infilteration.csv"
    columns_to_be_removed = ['Flow Bytes/s', ' Flow Packets/s']
    data = pd.read_csv(csv_data).drop(columns_to_be_removed, axis = 'columns') # dataset
    
    x_df = data.dropna()
    y_df = x_df.pop(" Label").apply(lambda s: 1 if s == "BENIGN" else 0)
    return x_df,y_df


if "outputs" not in os.listdir():
    os.mkdir("./outputs")

# x, y = clean_data(ds)
# x_train, x_test, y_train, y_test = train_test_split(
#     x, y, test_size=0.25, random_state=0
# )

def main():
    x, y = clean_data(ds)
    x_train, x_test, y_train,y_test = train_test_split(x,y, test_size=0.25, random_state=0)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--C",
        type=float,
        default=1.0,
        help="Inverse of regularization strength. Smaller values cause stronger regularization",
    )
    parser.add_argument(
        "--max_iter",
        type=int,
        default=10, # 100 # 1000
        help="Maximum number of iterations to converge",
    )
    args = parser.parse_args()
    run.log("Regularization Strength:", np.float(args.C))
    run.log("Max iterations:", np.int(args.max_iter))
    model = LogisticRegression(C=args.C, max_iter=args.max_iter).fit(x_train, y_train)
    accuracy = model.score(x_test, y_test)
    # run.log('Kernel type', np.string(args.kernel))
    # run.log('Penalty', np.float(args.penalty))
    run.log('Accuracy', np.float(accuracy))
    # run.log("overall accuracy", acc) # log a single value
    # run.log_list("errors", error_list) # log a list of values
    # run.log_row("boundaries", xmin=0, xmax=1, ymin=-1, ymax=1) # log arbitrary key/value pairs
    # run.log_image("AUC plot", plt) # log a matplotlib plot
    joblib.dump(model, './outputs/best-hd-model.pkl') # model.pkl
if __name__ == "__main__":
    main()


