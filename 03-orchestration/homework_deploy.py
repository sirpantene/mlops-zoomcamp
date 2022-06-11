import pickle

import pandas as pd

from dateutil.parser import parse
from dateutil.relativedelta import relativedelta
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

import mlflow

from prefect import flow, task
from prefect.task_runners import SequentialTaskRunner
from prefect.flow_runners import SubprocessFlowRunner
from prefect.deployments import DeploymentSpec
from prefect.orion.schemas.schedules import CronSchedule

@task
def get_paths(date):
    date_fmt_yymm = "%Y-%m"
    train_date = (parse(date) - relativedelta(months=2)).strftime(date_fmt_yymm)
    val_date = (parse(date) - relativedelta(months=1)).strftime(date_fmt_yymm)

    train_path = f"./data/fhv_tripdata_{train_date}.parquet"
    val_path = f"./data/fhv_tripdata_{val_date}.parquet"
    print(train_path, val_path)
    return train_path, val_path


@task
def read_data(path):
    df = pd.read_parquet(path)
    print(path, df.shape)
    return df


@task
def prepare_features(df, categorical, train=True):
    df['duration'] = df.dropOff_datetime - df.pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60
    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    mean_duration = df.duration.mean()
    if train:
        print(f"The mean duration of training is {mean_duration}")
    else:
        print(f"The mean duration of validation is {mean_duration}")

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    return df


@task
def train_model(df, categorical):
    train_dicts = df[categorical].to_dict(orient='records')
    dv = DictVectorizer()
    X_train = dv.fit_transform(train_dicts)
    y_train = df.duration.values

    print(f"The shape of X_train is {X_train.shape}")
    print(f"The DictVectorizer has {len(dv.feature_names_)} features")

    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_train)
    mse = mean_squared_error(y_train, y_pred, squared=False)
    print(f"The MSE of training is: {mse}")

    return lr, dv


@task
def run_model(df, categorical, dv, lr):
    val_dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(val_dicts)
    y_pred = lr.predict(X_val)
    y_val = df.duration.values

    mse = mean_squared_error(y_val, y_pred, squared=False)
    print(f"The MSE of validation is: {mse}")
    return


@flow(task_runner=SequentialTaskRunner())
def main(date=None):
    mlflow.set_tracking_uri("http://mlflow_server:5000")
    mlflow.set_experiment("03-orchestration")

    with mlflow.start_run():
        train_path, val_path = get_paths(date).result()

        categorical = ['PUlocationID', 'DOlocationID']

        df_train = read_data(train_path)
        df_train_processed = prepare_features(df_train, categorical)

        df_val = read_data(val_path)
        df_val_processed = prepare_features(df_val, categorical, False)

        # train the model
        lr, dv = train_model(df_train_processed, categorical).result()
        run_model(df_val_processed, categorical, dv, lr)

        dv_path = f"models/dv-{date}.bin"
        model_path = f"models/model-{date}.bin"
        with open(model_path, 'wb') as f_model, open(dv_path, 'wb') as f_dv:
            pickle.dump(lr, f_model)
            pickle.dump(dv, f_dv)
        mlflow.log_artifact(model_path, artifact_path="model")
        mlflow.log_artifact(dv_path, artifact_path="model")


schedule = CronSchedule(cron="0 9 15 * *")


DeploymentSpec(
    flow=main,
    name="model_training",
    schedule=schedule,
    flow_runner=SubprocessFlowRunner(),
    tags=["ml"]
)
