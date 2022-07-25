import pickle
import sys
from pathlib import Path

src_path = Path(__file__).parent.parent.resolve()
sys.path.append(str(src_path))
from dvclive.lgbm import DvcLiveCallback
import argparse

import pandas as pd
from joblib import dump
from sklearn.ensemble import RandomForestClassifier
from utils.load_params import load_params


def train(models_dir, model_fname, data_dir, random_state, **train_params):
    X_train = pd.read_pickle(data_dir / "X_train.pkl")
    y_train = pd.read_pickle(data_dir / "y_train.pkl")

    X_val = pd.read_pickle(data_dir / "X_val.pkl")
    y_val = pd.read_pickle(data_dir / "y_val.pkl")

    import lightgbm as lgb

    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, y_val)
    lgb_params = {"objective": "binary"}
    lgb_params["metric"] = "auc"
    lgb_params.update(train_params)
    clf = lgb.train(
        lgb_params, train_data, valid_sets=[val_data], callbacks=[DvcLiveCallback()]
    )

    models_dir.mkdir(exist_ok=True)

    clf.save_model(models_dir / model_fname, num_iteration=clf.best_iteration)


if __name__ == "__main__":
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument("--config", dest="config", required=True)
    args = args_parser.parse_args()

    params = load_params(params_path=args.config)
    data_dir = Path(params.base.data_dir)
    models_dir = Path(params.train.models_dir)
    model_fname = params.train.model_fname
    random_state = params.base.random_state
    cat_cols = params.base.cat_cols
    num_cols = params.base.num_cols
    train_params = params.train.lightgbm

    train(
        models_dir=models_dir,
        model_fname=model_fname,
        data_dir=data_dir,
        random_state=random_state,
        **train_params
    )
