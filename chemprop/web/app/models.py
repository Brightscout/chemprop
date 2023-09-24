"""Defines a number of model helper functions."""
import os
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.stats import percentileofscore

from chemprop.web.app import app, db
from chemprop.utils import load_args, load_checkpoint, load_scalers, load_task_names


MODELS_DICT = {}
DRUGBANK_DF = pd.DataFrame()
ATC_CODE_TO_DRUGBANK_INDICES = {}


def load_drugbank_reference() -> None:
    """Loads the reference set of DrugBank approved molecules with their model predictions."""
    # Load DrugBank data and save to global variable
    global DRUGBANK_DF, ATC_CODE_TO_SMILES
    print("-- LOADING DRUGBANK --")
    DRUGBANK_DF = pd.read_csv(app.config["DRUGBANK_PATH"])

    # Map ATC codes to all indices of the DRUGBANK_DF with that ATC code
    atc_code_to_drugbank_indices = defaultdict(set)
    for atc_column in [column for column in DRUGBANK_DF.columns if column.startswith("atc_")]:
        for index, atc_codes in DRUGBANK_DF[atc_column].dropna().items():
            for atc_code in atc_codes.split(";"):
                atc_code_to_drugbank_indices[atc_code.lower()].add(index)

    # Save ATC code to indices mapping to global variable and convert set to sorted list
    global ATC_CODE_TO_DRUGBANK_INDICES
    ATC_CODE_TO_DRUGBANK_INDICES = {
        atc_code: sorted(indices)
        for atc_code, indices in atc_code_to_drugbank_indices.items()
    }

# Create a decorator to load the DrugBank reference DataFrame if it is not already loaded
def preload_drugbank(func: callable) -> callable:
    """Decorator to load the DrugBank reference DataFrame if it is not already loaded."""

    def wrapper(*args, **kwargs):
        if len(DRUGBANK_DF) == 0:
            load_drugbank_reference()

        return func(*args, **kwargs)

    return wrapper


@preload_drugbank
def compute_drugbank_percentile(task_name: str, predictions: np.ndarray) -> np.ndarray:
    """Computes the percentile of the prediction compared to the DrugBank approved molecules.

    :param task_name: The name of the task (property) that is predicted.
    :param predictions: A 1D numpy array of predictions.
    """
    return percentileofscore(DRUGBANK_DF[task_name], predictions)


@preload_drugbank
def get_drugbank_dataframe(atc_code: str = 'all') -> pd.DataFrame:
    """Get the DrugBank reference DataFrame, optionally filtered by ATC code.

    :param atc_code: The ATC code to filter by. If 'all', returns the entire DrugBank.
    """
    if atc_code == 'all':
        return DRUGBANK_DF

    return DRUGBANK_DF.loc[ATC_CODE_TO_DRUGBANK_INDICES[atc_code]]


@preload_drugbank
def get_drugbank_unique_atc_codes() -> list[str]:
    """Get the unique ATC codes in the DrugBank reference set."""
    return sorted({
        atc_code.lower()
        for atc_column in [column for column in DRUGBANK_DF.columns if column.startswith("atc_")]
        for atc_codes in DRUGBANK_DF[atc_column].dropna().str.split(";")
        for atc_code in atc_codes
    })


def load_models() -> None:
    """Loads all of the models in the database into memory.

    Note: This will not work with alternate user IDs.
    """
    print("-- LOADING MODELS --")

    # Loop through each checkpoint ID and add the model ensemble to MODELS_DICT
    for ckpt_id in db.get_ckpts(app.config['DEFAULT_USER_ID']):
        # Get model paths for models in ensemble
        model_paths = [
            os.path.join(app.config["CHECKPOINT_FOLDER"], f'{model["id"]}.pt')
            for model in db.get_models(ckpt_id["id"])
        ]

        # Load train args
        train_args = load_args(model_paths[0])

        # Add task names, models, and scalers to MODELS_DICT
        MODELS_DICT[ckpt_id] = {
            "task_names": load_task_names(model_paths[0]),
            "models": [
                load_checkpoint(path=str(model_path)).eval()
                for model_path in model_paths
            ],
            "scalers": [
                load_scalers(path=str(model_path))[0] for model_path in model_paths
            ],
            "uses_fingerprints": train_args.features_path is not None
            or train_args.features_generator is not None,
        }


def get_models_dict() -> dict:
    """Gets a dictionary of models and their associated information (models, scalers, task names).

    :return: A dictionary of models and their associated information (models, scalers, task names).
    """
    if not MODELS_DICT:
        load_models()

    return MODELS_DICT
