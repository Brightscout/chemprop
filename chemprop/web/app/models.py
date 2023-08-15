"""Defines a number of model helper functions."""
import os

from chemprop.web.app import app, db
from chemprop.utils import load_args, load_checkpoint, load_scalers, load_task_names


MODELS_DICT = {}


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
