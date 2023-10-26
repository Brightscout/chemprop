"""Defines a number of routes/views for the flask app."""

import io
import os
import sys
import shutil
import time
import multiprocessing as mp
import tarfile
import zipfile
from functools import wraps
from pathlib import Path
from tempfile import TemporaryDirectory, NamedTemporaryFile
from typing import Callable, List, Tuple
from uuid import uuid4

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from chemfunc.molecular_fingerprints import compute_fingerprints
from flask import after_this_request, json, jsonify, redirect, render_template, request, Response, send_file, send_from_directory, session, url_for
from rdkit import Chem
from rdkit.Chem.Crippen import MolLogP
from rdkit.Chem.Descriptors import MolWt
from rdkit.Chem.QED import qed
from rdkit.Chem.rdMolDescriptors import CalcNumAtomStereoCenters, CalcNumHBA, CalcNumHBD, CalcTPSA
from tqdm import tqdm
from werkzeug.utils import secure_filename

from chemprop.web.app import app, db
from chemprop.web.app.models import (
    compute_drugbank_percentile,
    get_drugbank_dataframe,
    get_drugbank_unique_atc_codes,
    get_drugbank_tasks,
    get_models_dict
)

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))

from chemprop.args import TrainArgs
from chemprop.constants import MODEL_FILE_NAME, TRAIN_LOGGER_NAME
from chemprop.data import (
    get_data,
    get_header,
    get_smiles,
    get_task_names,
    MoleculeDataLoader,
    MoleculeDatapoint,
    MoleculeDataset,
    validate_data
)
from chemprop.train import run_training, predict as chemprop_predict
from chemprop.utils import create_logger, load_args

TRAINING = 0
PROGRESS = mp.Value('d', 0.0)
USER_TO_PREDS = {}


def check_not_demo(func: Callable) -> Callable:
    """
    View wrapper, which will redirect request to site
    homepage if app is run in DEMO mode.
    :param func: A view which performs sensitive behavior.
    :return: A view with behavior adjusted based on DEMO flag.
    """
    @wraps(func)
    def decorated_function(*args, **kwargs):
        if app.config['DEMO']:
            return redirect(url_for('predict'))
        return func(*args, **kwargs)

    return decorated_function


def check_allow_checkpoint_upload(func: Callable) -> Callable:
    """
    View wrapper, which will redirect request to site
    homepage if app is run without allowing checkpoint upload.
    :param func: A view which performs sensitive behavior.
    :return: A view with behavior adjusted based on ALLOW_CHECKPOINT_UPLOAD flag.
    """
    @wraps(func)
    def decorated_function(*args, **kwargs):
        if not app.config['ALLOW_CHECKPOINT_UPLOAD']:
            return redirect(url_for('predict'))
        return func(*args, **kwargs)

    return decorated_function


def progress_bar(args: TrainArgs, progress: mp.Value):
    """
    Updates a progress bar displayed during training.

    :param args: Arguments.
    :param progress: The current progress.
    """
    # no code to handle crashes in model training yet, though
    current_epoch = -1
    while current_epoch < args.epochs - 1:
        if os.path.exists(os.path.join(args.save_dir, 'verbose.log')):
            with open(os.path.join(args.save_dir, 'verbose.log'), 'r') as f:
                content = f.read()
                if 'Epoch ' + str(current_epoch + 1) in content:
                    current_epoch += 1
                    progress.value = (current_epoch + 1) * 100 / args.epochs
        else:
            pass
        time.sleep(0)


def find_unused_path(path: str) -> str:
    """
    Given an initial path, finds an unused path by appending different numbers to the filename.

    :param path: An initial path.
    :return: An unused path.
    """
    if not os.path.exists(path):
        return path

    base_name, ext = os.path.splitext(path)

    i = 2
    while os.path.exists(path):
        path = base_name + str(i) + ext
        i += 1

    return path


def name_already_exists_message(thing_being_named: str, original_name: str, new_name: str) -> str:
    """
    Creates a message about a path already existing and therefore being renamed.

    :param thing_being_named: The thing being renamed (ex. Data, Checkpoint).
    :param original_name: The original name of the object.
    :param new_name: The new name of the object.
    :return: A string with a message about the changed name.
    """
    return f'{thing_being_named} "{original_name} already exists. ' \
           f'Saving to "{new_name}".'


def get_upload_warnings_errors(upload_item: str) -> Tuple[List[str], List[str]]:
    """
    Gets any upload warnings passed along in the request.

    :param upload_item: The thing being uploaded (ex. Data, Checkpoint).
    :return: A tuple with a list of warning messages and a list of error messages.
    """
    warnings_raw = request.args.get(f'{upload_item}_upload_warnings')
    errors_raw = request.args.get(f'{upload_item}_upload_errors')
    warnings = json.loads(warnings_raw) if warnings_raw is not None else None
    errors = json.loads(errors_raw) if errors_raw is not None else None

    return warnings, errors


def format_float(value: float, precision: int = 4) -> str:
    """
    Formats a float value to a specific precision.

    :param value: The float value to format.
    :param precision: The number of decimal places to use.
    :return: A string containing the formatted float.
    """
    return f'{value:.{precision}f}'


def format_float_list(array: List[float], precision: int = 4) -> List[str]:
    """
    Formats a list of float values to a specific precision.

    :param array: A list of float values to format.
    :param precision: The number of decimal places to use.
    :return: A list of strings containing the formatted floats.
    """
    return [format_float(f, precision) for f in array]


#@app.route('/receiver', methods=['POST'])
@check_not_demo
def receiver():
    """Receiver monitoring the progress of training."""
    return jsonify(progress=PROGRESS.value, training=TRAINING)


@app.route('/')
def home():
    """Renders the home page."""
    return redirect(url_for('predict'))


#@app.route('/create_user', methods=['GET', 'POST'])
@check_not_demo
def create_user():
    """
    If a POST request is made, creates a new user.
    Renders the create_user page.
    """
    if request.method == 'GET':
        return render_template('create_user.html', users=db.get_all_users())

    new_name = request.form['newUserName']

    if new_name is not None:
        db.insert_user(new_name)

    return redirect(url_for('create_user'))


def render_train(**kwargs):
    """Renders the train page with specified kwargs."""
    data_upload_warnings, data_upload_errors = get_upload_warnings_errors('data')

    return render_template('train.html',
                           datasets=db.get_datasets(request.cookies.get('currentUser')),
                           cuda=app.config['CUDA'],
                           gpus=app.config['GPUS'],
                           data_upload_warnings=data_upload_warnings,
                           data_upload_errors=data_upload_errors,
                           users=db.get_all_users(),
                           **kwargs)


#@app.route('/train', methods=['GET', 'POST'])
@check_not_demo
def train():
    """Renders the train page and performs training if request method is POST."""
    global PROGRESS, TRAINING

    warnings, errors = [], []

    if request.method == 'GET':
        return render_train()

    # Get arguments
    data_name, epochs, ensemble_size, checkpoint_name = \
        request.form['dataName'], int(request.form['epochs']), \
        int(request.form['ensembleSize']), request.form['checkpointName']
    gpu = request.form.get('gpu')
    data_path = os.path.join(app.config['DATA_FOLDER'], f'{data_name}.csv')
    dataset_type = request.form.get('datasetType', 'regression')
    use_progress_bar = request.form.get('useProgressBar', 'True') == 'True'

    # Create and modify args
    args = TrainArgs().parse_args([
        '--data_path', data_path,
        '--dataset_type', dataset_type,
        '--epochs', str(epochs),
        '--ensemble_size', str(ensemble_size),
    ])

    # Get task names
    args.task_names = get_task_names(path=data_path, smiles_columns=args.smiles_columns)

    # Check if regression/classification selection matches data
    data = get_data(path=data_path, smiles_columns=args.smiles_columns)
    # Set the number of molecules through the length of the smiles_columns for now, we need to add an option to the site later

    targets = data.targets()
    unique_targets = {target for row in targets for target in row if target is not None}

    if dataset_type == 'classification' and len(unique_targets - {0, 1}) > 0:
        errors.append('Selected classification dataset but not all labels are 0 or 1. Select regression instead.')

        return render_train(warnings=warnings, errors=errors)

    if dataset_type == 'regression' and unique_targets <= {0, 1}:
        errors.append('Selected regression dataset but all labels are 0 or 1. Select classification instead.')

        return render_train(warnings=warnings, errors=errors)

    if gpu is not None:
        if gpu == 'None':
            args.cuda = False
        else:
            args.gpu = int(gpu)

    current_user = request.cookies.get('currentUser')

    if not current_user:
        # Use DEFAULT as current user if the client's cookie is not set.
        current_user = app.config['DEFAULT_USER_ID']

    ckpt_id, ckpt_name = db.insert_ckpt(checkpoint_name,
                                        current_user,
                                        args.dataset_type,
                                        args.epochs,
                                        args.ensemble_size,
                                        len(targets))

    with TemporaryDirectory() as temp_dir:
        args.save_dir = temp_dir

        if use_progress_bar:
            process = mp.Process(target=progress_bar, args=(args, PROGRESS))
            process.start()
            TRAINING = 1

        # Run training
        logger = create_logger(name=TRAIN_LOGGER_NAME, save_dir=args.save_dir, quiet=args.quiet)
        task_scores = run_training(args, data, logger)[args.metrics[0]]

        if use_progress_bar:
            process.join()

            # Reset globals
            TRAINING = 0
            PROGRESS = mp.Value('d', 0.0)

        # Check if name overlap
        if checkpoint_name != ckpt_name:
            warnings.append(name_already_exists_message('Checkpoint', checkpoint_name, ckpt_name))

        # Move models
        for root, _, files in os.walk(args.save_dir):
            for fname in files:
                if fname.endswith('.pt'):
                    model_id = db.insert_model(ckpt_id)
                    save_path = os.path.join(app.config['CHECKPOINT_FOLDER'], f'{model_id}.pt')
                    shutil.move(os.path.join(args.save_dir, root, fname), save_path)

    return render_train(trained=True,
                        metric=args.metric,
                        num_tasks=len(args.task_names),
                        task_names=args.task_names,
                        task_scores=format_float_list(task_scores),
                        mean_score=format_float(np.mean(task_scores)),
                        warnings=warnings,
                        errors=errors)


def render_predict(**kwargs):
    """Renders the predict page with specified kwargs"""
    checkpoint_upload_warnings, checkpoint_upload_errors = get_upload_warnings_errors('checkpoint')

    return render_template('predict.html',
                           checkpoints=db.get_ckpts(request.cookies.get('currentUser')),
                           cuda=app.config['CUDA'],
                           gpus=app.config['GPUS'],
                           max_molecules=app.config['MAX_MOLECULES'],
                           checkpoint_upload_warnings=checkpoint_upload_warnings,
                           checkpoint_upload_errors=checkpoint_upload_errors,
                           users=db.get_all_users(),
                           drugbank_atc_codes=['all'] + get_drugbank_unique_atc_codes(),
                           drugbank_tasks=get_drugbank_tasks(),
                           **kwargs)


def predict_all_models(
        smiles: list[str],
        num_workers: int = 0
) -> tuple[list[str], list[list[float]]]:
    """Make prediction with all the loaded models.

    TODO: Support GPU prediction.
    TODO: Handle invalid SMILES.

    :param smiles: A list of SMILES.
    :param num_workers: The number of workers for parallel data loading.
    :return: A tuple containing a list of task names and a list of predictions (num_molecules, num_tasks).
    """
    # Get models dict with models and association information
    models_dict = get_models_dict()

    # Determine fingerprints use
    uses_fingerprints_set = {model_dict["uses_fingerprints"] for model_dict in models_dict.values()}
    any_fingerprints_use = any(uses_fingerprints_set)
    all_fingerprints_use = all(uses_fingerprints_set)

    # Build data loader without fingerprints
    if not all_fingerprints_use:
        data_loader_without_fingerprints = MoleculeDataLoader(
            dataset=MoleculeDataset([
                MoleculeDatapoint(
                    smiles=[smile],
                ) for smile in smiles
            ]),
            num_workers=num_workers,
            shuffle=False
        )
    else:
        data_loader_without_fingerprints = None

    # Build dataloader with fingerprints
    if any_fingerprints_use:
        # TODO: Remove assumption of RDKit fingerprints
        fingerprints = compute_fingerprints(smiles, fingerprint_type='rdkit')

        data_loader_with_fingerprints = MoleculeDataLoader(
            dataset=MoleculeDataset(
                [
                    MoleculeDatapoint(
                        smiles=[smile],
                        features=fingerprint
                    )
                    for smile, fingerprint in zip(smiles, fingerprints)
                ]
            ),
            num_workers=num_workers,
            shuffle=False,
        )
    else:
        data_loader_with_fingerprints = None

    # Initialize lists to contain task names and predictions
    all_task_names = []
    all_preds = []

    # Loop through each ensemble and make predictions
    for model_name, model_dict in tqdm(models_dict.items(), desc='model ensembles'):
        # Get task names
        all_task_names += model_dict['task_names']

        # Select data loader based on features use
        if model_dict["uses_fingerprints"]:
            data_loader = data_loader_with_fingerprints
        else:
            data_loader = data_loader_without_fingerprints

        # Make predictions
        preds = [
            chemprop_predict(model=model, data_loader=data_loader)
            for model in tqdm(model_dict['models'], desc='individual models')
        ]

        # Scale predictions if needed (for regression)
        if model_dict['scalers'][0] is not None:
            preds = [
                scaler.inverse_transform(pred).astype(float)
                for scaler, pred in zip(model_dict['scalers'], preds)
            ]

        # Average ensemble predictions
        preds = np.mean(preds, axis=0).transpose()  # (num_tasks, num_molecules)
        all_preds += preds.tolist()

    # Transpose preds
    all_preds: list[list[float]] = np.array(all_preds).transpose().tolist()  # (num_molecules, num_tasks)

    return all_task_names, all_preds


def lipinski_rule_of_five(mol: Chem.Mol) -> float:
    """Determines how many of the Lipinski rules are satisfied by the molecule.

    :param mol: An RDKit molecule.
    :return: The number of Lipinski rules satisfied by the molecule.
    """
    return float(sum([
        MolWt(mol) <= 500,
        MolLogP(mol) <= 5,
        CalcNumHBA(mol) <= 10,
        CalcNumHBD(mol) <= 5
    ]))


def compute_physicochemical_properties(
        smiles: list[str],
) -> tuple[list[str], list[list[float]]]:
    """Compute physicochemical properties for a list of molecules.

    :param smiles: A list of SMILES.
    :return: A tuple containing a list of property names and a list of properties (num_molecules, num_properties).
    """
    # Compute RDKit molecules
    # TODO: avoid recomputing the RDKit molecules if they have to be computed for Chemprop anyway
    # TODO: handle invalid molecules
    mols = [Chem.MolFromSmiles(smile) for smile in smiles]

    # Set up physicochemical property functions
    property_functions = {
        "molecular_weight": MolWt,
        "logP": MolLogP,
        "hydrogen_bond_acceptors": CalcNumHBA,
        "hydrogen_bond_donors": CalcNumHBD,
        "Lipinski": lipinski_rule_of_five,
        "QED": qed,
        "stereo_centers": CalcNumAtomStereoCenters,
        "tpsa": CalcTPSA,
    }

    # Compute properties
    property_names = list(property_functions)
    properties = [
        [property_function(mol) for mol in mols]
        for property_name, property_function in property_functions.items()
    ]

    return property_names, properties


def create_drugbank_reference_plot(
        preds_df: pd.DataFrame,
        x_task: str | None = None,
        y_task: str | None = None,
        atc_code: str | None = None
) -> str:
    """Creates a 2D scatter plot of the DrugBank reference set vs the new set of molecules on two tasks.

    :param preds_df: A DataFrame containing the predictions on the new molecules.
    :param x_task: The name of the task to plot on the x-axis.
    :param y_task: The name of the task to plot on the y-axis.
    :param atc_code: The ATC code to filter the DrugBank reference set by.
    :return: A string containing the SVG of the plot.
    """
    # Set default values
    if x_task is None:
        x_task = "HIA_Hou"

    if y_task is None:
        y_task = "ClinTox"

    if atc_code is None:
        atc_code = "all"

    # Get DrugBank reference, optionally filtered ATC code
    drugbank = get_drugbank_dataframe(atc_code=atc_code)

    # Scatter plot of DrugBank molecules with density coloring
    sns.scatterplot(
        x=drugbank[x_task],
        y=drugbank[y_task],
        edgecolor=None,
        label="DrugBank Approved" + (" (ATC filter)" if atc_code != "all" else ""),
    )

    # Set input label
    input_label = "Input Molecule" + ("s" if len(preds_df) > 1 else "")

    # Scatter plot of new molecules
    if len(preds_df) > 0:
        sns.scatterplot(
            x=preds_df[x_task], y=preds_df[y_task], color="red", marker="*", s=200,
            label=input_label,
        )

    # Title
    plt.title(f"{input_label} vs DrugBank Approved" + (f"\nATC = {atc_code}" if atc_code != "all" else ""))

    # Save plot as svg to pass to frontend
    buf = io.BytesIO()
    plt.savefig(buf, format="svg")
    plt.close()
    buf.seek(0)
    drugbank_svg = buf.getvalue().decode('utf-8')

    return drugbank_svg


@app.route('/drugbank_plot', methods=['GET'])
def drugbank_plot():
    # Get requested ATC code
    session["atc_code"] = request.args.get('atc_code', default=session.get("atc_code"), type=str)

    # Get requested X and Y axes
    session["drugbank_x_task"] = request.args.get('x_task', default=session.get("drugbank_x_task"), type=str)
    session["drugbank_y_task"] = request.args.get('y_task', default=session.get("drugbank_y_task"), type=str)

    # Create DrugBank reference plot with ATC code
    drugbank_plot_svg = create_drugbank_reference_plot(
        preds_df=USER_TO_PREDS.get(session["user_id"], pd.DataFrame()),
        x_task=session["drugbank_x_task"],
        y_task=session["drugbank_y_task"],
        atc_code=session["atc_code"]
    )

    return jsonify({"svg": drugbank_plot_svg})


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    """Renders the predict page and makes predictions if the method is POST."""
    # Assign user ID to session
    if "user_id" not in session:
        session["user_id"] = uuid4().hex

    if request.method == 'GET':
        return render_predict()

    if request.form['textSmiles'] != '':
        smiles = request.form['textSmiles'].split()
    elif request.form['drawSmiles'] != '':
        smiles = [request.form['drawSmiles']]
    else:
        # Upload data file with SMILES
        data = request.files['data']
        data_name = secure_filename(data.filename)

        with TemporaryDirectory() as temp_dir:
            data_path = os.path.join(temp_dir, data_name)
            data.save(data_path)

            # Check if header is smiles
            possible_smiles = get_header(data_path)[0]
            smiles = [possible_smiles] if Chem.MolFromSmiles(possible_smiles) is not None else []

            # Get remaining smiles
            smiles.extend(get_smiles(data_path))

    # Error if too many molecules
    if app.config['MAX_MOLECULES'] is not None and len(smiles) > app.config['MAX_MOLECULES']:
        return render_predict(errors=[f'Received too many molecules. '
                                      f'Maximum number of molecules is {app.config["MAX_MOLECULES"]:,}.'])

    # TODO: validate that SMILES are valid, remove invalid ones, and put a warning if there are invalid ones

    # Make predictions
    task_names, preds = predict_all_models(smiles=smiles)
    num_tasks = len(task_names)

    # TODO: Display physicochemical properties (and compare to DrugBank)
    physchem_names, physchem_preds = compute_physicochemical_properties(smiles=smiles)

    # Compute DrugBank percentiles
    preds_numpy = np.array(preds).transpose()  # (num_tasks, num_molecules)
    drugbank_percentiles = np.stack([
        compute_drugbank_percentile(task_name=task_name, predictions=task_preds)
        for task_name, task_preds in zip(task_names, preds_numpy)
    ]).transpose()  # (num_molecules, num_tasks)

    # Save predictions
    preds_dicts = []
    for smiles_index, smile in enumerate(smiles):
        preds_dict = {"smiles": smile}

        for task_index, task_name in enumerate(task_names):
            preds_dict[task_name] = preds[smiles_index][task_index]
            preds_dict[f"{task_name}_drugbank_approved_percentile"] = drugbank_percentiles[smiles_index][task_index]

        preds_dicts.append(preds_dict)

    # Convert predictions to DataFrame
    preds_df = pd.DataFrame(preds_dicts)

    # Store predictions in memory
    # TODO: figure out how to remove predictions from memory once no longer needed (i.e., once session ends)
    USER_TO_PREDS[session["user_id"]] = preds_df

    # Handle invalid SMILES
    if all(p is None for p in preds):
        return render_predict(errors=['All SMILES are invalid'])

    # Replace invalid smiles with message
    invalid_smiles_warning = 'Invalid SMILES String'
    preds = [pred if pred is not None else [invalid_smiles_warning] * num_tasks for pred in preds]

    # Create DrugBank reference plot
    drugbank_plot_svg = create_drugbank_reference_plot(
        preds_df=USER_TO_PREDS[session["user_id"]],
        x_task=session.get("drugbank_x_task"),
        y_task=session.get("drugbank_y_task"),
        atc_code=session.get("atc_code")
    )

    cat_file = 'chemprop/web/app/categoryData.csv'
    cat_df = pd.read_csv(cat_file)

    dist_file = 'chemprop/web/app/distData.csv'
    dist_df = pd.read_csv(dist_file)

    meta_file = 'chemprop/web/app/metabolismData.csv'
    meta_df = pd.read_csv(meta_file)

    excretion_file = 'chemprop/web/app/excretionData.csv'
    excr_df = pd.read_csv(excretion_file)

    tox_file = 'chemprop/web/app/toxData.csv'
    tox_df = pd.read_csv(tox_file)

    return render_predict(predicted=True,
                          smiles=smiles,
                          num_smiles=min(10, len(smiles)),
                          show_more=max(0, len(smiles)-10),
                          task_names=task_names,
                          num_tasks=num_tasks,
                          cat = cat_df,
                          dist = dist_df,
                          meta = meta_df,
                          excr = excr_df,
                          tox = tox_df,
                          preds=preds,
                          drugbank_percentiles=drugbank_percentiles,
                          drugbank_plot=drugbank_plot_svg,
                          warnings=["List contains invalid SMILES strings"] if None in preds else None,
                          errors=["No SMILES strings given"] if len(preds) == 0 else None)


@app.route('/download_predictions')
def download_predictions() -> Response:
    """Downloads predictions as a .csv file."""
    preds_file = NamedTemporaryFile()

    @after_this_request
    def remove_file(response: Response) -> Response:
        preds_file.close()
        return response

    USER_TO_PREDS.get(session["user_id"], pd.DataFrame()).to_csv(preds_file.name, index=False)
    preds_file.seek(0)

    return send_file(preds_file.name, as_attachment=True, download_name='predictions.csv')


#@app.route('/data')
@check_not_demo
def data():
    """Renders the data page."""
    data_upload_warnings, data_upload_errors = get_upload_warnings_errors('data')

    return render_template('data.html',
                           datasets=db.get_datasets(request.cookies.get('currentUser')),
                           data_upload_warnings=data_upload_warnings,
                           data_upload_errors=data_upload_errors,
                           users=db.get_all_users())


#@app.route('/data/upload/<string:return_page>', methods=['POST'])
@check_not_demo
def upload_data(return_page: str):
    """
    Uploads a data .csv file.

    :param return_page: The name of the page to render to after uploading the dataset.
    """
    warnings, errors = [], []

    current_user = request.cookies.get('currentUser')

    if not current_user:
        # Use DEFAULT as current user if the client's cookie is not set.
        current_user = app.config['DEFAULT_USER_ID']

    dataset = request.files['dataset']

    with NamedTemporaryFile() as temp_file:
        dataset.save(temp_file.name)
        dataset_errors = validate_data(temp_file.name)

        if len(dataset_errors) > 0:
            errors.extend(dataset_errors)
        else:
            dataset_name = request.form['datasetName']
            # dataset_class = load_args(ckpt).dataset_type  # TODO: SWITCH TO ACTUALLY FINDING THE CLASS

            dataset_id, new_dataset_name = db.insert_dataset(dataset_name, current_user, 'UNKNOWN')

            dataset_path = os.path.join(app.config['DATA_FOLDER'], f'{dataset_id}.csv')

            if dataset_name != new_dataset_name:
                warnings.append(name_already_exists_message('Data', dataset_name, new_dataset_name))

            shutil.copy(temp_file.name, dataset_path)

    warnings, errors = json.dumps(warnings), json.dumps(errors)

    return redirect(url_for(return_page, data_upload_warnings=warnings, data_upload_errors=errors))


#@app.route('/data/download/<int:dataset>')
@check_not_demo
def download_data(dataset: int):
    """
    Downloads a dataset as a .csv file.

    :param dataset: The id of the dataset to download.
    """
    return send_from_directory(app.config['DATA_FOLDER'], f'{dataset}.csv', as_attachment=True)


#@app.route('/data/delete/<int:dataset>')
@check_not_demo
def delete_data(dataset: int):
    """
    Deletes a dataset.

    :param dataset: The id of the dataset to delete.
    """
    db.delete_dataset(dataset)
    os.remove(os.path.join(app.config['DATA_FOLDER'], f'{dataset}.csv'))
    return redirect(url_for('data'))


#@app.route('/checkpoints')
@check_not_demo
def checkpoints():
    """Renders the checkpoints page."""
    checkpoint_upload_warnings, checkpoint_upload_errors = get_upload_warnings_errors('checkpoint')

    return render_template('checkpoints.html',
                           checkpoints=db.get_ckpts(request.cookies.get('currentUser')),
                           checkpoint_upload_warnings=checkpoint_upload_warnings,
                           checkpoint_upload_errors=checkpoint_upload_errors,
                           users=db.get_all_users())


@app.route('/checkpoints/upload/<string:return_page>', methods=['POST'])
@check_not_demo
@check_allow_checkpoint_upload
def upload_checkpoint(return_page: str):
    """
    Uploads a checkpoint file or directory.

    .pt: single model upload.
    .zip: ensemble upload.
    .tar.gz: directory of ensembles upload.

    :param return_page: The name of the page to render after uploading the checkpoint file.
    """
    warnings, errors = [], []

    current_user = request.cookies.get('currentUser')

    if not current_user:
        # Use DEFAULT as current user if the client's cookie is not set.
        current_user = app.config['DEFAULT_USER_ID']

    ckpt = request.files['checkpoint']

    ckpt_name = request.form['checkpointName']
    ckpt_path = Path(ckpt.filename)

    # Collect paths to all uploaded checkpoints (and unzip if necessary)
    temp_dir = TemporaryDirectory()
    temp_dir_path = Path(temp_dir.name)

    if ckpt_path.suffix == '.pt':
        ckpt_path = temp_dir_path / MODEL_FILE_NAME
        ckpt.save(ckpt_path)
        ensemble_ckpt_dirs = [temp_dir_path]
        ensemble_names = [ckpt_name]

    elif ckpt_path.suffix == '.zip':
        ckpt_dir = temp_dir_path / 'models'
        zip_path = temp_dir_path / 'models.zip'
        ckpt.save(zip_path)

        with zipfile.ZipFile(zip_path, mode='r') as z:
            z.extractall(ckpt_dir)

        ensemble_ckpt_dirs = [ckpt_dir]
        ensemble_names = [ckpt_name]

    elif ckpt_path.name.endswith('.tar.gz'):
        ckpt_dir = temp_dir_path / 'models'
        tar_path = temp_dir_path / 'models.tar.gz'
        ckpt.save(tar_path)

        with tarfile.open(tar_path, 'r:gz') as tar:
            tar.extractall(ckpt_dir)

        ensemble_ckpt_dirs = list(ckpt_dir.iterdir())
        ensemble_names = [ckpt_dir.name for ckpt_dir in ensemble_ckpt_dirs]

    else:
        errors.append(f'Uploaded checkpoint(s) file must be either .pt or .zip or .tar.gz but got {ckpt_path.suffix}')
        ensemble_ckpt_dirs = []
        ensemble_names = []

    # Insert checkpoints into database
    for ensemble_ckpt_dir, ensemble_name in zip(ensemble_ckpt_dirs, ensemble_names):
        ckpt_paths = list(ensemble_ckpt_dir.glob('**/*.pt'))

        if len(ckpt_paths) > 0:
            ckpt_args = load_args(ckpt_paths[0])
            ckpt_id, new_ckpt_name = db.insert_ckpt(ensemble_name,
                                                    current_user,
                                                    ckpt_args.dataset_type,
                                                    ckpt_args.epochs,
                                                    len(ckpt_paths),
                                                    ckpt_args.train_data_size)

            for ckpt_path in ckpt_paths:
                model_id = db.insert_model(ckpt_id)
                model_path = Path(app.config['CHECKPOINT_FOLDER']) / f'{model_id}.pt'

                if ensemble_name != new_ckpt_name:
                    warnings.append(name_already_exists_message('Checkpoint', ckpt_name, new_ckpt_name))

                shutil.copy(ckpt_path, model_path)

    temp_dir.cleanup()

    warnings, errors = json.dumps(warnings), json.dumps(errors)

    return redirect(url_for(return_page, checkpoint_upload_warnings=warnings, checkpoint_upload_errors=errors))


#@app.route('/checkpoints/download/<int:checkpoint>')
@check_not_demo
def download_checkpoint(checkpoint: int):
    """
    Downloads a zip of model .pt files.

    :param checkpoint: The name of the checkpoint to download.
    """
    ckpt = db.query_db(f'SELECT * FROM ckpt WHERE id = {checkpoint}', one=True)
    models = db.get_models(checkpoint)

    model_data = io.BytesIO()

    with zipfile.ZipFile(model_data, mode='w') as z:
        for model in models:
            model_path = os.path.join(app.config['CHECKPOINT_FOLDER'], f'{model["id"]}.pt')
            z.write(model_path, os.path.basename(model_path))

    model_data.seek(0)

    return send_file(
        model_data,
        mimetype='application/zip',
        as_attachment=True,
        attachment_filename=f'{ckpt["ckpt_name"]}.zip',
        cache_timeout=-1
    )


#@app.route('/checkpoints/delete/<int:checkpoint>')
@check_not_demo
def delete_checkpoint(checkpoint: int):
    """
    Deletes a checkpoint file.

    :param checkpoint: The id of the checkpoint to delete.
    """
    db.delete_ckpt(checkpoint)
    return redirect(url_for('checkpoints'))
