"""
Runs the web interface version of Chemprop.
This allows for training and predicting in a web browser.
"""

import os

from tap import Tap  # pip install typed-argument-parser (https://github.com/swansonk14/typed-argument-parser)

from chemprop.data import set_cache_graph, set_cache_mol
from chemprop.web.app import app, db
from chemprop.web.utils import clear_temp_folder, set_root_folder



class WebArgs(Tap):
    host: str = '127.0.0.1'  # Host IP address
    port: int = 5000  # Port
    debug: bool = False  # Whether to run in debug mode
    demo: bool = False  # Display only demo features
    initdb: bool = False  # Initialize Database
    root_folder: str | None = None  # Root folder where web data and checkpoints will be saved (defaults to chemprop/web/app)
    allow_checkpoint_upload: bool = False  # Whether to allow checkpoint uploads
    max_molecules: int | None = None  # Maximum number of molecules for which to allow predictions


def setup_web(
        demo: bool = False,
        initdb: bool = False,
        root_folder: str | None = None,
        allow_checkpoint_upload: bool = False,
        max_molecules: int | None = None
) -> None:
    app.config['DEMO'] = demo
    app.config['ALLOW_CHECKPOINT_UPLOAD'] = allow_checkpoint_upload
    app.config['MAX_MOLECULES'] = max_molecules

    # Set up root folder and subfolders
    set_root_folder(
        app=app,
        root_folder=root_folder,
        create_folders=True
    )
    clear_temp_folder(app=app)

    db.init_app(app)

    # Initialize database
    if initdb or not os.path.isfile(app.config['DB_PATH']):
        with app.app_context():
            db.init_db()
            print("-- INITIALIZED DATABASE --")

    # Turn off caching to save memory (assumes no training, only prediction)
    set_cache_graph(False)
    set_cache_mol(False)


def chemprop_web() -> None:
    """Runs the Chemprop website locally.

    This is the entry point for the command line command :code:`chemprop_web`.
    """
    # Parse arguments
    args = WebArgs().parse_args()

    # Set up web app
    setup_web(
        demo=args.demo,
        initdb=args.initdb,
        root_folder=args.root_folder,
        allow_checkpoint_upload=args.allow_checkpoint_upload,
        max_molecules=args.max_molecules
    )

    # Run web app
    app.run(host=args.host, port=args.port, debug=args.debug)
