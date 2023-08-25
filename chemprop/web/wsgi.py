"""
Runs the web interface version of Chemprop.
Designed to be used for production only, along with Gunicorn.
"""
from flask import Flask

from chemprop.web.app import app
from chemprop.web.run import setup_web


def build_app(*args, **kwargs) -> Flask:
    setup_web(**kwargs)

    return app
