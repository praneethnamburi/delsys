# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys

sys.path.insert(0, os.path.abspath("../src"))

from delsys import __version__

project = "delsys"
copyright = "2026, Praneeth Namburi"
author = "Praneeth Namburi"
version = __version__
release = __version__

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.duration",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.napoleon",  # Google / NumPy style docstrings
    "sphinxcontrib.youtube",
]

# Prefix every section label with its document name so the same heading
# (e.g. "delsys") in index.md and the included README don't collide.
autosectionlabel_prefix_document = True

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_theme_options = {
    "collapse_navigation": False,
}
html_static_path = []

napoleon_use_param = True  # parameter types render in the description, not the signature
napoleon_use_rtype = True  # return type renders in the description, not the signature
napoleon_use_ivar = True   # ``Attributes:`` sections use ``:ivar:`` directives

autodoc_member_order = "bysource"
autodoc_default_options = {"ignore-module-all": True}
autodoc_typehints = "description"  # type hints fold into parameter descriptions

# Use a non-GUI matplotlib backend during the docs build so importing modules
# that touch pyplot doesn't try to open a window on the RTD builders.
import matplotlib

matplotlib.use("Agg")

from sphinx.ext.autodoc import between


def setup(app):
    # Register a sphinx.ext.autodoc.between listener to ignore everything
    # between lines that contain the word IGNORE.
    app.connect("autodoc-process-docstring", between("^.*IGNORE.*$", exclude=True))
    return app
