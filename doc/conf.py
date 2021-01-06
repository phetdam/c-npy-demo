# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

import os.path

# project (repository) root directory
PROJECT_ROOT = "/".join(
    os.path.dirname(os.path.abspath(__file__)).split("/")[:-1]
)

# -- Project information -----------------------------------------------------

project = "c_npy_demo"
copyright = "2021, Derek Huang"
author = "Derek Huang"

# no need to distinguish between release and version here. read from VERSION
with open(PROJECT_ROOT + "/VERSION") as vf:
    version = vf.read().strip()
release = version

# -- General configuration ---------------------------------------------------

# specifiy minimum sphinx version (3)
needs_sphinx = "3.0"

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your own.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx"
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build"]

# -- autodoc configuration ---------------------------------------------------

# set default options for autodoc directives. include __repr__ special member
# and any private members (names prepended with _), show class inheritance.
#
# note: since ignore-module-all is not set, only the members in __all__ in
# __init__.py will be looked for and their order will be maintained. since
# undoc-members was not specified, members with no docstring are skipped.
autodoc_default_options = {
    "members": True,
    "private-members": True,
    "show-inheritance": True,
    "special-members": "__repr__"
}

# -- autosummary configuration -----------------------------------------------

# set to True to generate stub files for any modules named in a file's
# autosummary directive(s). so far, only index.rst should have autosummary.
autosummary_generate = True

# -- intersphinx configuration -----------------------------------------------

# determines which external package documentations to link to
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None)
}

# -- Options for HTML output -------------------------------------------------

# html theme (my favorite theme)
html_theme = "sphinx_rtd_theme"
extensions.append(html_theme)

# HTML theme options for local RTD build
html_theme_options = {
    # don't display version on documentation sidebar header
    "display_version": False,
    # color for the sidebar navigation header (copied from touketsu; don't set)
    #"style_nav_header_background": "#a2c4cd"
}

# file for image to be used in sidebar logo (should not exceed 200 px in width)
#html_logo = "./_static/touketsu_logo.png" # copied from touketsu

# use emacs style for pygments highlighting in code blocks or inline code
pygments_style = "emacs"    # may change this later, who knows

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]