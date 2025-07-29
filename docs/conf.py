# Configuration file for the Sphinx documentation builder.
# Omega-Paradox Hive Recursion (Ω-PHR) Framework Documentation

import os
import sys
from datetime import datetime

# Add the project root to Python path for autodoc
sys.path.insert(0, os.path.abspath(".."))
sys.path.insert(0, os.path.abspath("../omega_phr"))

# -- Project information -----------------------------------------------------
project = "Omega-Paradox Hive Recursion (Ω-PHR)"
copyright = f"{datetime.now().year}, Omega-PHR Research Consortium"
author = "Omega-PHR Research Team"
version = "2.1.0"
release = "2.1.0-stable"

html_title = f"{project} v{version}"
html_short_title = "Ω-PHR Framework"

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.todo",
    "sphinx.ext.coverage",
    "sphinx.ext.githubpages",
    "sphinx_rtd_theme",
    "myst_parser",
]

templates_path = ["_templates"]
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "**/__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
]

source_suffix = {
    ".rst": None,
    ".md": "myst_parser",
}

language = "en"
master_doc = "index"

# -- Options for HTML output -------------------------------------------------
html_theme = "sphinx_rtd_theme"
html_theme_options = {
    "logo_only": False,
    "display_version": True,
    "prev_next_buttons_location": "both",
    "style_external_links": True,
    "collapse_navigation": False,
    "sticky_navigation": True,
    "navigation_depth": 4,
    "titles_only": False,
}

html_static_path = ["_static"]
html_last_updated_fmt = "%b %d, %Y at %H:%M:%S"
html_show_sourcelink = True
html_show_sphinx = True
html_show_copyright = True

# -- Extension configuration -------------------------------------------------
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = True
napoleon_include_special_with_doc = True
napoleon_use_param = True
napoleon_use_rtype = True

myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "html_admonition",
    "html_image",
    "linkify",
    "replacements",
    "smartquotes",
    "substitution",
    "tasklist",
]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
}

autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "special-members": "__init__",
    "undoc-members": True,
    "exclude-members": "__weakref__",
}

todo_include_todos = True

# Create _static directory if it doesn't exist
if not os.path.exists("_static"):
    os.makedirs("_static")
