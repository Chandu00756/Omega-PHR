# Configuration file for the Sphinx documentation builder.
# Advanced configuration for Omega-Paradox Hive Recursion (Ω-PHR) Documentation

import os
import sys
from datetime import datetime

# Add the project root to Python path for autodoc
sys.path.insert(0, os.path.abspath(".."))
sys.path.insert(0, os.path.abspath("../omega_phr"))
sys.path.insert(0, os.path.abspath("../services"))

# -- Project information -----------------------------------------------------
project = "Omega-Paradox Hive Recursion (Ω-PHR)"
copyright = f"{datetime.now().year}, Omega-PHR Research Consortium"
author = "Omega-PHR Research Team"
version = "2.1.0"
release = "2.1.0-stable"

# Project metadata
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
    "myst_parser",
    "sphinx_rtd_theme",
    "sphinx_copybutton",
    "sphinx_tabs.tabs",
    "sphinx_design",
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

# Source file configuration
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
html_css_files = ["custom.css"]

html_last_updated_fmt = "%b %d, %Y at %H:%M:%S"
html_show_sourcelink = True
html_show_sphinx = True
html_show_copyright = True

# -- Extension configuration -------------------------------------------------

# Napoleon settings for Google/NumPy docstrings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = True
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True

# MyST parser settings
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

# Intersphinx mapping
intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "asyncio": ("https://docs.python.org/3/library/asyncio.html", None),
}

# Autodoc settings
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "special-members": "__init__",
    "undoc-members": True,
    "exclude-members": "__weakref__",
}

# Todo extension settings
todo_include_todos = True

# Create _static directory if it doesn't exist
if not os.path.exists("_static"):
    os.makedirs("_static")

# -- General configuration ---------------------------------------------------
# Advanced extension configuration for comprehensive documentation

extensions = [
    # Core Sphinx extensions
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.todo",
    "sphinx.ext.coverage",
    "sphinx.ext.mathjax",
    "sphinx.ext.ifconfig",
    "sphinx.ext.githubpages",
    "sphinx.ext.inheritance_diagram",
    "sphinx.ext.graphviz",
    "sphinx.ext.duration",
    "sphinx.ext.autosectionlabel",
    # Third-party extensions for enhanced functionality
    "myst_parser",
    "sphinx_rtd_theme",
    "sphinx_copybutton",
    "sphinx_tabs.tabs",
    "sphinx_design",
    "sphinx_togglebutton",
    "sphinxcontrib.mermaid",
    "sphinx_external_toc",
]

# Template configuration
templates_path = ["_templates"]
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "**.ipynb_checkpoints",
    "**/node_modules",
    "**/__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
]

# Source file configuration
source_suffix = {
    ".rst": None,
    ".md": "myst_parser",
    ".txt": "myst_parser",
}

# Language and localization
language = "en"
locale_dirs = ["locale/"]
gettext_compact = False

# Master document
master_doc = "index"

# Numbering configuration
numfig = True
numfig_format = {
    "figure": "Figure %s",
    "table": "Table %s",
    "code-block": "Listing %s",
    "section": "Section %s",
}

# Cross-reference configuration
autosectionlabel_prefix_document = True
autosectionlabel_maxdepth = 3

# -- Options for HTML output -------------------------------------------------
# Advanced HTML theming and customization

html_theme = "sphinx_rtd_theme"
html_theme_path = []

# Advanced theme options
html_theme_options = {
    "analytics_id": "",  # Google Analytics ID
    "analytics_anonymize_ip": False,
    "logo_only": False,
    "display_version": True,
    "prev_next_buttons_location": "both",
    "style_external_links": True,
    "vcs_pageview_mode": "blob",
    "style_nav_header_background": "#2980B9",
    # Toc options
    "collapse_navigation": False,
    "sticky_navigation": True,
    "navigation_depth": 4,
    "includehidden": True,
    "titles_only": False,
    # Search options
    "canonical_url": "",
    "use_edit_page_button": True,
    "repository_url": "https://github.com/Chandu00756/Omega-PHR",
    "repository_branch": "main",
    # Social links
    "github_url": "https://github.com/Chandu00756/Omega-PHR",
    "gitlab_url": "",
    "bitbucket_url": "",
    "twitter_url": "",
}

# Custom CSS and JS
html_static_path = ["_static"]
html_css_files = [
    "custom.css",
    "code_highlight.css",
]
html_js_files = [
    "custom.js",
    (
        "https://cdnjs.cloudflare.com/ajax/libs/mermaid/8.13.8/mermaid.min.js",
        {"async": "async"},
    ),
]

# HTML output customization
html_logo = "_static/logo.png"
html_favicon = "_static/favicon.ico"
html_last_updated_fmt = "%b %d, %Y at %H:%M:%S"
html_show_sourcelink = True
html_show_sphinx = True
html_show_copyright = True
html_copy_source = True
html_use_opensearch = "https://omega-phr.readthedocs.io"

# Search configuration
html_search_language = "en"
html_search_options = {
    "type": "default",
    "scorer": "advanced",
}

# Page structure
html_sidebars = {
    "**": [
        "about.html",
        "navigation.html",
        "relations.html",
        "searchbox.html",
        "donate.html",
    ]
}

# -- Extension configuration -------------------------------------------------

# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = False
napoleon_type_aliases = None
napoleon_attr_annotations = True

# MyST parser settings
myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "deflist",
    "dollarmath",
    "html_admonition",
    "html_image",
    "linkify",
    "replacements",
    "smartquotes",
    "substitution",
    "tasklist",
]

# Intersphinx mapping
intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "grpc": ("https://grpc.github.io/grpc/python/", None),
}

# Todo extension settings
todo_include_todos = True

# Autodoc settings
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "special-members": "__init__",
    "undoc-members": True,
    "exclude-members": "__weakref__",
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

# Create _static directory if it doesn't exist
import os

if not os.path.exists("_static"):
    os.makedirs("_static")
