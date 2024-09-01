# Configuration file for the Sphinx documentation builder.

# ARS says: FOLLOW: https://sphinx-rtd-tutorial.readthedocs.io/en/latest/index.html


#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#

import os
import sys
import sphinx_rtd_theme

sys.path.insert(
    0, os.path.abspath("../../pricing/")
)  # To search modules there. /.. goes one step "up". Starting point is the conf.py path
sys.path.insert(0, os.path.abspath("../../risk/"))
sys.path.insert(0, os.path.abspath("../../data/"))
sys.path.append(os.path.abspath("../.."))  # To search modules in pypricing


# print(sys.path)

# The master toctree document.
master_doc = "index"

# -- Project information -----------------------------------------------------

project = "PyPricing"
copyright = "2024, AFS"
author = "AFS"

# The full version, including alpha/beta/rc tags
release = "2024"


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "numpydoc",  # needs to be loaded *after* autodoc
    # "sphinx.ext.napoleon",
    "sphinx_toolbox.installation",
    "sphinx_toolbox.latex",
    "sphinx.ext.autosummary",
    "sphinx.ext.graphviz",
    "sphinx.ext.inheritance_diagram",
    "sphinx.ext.autosectionlabel",
    "sphinx_design",
    "sphinx.ext.intersphinx",
    # 'myst_nb',
    # "myst_parser",
    "nbsphinx",
    # "sphinxcontrib.details"
]
# Numpydoc
numpydoc_show_class_members = False
numpydoc_class_members_toctree = False
numpydoc_attributes_as_param_listbool = True
numpydoc_xref_param_typebool = True

# Don't show the code
html_show_sourcelink = True

# Not show module
python_module_index = False

# Summary
autosummary_generate = True

# Docstrings of private methods
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "undoc-members": True,
    "private-members": True,
    "inherited-members": True,
    "show-inheritance": True,
    "special-members": "__add__, __sub__, __rmul__, __neg__",
}

html_sidebars = {
    "**": [
        "globaltoc.html",  # Índice general
        #'localtoc.html',   # Índice local para cada archivo
        "searchbox.html",  # Cuadro de búsqueda
    ]
}
# myst_enable_extensions = ["colon_fence"]  # TODO See options in https://myst-parser.readthedocs.io/en/latest/configuration.html

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "pydata_sphinx_theme"

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the documentation.
#
# https://github.com/readthedocs/sphinx_rtd_theme/blob/master/docs/configuring.rst#id9


html_logo = "Logo.png"
html_theme_options = {
    # 'logo_only': False,
    "display_version": False,
    # Table of contents options
    "collapse_navigation": False,
    "includehidden": True,
    "footer_center": "theme_version",
    "footer_end": ["footer.html"],
}
# This setting ensures that each section in your documentation is automatically assigned a unique label based on the document it belongs to.

autosectionlabel_prefix_document = True


# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
sys.path.insert(0, os.path.abspath("."))
# html_static_path = ["_build/html/_static"]
html_static_path = ["_static"]
html_css_files = ["custom.css"]


# show the members in the order they appear in the source code, you can use the autodoc_member_order option.

autodoc_typehints = "none"
autodoc_member_order = "bysource"


# -- Options for diagrams output -------------------------------------------------

graphviz_output_format = "svg"
inheritance_node_attrs = dict(
    shape="ellipse", fontsize=12, height=0.75, color="dodgerblue1", style="filled"
)
inheritance_graph_attrs = dict(fontsize=12, size='"16.0, 20.0"')
# graphviz_dot = 'C:\Program Files\Graphviz\\bin\dot.exe'  # Comment for Git webpage


# -- LaTeX -------------------------------------------------

# latex_engine = 'xelatex'
latex_engine = "pdflatex"
latex_elements = {
    "preamble": r"""
\usepackage[titles]{tocloft}
\cftsetpnumwidth {1.25cm}\cftsetrmarg{1.5cm}
\setlength{\cftchapnumwidth}{0.75cm}
\setlength{\cftsecindent}{\cftchapnumwidth}
\setlength{\cftsecnumwidth}{1.25cm}
""",
    "fncychap": r"\usepackage[Bjornstrup]{fncychap}",
    "printindex": r"\footnotesize\raggedright\printindex",
}
latex_show_urls = "footnote"


# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title,
#  author, documentclass [howto, manual, or own class]).
latex_documents = [(master_doc, "main.tex", project, author, "report")]
