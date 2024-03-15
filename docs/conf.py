# Sphinx conf.py for pellets documentation via readthedocs.io

project = 'pellets'
author = 'Teleoflexuous'
version = '0.1.0'
release = '0.1.0'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.doctest',
    'sphinx.ext.intersphinx',
    'sphinx.ext.viewcode',
]

templates_path = ['_templates']

html_theme = 'alabaster'
html_static_path = ['_static']

epub_show_urls = 'footnote'