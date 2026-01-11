"""Template loading for array HTML representations."""

from __future__ import annotations

import os.path

from jinja2 import Environment, FileSystemLoader, Template
from jinja2.exceptions import TemplateNotFound

from dask.utils import typename

FILTERS = {
    "type": type,
    "typename": typename,
}

TEMPLATE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")


def get_template(name: str) -> Template:
    """Load a Jinja2 template from the templates directory."""
    loader = FileSystemLoader([TEMPLATE_PATH])
    environment = Environment(loader=loader)
    environment.filters.update(FILTERS)

    try:
        return environment.get_template(name)
    except TemplateNotFound as e:
        raise TemplateNotFound(f"Unable to find {name} in {TEMPLATE_PATH}") from e
