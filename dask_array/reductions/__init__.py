from __future__ import annotations

from importlib import import_module

# This package intentionally lazy-loads its exports via __getattr__ instead of
# importing the implementation modules eagerly like other subpackages do.
# Worker-side unpickling of reduction functions (std/var/moment/all) imports
# this package re-entrantly and must not observe a partially initialized
# module (see commit e88b958). Don't "normalize" this to eager imports.
_EXPORTS = {
    "all": ("dask_array.reductions._common", "all"),
    "any": ("dask_array.reductions._common", "any"),
    "arg_reduction": ("dask_array.reductions._arg_reduction", "arg_reduction"),
    "argmax": ("dask_array.reductions._common", "argmax"),
    "argmin": ("dask_array.reductions._common", "argmin"),
    "cumprod": ("dask_array.reductions._cumulative", "cumprod"),
    "cumreduction": ("dask_array.reductions._cumulative", "cumreduction"),
    "cumsum": ("dask_array.reductions._cumulative", "cumsum"),
    "max": ("dask_array.reductions._common", "max"),
    "mean": ("dask_array.reductions._common", "mean"),
    "median": ("dask_array.reductions._common", "median"),
    "min": ("dask_array.reductions._common", "min"),
    "moment": ("dask_array.reductions._common", "moment"),
    "nanargmax": ("dask_array.reductions._common", "nanargmax"),
    "nanargmin": ("dask_array.reductions._common", "nanargmin"),
    "nancumprod": ("dask_array.reductions._cumulative", "nancumprod"),
    "nancumsum": ("dask_array.reductions._cumulative", "nancumsum"),
    "nanmax": ("dask_array.reductions._common", "nanmax"),
    "nanmean": ("dask_array.reductions._common", "nanmean"),
    "nanmedian": ("dask_array.reductions._common", "nanmedian"),
    "nanmin": ("dask_array.reductions._common", "nanmin"),
    "nannumel": ("dask_array.reductions._common", "nannumel"),
    "nanpercentile": ("dask_array.reductions._percentile", "nanpercentile"),
    "nanprod": ("dask_array.reductions._common", "nanprod"),
    "nanquantile": ("dask_array.reductions._common", "nanquantile"),
    "nanstd": ("dask_array.reductions._common", "nanstd"),
    "nansum": ("dask_array.reductions._common", "nansum"),
    "nanvar": ("dask_array.reductions._common", "nanvar"),
    "numel": ("dask_array.reductions._common", "numel"),
    "percentile": ("dask_array.reductions._percentile", "percentile"),
    "prod": ("dask_array.reductions._common", "prod"),
    "quantile": ("dask_array.reductions._common", "quantile"),
    "reduction": ("dask_array.reductions._reduction", "reduction"),
    "std": ("dask_array.reductions._common", "std"),
    "sum": ("dask_array.reductions._common", "sum"),
    "trace": ("dask_array.reductions._trace", "trace"),
    "var": ("dask_array.reductions._common", "var"),
    "_tree_reduce": ("dask_array.reductions._reduction", "_tree_reduce"),
}

__all__ = [
    "all",
    "any",
    "arg_reduction",
    "argmax",
    "argmin",
    "cumprod",
    "cumreduction",
    "cumsum",
    "max",
    "mean",
    "median",
    "min",
    "moment",
    "nanargmax",
    "nanargmin",
    "nancumprod",
    "nancumsum",
    "nanmax",
    "nanmean",
    "nanmedian",
    "nanmin",
    "nanpercentile",
    "nanprod",
    "nanquantile",
    "nanstd",
    "nansum",
    "nanvar",
    "percentile",
    "prod",
    "quantile",
    "reduction",
    "std",
    "sum",
    "trace",
    "var",
    "_tree_reduce",
]


def __getattr__(name):
    try:
        module_name, attr_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(name) from exc
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value
