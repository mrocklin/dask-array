"""Frisky graph helpers for dask-array.

There are deliberately no package-level re-exports: import layer classes and
the collectors from their submodules (``dask_array._frisky.blockwise``,
``dask_array._frisky.collect``, ...). This keeps the namespace inert — touching
it never imports ``dask_array._rust`` — and avoids a re-export catalog that
drifts as layers are added. The build-freshness check in
``dask_array._frisky.base`` runs when a layer module is actually used.
"""
