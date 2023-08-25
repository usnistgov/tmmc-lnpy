# Legacy API

For backward compatibility, we also provide the module {mod}`lnPi`. This module
is standalone (i.e., it is not a submodule of {mod}`lnpy`), so can be imported
using

```python
import lnPi
```

This is simply an interface to {mod}`lnpy` routines with some renaming.

```{eval-rst}

.. automodule:: lnPi
   :no-members:
   :no-inherited-members:
   :no-special-members:


.. autosummary::

    lnPi.MaskedlnPiDelayed
    lnPi.CollectionlnPi

.. autosummary::
    :toctree: generated/
    :template: autodocsumm/module.rst

    lnPi.maskedlnpi_legacy


```
