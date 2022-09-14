#############
API Reference
#############



Masked lnPi objects
===================
.. currentmodule:: lnpy

.. autosummary::
   :toctree: generated/
   :template: custom-class.rst

   lnPiArray
   lnPiMasked


Collection of lnPi
==================

.. currentmodule:: lnpy

.. autosummary::
   :toctree: generated/
   :template: custom-class.rst

   lnPiCollection


Ensemble properties
===================

.. currentmodule:: lnpy

.. autosummary::
   :toctree: generated/
   :template: custom-class.rst

   xGrandCanonical
   xCanonical

===============
Multiple phases
===============

Local free energy calculation
=============================

.. currentmodule:: lnpy

.. autosummary::
   :toctree: generated/
   :template: custom-class.rst

   wFreeEnergy
   wFreeEnergyCollection
   wFreeEnergyPhases


.. autosummary::
   :toctree: generated/

   merge_regions



Segmentation
============


.. autosummary::
   :toctree: generated/
   :template: custom-class.rst

   Segmenter
   PhaseCreator

   BuildPhases_mu
   BuildPhases_dmu

.. autosummary::
   :toctree: generated/

   peak_local_max_adaptive


Stability
=========

.. autosummary::
   :toctree: generated/
   :template: custom-class.rst

   Spinodals
   Binodals


   lnPiCollection.spinodal - Accessor to :class:`Spinodals`
   lnPiCollection.binodal - Accessor to :class:`Binodals`



Utilities
=========

.. currentmodule:: lnpy

.. autosummary::
   :toctree: generated/

   utils.mask_change_convention
   utils.masks_change_convention
   utils.labels_to_masks
   utils.masks_to_labels
   utils.distance_matrix
