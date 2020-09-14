API docs
========

This section contains the Documentation of the Application Programming
Interface (API) of 'conflict_model'.

The model pipeline
------------------
The main model pipeline consists of three steps.
First, create the XY data containing variable values and conflict classifier data.
Second, prepare the machine learning model, and third, run it n-times.

.. currentmodule:: conflict_model

.. autosummary::
   :toctree: generated/
   :nosignatures:

   pipeline.create_XY
   pipeline.prepare_ML
   pipeline.run

The various models
------------------
Various modelling approaches can be chosen, depending on the envisaged analysis.

.. currentmodule:: conflict_model

.. autosummary::
   :toctree: generated/
   :nosignatures:

   models.all_data
   models.leave_one_out
   models.single_variables
   models.dubbelsteen

.. note::

    the 'leave_one_out' and 'single_variables' models are only tested in beta-state.

Selecting polygons and conflicts
--------------------------------
Not all polygons and conflicts stored in the input data may be used for an analysis. 
Based on various settings, a sub-dataset can be selected.

.. currentmodule:: conflict_model

.. autosummary::
   :toctree: generated/
   :nosignatures:

   selection.select
   selection.filter_conflict_properties
   selection.select_period
   selection.clip_to_extent
   selection.climate_zoning