The model pipeline
=========================

The main model pipeline consists of three steps.
First, create the XY data containing variable values and conflict classifier data.
Second, prepare the machine learning model, and third, run it n-times.

.. currentmodule:: conflict_model

.. autosummary::
   :nosignatures:

   pipeline.create_XY
   pipeline.prepare_ML
   pipeline.run
