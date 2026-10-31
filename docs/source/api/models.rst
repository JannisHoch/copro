Main Model
===========

The models class
---------------------------
The models class contains all the steps required to fit and evaluate the model.
It essentially wraps the much of the functionality of the :ref:`Machine Learning` class.

.. currentmodule:: copro

.. autoclass:: models.MainModel
    :members:
    :undoc-members:
    :show-inheritance:

Running projections
-------------------
Projections are made from the deploy model, fitted on all rows by ``copro/scripts/deploy.py``.

.. autofunction:: models.run_prediction
