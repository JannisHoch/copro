Using Copro
============

Model workflow
---------------

`copro` trains a Random Forest classifier model to predict the probability of conflict occorrance.
To that end, it needs conflict data (from UCDP) and a set of features describing potential conflict drivers.
The temporal resolution of the model is annual, i.e., feature data should contain annual data too and conflicts are predicted for each year.

The model is trained on a training set, and evaluated on a test set.
It is possible to use multiple model instances to account for variations due to the train/test split.
This is done with the `n_runs` key in the `[machine_learning]` section of the :ref:`Reference configuration file`. 
Overall model performance is evaluated by averaging the results of all model instances.

The final conflict state of the reference period is used as initial conditions for the prediction.
Each model instance starts from there and forward-predicts conflict occurance probability for the prediction period.

Via the command line
---------------------
The most convienient way to use `copro` is via the command line. 
This allows you to run the model with a single command, and to specify all model configurations in one file.

.. toctree::
   :maxdepth: 1

   cli.rst
   config.rst

API
----
For bespoke applications, it is possible to use `copro` as a library.
Please find more information in the :ref:`API documentation`.