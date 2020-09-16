API docs
========

This section contains the Documentation of the Application Programming
Interface (API) of 'conflict_model'.

The model pipeline
------------------

.. currentmodule:: conflict_model

.. autosummary::
   :toctree: generated/
   :nosignatures:

   pipeline.create_XY
   pipeline.prepare_ML
   pipeline.run

The various models
------------------

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

.. currentmodule:: conflict_model

.. autosummary::
   :toctree: generated/
   :nosignatures:

   selection.select
   selection.filter_conflict_properties
   selection.select_period
   selection.clip_to_extent
   selection.climate_zoning

Machine learning
----------------

.. currentmodule:: conflict_model

.. autosummary::
   :toctree: generated/
   :nosignatures:

   machine_learning.define_scaling
   machine_learning.define_model
   machine_learning.split_scale_train_test_split
   machine_learning.fit_predict

Variable values
------------------------

.. currentmodule:: conflict_model

.. autosummary::
   :toctree: generated/
   :nosignatures:

   variables.nc_with_float_timestamp
   variables.nc_with_continous_datetime_timestamp

.. warning::

   Reading files with a float timestamp will most likely be deprecated in near future.

Work with conflict data
------------------------

.. currentmodule:: conflict_model

.. autosummary::
   :toctree: generated/
   :nosignatures:

   conflict.conflict_in_year_bool
   conflict.get_poly_ID
   conflict.get_poly_geometry
   conflict.split_conflict_geom_data
   conflict.get_pred_conflict_geometry

XY-Data
---------

.. currentmodule:: conflict_model

.. autosummary::
   :toctree: generated/
   :nosignatures:

   data.initiate_XY_data
   data.fill_XY
   data.split_XY_data

Model evaluation
-----------------

.. currentmodule:: conflict_model

.. autosummary::
   :toctree: generated/
   :nosignatures:

   evaluation.init_out_dict
   evaluation.fill_out_dict
   evaluation.init_out_df
   evaluation.fill_out_df
   evaluation.evaluate_prediction
   evaluation.polygon_model_accuracy
   evaluation.init_out_ROC_curve
   evaluation.plot_ROC_curve_n_times
   evaluation.plot_ROC_curve_n_mean
   evaluation.correlation_matrix
   evaluation.categorize_polys

Plotting
---------

.. currentmodule:: conflict_model

.. autosummary::
   :toctree: generated/
   :nosignatures:

   plots.plot_active_polys
   plots.plot_metrics_distribution
   plots.plot_nr_and_dist_pred
   plots.plot_frac_and_nr_conf
   plots.plot_frac_pred
   plots.plot_scatterdata
   plots.plot_correlation_matrix
   plots.plot_categories

Auxiliary functions
---------------------

.. currentmodule:: conflict_model

.. autosummary::
   :toctree: generated/
   :nosignatures:

   utils.get_geodataframe
   utils.show_versions
   utils.parse_settings
   utils.make_output_dir
   utils.download_PRIO
   utils.initiate_setup
   utils.create_artificial_Y
   utils.global_ID_geom_info
   utils.get_conflict_datapoints_only