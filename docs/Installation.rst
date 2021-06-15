Installation
=========================

From GitHub
------------

To install CoPro from GitHub, first clone the code. It is advised to create a separate environment first. 

.. note::

    We recommend to use Anaconda or Miniconda to install CoPro as this was used to develop and test the model.
    For installation instructions, see `here <https://docs.anaconda.com/anaconda/install/>`_.

.. code-block:: console

    $ git clone https://github.com/JannisHoch/copro.git
    $ cd path/to/copro
    $ conda env create -f environment.yml

It is now possible to activate this environment with

.. code-block:: console

    $ conda activate copro

To install CoPro in editable mode in this environment, run this command next in the CoPro-folder:

.. code-block:: console

    $ pip install -e .

From PyPI
------------

.. todo::

    This is not yet supported. Feel invited to provide a pull request enabling installation via PyPI.

From conda
------------

.. todo::

    This is not yet supported. Feel invited to provide a pull request enabling installation via conda.