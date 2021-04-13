Pytorch Lantern Template
========================

Project template for use with
`pytorch-lantern <https://github.com/Aiwizo/pytorch-lantern>`__ .


Usage
-----

Install `cookiecutter <https://github.com/cookiecutter/cookiecutter>`_
and `poetry <https://github.com/python-poetry/poetry>`_:

.. code-block::

    pip install cookiecutter
    curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python

Setup project:

.. code-block::

    cookiecutter https://github.com/aiwizo/pytorch-lantern-template.git
    cd <new-project>
    poetry install


Code structure
--------------

What each directory in the newly-initialized project contains:

``problem``
~~~~~~~~~~~

Functions to supply the data in a natural (human-readable) form. No model specific functionality.

-  Splits data in train / compare
-  Works with the natural format of the examples

``datastream``
~~~~~~~~~~~~~~

Adapts the data from ``problem`` for training and evaluation. Keeps data in a natural form.

-  Splits train data in train / early\_stopping
-  Augments data
-  Works with the natural format of the examples just like problem
-  Create informative batches (oversample, stratify, etc)

``model``
~~~~~~~~~~~~~~~~

Contains the model used to solve the problem as well as functions to convert the data between human-readable to model-readable.

-  Preprocessing natural data into model-interpretable data
-  Model
-  Predictions - model output
-  Loss
-  Prediction visualization

``splits``
~~~~~~~~~~

Contains train/compare and train/early stopping splits as ``.json`` files.

``tools``
~~~~~~~~~

Utility functions that don't fit elsewhere.

``operations``
~~~~~~~~~~~~~~

Operations available from the `guild.ai <https://guild.ai/>`__ CLI.

Development
-----------

Setup test environment

.. code-block:: shell

    ./test/create.sh

Download and prepare dataset

.. code-block:: shell

    ./test/prepare.sh

Run tests in any order

.. code-block:: shell

    ./test/run.sh
    ./test/test.sh
    ./test/lint.sh

It can be useful to chain create and run when developing

.. code-block:: shell

    ./test/run.sh && ./test/run.sh
