Data factory
=======================

The ``data_factory_bd`` module is responsible for extracting data using the ``beautifulData`` library. Once the data is extracted, the module formats it
appropriately and instantiates objects from the PyPricing library. For instance, it performs bootstrapping using LIBOR rates for various tenors and USDA3L
swap rates to create the proper `discount curve <../MonteCarlo_engine/Discount_curves/discount_curves.html>`_.

Currently, there are two classes: ``DataFactory`` and ``DataFactoryBeautifulData``. The former is an older class that does not use ``beautifulData``; instead,
it reads data from local Excel files, which is considered a bad practice for several reasons. The latter class fetches data from the new database using
``beautifulData`` and is the one that will be used going forward. ``DataFactory`` will be maintained during the transition period.


The code architecture and docstrings are shown in the following link:

.. toctree:: 
   :maxdepth: 1

   Code <data_factory_bd>

