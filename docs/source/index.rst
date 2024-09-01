.. PyPricing documentation master file, created by
   sphinx-quickstart on Fri Dec 16 12:46:40 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.
   [ARS] Tutorial: https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html


AFS PyPricing
===============
The need for an independent valuation is unquestionable. For asset managers, banks, insurance companies, and all types of financial institutions (and even non-financial ones), understanding the value of their assets, as well as other market assets they might be interested in, is paramount for making informed investment decisions.

In response to the changing dynamics of financial markets and the increasing complexity of pricing and risk management, our platform has been developed to meet this need. Traditional pricing platforms often struggle to incorporate new technologies and lack the flexibility to effectively address today’s challenges. However, our platform steps in as a comprehensive and adaptable solution. It enables the valuation of a wide array of products. This accomplishment is realized by employing a variety of models, all the while prioritizing computational efficiency. Through this innovative approach, we aim to provide the accurate insights necessary for sound decision-making in an ever-evolving financial landscape.

Platform workflow:
-------------------
The operation of the platform can be observed in greater detail at the following link: 

.. toctree::
   :maxdepth: 2

   Platform_workflow/platform_workflow__main


.. _cards-clickable:


.. raw:: html

   <style>
       .grid {
           display: grid;
           grid-template-columns: repeat(2, 1fr);
           grid-template-rows: repeat(4, 1fr);
           grid-gap: 10px; /* Ajusta el valor según el espaciado deseado */
           gap: 100px 100px;
       }
   </style>


Products:
----------

.. grid:: 2
    :gutter: 1

    .. grid-item-card::  Structured products
      :link: Products/Structured_Products/structured_main.html
      :text-align: center

      :octicon:`columns;5em;sd-text-info`
      ^^^
      See ``Structured products``.

    .. grid-item-card::  Interest rate products
      :link: Products/Interest_Rate_Products/irproducts_main.html
      :text-align: center

      :octicon:`pulse;5em;sd-text-info`
      ^^^
      See ``Interest rate products``.

.. grid:: 2
    :gutter: 1

    .. grid-item-card::  Credit derivatives
      :link: Products/Credit/credit_main.html
      :text-align: center

      :octicon:`arrow-switch;5em;sd-text-info`
      ^^^
      See ``Credit derivatives``.

    .. grid-item-card::  Fixed income
      :link: Products/Fixed_Income/fixedincome_main.html
      :text-align: center

      :octicon:`pin;5em;sd-text-info`
      ^^^
      See ``Fixed income``.

Underlying models:
--------------------

.. grid:: 1
    :gutter: 1

    .. grid-item-card::  Underlyings
      :link: Underlying_models/Underlyings/underlyings_main.html
      :text-align: center

      :octicon:`stack;5em;sd-text-info`
      ^^^
      See ``Underlyings``.


.. grid:: 2
    :gutter: 1

    .. grid-item-card::  Interest rate models
      :link: Underlying_models/Interest_Rate_Models/irmodels_main.html
      :text-align: center

      :octicon:`number;5em;sd-text-info`
      ^^^
      See ``Interest rate models``.

    .. grid-item-card::  Rate curves
      :link: Underlying_models/Rate_Curves/ratecurves_main.html
      :text-align: center

      :octicon:`unfold;5em;sd-text-info`
      ^^^
      See ``Rate curves``.





Monte Carlo engine:
--------------------------------

.. grid:: 2
    :gutter: 1

    .. grid-item-card::  Monte Carlo
      :link: MonteCarlo_engine/MonteCarlo_engines/mc_main.html
      :text-align: center

      :octicon:`gear;5em;sd-text-info`
      ^^^
      See ``Monte Carlo``.

    .. grid-item-card::  Discount curves
      :link: MonteCarlo_engine/Discount_curves/discount_curves_main.html
      :text-align: center

      :octicon:`graph;5em;sd-text-info`
      ^^^
      See ``Discount curves``.


Risk and others:
--------------------------------

.. grid:: 2
    :gutter: 1

    .. grid-item-card::  Risk
      :link: Risk/risk_main.html
      :text-align: center

      :octicon:`alert;5em;sd-text-info`
      ^^^
      See ``Risk``.


    .. grid-item-card::  Others
      :link: Others/others_main.html
      :text-align: center

      :octicon:`server;5em;sd-text-info`
      ^^^
      See ``Others``.

.. contents:: Table of Contents

Products
============

.. toctree::
   :maxdepth: 3

   Products/products_main


Underlying Models
======================

.. toctree::
   :maxdepth: 3

   Underlying_models/Underlyingmodels_main

Monte Carlo Engine
====================

.. toctree::
   :maxdepth: 3

   MonteCarlo_engine/MonteCarloengine_main

Risk
==========

.. toctree::
   :maxdepth: 1

   Risk/risk_main

Data Factory
==============

.. toctree::
   :maxdepth: 1

   Data_Factory/datafactory_main


Others
==========

.. toctree::
   :maxdepth: 2

   Others/others_main

Changelog
---------

.. toctree::
   :maxdepth: 1

   changelog

Testing
-------

.. toctree:: 2
   :maxdepth: 1

   testing





..
   API Reference:
   --------------


   .. rubric:: Footnotes

   .. [#f1] Under construction.
