Rate curves
=======================

Rate Curves serve as the underying for interest rate derivatives. In comparison with :py:meth:`underlyings.py <data.underlyings>`, models and curves are separated into two
different modules, :py:meth:`ir_models <pricing.ir_models>` and :py:meth:`ratecurves.py <pricing.ratecurves>`, where the latter uses the former to construct the rates, although other methods are implemented
independently of the interest rate model used.

.. toctree::
   :maxdepth: 1
   :titlesonly:

   Jupyter <Rate_Curves_Documentation>
   Code <ratecurves>

