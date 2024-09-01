Changelog
=========

January 2024
------------------
- **structured.py:**

  - Add a method for computing the Greeks and PnL of a :py:meth:`Vanilla <structured.Vanilla>` option in different scenarios.

  - Extend methods from a :py:meth:`Vanilla <structured.Vanilla>` option to a collection of them using the new class :py:meth:`VanillaStrategy <structured.VanillaStrategy>`.

  - Add basic docstrings for new methods in classes :py:meth:`Vanilla <structured.Vanilla>` and :py:meth:`VanillaStrategy <structured.VanillaStrategy>`.
  
  - Change :py:meth:`structured.Structured.__add__` method to :py:meth:`structured.Vanilla.create_strategy`.

  - Modify the volatility input of :py:meth:`structured.Vanilla.get_risk_matrix` to values over 1 instead of percentage.

  - Improve some methods in classes :py:meth:`Structured <structured.Structured>` and :py:meth:`MCProduct <structured.MCProduct>`.
  
  - Correct code in classes :py:meth:`Airbag <structured.Airbag>` and :py:meth:`Lookback <structured.Lookback>`.

- **test_structured.py:**

  - Create unit tests for :py:meth:`Lookback <structured.Lookback>` options.

  - Change old unit tests according to code modifications.

  - Add a unit test concerning :py:meth:`structured.Vanilla.get_risk_matrix`.

- **Structured_Products_Documentation.ipynb:**

  - Revise the documentation of several classes to align with the new conventions on nominal and strike.

  - Add a warning for all classes that continue to use the old convention regarding nominal and strike.

  - Correct errors in ``Airbag options`` and ``Lookback option with floating strike``.

- **discount_curves.py:**

  - Improve :py:meth:`ShiftedCurve <discount_curves.ShiftedCurve>` with a better scaling method.

- **Code Examples - Structured (BeautifulData).ipynb:**

  - Replicate the old notebook **Code examples – Structured Products.ipynb** using BeautifulData.

- **calendars.py:**

  - Add basic docstrings for classes :py:meth:`DayCountCalendar <calendars.DayCountCalendar>` and :py:meth:`MonthYearCalendar <calendars.MonthYearCalendar>`

- **strategy_simulation.py:**

  - Implement the class :py:meth:`Strategy <mc_engines.Strategy>` to model dynamic strategies.

  - Implement the class :py:meth:`DeltaHedging <mc_engines.DeltaHedging>` as a rebalancing rule for dynamic strategies.

December 2023
------------------
- **underlyings.py:**

  - Solve RuntimeWarning due to square root of negative numbers.

  - Add basic docstrings for methods in classes :py:meth:`NormalAsset <underlyings.NormalAsset>` and :py:meth:`LognormalAsset <underlyings.LognormalAsset>`.

  - Correct the simulation of the underlying assets following :py:meth:`NormalAsset <underlying.NormalAsset>` and :py:meth:`LognormalAsset <underlying.LognormalAsset>` dynamics within the scope of Monte Carlo method.

- **structured.py:**

  - Include and document a new attribute :py:attr:`Vanilla.implied_volatility` for :py:meth:`Vanilla <structured.Vanilla>`.

  - Add the method :py:meth:`structured.Vanilla.get_greek_table` to obtain a table of the Greeks.

  - Update the code and documentation in accordance with the new convention regarding nominal and strike.
  
  - Add a warning for all classes that continue to use the old convention regarding nominal and strike.

- **test_structured.py:**

  - Create unit tests to verify the accurate pricing of the sum of two vanilla call options with Monte Carlo.

  - Adjust the tests in accordance with the new convention regarding nominal and strike.

  - Add tests concerning the new attribute :py:attr:`Vanilla.implied_volatility`.

- **test_callable.py:**

  - Adjust the tests in accordance with the new convention regarding nominal and strike.

- **test_underlyings.py:**

  - Adjust the tests in accordance with the new convention regarding nominal and strike.

- **Underlying_Documentation.ipynb:**

  - Document the correction related to the simulation of underlying assets following normal and lognormal dynamics within the scope of Monte Carlo simulations. 

- **mc_engines.py:**

  - Add basic docstrings for methods in the class :py:meth:`DiffusionMC <mc_engines.DiffusionMC>`.

  - Revise the method for computing the underlying forward price in :py:meth:`DiffusionMC <mc_engines.DiffusionMC>`, :py:meth:`DeterministicVolDiffusionMC <mc_engines.DeterministicVolDiffusionMC>` and :py:meth:`StochasticVolDiffusionMC <mc_engines.StochasticVolDiffusionMC>`.

- **discount_curves.py:**

  - Rectify an issue within the method :py:meth:`YieldCurve.fit <discount_curves.YieldCurve.fit>`.

- **Code Examples - Discount Curves_Ric_updated.ipynb**

  - Update the old notebook **Code Examples - Discount Curves.ipynb** with new features.

November 2023
------------------

- **implied_volatility.py:**

  - Solve circular import problem.

- **structured.py:**

  - Implement analytical formulas for Delta and theta Greeks with Bachelier model in :py:meth:`Call <structured.Call>` and :py:meth:`Put <structured.Put>`.

  - Implement analytical formulas for rho, Vega and Gamma Greeks with Black-Scholes and Bachelier models in :py:meth:`Call <structured.Call>` and :py:meth:`Put <structured.Put>`.

  - Add basic docstrings for methods computing price, Delta, theta, rho, Vega and Gamma Greeks in classes :py:meth:`Call <structured.Call>` and :py:meth:`Put <structured.Put>`.

  - Solve FutureWarning on compatible data.

- **test_structured.py:**

  - Create unit tests for Delta and theta Greeks of vanilla options for Bachelier model.

  - Create unit tests for rho, Vega and Gamma Greeks of vanilla options for both Black-Scholes and Bachelier models.

- **mc_engines.py:**

  - Solve a problem for the computation of forward price in class :py:meth:`DeterministicVolDiffusionMC <mc_engines.DeterministicVolDiffusionMC>`.

  - Document methods available to compute theta Greek with Monte Carlo in :py:meth:`MCProduct <structured.MCProduct>`.

  - Implement rho, Vega and Gamma computation with Monte Carlo in :py:meth:`MCProduct <structured.MCProduct>`.

- **data_factory_bd.py:**

  - Add a parameter to the import function of discount curves to specify the method to be used for interpolating the data.

- **discount_curves.py:**

  - Add the Piecewise Cubic Hermite Interpolating Polynomial interpolation method to class :py:meth:`YieldCurve <discount_curves.YieldCurve>`.

- **underlyings.py:**

  - Implement a new method for computing implied volatilities from options by using py_vollib_vectorized library.

- **credit.py:**

  - Solve DeprecationWarning for NumPy 1.25.

- **ir_models.py**

  - Solve RuntimeWarning caused by division by zero.

- **test_ir_products.py:**

  - Solve DeprecationWarning regarding NumPy 1.25 array to scalar conversion.

September-October 2023
------------------
- **underlyings.py:**

  - Add basic docstrings for methods in class :py:meth:`Underlying <underlyings.Underlying>`.

- **structured.py:**

  - Add basic docstrings for methods in classes :py:meth:`ZCBond <structured.ZCBond>` and :py:meth:`Derivative <structured.Derivative>`.


- **test_structured.py:**

  - Create unit tests for Bachelier and Black-Scholes vanilla option prices.

- **implied_volatility.py:**

  - Create and document classes and methods to handle the fitting of volatility smile and volatility surface.

  - Add possibility of use a :py:meth:`DiscountCurve <discount_curves.DiscountCurve>` instead of a constant interest rate.

- **excel_stuff.py:**

  - Add basic docstrings for methods in class :py:meth:`ToleranceTableObject <excel_stuff.ToleranceTableObject>`.

- **ratecurves.py:**

  - Add basic docstrings for class :py:meth:`ForwardRate <ratecurves.ForwardRate>` and its methods.

- **Structured_Products_Documentation.ipynb:**

  - Revise documentation according to the newly chosen conventions.

- **ir_models.py:**

  - Add basic docstrings to several methods in class :py:meth:`ShortRateModel <ir_models.ShortRateModel>`.

- **mc_engines.py:**

  - Add basic docstring in class :py:meth:`RegressionMC <mc_engines.RegressionMC>`.

- **data_factory_bd:**

  - Improve import of underlyings.

- **specs.py:**

  - Add EURIBOR to available discount curves.

August  2023
------------------
- **irproducts.py:**

  - Add basic docstrings.

- **ratecurves.py:**

  - Add basic docstrings.

- **mc_engines.py**

  - Improve classes :py:meth:`DiffusionMC <mc_engines.DiffusionMC>`, :py:meth:`SRDeterministicVolDiffusionMC <mc_engines.SRDeterministicVolDiffusionMC>`.

  - Document class ``SRDeterministicVolDiffusionMC``.

- **callable.py:**

  - Add :py:meth:`Callable <callable.Callable>`, :py:meth:`AmericanFromEuropean <callable.AmericanFromEuropean>` and :py:meth:`AmericanVanillaOption <callable.AmericanVanillaOption>`  classes.

- **data_factory_bd.py:**

  - Update class :py:meth:`DataFactoryBeautifulData <data.data_factory_bd.DataFactoryBeautifulData>` for importing underlyings and discount curves using beautifulData.

  - Add Module to GitLab documentation.

- **Workspace**

  - PyPricing and beautifulData available in the workspace (codeserver).

July  2023
------------------
- **mc_engines.py:**

  - Add  :py:meth:`get_delta <structured.Call.get_delta>` and  :py:meth:`get_theta <structured.Call.get_theta>` functions using MC for a general product.

  - Define multicurve functions.

- **discount_curves.py:**

  - Correct the function ``CubicDC``, :py:meth:`YieldCurve <discount_curves.YieldCurve>`.

  - Add code example of CDS.

  - Add several methods and change :py:meth:`CDSCurve <discount_curves.CDSCurve>` class for pricing CDS contracts.

  - Add docstrings.

- **credit.py:**

  - Define  :py:meth:`get_rpvp_par_spread <credit.CDS.get_rpvp_par_spread>`.

  - Change :py:meth:`get_px <credit.CDS.get_px>` so the accrued coupon is properly computed.

- **test_credit.py,** **test_discount_curves.py,** **test_ir_products.py,** **test_structured.py** and **test_underlying.py:**

  - Create unit tests for each script of the platform.

June  2023
------------------
- **structured.py:**

  - Introduction of abstract methods.

- **underlyings.py:**

  - Introduction of abstract methods.

- **Underlyings Documentation.ipynb:**

  - Documentation of ``NormalAsset`` and ``LogNormalAsset`` functions.

  - Code examples.


- **mc_engines.py:**

  - Added :py:meth:`DiffusionMC <mc_engines.DiffusionMC>` class.

- **discount_curves.py:**

  - Improve the efficiency and architecture of :py:meth:`SWICurve <discount_curves.SWICurve>` class.

  - Corrected conceptual errors in the :py:meth:`fit_seasonality_adjustment <discount_curves.SWICurve.fit_seasonality_adjustment>`  in :py:meth:`SWICurve <discount_curves.SWICurve>` class.

- **data_factory.py:**

  - :py:meth:`DataFactory <data.data_factory_bd.DataFactory>` class documented.

  - Added ``asset_kind`` argument in  :py:meth:`import_underlying <data.data_factory_bd.DataFactory.import_underlying>` method of :py:meth:`DataFactory <data.data_factory_bd.DataFactory>` class.

  -  :py:meth:`DataFactoryBeautifulData <data.data_factory_bd.DataFactoryBeautifulData>` class introduced and documented.

- **functions.py:**

  - Script added.

- **specs.py:**

  - Script added.

May  2023
----------

- **Gitlab Documentation page** created using Sphinx.

- **structured.py:**

  - Implementation of :py:meth:`Condor <structured.Condor>` class.
  - Implementation of Greeks functions.

- **Structured Products Documentation.ipynb:**

  - Documentation ``ProductFromFunction`` class.

- **underlyings.py:**

  - Implemented the Euler method for the SABR model using the :py:meth:`generate_paths_euler <data.underlyings.SABR.generate_paths_euler>` function.

  - Implemented the Euler method for the Multiasset Heston model using the :py:meth:`generate_paths_euler <data.underlyings.MultiAssetHeston.generate_paths_euler>` function.
  
  - Created :py:meth:`option_price_functions <data.underlyings.Heston.option_price_functions>`, :py:meth:`option_price_functions <data.underlyings.VolModel.option_price_functions>`, :py:meth:`fit_to_options <data.underlyings.Heston.fit_to_options>`, :py:meth:`fit_to_options <data.underlyings.VolModel.fit_to_options>` and :py:meth:`compute_implied_vol <data.underlyings.VolModel.compute_implied_vol>`   functions for calibrating  :py:meth:`Heston <data.underlyings.Heston>` and :py:meth:`VolModel <data.underlyings.VolModel>` classes.

  - Implemented the :py:meth:`MultiAssetHeston <data.underlyings.MultiAssetHeston>` method using the :py:meth:`compute_corr_matrix <data.underlyings.MultiAssetHeston.compute_corr_matrix>` and :py:meth:`generate_paths_for_pricing <data.underlyings.MultiAssetHeston.generate_paths_for_pricing>` functions.

- **Underlyings Documentation.ipynb:**

  - Documentation of the ``SABR`` class. 

  - Documentation of the path simulations.

- **mc_engines.py:**

  - Reduced computation time by adapting Monte Carlo to include simulation dates through the creation of the  :py:meth:`StochasticVolDiffusionMC <mc_engines.StochasticVolDiffusionMC>` class.

  - ``no_calcs`` introduced for avoiding MemoryErrors.


- **discount_curves.py:**

  - Appropriate arguments for :py:meth:`get_value <discount_curves.DiscountCurve.get_value>` from :py:meth:`DiscountCurve <discount_curves.DiscountCurve>` class.

  - Added docstrings for previously implemented methods.

  - Creation of the code examples.
  
  - Corrected the retrieval of the ``calendars`` attribute.


April  2023
------------------
- **structured.py:**

  - Implementation of  :py:meth:`Butterfly <structured.Butterfly>`, :py:meth:`Straddle <structured.Straddle>` and :py:meth:`Strangle <structured.Strangle>` classes.

  - Implementation :py:meth:`get_px <structured.Lookback.get_px>` in the :py:meth:`Lookback <structured.Lookback>` class.

- **Structured Products Documentation.ipynb:**

  - Documentation of ``Airbag`` and ``Lookback`` classes.

- **underlyings.py:**

  - Implementation of the Heston model through the addition of the methods :py:meth:`generate_paths_vols <data.underlyings.Heston.generate_paths_vols>` and :py:meth:`generate_paths <data.underlyings.Heston.generate_paths>`.

  - Implementation of the SABR model through the addition of the methods :py:meth:`generate_paths_vols <data.underlyings.SABR.generate_paths_vols>` and :py:meth:`generate_paths <data.underlyings.SABR.generate_paths>`.

  - Added docstrings for previously implemented methods.

- **Underlyings Documentation.ipynb:**

  - Creation of the document.

- **mc_engines.py:**

  - Added docstrings for previously implemented methods.

  - Modifications of :py:meth:`RegressionMC <mc_engines.RegressionMC>` and :py:meth:`SRRegressionMC <mc_engines.SRRegressionMC>`classes.

  - Adapted Monte Carlo to include more simulation dates.

- **MC engines Documentation.ipynb:**

  - Creation of the document. Commented ``RegressionMC`` and  ``SRRegressionMC`` classes.


March 2023
------------------
- **structured.py:**

  - Implementation of :py:meth:`ProductFromFunction <structured.ProductFromFunction>` class.

- **Interest Rate Products Documentation.ipynb:**

  - Documentation and code example implementation.

- **ir_models.py:**

  - Arquitecture of Hull-White two factors as a particular case of G2++.

- **underlyings.py:**

  - Optimization :py:meth:`generate_paths <data.underlyings.ExposureIndex.generate_paths>` and  :py:meth:`_get_index <data.underlyings.ExposureIndex._get_index>` methods of the :py:meth:`ExposureIndex <data.underlyings.ExposureIndex>` class.

- **ratecurves.py:**

  - Corrections in the :py:meth:`FixedTenorRate <ratecurves.FixedTenorRate>` class.

  - Adjunstment in the private functions :py:meth:`_get_index <ratecurves.Underlying._get_index>` and  :py:meth:`_get_ewma_vol_abstract <ratecurves.Underlying._get_ewma_vol_abstract>`  of :py:meth:`Underlying <ratecurves.Underlying>`  class.

- **Rate Curves Documentation.ipynb:**

  - Documentation in  ``FixedTenorRate`` class.

February 2023
------------------

- **structured.py:**

  - Payoff of  :py:meth:`KnockOutContingentPayment <structured.KnockOutContingentPayment>` optimized (broadcasting and vectorization).

  - Docstrings added.

- **irproducts.py:**

  - Population of the module. In particular, creation of products as the classes :py:meth:`CMSSpreadForward <irproducts.CMSSpreadForward>`, :py:meth:`TARN <irproducts.TARN>`, :py:meth:`CMSSpreadSwap <irproducts.CMSSpreadSwap>`, :py:meth:`RangeAccrual <irproducts.RangeAccrual>`.

- **Interest Rate Products Documentation.ipynb:**

  - Documentation of products as the classes ``CMSSpreadForward``, ``TARN``, ``CMSSpreadSwap``, ``RangeAccrual``.

- **ratecurves.py:**

  - Population of the module. In particular, creation of rate classes and operations as the classes :py:meth:`MultiRate <ratecurves.MultiRate>`, :py:meth:`CumulativeRate <ratecurves.CumulativeRate>`, :py:meth:`DifferenceRate <ratecurves.DifferenceRate>`, :py:meth:`TARNRate <ratecurves.TARNRate>`, :py:meth:`KnockOutRate <ratecurves.KnockOutRate>`.

- **Rate Curves Documentation.ipynb:**

  - Documentation of rate classes and operations as the classes ``MultiRate``, ``CumulativeRate``, ``DifferenceRate``, ``TARNRate``, ``KnockOutRae``.

January 2023
------------------

- **structured.py:**

  - Fixed typos in :py:meth:`Airbag <structured.Airbag>` class.

  - Added :py:meth:`plot_payoff <structured.Structured.plot_payoff>` method.

  - Modified ``pastmatters`` atribute.

- **Structured Products Documentation.ipynb:**

  - Documentation revised and improved.

- **ir_models.py:**

  - Implementation of the semi-analytical formula :py:meth:`swaption_price_function <ir_models.G2PlusPlusShortRate.swaption_price_function>`  and approximation  :py:meth:`swaption_price_function_SP <ir_models.G2PlusPlusShortRate.swaption_price_function_SP>` to check that the flow of **mc_engines.py**
    for IR products works properly (i.e., explicit formulas coincide with MC).

- **ratecurves.py:**

  - Docstrings and documentation.

  - Creation of :py:meth:`SwapRate.generate_rates <ratecurves.SwapRate.generate_rates>`.

- **mc_engines.py:**

  - Correction of numeraire and terminal measure adjustments.

  - Modification of the class :py:meth:`SRDeterministicVolDiffusionMC <mc_engines.SRDeterministicVolDiffusionMC>` so that the flow with **ratecurves.py** and **ir_models.py** works and can price IR products.

  - Introduction of plotly in :py:meth:`StatisticsGatherer.histogram <mc_engines.StatisticsGatherer.histogram>` method.





