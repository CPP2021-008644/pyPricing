import numpy as np
from scipy import optimize
from scipy.stats import gmean  # multivariate_normal, norm
import plotly.express as px  # For the plot_payoff method
from datetime import timedelta
import copy
from itertools import product

try:
    from ..data.underlyings import *  # Import needed for the workspace. In this case we need the parent package since underlyings

    # is in a different subpackage (data)
except (ImportError, ModuleNotFoundError, ValueError):
    from data.underlyings import *  # (Absolute) local import
try:
    from ..data.specs import *
except (ImportError, ModuleNotFoundError, ValueError):
    from data.specs import *  # (Absolute) local import
try:
    from .mc_engines import MC1d, DeterministicVolDiffusionMC, DiffusionMC
except (ImportError, ModuleNotFoundError, ValueError):
    from pricing.mc_engines import MC1d, DeterministicVolDiffusionMC, DiffusionMC
try:
    from .discount_curves import CRDC, ShiftedCurve
except (ImportError, ModuleNotFoundError, ValueError):
    from pricing.discount_curves import CRDC, ShiftedCurve
try:
    from .excel_stuff import ToleranceTableObject
except (ImportError, ModuleNotFoundError, ValueError):
    from pricing.excel_stuff import ToleranceTableObject
try:
    from . import functions as afsfun
except (ImportError, ModuleNotFoundError, ValueError):
    import pricing.functions as afsfun

zero_credit_risk = CRDC(0)


# ------------------------------------------------------------------------------------
# Generic code
# ------------------------------------------------------------------------------------


class Structured(ToleranceTableObject, ABC):
    # the general structured products class provides the induction law at the most general level
    # .pxfun and .get_px are a nice logical circle!
    # It breaks in subclasses, when we define .get_px for the particular case;
    # or, more importantly, breaks when adding objects, by defining .pxfun
    # ditto for .run_mc and .generate_tolerances_table
    """
    This class provides the "induction law" for structured products, into which we feed simple products.
    
    Parameters
    ----------
    obsdates : pandas.DatetimeIndex or list of strings
        Observation dates.
    
    pastmatters : bool
        True for path dependent structured products, otherwise False.

    pay_dates : pandas.DatetimeIndex or list of strings, optional
        Dates at which payments occur. If None, `pay_dates` equals `obsdates` (defaults to None).
        
    credit_curve : pricing.discount_curves.CRDC
        The credit spread curve (default is zero_credit_risk = CRDC(0), namely there is no credit spread).
        
    monotonicity_price_function : string, optional
        Monotonicity of the price function of the structured product. It can be `increasing`, `decreasing` \
        or, for non definite monotonicity, None. This attribute is beneficial for calculating risk metrics, such as VaR. \
        Default is None.
    """

    def __init__(
        self,
        obsdates,
        pastmatters,
        pay_dates=None,
        credit_curve=zero_credit_risk,
        monotonicity_price_function=None,
    ):
        self.obsdates = pd.to_datetime(obsdates).sort_values()
        self.maturity = self.obsdates[-1]
        self.pastmatters = pastmatters
        # For path dependent options, self.pastmatters = True
        self.credit_curve = credit_curve
        if pay_dates is None:
            self.pay_dates = self.obsdates
        else:
            self.pay_dates = pd.to_datetime(pay_dates).sort_values()
        self.monotonicity_price_function = monotonicity_price_function
        # self.strike = np.nan

    def __add__(self, other):
        if hasattr(self, "underlying") and hasattr(other, "underlying"):
            if self.underlying != other.underlying:
                print("Assets with different underlyings, operation not allowed.")
                return None

        if hasattr(self, "calendar") and hasattr(other, "calendar"):
            if self.calendar != other.calendar:
                print("Assets with different calendars, operation not allowed.")
                return None

        obsdates = self.obsdates.union(other.obsdates).sort_values()
        z = Structured(
            obsdates=obsdates, pastmatters=max(self.pastmatters, other.pastmatters)
        )
        # The resulting product will have pastmatters = True if one of the pieces has it.
        z.__class__.__name__ = (
            self.__class__.__name__ + " + " + other.__class__.__name__
        )
        setattr(z, "payoff", lambda *x: self.payoff(*x) + other.payoff(*x))
        setattr(
            z,
            "get_px",
            lambda dates, discount_curve, no_sims, no_calcs: self.get_px(
                dates, discount_curve, no_sims, no_calcs
            )
            + other.get_px(dates, discount_curve, no_sims, no_calcs),
        )
        setattr(
            z,
            "get_delta",
            lambda dates, discount_curve, no_sims, no_calcs, monte_carlo: self.get_delta(
                dates, discount_curve, no_sims, no_calcs, monte_carlo
            )
            + other.get_delta(dates, discount_curve, no_sims, no_calcs, monte_carlo),
        )
        setattr(
            z,
            "get_gamma",
            lambda dates, discount_curve, no_sims, no_calcs, monte_carlo: self.get_gamma(
                dates, discount_curve, no_sims, no_calcs, monte_carlo
            )
            + other.get_gamma(dates, discount_curve, no_sims, no_calcs, monte_carlo),
        )
        setattr(
            z,
            "get_vega",
            lambda dates, discount_curve, no_sims, no_calcs, monte_carlo: self.get_vega(
                dates, discount_curve, no_sims, no_calcs, monte_carlo
            )
            + other.get_vega(dates, discount_curve, no_sims, no_calcs, monte_carlo),
        )
        setattr(
            z,
            "get_rho",
            lambda dates, discount_curve, no_sims, no_calcs, monte_carlo: self.get_rho(
                dates, discount_curve, no_sims, no_calcs, monte_carlo
            )
            + other.get_rho(dates, discount_curve, no_sims, no_calcs, monte_carlo),
        )
        setattr(
            z,
            "get_theta",
            lambda dates, discount_curve, no_sims, no_calcs, maturity_method, monte_carlo, constant_curve, refit: self.get_theta(
                dates,
                discount_curve,
                no_sims,
                no_calcs,
                maturity_method,
                monte_carlo,
                constant_curve,
                refit,
            )
            + other.get_theta(
                dates,
                discount_curve,
                no_sims,
                no_calcs,
                maturity_method,
                monte_carlo,
                constant_curve,
                refit,
            ),
        )
        setattr(
            z,
            "compute_discount_tolerances",
            lambda *x: self.compute_discount_tolerances(*x)
            + other.compute_discount_tolerances(*x),
        )
        setattr(
            z,
            "compute_equity_tolerances",
            lambda *x: self.compute_equity_tolerances(*x)
            + other.compute_equity_tolerances(*x),
        )

        if hasattr(self, "underlying") or hasattr(other, "underlying"):
            try:
                z.underlying = self.underlying
            except:
                z.underlying = other.underlying

        if hasattr(self, "calendar") or hasattr(other, "calendar"):
            try:
                z.calendar = self.calendar
            except:
                z.calendar = other.calendar

        if hasattr(self, "implied_volatility") or hasattr(other, "implied_volatility"):
            z.implied_volatility = None

        return z

    def __sub__(self, other):
        if hasattr(self, "underlying") and hasattr(other, "underlying"):
            if self.underlying != other.underlying:
                print("Assets with different underlyings, operation not allowed.")
                return None

        if hasattr(self, "calendar") and hasattr(other, "calendar"):
            if self.calendar != other.calendar:
                print("Assets with different calendars, operation not allowed.")
                return None

        obsdates = self.obsdates.union(other.obsdates).sort_values()
        z = Structured(
            obsdates=obsdates, pastmatters=max(self.pastmatters, other.pastmatters)
        )
        z.__class__.__name__ = (
            self.__class__.__name__ + " - " + other.__class__.__name__
        )
        setattr(z, "payoff", lambda *x: self.payoff(*x) - other.payoff(*x))
        setattr(
            z,
            "get_px",
            lambda dates, discount_curve, no_sims, no_calcs: (
                self.get_px(dates, discount_curve, no_sims, no_calcs)
                - other.get_px(dates, discount_curve, no_sims, no_calcs)
            ),
        )
        setattr(
            z,
            "get_delta",
            lambda dates, discount_curve, no_sims, no_calcs, monte_carlo: self.get_delta(
                dates, discount_curve, no_sims, no_calcs, monte_carlo
            )
            - other.get_delta(dates, discount_curve, no_sims, no_calcs, monte_carlo),
        )
        setattr(
            z,
            "get_gamma",
            lambda dates, discount_curve, no_sims, no_calcs, monte_carlo: self.get_gamma(
                dates, discount_curve, no_sims, no_calcs, monte_carlo
            )
            - other.get_gamma(dates, discount_curve, no_sims, no_calcs, monte_carlo),
        )
        setattr(
            z,
            "get_vega",
            lambda dates, discount_curve, no_sims, no_calcs, monte_carlo: self.get_vega(
                dates, discount_curve, no_sims, no_calcs, monte_carlo
            )
            - other.get_vega(dates, discount_curve, no_sims, no_calcs, monte_carlo),
        )
        setattr(
            z,
            "get_rho",
            lambda dates, discount_curve, no_sims, no_calcs, monte_carlo: self.get_rho(
                dates, discount_curve, no_sims, no_calcs, monte_carlo
            )
            - other.get_rho(dates, discount_curve, no_sims, no_calcs, monte_carlo),
        )
        setattr(
            z,
            "get_theta",
            lambda dates, discount_curve, no_sims, no_calcs, maturity_method, monte_carlo, constant_curve, refit: self.get_theta(
                dates,
                discount_curve,
                no_sims,
                no_calcs,
                maturity_method,
                monte_carlo,
                constant_curve,
                refit,
            )
            - other.get_theta(
                dates,
                discount_curve,
                no_sims,
                no_calcs,
                maturity_method,
                monte_carlo,
                constant_curve,
                refit,
            ),
        )
        setattr(
            z,
            "compute_discount_tolerances",
            lambda *x: self.compute_discount_tolerances(*x)
            - other.compute_discount_tolerances(*x),
        )
        setattr(
            z,
            "compute_equity_tolerances",
            lambda *x: self.compute_equity_tolerances(*x)
            - other.compute_equity_tolerances(*x),
        )

        if hasattr(self, "underlying") or hasattr(other, "underlying"):
            try:
                z.underlying = self.underlying
            except:
                z.underlying = other.underlying

        if hasattr(self, "calendar") or hasattr(other, "calendar"):
            try:
                z.calendar = self.calendar
            except:
                z.calendar = other.calendar

        if hasattr(self, "implied_volatility") or hasattr(other, "implied_volatility"):
            z.implied_volatility = None

        return z

    def __rmul__(self, other):
        obsdates = self.obsdates
        z = Structured(obsdates=obsdates, pastmatters=self.pastmatters)
        # Here we do not include other.pastmatters since this is a multiplication by a float.
        z.__class__.__name__ = str(other) + " * (" + self.__class__.__name__ + ")"
        setattr(z, "payoff", lambda *x: other * self.payoff(*x))
        setattr(
            z,
            "get_px",
            lambda *x: other * self.get_px(*x),
        )
        setattr(
            z,
            "get_delta",
            lambda *x: other * self.get_delta(*x),
        )
        setattr(
            z,
            "get_gamma",
            lambda *x: other * self.get_gamma(*x),
        )
        setattr(
            z,
            "get_vega",
            lambda *x: other * self.get_vega(*x),
        )
        setattr(
            z,
            "get_rho",
            lambda *x: other * self.get_rho(*x),
        )
        setattr(
            z,
            "get_theta",
            lambda *x: other * self.get_theta(*x),
        )
        setattr(
            z,
            "compute_discount_tolerances",
            lambda *x: other * self.compute_discount_tolerances(*x),
        )
        setattr(
            z,
            "compute_equity_tolerances",
            lambda *x: other * self.compute_equity_tolerances(*x),
        )
        if hasattr(self, "underlying"):
            z.underlying = self.underlying
        if hasattr(self, "calendar"):
            z.calendar = self.calendar
        if hasattr(self, "implied_volatility"):
            z.implied_volatility = self.implied_volatility
        return z

    def __neg__(self):
        obsdates = self.obsdates
        z = Structured(obsdates=obsdates, pastmatters=self.pastmatters)
        z.__class__.__name__ = " - " + self.__class__.__name__
        setattr(z, "payoff", lambda *x: -self.payoff(*x))
        setattr(
            z,
            "get_px",
            lambda *x: -self.get_px(*x),
        )
        setattr(
            z,
            "get_delta",
            lambda *x: -self.get_delta(*x),
        )
        setattr(
            z,
            "get_gamma",
            lambda *x: -self.get_gamma(*x),
        )
        setattr(
            z,
            "get_vega",
            lambda *x: -self.get_vega(*x),
        )
        setattr(
            z,
            "get_rho",
            lambda *x: -self.get_rho(*x),
        )
        setattr(
            z,
            "get_theta",
            lambda *x: -self.get_theta(*x),
        )
        setattr(
            z,
            "compute_discount_tolerances",
            lambda *x: -self.compute_discount_tolerances(*x),
        )
        setattr(
            z,
            "compute_equity_tolerances",
            lambda *x: -self.compute_equity_tolerances(*x),
        )
        if hasattr(self, "underlying"):
            z.underlying = self.underlying
        if hasattr(self, "calendar"):
            z.calendar = self.calendar
        if hasattr(self, "implied_volatility"):
            z.implied_volatility = self.implied_volatility
        return z

    def run_mc(
        self, mc_engine, dates, discount_curve, no_sims, statistics_gatherer=None
    ):
        """
        Run a Monte Carlo simulation for a given instrument.

        If the instrument has an underlying, the Monte Carlo engine's price method is used.
        Otherwise, the instrument's get_px method is called.

        Parameters
        ----------
        mc_engine : object
            The Monte Carlo engine to use for pricing.
        dates : pandas.DatetimeIndex, list of strings, pandas.Timestamp, or string
            The dates for which to calculate prices. It can be a single date (as a pandas.Timestamp or its string representation) or an array-like object (as a pandas.DatetimeIndex
            or a list of its string representation) containing dates.

        discount_curve : object
            The discount curve to use for pricing.
        no_sims : int
            The number of simulations to run.
        statistics_gatherer : object
            An optional statistics gatherer. If provided, will be passed to the Monte Carlo engine.

        Returns
        -------
        float or numpy.ndarray
            The simulated prices for the instrument.
        """
        if hasattr(self, "underlying"):
            return mc_engine.price(
                self, dates, discount_curve, no_sims, statistics_gatherer
            )
        else:
            return self.get_px(dates=dates, discount_curve=discount_curve)

    def payoff(self, prices, dates_dic, n):
        """
        Return the payoff of a Structured product.

        Parameters
        ----------
        prices: numpy.ndarray
            Prices of the underlying. The first index of price denotes the simulation, the second one the (full) observation dates and the third one the valuation dates.
            The fourth one is for the asset number.
        dates_dic: dict
            Dictionary (or pandas.Series) which assigns an index to each date.
        n: int
            Number of observation dates previous to a fixed observation date.
        Returns
        -------
        numpy.ndarray
            3 dimensional numpy.ndarray with the payoffs.

        See Also
        ----------
        JupyterNotebook: Structured Products Documentation.
        """
        pass

    def get_px(self, dates, discount_curve, no_sims=None, no_calcs=None):
        """
        Compute the price.

        Notes
        -----
            Note that when inheriting from Derivatives and MCProduct, we need to inherit from MCProduct first. Otherwise, the abstract method get_px will try to be used.
        """
        pass

    def generate_prices(
        self, dates, discount_curve, vols, no_sims, no_sims_price, no_calcs_price
    ):
        """
        Simulate prices of a product the day after ``dates`` for computing \
        Value at Risk (VaR) and Expected Shortfall (ES). We neglect the interest rate \
        in the simulation of the underlying.

        dates : pandas.DatetimeIndex, list of strings, pandas.Timestamp, or string
            Dates at which VaR or ES is computed. It can be a single date (as a pandas.Timestamp or its string representation) or an array-like object (as a
            pandas.DatetimeIndex or a list of its string representation) containing dates.
        discount_curve : discount_curves.DiscountCurve
            Gives the discounts for the computations.
        time_horizon : int, optional
            Time horizon for the VaR or ES in days (defaults to 1).
        conf_level : float, optional
            Confidence level for the VaR or ES. The value pertains to the distribution of the profit function (defaults to 0.01).
        no_sims : int, optional
            Number of simulations for Monte Carlo valuation of VaR (defaults to 10**4).
        no_sims_price : int, optional
            Number of simulations for Monte Carlo valuation of product's price (defaults to 10**6).
        no_calcs : int, optional
            The product's price is computed ``no_calcs`` times and then the mean value is returned (defaults to 1).
            With this we can effectively prevent memory errors ("MemoryError: Unable to allocate...") in Python when increasing the number of simulations.

        Returns
        -------
        list of numpy.ndarray
            A list containing ``no_sims`` numpy.ndarray's, each representing simulated prices for the corresponding date in ``dates``.
        """
        dates = afsfun.dates_formatting(dates)
        prices_underlying = pd.DataFrame(
            self.underlying.get_value(dates).values, index=dates, columns=["Price"]
        )
        dates_plus = dates + pd.tseries.offsets.BDay()
        tau = self.calendar.interval(dates=dates, future_dates=dates_plus)
        if isinstance(self.underlying, LognormalAsset):
            all_prices_underlying_stress = np.transpose(
                prices_underlying.values
            ) * np.exp(
                -1 / 2 * (vols**2 * tau).values
                + (vols * np.sqrt(tau)).values * np.random.randn(no_sims, dates.size)
            )
        elif isinstance(self.underlying, NormalAsset):
            all_prices_underlying_stress = np.transpose(prices_underlying.values) + (
                vols * np.sqrt(tau)
            ).values * np.random.randn(no_sims, dates.size)
        all_prices_underlying_stress = np.transpose(all_prices_underlying_stress)
        all_prices_stress = []
        for no_sim in range(no_sims):
            prices_underlying_stress = all_prices_underlying_stress[:, no_sim]
            prices_underlying_stress = pd.DataFrame(
                prices_underlying_stress, index=dates_plus, columns=["Price"]
            )
            self.underlying.set_data(prices_underlying_stress)
            prices_stress = self.get_px(
                dates=dates_plus,
                discount_curve=discount_curve,
                no_sims=no_sims_price,
                no_calcs=no_calcs_price,
            )
            all_prices_stress.append(prices_stress.values)
        return all_prices_stress

    def get_var(
        self,
        dates,
        discount_curve,
        time_horizon=1,
        conf_level=0.01,
        no_sims=10**4,
        no_sims_price=10**6,
        no_calcs_price=1,
    ):
        """
        Calculate the Value at Risk (VaR) of a structured product relative to a \
        specific time horizon and confidence level for the given dates.
        
        The method is implemented assuming lognormal dynamics (`LognormalAsset`) or \
        Gaussian dynamics (`NormalAsset`) for the underlying price :math:`X`, and \
        consider the latter as the sole risk factor. We neglect the interest rate \
        in the simulation of the underlying. If the price function of the \
        structured product :math:`\\pi(X)` is monotonically increasing or decreasing, \
        the :math:`\\alpha` confidence level VaR at 1 day at time :math:`t`\
        :math:`\\textrm{VaR}^{\\textrm{1 day}}_\\alpha (\\Delta \\pi_{t+\\textrm{1 day}}(X_t))` \
        is calculated analytically based on the following relationships:
        
        .. math::
            \\textrm{VaR}^{\\textrm{1 day}}_\\alpha (\\Delta \\pi_{t+\\textrm{1 day}}(X_t)) \\equiv F^{-1}_{\\Delta \\pi_{t+\\textrm{1 day}}(X_t)} (\\alpha) = \\Delta \\pi_{t+\\textrm{1 day}} (F^{-1}_{X_t} (\\alpha)) \\quad \\textrm{for $\\pi$ increasing} \,,
            
        .. math::
            \\textrm{VaR}^{\\textrm{1 day}}_\\alpha ( \\Delta \\pi_{t+\\textrm{1 day}}(X_t)) \\equiv F^{-1}_{\\Delta \\pi_{t+\\textrm{1 day}}(X_t)} (\\alpha) = \\Delta \\pi_{t+\\textrm{1 day}}(F^{-1}_{X_t} (1 - \\alpha)) \\quad \\textrm{for $\\pi$ decreasing} \,,

        where:

        - :math:`\\Delta \\pi_{t+\\textrm{1 day}}(X_t) \\equiv \\pi(X_{t+\\textrm{1 day}}) - \\pi(X_t)` is the product profit function between :math:`t` and :math:`t+\\textrm{1 day}`,
        - :math:`F^{-1}_{X_t} (\\alpha)` is the inverse cumulative distribution \
        function of the random variable :math:`X_t` evaluated at :math:`\\alpha`.
        
        If the price function of the structured product has got no definite \
        monotonicity, a Monte Carlo simulations approach is used. The number of \
        simulations is ruled by the parameter input ``no_sims``.
        
        For time horizon :math:`T` longer than one day, the VaR is computed as:
        
        .. math::
            \\textrm{VaR}^T_\\alpha (\\Delta \\pi_{t+T}(X_t)) = \\sqrt{T} \\cdot \\textrm{VaR}^{\\textrm{1 day}}_\\alpha (\\Delta \\pi_{t+\\textrm{1 day}}(X_t)) \,.
        
        Parameters
        ----------
        dates : pandas.DatetimeIndex, list of strings, pandas.Timestamp, or string
            Dates at which Value at Risk (VaR) is computed. It can be a single date (as a pandas.Timestamp or its string representation) or an array-like object (as a
            pandas.DatetimeIndex or a list of its string representation) containing dates.
        discount_curve : discount_curves.DiscountCurve
            Gives the discounts for the computations.
        time_horizon : int, optional
            Time horizon for the VaR in days (defaults to 1).
        conf_level : float, optional
            Confidence level for the VaR. The value pertains to the distribution of the profit function (defaults to 0.01).
        no_sims : int, optional
            Number of simulations for Monte Carlo valuation of VaR (defaults to 10**4).
        no_sims_price : int, optional
            Number of simulations for Monte Carlo valuation of product's price (defaults to 10**6).
        no_calcs : int, optional
            The product's price is computed ``no_calcs`` times and then the mean value is returned (defaults to 1).
            With this we can effectively prevent memory errors ("MemoryError: Unable to allocate...") in Python when increasing the number of simulations.

        Returns
        ----------
        pandas.Series
            Value at Risk (VaR) for the given dates.

        References
        ----------
        - [McNeil et al., 2015] McNeil, J. A., Frey, R., Embrechts, P. (2015). Quantitative Risk Management: Concepts, Techniques and Tools.

        - [Privault, 2024] Privault, N. (2024) Notes on Financial Risk and Analytics.
        """
        dates = afsfun.dates_formatting(dates)
        prices = self.get_px(
            dates=dates,
            discount_curve=discount_curve,
            no_sims=no_sims_price,
            no_calcs=no_calcs_price,
        )
        prices_underlying = pd.DataFrame(
            self.underlying.get_value(dates).values, index=dates, columns=["Price"]
        )
        dates_plus = dates + pd.tseries.offsets.BDay()
        prices_underlying_plus = pd.DataFrame(
            self.underlying.get_value(dates_plus).values,
            index=dates_plus,
            columns=["Price"],
        )
        tau = self.calendar.interval(dates=dates, future_dates=dates_plus)
        vols = self.underlying.get_vol(dates=dates)
        if (
            self.monotonicity_price_function == "decreasing"
            or self.monotonicity_price_function == "increasing"
        ):
            if self.monotonicity_price_function == "decreasing":
                conf_level = 1 - conf_level
            if isinstance(self.underlying, LognormalAsset):
                # TODO: introduce parameter, argument of method with default value,
                # taking into account the drift and Ito's term.
                # More general distributions after a full review?
                prices_underlying_stress = np.transpose(
                    prices_underlying.values
                ) * np.exp(
                    -1 / 2 * (vols**2 * tau).values
                    + (vols * np.sqrt(tau)).values * norm.ppf(conf_level)
                )
            elif isinstance(self.underlying, NormalAsset):
                prices_underlying_stress = np.transpose(prices_underlying.values) + (
                    vols * np.sqrt(tau)
                ).values * norm.ppf(conf_level)
            prices_underlying_stress = np.transpose(prices_underlying_stress)
            prices_underlying_stress = pd.DataFrame(
                prices_underlying_stress, index=dates_plus, columns=["Price"]
            )
            self.underlying.set_data(prices_underlying_stress)
            prices_stress = self.get_px(
                dates=dates_plus,
                discount_curve=discount_curve,
                no_sims=no_sims_price,
                no_calcs=no_calcs_price,
            )
        else:
            all_prices = self.generate_prices(
                dates, discount_curve, vols, no_sims, no_sims_price, no_calcs_price
            )
            prices_stress = np.sort(
                np.partition(
                    np.transpose(all_prices),
                    int(no_sims * conf_level),
                    axis=1,
                )
            )[:, int(no_sims * conf_level)]
            prices_stress = pd.Series(prices_stress, dates)
        var = (prices - prices_stress.values) * np.sqrt(time_horizon)
        self.underlying.set_data(prices_underlying_plus)
        return var

    def get_es(
        self,
        dates,
        discount_curve,
        time_horizon=1,
        conf_level=0.01,
        no_sims=10**4,
        no_sims_price=10**4,
        no_calcs_price=1,
    ):
        """
        Calculate the Expected Shortfall (ES) of a structured product relative to a \
        specific time horizon and confidence level for the given dates.

        The method considers the underlying price :math:`X` as the sole risk factor. \
        In this context, the :math:`\\alpha` confidence level ES \
        :math:`\\textrm{ES}^T_\\alpha (\\Delta \\pi_{t+T}(X_t))` \
        at time :math:`t` with time horizon :math:`T` is calculated as:

        .. math::
            \\textrm{ES}^T_\\alpha (\\Delta \\pi_{t+T}(X_t)) \\equiv \\frac{1}{\\alpha} \\int_0^{\\alpha} \\textrm{VaR}^T_\\gamma (\\Delta \\pi_{t+T}(X_t)) \, \\textrm{d}\\gamma = \\frac{1}{N} \\sum_{i=0}^N \\textrm{VaR}^T_{\\frac{i}{N}\\alpha} (\\Delta \\pi_{t+T}(X_t)) + o_\\infty (N^0) \,,

        where:

        - :math:`\\Delta \\pi_{t+\\textrm{1 day}}(X_t) \\equiv \\pi(X_{t+\\textrm{1 day}}) - \\pi(X_t)` is the product profit function between :math:`t` and :math:`t+\\textrm{1 day}`,
        - :math:`N = \\textit{no_sims} \\cdot \\textit{conf_level}` is the number of intervals `\\alpha` is divided in,
        - :math:`o_\\infty (N^0)` is a quantity :math:`x` such that :math:`\\lim\\limits_{N \\to \\infty} \\frac{x}{N^0} = \\lim\\limits_{N \\to \\infty} x = 0`.

        Parameters
        ----------
        dates : pandas.DatetimeIndex, list of strings, pandas.Timestamp, or string
            Dates at which the Expected Shortfall (ES) is computed. It can be a single date (as a pandas.Timestamp or its string representation) or an array-like object (as a
            pandas.DatetimeIndex or a list of its string representation) containing dates.
        discount_curve : discount_curves.DiscountCurve
            Gives the discounts for the computations.
        time_horizon : int, optional
            Time horizon for the ES in days (defaults to 1).
        conf_level : float, optional
            Confidence level for the ES. The value pertains to the distribution of the profit function (defaults to 0.01).
        no_sims : int, optional
            Number of simulations for Monte Carlo valuation of the ES (defaults to 10**4).
        no_sims_price : int, optional
            Number of simulations for Monte Carlo valuation of product's price (defaults to 10**6).
        no_calcs : int, optional
            The product's price is computed ``no_calcs`` times and then the mean value is returned (defaults to 1).
            With this we can effectively prevent memory errors ("MemoryError: Unable to allocate...") in Python when increasing the number of simulations.

        Returns
        ----------
        pandas.Series
            Expected Shortfall (ES) for the given dates.

        References
        ----------
        - [McNeil et al., 2015] McNeil, J. A., Frey, R., Embrechts, P. (2015). Quantitative Risk Management: Concepts, Techniques and Tools.

        - [Privault, 2024] Privault, N. (2024) Notes on Financial Risk and Analytics.
        """
        dates = afsfun.dates_formatting(dates)
        prices = self.get_px(
            dates=dates,
            discount_curve=discount_curve,
            no_sims=no_sims_price,
            no_calcs=no_calcs_price,
        )
        dates_plus = dates + pd.tseries.offsets.BDay()
        prices_underlying_plus = pd.DataFrame(
            self.underlying.get_value(dates_plus).values,
            index=dates_plus,
            columns=["Price"],
        )
        vols = self.underlying.get_vol(dates=dates)
        if (
            self.monotonicity_price_function == "decreasing"
            or self.monotonicity_price_function == "increasing"
        ):
            vars = []
            for i in range(1, int(conf_level * no_sims) + 1):
                var = self.get_var(
                    dates=dates,
                    discount_curve=discount_curve,
                    time_horizon=time_horizon,
                    conf_level=conf_level * i / (int(conf_level * no_sims)),
                    no_sims=no_sims,
                    no_sims_price=no_sims_price,
                    no_calcs_price=no_calcs_price,
                ).values
                vars.append(var)
            vars = np.array(vars)
            es = np.mean(vars, axis=0)
        else:
            all_prices = self.generate_prices(
                dates, discount_curve, vols, no_sims, no_sims_price, no_calcs_price
            )
            prices_stress = np.sort(
                np.partition(
                    np.transpose(all_prices),
                    int(no_sims * conf_level),
                    axis=1,
                )
            )[:, : int(no_sims * conf_level)]
            prices_stress = np.transpose(prices_stress)
            es = np.mean(
                (prices.values - prices_stress) * np.sqrt(time_horizon), axis=0
            )
            self.underlying.set_data(prices_underlying_plus)
        es = pd.Series(es, dates)
        return es

    def plot_payoff(self, mins, maxs, past_values="zero"):
        """
        Return a plot of the payoff as a function of the price :math:`S_T`.

        Parameters
        ----------
        mins : float
            Minimum value of :math:`S_T` to be plotted.
        maxs : float
            Maximum value of :math:`S_T` to be plotted.
        past_values : str, default = "zero"
            If past_values="zero", for path dependent options we set :math:`S_t = 0` for all :math:`t < T`.
            If past_values="copy", for path dependent options we set :math:`S_t = S_T` for all :math:`t < T`. This method is preferred when the product averages the prices over
            all the observation dates
        Returns
        -------
        html
            Plot of the payoff as a function of the price :math:`S_T`.

        """
        num = 10**3  # Number of points.
        lenobs = len(self.obsdates)
        if lenobs > 1:
            prices = np.zeros((num, lenobs, 1, 1))
            if past_values == "zero":
                prices[:, -1] = np.linspace(
                    start=mins, stop=maxs, num=num, endpoint=True
                ).reshape(
                    num, 1, 1
                )  # We generate a uniformly distributed array with only one
                # observation date (maturity, last column) and num of simulations. The prices for different observation dates are assumed to be zero.
            elif past_values == "copy":
                # Copy value of prices in all the observations so that, when means are used in path dependant products, the correct strike is achieved
                prices[:, :] = np.repeat(
                    np.linspace(start=mins, stop=maxs, num=num, endpoint=True).reshape(
                        num, 1, 1, 1
                    ),
                    lenobs,
                    axis=1,
                )
            else:
                raise NameError(
                    f"{past_values} is not a correct keyword for past_values."
                )

            dates_dic = pd.Series(
                {date: num for num, date in enumerate(self.obsdates)}
            )  # The observation dates are the only ones contained in dates_dic,
            payoff_matrix = self.payoff(prices, dates_dic, 0)
            payoff = payoff_matrix[
                :, lenobs - 1, 0
            ]  # We recover the non-zero values of the payoff matrix.
        else:
            prices = np.linspace(start=mins, stop=maxs, num=num, endpoint=True).reshape(
                num, 1, 1, 1
            )  # We generate a uniformly distributed array with only one
            # observation date (maturity) and num of simulations.
            dates_dic = pd.Series({pd.Timestamp(self.maturity): 0})  # We set index=0.
            payoff_matrix = self.payoff(prices, dates_dic, 0)
            payoff = payoff_matrix.reshape(num)
        prices = np.linspace(
            start=mins, stop=maxs, num=num, endpoint=True
        )  # The uniformly distributed array of prices for the plot.
        ps = pd.Series(payoff, index=prices)  # We create a Pandas Series for the plot
        fig = px.line(ps)
        if lenobs > 1:
            if past_values == "copy":
                fig.update_layout(
                    xaxis_title="$S_T$",
                    yaxis_title="Payoff",
                    legend=dict(title=""),
                    title=f"Payoff {self.__class__.__name__}. The value of S_t for t < T is set to S_T.",
                    showlegend=False,
                )
            else:
                fig.update_layout(
                    xaxis_title="$S_T$",
                    yaxis_title="Payoff",
                    legend=dict(title=""),
                    title=f"Payoff {self.__class__.__name__}. The value of S_t for t < T is set to zero.",
                    showlegend=False,
                )
        else:
            fig.update_layout(
                xaxis_title="$S_T$",
                yaxis_title=f"Payoff",
                legend=dict(title=""),
                title=f"Payoff {self.__class__.__name__}",
                showlegend=False,
            )
        return fig.show()


class Derivative(Structured):
    """
    date format: %Y%m%d
    """

    def __init__(
        self,
        underlying,
        obsdates,
        pastmatters,
        calendar,
        pay_dates=None,
        credit_curve=zero_credit_risk,
        monotonicity_price_function=None,
    ):
        Structured.__init__(
            self,
            obsdates=obsdates,
            pastmatters=pastmatters,
            pay_dates=pay_dates,
            credit_curve=credit_curve,
            monotonicity_price_function=monotonicity_price_function,
        )
        self.underlying = underlying
        self.calendar = calendar

    def get_underlying_fpx(self, dates, discount_curve):
        """
        Compute the forward price of the underlying asset at given dates.

        Parameters
        ----------
        dates : pandas.DatetimeIndex, list of strings, pandas.Timestamp, or string
            Dates at which to compute the forward price. It can be a single date (as a pandas.Timestamp or its string representation) or an array-like object
            (as a pandas.DatetimeIndex or a list of its string representation) containing dates.
        discount_curve : pricing.discount_curves.DiscountCurve
            Gives the discounts for the computations.

        Returns
        -------
        pandas.Series
            The forward price.
        """
        # This methods exists for uniformity with Quanto options (it avoids having to re-write option functions)
        fpx = self.underlying.get_fpx(
            dates=dates,
            fdate=self.maturity,
            discountcurve=discount_curve,
            calendar=self.calendar,
        )
        return fpx

    def compute_discount_tolerances(self, dates, discount_curve):
        """
        Construct a table filled with the computed tolerances values of the derivative at the given dates.
        The tolerances are computed as the difference between the price of the product with a slight modification (i.e., a bump)
        of the value of interest rate (or credit spread) and the original price.

        Parameters
        ----------
        dates : pandas.DatetimeIndex, list of strings, pandas.Timestamp, or string
            Dates at which to compute the tolerances. It can be a single date (as a pandas.Timestamp or its string representation) or an array-like object
            (as a pandas.DatetimeIndex or a list of its string representation) containing dates.
        discount_curve : pricing.discount_curves.DiscountCurve
            Gives the discounts for the computations.

        Returns
        -------
        pandas.Dataframe
            The table with tolerances. Every row represents a date, while the different kind of positive and negative bumps are on the columns.

        Notes
        ----------
        Let us make an example for the interest rate tolerance in order to explain how it is computed mathematically:

        .. math::
            \\Delta V = V(t, D + \\Delta D, \\ldots) - V(t, D, \\ldots)\\,,

        where:

        - :math:`\\Delta V` is the tolerance for interest rates,
        - :math:`V(t, D, \\ldots)` is the price of the derivative at time `t` without the interest rate bump.
          Its price depends on the discount curve used, :math:`D`, and, possibly, other parameters.
        - :math:`V(t, D + \\Delta D, \\ldots)` is the price of the derivative at time `t`
          with the interest rate bump, :math:`\\Delta D`, represented as a change in the discount curve.

        See Also
        --------
        discount_curves.YieldCurve.fit
            For more details on how the bumps are calculated.

        """
        # this is separated because it is redefined for products for which get_px only run with mc
        dates = afsfun.dates_formatting(dates)
        int_header = pd.MultiIndex.from_product(
            [["Tipos de Interés"], ["-5bp", "+5bp"]]
        )
        cred_header = pd.MultiIndex.from_product(
            [["Spread Crédito"], ["-20bp", "+20bp"]]
        )
        headers = pd.MultiIndex.union(int_header, cred_header, sort=False)
        table = pd.DataFrame(index=headers, columns=dates).transpose()

        prices = self.get_px(dates=dates, discount_curve=discount_curve)
        discount_curve.p = -1
        if hasattr(self.underlying, "curve"):
            if self.underlying.spot_curve.is_ir_curve:
                self.underlying.spot_curve.p = -1
        table["Tipos de Interés", "-5bp"] = (
            self.get_px(dates=dates, discount_curve=discount_curve) - prices
        )
        discount_curve.p = 1
        if hasattr(self.underlying, "curve"):
            if self.underlying.spot_curve.is_ir_curve:
                self.underlying.spot_curve.p = 1
        table["Tipos de Interés", "+5bp"] = (
            self.get_px(dates=dates, discount_curve=discount_curve) - prices
        )
        discount_curve.p = 0
        if hasattr(self.underlying, "curve"):
            if self.underlying.spot_curve.is_ir_curve:
                self.underlying.spot_curve.p = 0
        self.credit_curve.p = -1
        if hasattr(self.underlying, "curve"):
            if self.underlying.spot_curve.is_credit_curve:
                self.underlying.spot_curve.p = -1
        table["Spread Crédito", "-20bp"] = (
            self.get_px(dates=dates, discount_curve=discount_curve) - prices
        )
        self.credit_curve.p = 1
        if hasattr(self.underlying, "curve"):
            if self.underlying.spot_curve.is_credit_curve:
                self.underlying.spot_curve.p = 1
        table["Spread Crédito", "+20bp"] = (
            self.get_px(dates=dates, discount_curve=discount_curve) - prices
        )
        self.credit_curve.p = 0
        if hasattr(self.underlying, "curve"):
            if self.underlying.spot_curve.is_credit_curve:
                self.underlying.spot_curve.p = 0
        return table

    def compute_equity_tolerances(self, dates, discount_curve):
        """
        Construct a table filled with the computed tolerances values of the derivative at the given dates.
        The tolerances are computed as the difference between the price of the product with a slight modification (i.e., a bump)
        of the value of equity volatility (or equity dividends) and the original price.

        Parameters
        ----------
        dates : pandas.DatetimeIndex, list of strings, pandas.Timestamp, or string
            Dates at which to compute the tolerances. It can be a single date (as a pandas.Timestamp or its string representation) or an array-like object
            (as a pandas.DatetimeIndex or a list of its string representation) containing dates.
        discount_curve : pricing.discount_curves.DiscountCurve
            Gives the discounts for the computations.

        Returns
        -------
        pandas.Dataframe
            The table with tolerances. The index of this data frame is ``dates``, while the different kind of
            positive and negative bumps are the columns.

        Notes
        ----------
        Let us make an example for the equity volatility tolerance in order to explain how it is computed mathematically:

        .. math::
            \\Delta V = V(t, \\sigma + \\Delta \\sigma, \\ldots) - V(t, \\sigma, \\ldots)\\,,

        where:

        - :math:`\\Delta V` is the tolerance for equity volatility,
        - :math:`V(t, \\sigma, \\ldots)` is the price of the derivative at time `t` without the equity volatility bump.
          Its price depends on the equity volatility used, :math:`\\sigma`, and, possibly, other parameters.
        - :math:`V(t, \\sigma + \\Delta \\sigma, \\ldots)` is the price of the derivative at time `t`
          with the equity volatility bump, :math:`\\Delta \\sigma`, represented as a change in the equity volatility.

        """

        dates = afsfun.dates_formatting(dates)
        vol_header = pd.MultiIndex.from_product(
            [["EQ Volatilidad"], ["-200bp", "+200bp"]]
        )
        div_header = pd.MultiIndex.from_product(
            [["EQ Dividendos"], ["-100bp", "+100bp"]]
        )
        eq_headers = pd.MultiIndex.union(vol_header, div_header, sort=False)
        table = pd.DataFrame(index=eq_headers, columns=dates).transpose()

        prices = self.get_px(dates, discount_curve)
        if hasattr(self.underlying, "get_divrate"):
            divs = pd.DataFrame(
                self.underlying.get_divrate(dates).values,
                index=dates,
                columns=["Dividend Rate"],
            )
            self.underlying.set_data(divs - 0.01)
            table["EQ Dividendos", "-100bp"] = (
                self.get_px(dates, discount_curve) - prices
            )
            self.underlying.set_data(divs + 0.01)
            table["EQ Dividendos", "+100bp"] = (
                self.get_px(dates, discount_curve) - prices
            )
            self.underlying.set_data(divs)
        else:
            table["EQ Dividendos", "-100bp"] = float(0)
            table["EQ Dividendos", "+100bp"] = float(0)
        vols = pd.DataFrame(
            self.underlying.get_vol(dates).values, index=dates, columns=["Volatility"]
        )
        self.underlying.set_data(vols - 0.02)
        table["EQ Volatilidad", "-200bp"] = self.get_px(dates, discount_curve) - prices
        self.underlying.set_data(vols + 0.02)
        table["EQ Volatilidad", "+200bp"] = self.get_px(dates, discount_curve) - prices
        self.underlying.set_data(vols)
        return table


class MCProduct:
    def __init__(self):
        self.calendar = None
        self.underlying = None
        self.maturity = None

    def get_px(self, dates, discount_curve, no_sims=10**6, no_calcs=10):
        # We could just use MCnd by default, since it works with one asset also
        """
        Compute the price.

        Parameters
        ----------
        dates : pandas.DatetimeIndex, list of strings, pandas.Timestamp, or string
            Valuation dates. It can be a single date (as a pandas.Timestamp or its string representation) or an array-like object (as a pandas.DatetimeIndex or a list of its string
            representation) containing dates.
        discount_curve : pricing.discount_curves.DiscountCurve
            Gives the discounts for the computations.
        no_sims: int, optional
            Number of simulations (defaults to 10**6).
        no_calcs : int, optional
            The price is computed ``no_calcs`` times and then the mean value is returned (defaults to 10).
            With this we can effectively prevent memory errors ("MemoryError: Unable to allocate...") in Python when increasing the number of simulations.

        Returns
        -------
        pandas.Series
            The calculated price of the product.
        """

        if type(self.underlying) == list:  # TODO: remove this
            x = MC1d()
        else:
            x = DiffusionMC()

        price = x.price(
            self,
            dates=dates,
            discount_curve=discount_curve,
            no_sims=no_sims,
            no_calcs=no_calcs,
        )

        return price

    def get_delta(
        self, dates, discount_curve, no_sims=10**6, no_calcs=10, monte_carlo=False
    ):
        """
        Calculate the `discrete` delta of a product. Delta is the derivative of the product's value w.r.t. the underlying.

        Parameters
        ----------
        dates : pandas.DatetimeIndex, list of strings, pandas.Timestamp, or string
            Dates at which delta is calculated. It can be a single date (as a pandas.Timestamp or its string representation) or an array-like object (as a pandas.DatetimeIndex or a
            list of its string representation) containing dates.
        discount_curve : pricing.discount_curves.DiscountCurve
            Provides the discounts for the computation of the Greek.
        no_sims: int, optional
            Number of simulations (defaults to 10**6).
        no_calcs : int, optional
            The Greek is computed ``no_calcs`` times and then the mean value is returned (defaults to 10).
            With this we can effectively prevent memory errors ("MemoryError: Unable to allocate...") in Python when increasing the number of simulations.
        monte_carlo: bool, optional
            If ``True``, uses Monte Carlo simulations for computation, even if analytical formulas are available (defaults to False).

        Returns
        -------
        pandas.Series
            Returns the delta for the given dates.

        References
        ----------
        - [Andersen and Piterbarg, 2010] Andersen, L. B. G., and Piterbarg, V.V. (2010). Interest Rate Modeling.

        """
        dates = afsfun.dates_formatting(dates)

        prices_underlying = pd.DataFrame(
            self.underlying.get_value(dates).values, index=dates, columns=["Price"]
        )
        central_delta_est = pd.Series(data=0, index=dates)
        if (
            getattr(type(self), "get_px") != getattr(MCProduct, "get_px")
            and not monte_carlo
        ):
            h = prices_underlying * 0.005  # Table 1.1 of Greeks Calculations in DLIB.
            self.underlying.set_data(prices_underlying + h)
            prices_plus = self.get_px(dates, discount_curve)
            self.underlying.set_data(prices_underlying - h)
            prices_minus = self.get_px(dates, discount_curve)
            h = h.squeeze(axis=1)  # Transform to a series, like prices
            central_delta_est = (prices_plus - prices_minus) / (2 * h)
            self.underlying.set_data(prices_underlying)  # Restores the initial values

        else:
            engine = DiffusionMC()
            for i in range(no_calcs):
                h = (
                    prices_underlying * 0.005
                )  # See also Table 1.1 of Greeks Calculations in DLIB.
                self.underlying.set_data(prices_underlying + h)
                prices_plus = engine.price(
                    self, dates, discount_curve, no_sims=no_sims
                )  # We use engine as we need to mantain the draws, as explained below
                self.underlying.set_data(prices_underlying - h)
                prices_minus = engine.price(
                    self, dates, discount_curve, no_sims=no_sims, reuse_draws=True
                )
                engine.draws = None  # Clear the draws, used the same for both computations. See [Andersen and Piterbarg], , p.134-135.
                h = h.squeeze(axis=1)  # Transform to a series, like prices
                central_delta_est += (prices_plus - prices_minus) / (2 * h)
            self.underlying.set_data(prices_underlying)  # Restores the initial values
            central_delta_est /= no_calcs

        return central_delta_est

    def get_vega(
        self, dates, discount_curve, no_sims=10**6, no_calcs=10, monte_carlo=False
    ):
        """
        Calculate the `discrete` vega of a product. Vega is the derivative of the product's value w.r.t. the underlying volatility.

        Parameters
        ----------
        dates : pandas.DatetimeIndex, list of strings, pandas.Timestamp, or string
            Dates at which vega is calculated. It can be a single date (as a pandas.Timestamp or its string representation) or an array-like object (as a pandas.DatetimeIndex or a
            list of its string representation) containing dates.
        discount_curve : pricing.discount_curves.DiscountCurve
            Provides the discounts for the computation of the Greek.
        no_sims: int, optional
            Number of simulations (defaults to 10**6).
        no_calcs : int, optional
            The Greek is computed ``no_calcs`` times and then the mean value is returned (defaults to 10).
            With this we can effectively prevent memory errors ("MemoryError: Unable to allocate...") in Python when increasing the number of simulations.
        monte_carlo: bool, optional
            If ``True``, uses Monte Carlo simulations for computation, even if analytical formulas are available (defaults to False).

        Returns
        -------
        pandas.Series
            Returns the vega for the given dates.

        References
        ----------
        - [Andersen and Piterbarg, 2010] Andersen, L. B. G., and Piterbarg, V.V. (2010). Interest Rate Modeling.

        """
        dates = afsfun.dates_formatting(dates)

        vols_underlying = pd.DataFrame(
            self.underlying.get_vol(dates).values, index=dates, columns=["Volatility"]
        )
        central_vega_est = pd.Series(data=0, index=dates)
        if isinstance(self.underlying, LognormalAsset):
            h = 0.0025  # See also Table 1.1 of Greeks Calculations in DLIB.
        elif isinstance(self.underlying, NormalAsset):
            h = 0.00025
        else:
            h = 0.0025
        if (
            getattr(type(self), "get_px") != getattr(MCProduct, "get_px")
            and not monte_carlo
        ):
            self.underlying.set_data(vols_underlying + h)
            prices_plus = self.get_px(dates, discount_curve)
            self.underlying.set_data(vols_underlying - h)
            prices_minus = self.get_px(dates, discount_curve)
            # h = h.squeeze(axis=1)  # Transform to a series, like prices
            central_vega_est = (prices_plus - prices_minus) / (2 * h)
            self.underlying.set_data(vols_underlying)  # Restores the initial values

        else:
            engine = DiffusionMC()
            for i in range(no_calcs):
                self.underlying.set_data(vols_underlying + h)
                prices_plus = engine.price(
                    self, dates, discount_curve, no_sims=no_sims
                )  # We use engine as we need to mantain the draws, as explained below
                self.underlying.set_data(vols_underlying - h)
                prices_minus = engine.price(
                    self, dates, discount_curve, no_sims=no_sims, reuse_draws=True
                )
                engine.draws = None  # Clear the draws, used the same for both computations. See [Andersen and Piterbarg], , p.134-135.
                # h = h.squeeze(axis=1)  # Transform to a series, like prices
                central_vega_est += (prices_plus - prices_minus) / (2 * h)
            self.underlying.set_data(vols_underlying)  # Restores the initial values
            central_vega_est /= no_calcs

        return central_vega_est

    def get_rho(
        self, dates, discount_curve, no_sims=10**6, no_calcs=10, monte_carlo=False
    ):
        """
        Calculate the `discrete` rho of a product. Rho is the derivative of the product's value w.r.t. the interest rate.

        Parameters
        ----------
        dates : pandas.DatetimeIndex, list of strings, pandas.Timestamp, or string
            Dates at which rho is calculated. It can be a single date (as a pandas.Timestamp or its string representation) or an array-like object (as a pandas.DatetimeIndex or a
            list of its string representation) containing dates.
        discount_curve : pricing.discount_curves.DiscountCurve
            Provides the discounts for the computation of the Greek.
        no_sims: int, optional
            Number of simulations (defaults to 10**6).
        no_calcs : int, optional
            The Greek is computed ``no_calcs`` times and then the mean value is returned (defaults to 10).
            With this we can effectively prevent memory errors ("MemoryError: Unable to allocate...") in Python when increasing the number of simulations.
        monte_carlo: bool, optional
            If ``True``, uses Monte Carlo simulations for computation, even if analytical formulas are available (defaults to False).

        Returns
        -------
        pandas.Series
            Returns the rho for the given dates.

        References
        ----------
        - [Andersen and Piterbarg, 2010] Andersen, L. B. G., and Piterbarg, V.V. (2010). Interest Rate Modeling.

        """
        dates = afsfun.dates_formatting(dates)
        central_rho_est = pd.Series(data=0, index=dates)
        if (
            getattr(type(self), "get_px") != getattr(MCProduct, "get_px")
            and not monte_carlo
        ):
            discount_curve.p = -1
            prices_plus = self.get_px(dates=dates, discount_curve=discount_curve)
            discount_curve.p = 1
            prices_minus = self.get_px(dates=dates, discount_curve=discount_curve)
            central_rho_est = (prices_plus - prices_minus) / (2 * 0.0005)
            discount_curve.p = 0  # Restores the initial values

        else:
            engine = DiffusionMC()
            for i in range(no_calcs):
                discount_curve.p = -1
                prices_plus = engine.price(
                    self, dates, discount_curve, no_sims=no_sims
                )  # We use engine as we need to mantain the draws, as explained below
                discount_curve.p = 1
                prices_minus = engine.price(
                    self, dates, discount_curve, no_sims=no_sims, reuse_draws=True
                )
                engine.draws = None  # Clear the draws, used the same for both computations. See [Andersen and Piterbarg], , p.134-135.
                central_rho_est += (prices_plus - prices_minus) / (2 * 0.0005)
            discount_curve.p = 0  # Restores the initial values
            central_rho_est /= no_calcs

        return central_rho_est

    def get_gamma(
        self, dates, discount_curve, no_sims=10**6, no_calcs=10, monte_carlo=False
    ):
        """
        Calculate the `discrete` gamma of a product. Gamma is the second derivative of the product's value w.r.t. the underlying value.

        Parameters
        ----------
        dates : pandas.DatetimeIndex, list of strings, pandas.Timestamp, or string
            Dates at which gamma is calculated. It can be a single date (as a pandas.Timestamp or its string representation) or an array-like object (as a pandas.DatetimeIndex or a
            list of its string representation) containing dates.
        discount_curve : pricing.discount_curves.DiscountCurve
            Provides the discounts for the computation of the Greek.
        no_sims: int, optional
            Number of simulations (defaults to 10**6).
        no_calcs : int, optional
            The Greek is computed ``no_calcs`` times and then the mean value is returned (defaults to 10).
            With this we can effectively prevent memory errors ("MemoryError: Unable to allocate...") in Python when increasing the number of simulations.
        monte_carlo: bool, optional
            If ``True``, uses Monte Carlo simulations for computation, even if analytical formulas are available (defaults to False).

        Returns
        -------
        pandas.Series
            Returns the gamma for the given dates.

        References
        ----------
        - [Andersen and Piterbarg, 2010] Andersen, L. B. G., and Piterbarg, V.V. (2010). Interest Rate Modeling.

        """
        dates = afsfun.dates_formatting(dates)

        prices_underlying = pd.DataFrame(
            self.underlying.get_value(dates).values, index=dates, columns=["Price"]
        )
        central_gamma_est = pd.Series(data=0, index=dates)
        if (
            getattr(type(self), "get_px") != getattr(MCProduct, "get_px")
            and not monte_carlo
        ):
            prices_middle = self.get_px(dates, discount_curve)
            h = prices_underlying * 0.01  # Table 1.1 of Greeks Calculations in DLIB.
            self.underlying.set_data(prices_underlying + h)
            prices_plus = self.get_px(dates, discount_curve)
            self.underlying.set_data(prices_underlying - h)
            prices_minus = self.get_px(dates, discount_curve)
            h = h.squeeze(axis=1)  # Transform to a series, like prices
            central_gamma_est = (prices_plus - 2 * prices_middle + prices_minus) / h**2
            self.underlying.set_data(prices_underlying)  # Restores the initial values

        else:
            engine = DiffusionMC()
            for i in range(no_calcs):
                prices_middle = engine.price(
                    self, dates, discount_curve, no_sims=no_sims
                )
                h = (
                    prices_underlying * 0.01
                )  # See also Table 1.1 of Greeks Calculations in DLIB.
                self.underlying.set_data(prices_underlying + h)
                prices_plus = engine.price(
                    self, dates, discount_curve, no_sims=no_sims, reuse_draws=True
                )  # We use engine as we need to mantain the draws, as explained below
                self.underlying.set_data(prices_underlying - h)
                prices_minus = engine.price(
                    self, dates, discount_curve, no_sims=no_sims, reuse_draws=True
                )
                engine.draws = None  # Clear the draws, used the same for both computations. See [Andersen and Piterbarg], , p.134-135.
                h = h.squeeze(axis=1)  # Transform to a series, like prices
                central_gamma_est += (
                    prices_plus - 2 * prices_middle + prices_minus
                ) / h**2
                self.underlying.set_data(
                    prices_underlying
                )  # Restores the initial values
            central_gamma_est /= no_calcs

        return central_gamma_est

    def get_theta(
        self,
        dates,
        discount_curve,
        no_sims=10**7,
        no_calcs=10,
        maturity_method=False,
        monte_carlo=False,
        constant_curve=False,
        refit=False,
    ):
        """
        Calculate the `discrete` theta of a product. Theta is the derivative of the product's value w.r.t. time.

        Parameters
        ----------
        dates : pandas.DatetimeIndex, list of strings, pandas.Timestamp, or string
            Dates at which theta is calculated. It can be a single date (as a pandas.Timestamp or its string representation) or an array-like object (as a pandas.DatetimeIndex or a
            list of its string representation) containing dates.
        discount_curve :  pricing.discount_curves.DiscountCurve
            Provides the discounts for the computation of the Greek.
        no_sims: int, optional
            Number of simulations (defaults to 10**7).
        no_calcs : int, optional
            The Greek is computed ``no_calcs`` times and then the mean value is returned (defaults to 10).
            With this we can effectively prevent memory errors ("MemoryError: Unable to allocate...") in Python when increasing the number of simulations.
        maturity_method : bool, optional
            If True, the derivative is computed w.r.t. -T, minus maturity, which should be the same if the price depends on t as (T-t), the tenor.
            This is common for vanilla options (defaults to False).
        monte_carlo: bool, optional
            If ``True``, uses Monte Carlo simulations for computation, even if analytical formulas are available (defaults to False).
        constant_curve : bool, optional
            If True, keeps the discount curve constant while computing the theta (defaults to False). Only applicable if the price depends only on :math:`r_t`, where :math:`t` is the valuation date.
        refit : bool, optional
            If True, refit the discount curve while shifting. This is applicable when shifting the curve to compute theta. If False, the curve is scaled (defaults to False).

        Returns
        -------
        pandas.Series
            Returns the theta for the given dates. It is a pandas.Series where the index corresponds to the input dates and the values are the computed thetas.

        Notes
        -----
        It actually computes a `discrete` theta in the sense of (2.15) of [Andersen and Piterbarg, 2010].
        The time step is chosen as small as possible, i.e., the previous and next business day.

        Note
        --------
        - ``maturity_method`` only works if the interest rate and the dividend yield are constant, because it relies on the assumption \
            :math:`\\pi = \\pi(T - t)`, where :math:`\\pi` is the price of the derivative at time `t` and `T` is the maturity. \
            In this case :math:`\\Theta \\equiv \\frac{\\partial \\pi (T-t)}{\\partial t} = \
            \\frac{\\partial \\pi}{\\partial (T-t)} \\frac{\\partial (T-t)}{\\partial t} = \
            \\frac{\\partial \\pi}{\\partial (T-t)} \\left(- \\frac{\\partial (T-t)}{\\partial T} \\right) = \
            -\\frac{\\partial \\pi (T-t)}{\\partial T}` \
            and is practically computed via central estimation method, :math:`\\hat{\\Theta}(t_+, t_-) = \
            - \\frac{\\pi(t_+) - \\pi(t_-)}{t_+ - t_-}`, \
            where :math:`t_+` is the day following `t` and :math:`t_-` is the day preceding `t`. \
            When one between the interest rate and the dividend yield is non constant, the price of the derivative usually depends on \
            factors like :math:`\\mathrm{e}^{-\\int^T_t q(s) \\, \\mathrm{d}s}` or \
            the discount factor :math:`\\mathrm{e}^{-\\int^T_t r(s) \\, \\mathrm{d}s}`, \
            which makes the initial assumption wrong and the method not applicable.
            
        - ``constant_curve`` only works if the derivative price :math:`\\pi` is not path dependent, \
            namely depends on the valuation and maturity dates, but not on other dates. \
            Given the valuation and maturity dates, respectively :math:`t_0` and :math:`T_0`, \
            we define :math:`\\bar{r} (t_0, T_0) \\equiv - \\frac{\\ln D(t_0, T_0)}{T_0 - t_0}`, \
            where `D(t, T)` is the original discount curve. \
            We build a new discount curve as :math:`\\bar{D}(t, T) \\equiv \\mathrm{e}^{- \\bar{r}(t_0, T_0) (T-t)}`, \
            which satisfies the condition :math:`\\bar{D}(t_0, T_0) = D(t_0, T_0)` by construction. \
            Eventually, we check the equivalence :math:`\\pi(t, D, \\dots) = \\pi(t, \\bar{D}, \\dots)`, \
            that is to say the derivative price is independent of the discount curve points \
            different from valuation and maturity dates.

        References
        ----------
        - [Andersen and Piterbarg, 2010] Andersen, L. B. G., and Piterbarg, V.V. (2010). Interest Rate Modeling.
        """
        dates = afsfun.dates_formatting(dates)
        x = DiffusionMC()
        central_theta_est = pd.Series(data=float(0), index=dates)
        underlying_dates = self.underlying.get_dates()
        if maturity_method:
            # We compute the derivative w.r.t. to -T, minus maturity, which should be the same if the price depends on t as (T-t), the tenor. This happens for vanilla options.
            original_maturity = self.maturity
            mat_plus = original_maturity + pd.DateOffset(days=1)
            self.maturity = mat_plus
            prices_p = self.get_px(dates, discount_curve)
            mat_minus = original_maturity - pd.DateOffset(days=1)
            self.maturity = mat_minus
            prices_m = self.get_px(dates, discount_curve)
            self.maturity = original_maturity
            time_int_mat = self.calendar.interval(mat_minus, mat_plus)
            theta = -(1 / time_int_mat) * (prices_p.values - prices_m.values)
            central_theta_est = pd.Series(data=theta, index=dates)
        else:
            # Theta is the derivative w.r.t. time only, so we must make other parameters constant.
            for (
                date
            ) in dates:  # Not doing this loop would assign two values to the same date
                date = pd.DatetimeIndex([date])  # To work with DatetimeIndex
                date_bplus = pd.DatetimeIndex(date + pd.tseries.offsets.BDay())
                # Business days so we have the price for the underlying
                target_index = underlying_dates.get_loc(date[0])
                # Find the index of the target date in the DatetimeIndex
                post_date = underlying_dates[target_index + 1]
                # This is the one that will be used for the shift below
                assert post_date == date_bplus
                date_bminus = pd.DatetimeIndex(date - pd.tseries.offsets.BDay())
                # Business days so we have the price for the underlying
                prev_date = underlying_dates[target_index - 1]
                assert prev_date == date_bminus
                time_int = self.calendar.interval(date_bminus, date_bplus)

                plus_data_total = self.underlying.data.shift(1, freq=None).dropna(
                    axis="rows", how="all"
                )
                # This method create NaN rows
                assert np.all(
                    plus_data_total.loc[date_bplus].values
                    == self.underlying.data.loc[date]
                )
                minus_data_total = self.underlying.data.shift(-1, freq=None).dropna(
                    axis="rows", how="all"
                )
                assert np.all(
                    minus_data_total.loc[date_bminus].values
                    == self.underlying.data.loc[date]
                )
                original_data_total = self.underlying.data

                if constant_curve:
                    # Constant discount curve if the price only depends on r_t
                    r = -np.log(
                        discount_curve.get_value(date, self.maturity)
                    ) / self.calendar.interval(date, self.maturity)
                    assert CRDC(r, calendar=self.calendar).get_value(
                        date, future_dates=self.maturity
                    ) == discount_curve.get_value(date, future_dates=self.maturity)
                    if (
                        not self.get_px(date, CRDC(r, calendar=self.calendar)).values
                        == self.get_px(date, discount_curve).values
                    ):
                        raise ValueError(
                            "Value depends on the whole DC curve, constant_curve could not be appropriate."
                        )
                    cte_curve = CRDC(r, calendar=self.calendar)
                    plus_curve = cte_curve
                    curve = plus_curve
                    minus_curve = plus_curve
                else:
                    if refit:
                        plus_curve = ShiftedCurve(
                            discount_curve, date, delay=1, method="refit"
                        )
                        minus_curve = ShiftedCurve(
                            discount_curve, date, delay=-1, method="refit"
                        )
                    else:
                        plus_curve = ShiftedCurve(
                            discount_curve, date, method="scaling"
                        )
                        minus_curve = ShiftedCurve(
                            discount_curve,
                            date,
                            method="scaling",
                        )
                        assert (
                            minus_curve.get_value(date_bminus, date_bminus)
                            == discount_curve.get_value(date, date)
                            == plus_curve.get_value(date_bplus, date_bplus)
                            == 1
                        )
                    curve = discount_curve

                if (
                    getattr(type(self), "get_px") != getattr(MCProduct, "get_px")
                    and not monte_carlo
                ):
                    # Uses analytical prices if they are available
                    self.underlying.set_data(plus_data_total)
                    prices_plus_an = self.get_px(date_bplus, plus_curve)
                    self.underlying.set_data(original_data_total)
                    prices_an = self.get_px(date, curve)
                    self.underlying.set_data(minus_data_total)
                    prices_minus_an = self.get_px(date_bminus, minus_curve)
                    central_theta_est_an = (
                        prices_plus_an.values - prices_minus_an.values
                    ) / time_int

                    if self.calendar.interval(
                        date_bminus, date
                    ) != self.calendar.interval(date, date_bplus):
                        # Extrapolation methods, see Glasserman
                        time_int_forward = self.calendar.interval(date, date_bplus)
                        forward_theta_est_an = (
                            prices_plus_an.values - prices_an.values
                        ) / time_int_forward
                        central_theta_est_an = (
                            time_int_forward * central_theta_est_an
                            + (time_int - 2 * time_int_forward) * forward_theta_est_an
                        ) / (time_int - time_int_forward)
                        # See my note on p.20 of Greeks calculations in DLIB. The idea is to cancel out the second order term of the Taylor expansion

                    central_theta_est.loc[date] = central_theta_est_an
                else:
                    central_theta_est_mc = 0
                    for i in range(no_calcs):
                        # Same comments as in delta_method, we use x.price instead of get_px to maintain the draws.
                        self.underlying.set_data(plus_data_total)
                        prices_plus = x.price(
                            self, date_bplus, plus_curve, no_sims=no_sims
                        )
                        self.underlying.set_data(minus_data_total)
                        prices_minus = x.price(
                            self,
                            date_bminus,
                            minus_curve,
                            no_sims=no_sims,
                            reuse_draws=True,
                        )
                        central_theta_est_mc_temp = (
                            prices_plus.values - prices_minus.values
                        ) / time_int
                        if self.calendar.interval(
                            date_bminus, date
                        ) != self.calendar.interval(date, date_bplus):
                            # Extrapolation methods, see Glasserman
                            time_int_forward = self.calendar.interval(date, date_bplus)
                            self.underlying.set_data(original_data_total)
                            prices = x.price(
                                self, date, curve, no_sims=no_sims, reuse_draws=True
                            )
                            forward_theta_est = (
                                prices_plus.values - prices.values
                            ) / time_int_forward
                            central_theta_est_mc += (
                                time_int_forward * central_theta_est_mc_temp
                                + (time_int - 2 * time_int_forward) * forward_theta_est
                            ) / (time_int - time_int_forward)
                            # See my note on p.20 of Greeks calculations in DLIB. The idea is to cancel out the second order term of the Taylor expansion
                        else:
                            central_theta_est_mc += central_theta_est_mc_temp
                        x.draws = None  # Clear the draws, used the same for both computations. See [Andersen and Piterbarg, 2010], p.134-135.
                    central_theta_est_mc /= no_calcs
                    central_theta_est.loc[date] = central_theta_est_mc

                self.underlying.set_data(original_data_total)
                # Restores the initial values

                # # [For educational purposes] Without making other parameters constant, the total variation is much higher
                # prices_plus_an = self.get_px(date_bplus, discount_curve)
                # prices_minus_an = self.get_px(date_bminus, discount_curve)
                # total_variation = (prices_plus_an.values - prices_minus_an.values)/time_int

        return central_theta_est

    def compute_discount_tolerances(self, dates, discount_curve, no_sims=10**5):
        dates = afsfun.dates_formatting(dates)
        # We could just use MCnd by default, since it works with one asset also
        if type(self.underlying) == list:
            x = MC1d()
        else:
            x = DeterministicVolDiffusionMC()
        simulation_data = x.generate_paths_for_pricing(
            self, dates=dates, discount_curve=discount_curve, no_sims=no_sims
        )
        int_header = pd.MultiIndex.from_product(
            [["Tipos de Interés"], ["+5bp", "-5bp"]]
        )
        cred_header = pd.MultiIndex.from_product(
            [["Spread Crédito"], ["+20bp", "-20bp"]]
        )
        headers = pd.MultiIndex.union(int_header, cred_header, sort=False)
        table = pd.DataFrame(index=headers, columns=dates).transpose()
        prices = x.compute_price_from_simulations(
            self, discount_curve=discount_curve, simulation_data=simulation_data
        )
        discount_curve.p = -1
        if hasattr(self.underlying, "curve"):
            if self.underlying.spot_curve.is_ir_curve:
                self.underlying.spot_curve.p = -1
        table["Tipos de Interés", "-5bp"] = (
            x.compute_price_from_simulations(
                self, discount_curve=discount_curve, simulation_data=simulation_data
            )
            - prices
        )
        discount_curve.p = 1
        if hasattr(self.underlying, "curve"):
            if self.underlying.spot_curve.is_ir_curve:
                self.underlying.spot_curve.p = 1
        table["Tipos de Interés", "+5bp"] = (
            x.compute_price_from_simulations(
                self, discount_curve=discount_curve, simulation_data=simulation_data
            )
            - prices
        )
        discount_curve.p = 0
        if hasattr(self.underlying, "curve"):
            if self.underlying.spot_curve.is_ir_curve:
                self.underlying.spot_curve.p = 0

        self.credit_curve.p = -1
        if hasattr(self.underlying, "curve"):
            if self.underlying.spot_curve.is_credit_curve:
                self.underlying.spot_curve.p = -1
        table["Spread Crédito", "-20bp"] = (
            x.compute_price_from_simulations(
                self, discount_curve=discount_curve, simulation_data=simulation_data
            )
            - prices
        )
        self.credit_curve.p = 1
        if hasattr(self.underlying, "curve"):
            if self.underlying.spot_curve.is_credit_curve:
                self.underlying.spot_curve.p = 1
        table["Spread Crédito", "+20bp"] = (
            x.compute_price_from_simulations(
                self, discount_curve=discount_curve, simulation_data=simulation_data
            )
            - prices
        )
        self.credit_curve.p = 0
        if hasattr(self.underlying, "curve"):
            if self.underlying.spot_curve.is_credit_curve:
                self.underlying.spot_curve.p = 0
        return table


# ------------------------------------------------------------------------------------
# Products
# ------------------------------------------------------------------------------------


class ProductFromFunction(MCProduct, Derivative):
    """
    Class for instantiating an object from its payoff at maturity. The product can be path-dependent (exotic) but it
    should have only one payment (at maturity).

    Parameters
    ----------
    func : function object
        Payoff at maturity. For efficiency reasons, it is better to implement it using ``numpy.where`` (see Examples).
        Function :math:`f(S)` defined in JupyterNotebook: Structured Products Documentation.

    pastmatters : bool
        True for path dependent options otherwise, Vanilla, False (default value).

    obsdates : pandas.DatetimeIndex or list
        Observation dates. If ``pastmatters = False`` we define maturity as the last element of observation dates.

    func_asset : function object
        Function for several objects (Function :math:`\phi_\mathcal{L}` defined in JupyterNotebook: Structured Products
        Documentation.) By default, numpy.min.

    Examples
    --------
    >>> def func(s):
    >>>     return numpy.where(s < -100, s, numpy.where(s < 0, -s, numpy.where(s < 100, 1.3*s, 130)))
    Note
    --------
    For func any piecewise function can be constructed using (nested) numpy.where's. For instance, we give the payoff of
    "Trigger Dual Directional/Twin Win Note" defined in DLIB documentation of Bloomberg.
    """

    def __init__(
        self,
        underlying,
        obsdates,
        func,
        calendar,
        pastmatters=False,
        func_asset=np.min,
        nominal=100,
        credit_curve=zero_credit_risk,
        implied_volatility=None,
        monotonicity_price_function=None,
    ):
        self.nominal = nominal
        obsdates = pd.to_datetime(obsdates).sort_values()
        self.func = func
        self.func_asset = func_asset
        Derivative.__init__(
            self,
            underlying=underlying,
            obsdates=obsdates,
            pastmatters=pastmatters,
            calendar=calendar,
            credit_curve=credit_curve,
        )
        self.implied_volatility = implied_volatility
        self.monotonicity_price_function = monotonicity_price_function

    def payoff(self, prices, dates_dic, n):
        payoffs_matrix = np.zeros(
            (prices.shape[0], prices.shape[1] - n, prices.shape[2])
        )
        index = dates_dic[self.maturity]
        if self.pastmatters:
            indices = [dates_dic[date] for date in self.obsdates]
            s = prices[:, indices]
        else:
            s = prices[:, index]
        payoff_total = self.func_asset(self.func(s), axis=-1)
        payoffs_matrix[:, index - n] = (
            payoff_total  # There is only one payment (at maturity).
        )
        return payoffs_matrix


class ZCBond(Structured):
    # Zero coupon bonds are typically included in structured products
    def __init__(self, maturity, calendar, nominal=100, credit_curve=zero_credit_risk):
        """
        date format: %Y%m%d
        """
        self.nominal = nominal
        self.maturity = pd.to_datetime(maturity)
        self.calendar = calendar
        obsdates = pd.to_datetime([maturity])

        Structured.__init__(
            self, obsdates=obsdates, pastmatters=False, credit_curve=credit_curve
        )

    def payoff(self, prices, dates_dic, n):
        payoffs_matrix = np.zeros(
            (prices.shape[0], prices.shape[1] - n, prices.shape[2])
        )
        payoffs_matrix[:, -1] = np.ones((prices.shape[0], prices.shape[2]))
        return self.nominal * payoffs_matrix

    def get_px(self, dates, discount_curve, no_sims=None, no_calcs=None):
        """
        Calculates the prices of Zero Coupon Bond based on the Black-Scholes formula.

        Parameters
        ----------
        dates : pandas.DatetimeIndex, list of strings, pandas.Timestamp, or string
            Valuation dates. It can be a single date (as a pandas.Timestamp or its string representation) or an array-like object (as a pandas.DatetimeIndex or a list of its string
            representation) containing dates.
        discount_curve : pricing.discount_curves.DiscountCurve
            Gives the discounts for the computations.
        no_sims : int, optional
            Ignored in this implementation (defaults to None).
        no_calcs : int, optional
            Ignored in this implementation (defaults to None).

        Returns
        ----------
        pandas.Series
            The prices of the options.
        """
        fd = self.maturity
        dates = afsfun.dates_formatting(dates)
        if np.max(dates > fd):
            missing = dates[dates > fd]
            dates = dates[dates <= fd]
            print("Eliminating valuation dates past future date:")
            for date in missing:
                print(date.strftime("%Y-%m-%d"))
        value = self.nominal * discount_curve.get_value(
            dates=dates, future_dates=self.maturity, calendar=self.calendar
        )
        prices = pd.Series(value, index=dates)
        return prices

    def get_delta(
        self, dates, discount_curve, no_sims=None, no_calcs=None, monte_carlo=None
    ):
        """
        Calculate the delta of a Zero Coupon Bond. Delta is the derivative of the latter's value w.r.t. the underlying.
        It is consistently set to zero.

        Parameters
        ----------
        dates : pandas.DatetimeIndex, list of strings, pandas.Timestamp, or string
            Dates at which delta is calculated. It can be a single date (as a pandas.Timestamp or its string representation) or an array-like object (as a pandas.DatetimeIndex or a
            list of its string representation) containing dates.
        discount_curve : pricing.discount_curves.DiscountCurve
            Provides the discounts for the computation of the Greek. Not used in this method.
        no_sims: int, optional
            Ignored in this implementation (defaults to None).
        no_calcs : int, optional
            Ignored in this implementation (defaults to None).
        monte_carlo: bool, optional
            Ignored in this implementation (defaults to None).

        Returns
        -------
        pandas.Series
            Returns zero for the given dates.
        """
        dates = afsfun.dates_formatting(dates)
        value = np.zeros(dates.values.size)
        delta = pd.Series(value, index=dates)
        return delta

    def get_gamma(
        self, dates, discount_curve, no_sims=None, no_calcs=None, monte_carlo=None
    ):
        """
        Calculate the gamma of a Zero Coupon Bond. Gamma is the second derivative of the latter's value w.r.t. the underlying.
        It is consistently set to zero.

        Parameters
        ----------
        dates : pandas.DatetimeIndex, list of strings, pandas.Timestamp, or string
            Dates at which gamma is calculated. It can be a single date (as a pandas.Timestamp or its string representation) or an array-like object (as a pandas.DatetimeIndex or a
            list of its string representation) containing dates.
        discount_curve : pricing.discount_curves.DiscountCurve
            Provides the discounts for the computation of the Greek. Not used in this method.
        no_sims: int, optional
            Ignored in this implementation (defaults to None).
        no_calcs : int, optional
            Ignored in this implementation (defaults to None).
        monte_carlo: bool, optional
            Ignored in this implementation (defaults to None).

        Returns
        -------
        pandas.Series
            Returns zero for the given dates.
        """

        dates = afsfun.dates_formatting(dates)
        value = np.zeros(dates.values.size)
        gamma = pd.Series(value, index=dates)
        return gamma

    def get_vega(
        self, dates, discount_curve, no_sims=None, no_calcs=None, monte_carlo=None
    ):
        """
        Calculate the vega of a Zero Coupon Bond. Vega is the derivative of the latter's value w.r.t. the underlying volatility.

        Parameters
        ----------
        dates : pandas.DatetimeIndex, list of strings, pandas.Timestamp, or string
            Dates at which vega is calculated. It can be a single date (as a pandas.Timestamp or its string representation) or an array-like object (as a pandas.DatetimeIndex or a
            list of its string representation) containing dates.
        discount_curve : pricing.discount_curves.DiscountCurve
            Provides the discounts for the computation of the Greek. Not used in this method.
        no_sims: int, optional
            Ignored in this implementation (defaults to None).
        no_calcs : int, optional
            Ignored in this implementation (defaults to None).
        monte_carlo: bool, optional
            Ignored in this implementation (defaults to None).

        Returns
        -------
        pandas.Series
            Returns zero for the given dates.
        """

        dates = afsfun.dates_formatting(dates)
        value = np.zeros(dates.values.size)
        vega = pd.Series(value, index=dates)
        return vega

    def get_rho(
        self, dates, discount_curve, no_sims=10**6, no_calcs=10, monte_carlo=False
    ):
        """
        Calculate the `discrete` rho of a Zero Coupon Bond. Rho is the derivative of the latter's value w.r.t. the interest rate.

        Parameters
        ----------
        dates : pandas.DatetimeIndex, list of strings, pandas.Timestamp, or string
            Dates at which rho is calculated. It can be a single date (as a pandas.Timestamp or its string representation) or an array-like object (as a pandas.DatetimeIndex or a
            list of its string representation) containing dates.
        discount_curve : pricing.discount_curves.DiscountCurve
            Provides the discounts for the computation of the Greek.
        no_sims: int, optional
            Number of simulations (defaults to 10**6).
        no_calcs : int, optional
            The Greek is computed ``no_calcs`` times and then the mean value is returned (defaults to 10).
            With this we can effectively prevent memory errors ("MemoryError: Unable to allocate...") in Python when increasing the number of simulations.
        monte_carlo: bool, optional
            If ``True``, uses Monte Carlo simulations for computation, even if analytical formulas are available (defaults to False).

        Returns
        -------
        pandas.Series
            Returns the rho for the given dates.
        """
        dates = afsfun.dates_formatting(dates)
        central_rho_est = pd.Series(data=0, index=dates)
        if (
            getattr(type(self), "get_px") != getattr(MCProduct, "get_px")
            and not monte_carlo
        ):
            discount_curve.p = -1
            prices_plus = self.get_px(dates=dates, discount_curve=discount_curve)
            discount_curve.p = 1
            prices_minus = self.get_px(dates=dates, discount_curve=discount_curve)
            central_rho_est = (prices_plus - prices_minus) / (2 * 0.0005)
            discount_curve.p = 0  # Restores the initial values

        else:
            engine = DiffusionMC()
            for i in range(no_calcs):
                discount_curve.p = -1
                prices_plus = engine.price(
                    self, dates, discount_curve, no_sims=no_sims
                )  # We use engine as we need to mantain the draws, as explained below
                discount_curve.p = 1
                prices_minus = engine.price(
                    self, dates, discount_curve, no_sims=no_sims, reuse_draws=True
                )
                engine.draws = None  # Clear the draws, used the same for both computations. See [Andersen and Piterbarg], , p.134-135.
                central_rho_est += (prices_plus - prices_minus) / (2 * 0.0005)
            discount_curve.p = 0  # Restores the initial values
            central_rho_est /= no_calcs

        return central_rho_est

    def get_theta(
        self,
        dates,
        discount_curve,
        no_sims=None,
        no_calcs=None,
        maturity_method=None,
        monte_carlo=None,
        constant_curve=False,
        refit=None,
    ):
        """
        Calculate the analytical theta of a Zero Coupon Bond. Theta is the derivative of the latter's value w.r.t. time.

        Parameters
        ----------
        dates : pandas.DatetimeIndex, list of strings, pandas.Timestamp, or string
            Dates at which theta is calculated. It can be a single date (as a pandas.Timestamp or its string representation) or an array-like object (as a pandas.DatetimeIndex or a
            list of its string representation) containing dates.
        discount_curve :  pricing.discount_curves.DiscountCurve
            Provides the discounts for the computation of the Greek.
        no_sims: int, optional
            Ignored in this implementation (default to None).
        no_calcs : int, optional
            Ignored in this implementation (default to None).
        maturity_method : bool, optional
            Ignored in this implementation (defaults to None).
        monte_carlo: bool, optional
            Ignored in this implementation (defaults to None).
        constant_curve : bool, optional
            If True, keeps the discount curve constant while computing the theta (defaults to False).
        refit : bool, optional
            Ignored in this implementation (defaults to None).

        Returns
        -------
        pandas.Series
            Returns the theta for the given dates. It is a pandas.Series where the index corresponds to the input dates and the values are the computed thetas.

        See Also
        --------
        ir_models.py
            See this module for stochastic interest rate models.
        """
        dates = afsfun.dates_formatting(dates)
        if constant_curve:
            r = -np.log(
                discount_curve.get_value(dates, self.maturity)
            ) / self.calendar.interval(dates, self.maturity)
        else:
            dates_p = pd.to_datetime(dates) + pd.tseries.offsets.BDay()
            dates_m = pd.to_datetime(dates) - pd.tseries.offsets.BDay()
            time_int_mat = self.calendar.interval(dates_m, dates_p)
            discount_p = discount_curve.get_value(dates_p, self.maturity)
            discount_m = discount_curve.get_value(dates_m, self.maturity)
            deriv_df = (discount_p - discount_m) / time_int_mat
            r = deriv_df / discount_curve.get_value(dates, self.maturity)
        theta = self.nominal * r * discount_curve.get_value(dates, self.maturity)
        theta = pd.Series(data=theta, index=dates)
        return theta

    def compute_equity_tolerances(self, dates, discount_curve=None):
        """
        Construct a table for tolerances of the zero coupon bond at given dates filled with zeros,
        because changes in equity volatility and dividends does not affect zero coupon bonds prices.

        Parameters
        ----------
        dates : pandas.DatetimeIndex, list of strings, pandas.Timestamp, or string
            Dates at which to compute the tolerances. It can be a single date (as a pandas.Timestamp or its string representation) or an array-like object
            (as a pandas.DatetimeIndex or a list of its string representation) containing dates.
        discount_curve : pricing.discount_curves.DiscountCurve
            Gives the discounts for the computations.

        discount_curve : pricing.discount_curves.DiscountCurve, optional
            Ignored in this implementation (defaults to None).

        Returns
        -------
        pandas.Dataframe
            The table with tolerances. The index of this data frame is ``dates``, while the different kind of
            positive and negative bumps (identically zero) are the columns.
        """

        dates = afsfun.dates_formatting(dates)
        vol_header = pd.MultiIndex.from_product(
            [["EQ Volatilidad"], ["-200bp", "+200bp"]]
        )
        div_header = pd.MultiIndex.from_product(
            [["EQ Dividendos"], ["-100bp", "+100bp"]]
        )
        eq_headers = pd.MultiIndex.union(vol_header, div_header, sort=False)
        table = pd.DataFrame(index=eq_headers, columns=dates).transpose()

        table["EQ Dividendos", "-100bp"] = float(0)
        table["EQ Dividendos", "+100bp"] = float(0)
        table["EQ Volatilidad", "-200bp"] = float(0)
        table["EQ Volatilidad", "+200bp"] = float(0)

        return table


class Forward(Derivative):
    """
    date format: %Y%m%d
    """

    def __init__(
        self,
        underlying,
        maturity,
        strike,
        calendar,
        nominal=100,
        credit_curve=zero_credit_risk,
    ):
        self.strike = strike
        self.nominal = nominal

        Derivative.__init__(
            self,
            underlying=underlying,
            obsdates=[maturity],
            pastmatters=False,
            calendar=calendar,
            credit_curve=credit_curve,
        )

    def payoff(self, prices, dates_dic, n):
        payoffs_matrix = np.zeros(
            (prices.shape[0], prices.shape[1] - n, prices.shape[2])
        )
        index = dates_dic[self.maturity]
        prices = prices[:, index]  # We take the subarray with the dates at maturity.
        if self.strike:
            payoffs = prices - self.strike
        else:
            payoffs = prices
        payoffs_matrix[:, index - n] = np.min(
            payoffs, axis=-1
        )  # When the price depends on several underlyings it is defined as the minimum over them.
        return self.nominal * payoffs_matrix

    def get_px(self, dates, discount_curve, no_sims=None, no_calcs=None):
        dates = afsfun.dates_formatting(dates)
        px = self.get_underlying_fpx(dates=dates, discount_curve=discount_curve)
        dates = px.index
        discount = discount_curve.get_value(
            dates=dates, future_dates=self.maturity, calendar=self.calendar
        )
        credit = self.credit_curve.get_value(
            dates=dates, future_dates=self.maturity, calendar=self.calendar
        )
        return discount * credit * (px - self.strike) * self.nominal


# ------------------------------------------------------------------------------------
# Vanilla
# ------------------------------------------------------------------------------------


class Vanilla(Derivative, MCProduct):
    """
    This class precomputes everything that is common for puts and calls. EXCEPT FOR IMPVOL which uses .set_, which
    requires [[pandas.datetime],[values]].

    This class is inherited from classes Derivative and MCProduct.

    Parameters
    ----------
    underlying : str
        The underlying asset's symbol.
    strike : float
        The strike price of the  option.
    maturity : pandas.Timestamp
        The expiration date of the option.
    kind : str
        The type of option, either 'call' or 'put'.
    calendar : data.calendars.DayCountCalendar
        Calendar convention for counting days.
    nominal : float, optional
        Number of underlying shares for each call option contract (default is 100.0).
    credit_curve : pricing.discount_curves.CRDC
        The credit spread curve (default is zero_credit_risk = CRDC(0), namely there is no credit spread).
    alpha : float, optional
        Quantity added to the strike price (default is 0).
    phi : str, optional
        If phi = "min" or None, self.phi is set to numpy.min; if phi = "max", self.phi is set to numpy.max.
        Otherwise it raises a ValueError "Value of self.phi must be 'min' or 'max'".
        This attribute is used when the price depends on several underlyings (default is None).
    implied_volatility : float, optional
        If None, the method get_vol is exploited to obtain the value of implied volatility from data.
        Otherwise, the input value is accepted as implied volatility for the option (default is None).
    monotonicity_price_function : string, optional
        Monotonicity of the price function of the vanilla option. It can be `increasing`, `decreasing` \
        or, for non definite monotonicity, None. This attribute is beneficial for calculating risk metrics, such as VaR. \
        Default is None.

    Notes
    ----------
    - date format: %Y%m%d
    - prices are in percentage of nominal
    """

    def __init__(
        self,
        underlying,
        strike,
        maturity,
        kind,
        calendar,
        nominal=100.0,
        credit_curve=zero_credit_risk,
        alpha=0,
        phi=None,
        implied_volatility=None,
        monotonicity_price_function=None,
    ):
        self.strike = np.array(strike)
        self.nominal = nominal
        maturity = pd.to_datetime(maturity)
        Derivative.__init__(
            self,
            underlying=underlying,
            obsdates=[maturity],
            pastmatters=False,
            calendar=calendar,
            credit_curve=credit_curve,
            monotonicity_price_function=monotonicity_price_function,
        )

        self.kind = kind
        self.N = np.vectorize(norm.cdf)
        self.f = np.vectorize(norm.pdf)
        self.alpha = alpha
        self.implied_volatility = implied_volatility
        if phi is None or phi == "min":
            self.phi = np.min
        elif phi == "max":
            self.phi = np.max
        else:
            raise ValueError("Value of self.phi must be 'min' or 'max'")

    def payoff(self, prices, dates_dic, n):
        payoffs_matrix = np.zeros(
            (prices.shape[0], prices.shape[1] - n, prices.shape[2])
        )
        index = dates_dic[self.maturity]
        prices = prices[:, index]
        if prices.ndim == 3:
            prices = self.phi(
                prices, axis=-1, keepdims=False
            )  # When the price depends on several underlyings it is defined as the minimum over them.
        if self.strike.all():  # .all() for MultiAsset
            payoffs = (prices - self.strike) * (prices >= self.strike) - (
                prices - self.strike
            ) * (self.kind == "put")
        else:  # TODO: If one is zero but not the other we should have to use different formulas.
            payoffs = prices * (prices >= self.strike) - prices * (self.kind == "put")
        payoffs_matrix[:, index - n] = payoffs
        return self.nominal * payoffs_matrix

    def get_px(self, dates, discount_curve, no_sims=None, no_calcs=None):
        # TODO: Add parameters no_sims and no_calcs in docstring. They are added here to standardize the get_px method.
        """
        Compute the price of an option.

        Parameters
        ----------
        dates : pandas.DatetimeIndex, list of strings, pandas.Timestamp, or string
            Dates at which forward price is computed. It can be a single date (as a pandas.Timestamp or its string representation) or an array-like object (as a
            pandas.DatetimeIndex or a list of its string representation) containing dates.
        discount_curve : pricing.discount_curves.DiscountCurve
            Gives the discounts for the computations.
        no_sims : int, optional
            Ignored in this implementation (defaults to None).
        no_calcs : int, optional
            Ignored in this implementation (defaults to None).

        Returns
        -------
        pandas.Series
            The function's output(a tuple) varies depending on the type of asset:
                - For LognormalAsset, the function returns a tuple containing pandas.Series of forward prices, d1, d2, discount factors, and volatilities.
                - For NormalAsset, the function returns a tuple containing pandas.Series of zero rates, volatilities, d values and discount factors.
        """

        alpha = self.alpha
        # check if data exists for all dates, so that it doesn't give error multiple times
        if isinstance(self.underlying, LognormalAsset):
            # among other things, using fpx and not px allows us to use this method for caps and swaptions
            # (no need to worry about r as it does not show up)
            fpx = alpha + self.get_underlying_fpx(
                dates=dates, discount_curve=discount_curve
            )
            dates = fpx.index
            tau = self.calendar.interval(dates=dates, future_dates=self.maturity)
            if self.implied_volatility is None:
                vol = self.underlying.get_vol(dates=dates)
            else:
                vol = self.implied_volatility
            d1 = (
                np.log((fpx + alpha) / (self.strike + alpha)) + 0.5 * (vol**2) * tau
            ) / (vol * np.sqrt(tau))
            d2 = d1 - vol * np.sqrt(tau)

            discount = discount_curve.get_value(
                dates=dates, future_dates=self.maturity, calendar=self.calendar
            )
            credit = self.credit_curve.get_value(
                dates=dates, future_dates=self.maturity, calendar=self.calendar
            )
            discount *= credit
            return fpx, d1, d2, discount, vol
        elif isinstance(self.underlying, NormalAsset) and not self.underlying.yieldsdiv:
            # Normal is only used for forward and swap rates, so we use the Bachelier formula for zero interest rates
            fpx = self.get_underlying_fpx(dates=dates, discount_curve=discount_curve)
            tau = self.calendar.interval(dates=dates, future_dates=self.maturity)
            if self.implied_volatility is None:
                vol = self.underlying.get_vol(
                    dates=dates, tenors=tau, strike=self.strike
                )
            else:
                vol = self.implied_volatility
            d = (fpx - self.strike) / (vol * np.sqrt(tau))
            discount = discount_curve.get_value(
                dates=dates, future_dates=self.maturity, calendar=self.calendar
            )
            credit = self.credit_curve.get_value(
                dates=dates, future_dates=self.maturity, calendar=self.calendar
            )
            discount *= credit
            return fpx, vol * np.sqrt(tau), d, discount
        else:
            raise TypeError("Underlying dynamics not supported")

    def get_gamma(self, dates, discount_curve):
        """
        Calculate the gamma of an option. Gamma is the second derivative
        of the option's price with respect to the underlying asset's price.

        Parameters
        ----------
        dates : pandas.DatetimeIndex, list of strings, pandas.Timestamp, or string
            Dates at which gamma is calculated. It can be a single date (as a pandas.Timestamp or its string representation) or an array-like object (as a pandas.DatetimeIndex or a
            list of its string representation) containing dates.
        discount_curve : pricing.discount_curves.DiscountCurve
            Provides the discounts for the computation of the gamma.

        Returns
        -------
        pandas.Series
            Returns gamma for the given dates if the underlying asset is of type LognormalAsset.
            For a basket of assets or other unsupported types, a message is printed and ``None`` is returned.

        Notes
        -----
        This method is not applicable for a basket of assets; in such cases, Monte Carlo methods
        are suggested. If the underlying asset is not of type LognormalAsset, the method also
        returns ``None``, indicating unsupported asset dynamics.
        """
        if type(self.underlying) is list:
            print("Method not available for baskets; try Monte Carlo instead.")
            return None
        if isinstance(self.underlying, LognormalAsset):
            fpx, d1, d2, discount, vol = Vanilla.get_px(
                self, dates=dates, discount_curve=discount_curve
            )
            dates = fpx.index
            tau = self.calendar.interval(dates=dates, future_dates=self.maturity)
            q = self.underlying.get_divrate(dates=dates).values
            px = self.underlying.get_value(dates=dates)
            # needs to be adjusted to yield percentage of nominal, like price
            # Vanilla.get_px already adjusts data by alpha
            gamma = np.exp(-q * tau) * self.f(d1) / (px * vol * np.sqrt(tau))
            return self.nominal * gamma
        else:
            raise TypeError("Underlying dynamics not supported")

    def get_vega(self, dates, discount_curve):
        """
        Calculate the vega of an option. Vega is the derivative of the option value with respect to the volatility of the underlying asset.

        Parameters
        ----------
        dates : pandas.DatetimeIndex, list of strings, pandas.Timestamp, or string
            Dates at which vega is calculated. It can be a single date (as a pandas.Timestamp or its string representation) or an array-like object (as a pandas.DatetimeIndex or a
            list of its string representation) containing dates.
        discount_curve : pricing.discount_curves.DiscountCurve
            Provides the discounts for the computation of the vega.

        Returns
        -------
        pandas.Series
            Returns vega for the given dates if the underlying asset is of type LognormalAsset. For a basket of assets or other unsupported types, a message is printed and ``None``
            is returned.

        Notes
        -----
        This method is not applicable for a basket of assets; in such cases, Monte Carlo methods
        are suggested. If the underlying asset is not of type LognormalAsset, the method ``also``
        returns None, indicating unsupported asset dynamics.
        """
        if type(self.underlying) is list:
            print("Method not available for baskets; try Monte Carlo instead.")
            return None
        if isinstance(self.underlying, LognormalAsset):
            fpx, d1, d2, discount, vol = Vanilla.get_px(self, dates, discount_curve)
            dates = fpx.index
            px = self.underlying.get_value(dates=dates)
            tau = self.calendar.interval(dates=dates, future_dates=self.maturity)
            q = self.underlying.get_divrate(dates=dates).values
            # Vanilla.get_px already adjusts data by alpha
            vega = px * np.sqrt(tau) * np.exp(-q * tau) * self.f(d1)
            return self.nominal * vega
        else:
            raise TypeError("Underlying dynamics not supported")

    def fit_alpha(self, date, discount_curve, px):
        if type(self.underlying) is list:
            print("Method not available for baskets; try Monte Carlo instead.")
            return None

        def difference(a):
            self.alpha = a
            diff = px - self.get_px(dates=date, discount_curve=discount_curve).values[0]
            return diff

        self.alpha = optimize.newton(difference, 0.001)

    def get_vanna(self, dates, discount_curve):
        """
        Calculate the vanna of an option. Vanna is the second order derivative of the option value, once to the underlying spot price and once to volatility.

        Parameters
        ----------
        dates : pandas.DatetimeIndex, list of strings, pandas.Timestamp, or string
            Dates at which vanna is calculated. It can be a single date (as a pandas.Timestamp or its string representation) or an array-like object (as a pandas.DatetimeIndex or a
            list of its string representation) containing dates.
        discount_curve : pricing.discount_curves.DiscountCurve
            Provides the discounts for the computation of the vanna.

        Returns
        -------
        pandas.Series
            Returns vanna for the given dates if the underlying asset is of type LognormalAsset. For a basket of assets or other unsupported types, a message is printed and
            ``None`` is returned.

        Notes
        -----
        This method is not applicable for a basket of assets; in such cases, Monte Carlo methods are suggested. If the underlying asset is not of type LognormalAsset, the method also
        returns None, indicating unsupported asset dynamics.
        """
        if type(self.underlying) is list:
            print("Method not available for baskets; try Monte Carlo instead.")
            return None
        if isinstance(self.underlying, LognormalAsset):
            fpx, d1, d2, discount, vol = Vanilla.get_px(
                self, dates=dates, discount_curve=discount_curve
            )
            tau = self.calendar.interval(dates=dates, future_dates=self.maturity)
            q = self.underlying.get_divrate(dates=dates)
            # Vanilla.get_px already adjusts data by alpha
            # vanna = Vanilla.get_vega(self, dates, discount_curve) /  px * (1 - d1/(vol * np.sqrt(tau)))
            vanna = -np.exp(-q * tau) * self.f(d1) * d2 / vol
            return self.nominal * vanna
        else:
            raise TypeError("Underlying dynamics not supported")

    def get_vomma(self, dates, discount_curve):
        """
        Calculate the vomma of an option. Vomma is the second order sensitivity to volatility.

        Parameters
        ----------
        dates : pandas.DatetimeIndex, list of strings, pandas.Timestamp, or string
            Dates at which vomma is calculated. It can be a single date (as a pandas.Timestamp or its string representation) or an array-like object (as a pandas.DatetimeIndex or a
            list of its string representation) containing dates.
        discount_curve : pricing.discount_curves.DiscountCurve
            Provides the discounts for the computation of the vomma.

        Returns
        -------
        pandas.Series
            Returns vomma for the given dates if the underlying asset is of type LognormalAsset. For a basket of assets or other unsupported types, a message is printed and
            ``None`` is returned.

        Notes
        -----
        This method is not applicable for a basket of assets; in such cases, Monte Carlo methods are suggested. If the underlying asset is not of type LognormalAsset, the method also
        returns ``None``, indicating unsupported asset dynamics.
        """

        if type(self.underlying) is list:
            print("Method not available for baskets; try Monte Carlo instead.")
            return None
        if isinstance(self.underlying, LognormalAsset):
            fpx, d1, d2, discount, vol = Vanilla.get_px(
                self, dates=dates, discount_curve=discount_curve
            )
            dates = fpx.index
            # Vanilla.get_px already adjusts data by alpha
            # vomma = (fpx / (self.strike + alpha)) * self.f(d1) * np.sqrt(tau) * d1 * d2 / vol
            vomma = (
                Vanilla.get_vega(self, dates=dates, discount_curve=discount_curve)
                / self.nominal
                * d1
                * d2
                / vol
            )
            return self.nominal * vomma
        else:
            raise TypeError("Underlying dynamics not supported")

    def get_veta(self, dates, discount_curve):
        """
        Calculate the veta of an option. Veta is the second derivative of the value function; once to volatility and once to time.

        Parameters
        ----------
        dates : pandas.DatetimeIndex, list of strings, pandas.Timestamp, or string
            Dates at which veta is calculated. It can be a single date (as a pandas.Timestamp or its string representation) or an array-like object (as a pandas.DatetimeIndex or a
            list of its string representation) containing dates.
        discount_curve : pricing.discount_curves.DiscountCurve
            Provides the discounts for the computation of the veta.

        Returns
        -------
        pandas.Series
            Returns veta for the given dates if the underlying asset is of type LognormalAsset. For a basket of assets or other unsupported types, a message is printed
            and ``None`` is returned.

        Notes
        -----
        This method is not applicable for a basket of assets; in such cases, Monte Carlo methods
        are suggested. If the underlying asset is not of type LognormalAsset, the method also
        returns ``None``, indicating unsupported asset dynamics.
        """

        if type(self.underlying) is list:
            print("Method not available for baskets; try Monte Carlo instead.")
            return None
        if isinstance(self.underlying, LognormalAsset):
            fpx, d1, d2, discount, vol = Vanilla.get_px(
                self, dates=dates, discount_curve=discount_curve
            )
            dates = fpx.index
            tau = self.calendar.interval(dates=dates, future_dates=self.maturity)
            px = self.underlying.get_value(dates=dates)
            q = self.underlying.get_divrate(dates=dates)
            r = -np.log(discount) / tau
            # Vanilla.get_px already adjusts data by alpha
            veta = (
                -px
                * np.exp(-q * tau)
                * self.f(d1)
                * np.sqrt(tau)
                * (q + (r - q) * d1 / (vol * np.sqrt(tau)) - (1 + d1 * d2) / (2 * tau))
            )
            return self.nominal * veta
        else:
            raise TypeError("Underlying dynamics not supported")

    def get_speed(self, dates, discount_curve):
        """
        Calculate the speed of an option. Speed is the third derivative of the value function with respect to the underlying spot price.

        Parameters
        ----------
        dates : pandas.DatetimeIndex, list of strings, pandas.Timestamp, or string
            Dates at which speed is calculated. It can be a single date (as a pandas.Timestamp or its string representation) or an array-like object (as a pandas.DatetimeIndex or a
            list of its string representation) containing dates.
        discount_curve : pricing.discount_curves.DiscountCurve
            Provides the discounts for the computation of the speed.

        Returns
        -------
        pandas.Series
            Returns speed for the given dates if the underlying asset is of type LognormalAsset. For a basket of assets or other unsupported types, a message is printed
            and ``None`` is returned.

        Notes
        -----
        This method is not applicable for a basket of assets; in such cases, Monte Carlo methods
        are suggested. If the underlying asset is not of type LognormalAsset, the method also
        returns ``None``, indicating unsupported asset dynamics.
        """

        if type(self.underlying) is list:
            print("Method not available for baskets; try Monte Carlo instead.")
            return None
        if isinstance(self.underlying, LognormalAsset):
            fpx, d1, d2, discount, vol = Vanilla.get_px(
                self, dates=dates, discount_curve=discount_curve
            )
            dates = fpx.index
            tau = self.calendar.interval(dates=dates, future_dates=self.maturity)
            px = self.underlying.get_value(dates=dates)
            # Vanilla.get_px already adjusts data by alpha
            speed = (
                -Vanilla.get_gamma(self, dates=dates, discount_curve=discount_curve)
                / self.nominal
                / px
                * (d1 / (vol * np.sqrt(tau)) + 1)
            )
            return self.nominal * speed
        else:
            raise TypeError("Underlying dynamics not supported")

    def get_zomma(self, dates, discount_curve):
        """
        Calculate the zomma of an option. Zomma is the third derivative of the option value, twice to underlying asset price and once to volatility.

        Parameters
        ----------
        dates : pandas.DatetimeIndex, list of strings, pandas.Timestamp, or string
            Dates at which zomma is calculated. It can be a single date (as a pandas.Timestamp or its string representation) or an array-like object (as a pandas.DatetimeIndex or a
            list of its string representation) containing dates.
        discount_curve : pricing.discount_curves.DiscountCurve
            Provides the discounts for the computation of the zomma.

        Returns
        -------
        pandas.Series
            Returns zomma for the given dates if the underlying asset is of type LognormalAsset. For a basket of assets or other unsupported types, a message is printed
            and ``None`` is returned.

        Notes
        -----
        This method is not applicable for a basket of assets; in such cases, Monte Carlo methods
        are suggested. If the underlying asset is not of type LognormalAsset, the method also
        returns ``None``, indicating unsupported asset dynamics.
        """

        if type(self.underlying) is list:
            print("Method not available for baskets; try Monte Carlo instead.")
            return None
        if isinstance(self.underlying, LognormalAsset):
            fpx, d1, d2, discount, vol = Vanilla.get_px(
                self, dates=dates, discount_curve=discount_curve
            )
            dates = fpx.index
            zomma = (
                Vanilla.get_gamma(self, dates=dates, discount_curve=discount_curve)
                / self.nominal
                * (d1 * d2 - 1)
                / vol
            )
            return self.nominal * zomma
        else:
            raise TypeError("Underlying dynamics not supported")

    def get_color(self, dates, discount_curve):
        """
        Calculate the color of an option. Color is a third-order derivative of the option value, twice to underlying asset price and once to time.

        Parameters
        ----------
        dates : pandas.DatetimeIndex, list of strings, pandas.Timestamp, or string
            Dates at which color is calculated. It can be a single date (as a pandas.Timestamp or its string representation) or an array-like object (as a pandas.DatetimeIndex or a
            list of its string representation) containing dates.
        discount_curve : pricing.discount_curves.DiscountCurve
            Provides the discounts for the computation of the color.

        Returns
        -------
        pandas.Series
            Returns color for the given dates if the underlying asset is of type LognormalAsset. For a basket of assets or other unsupported types, a message is printed
            and ``None`` is returned.

        Notes
        -----
        This method is not applicable for a basket of assets; in such cases, Monte Carlo methods
        are suggested. If the underlying asset is not of type LognormalAsset, the method also
        returns ``None``, indicating unsupported asset dynamics.
        """
        if type(self.underlying) is list:
            print("Method not available for baskets; try Monte Carlo instead.")
            return None
        if isinstance(self.underlying, LognormalAsset):
            fpx, d1, d2, discount, vol = Vanilla.get_px(
                self, dates=dates, discount_curve=discount_curve
            )
            dates = fpx.index
            tau = self.calendar.interval(dates=dates, future_dates=self.maturity)
            px = self.underlying.get_value(dates=dates)
            r = -np.log(discount) / tau
            q = self.underlying.get_divrate(dates=dates)
            # Vanilla.get_px already adjusts data by alpha
            color = -(
                -np.exp(-q * tau)
                * self.f(d1)
                / (2 * px * tau * vol * np.sqrt(tau))
                * (
                    2 * q * tau
                    + 1
                    + (2 * (r - q) * tau - d2 * vol * np.sqrt(tau))
                    * d1
                    / (vol * np.sqrt(tau))
                )
            )
            return self.nominal * color
        else:
            raise TypeError("Underlying dynamics not supported")

    def get_ultima(self, dates, discount_curve):
        """
        Calculate the ultima of an option. Ultima is a third-order derivative of the option value to volatility.

        Parameters
        ----------
        dates : pandas.DatetimeIndex, list of strings, pandas.Timestamp, or string
            Dates at which ultima is calculated. It can be a single date (as a pandas.Timestamp or its string representation) or an array-like object (as a pandas.DatetimeIndex or a
            list of its string representation) containing dates.
        discount_curve : pricing.discount_curves.DiscountCurve
            Provides the discounts for the computation of the ultima.

        Returns
        -------
        pandas.Series
            Returns ultima for the given dates if the underlying asset is of type LognormalAsset. For a basket of assets or other unsupported types, a message is printed
            and ``None`` is returned.

        Notes
        -----
        This method is not applicable for a basket of assets; in such cases, Monte Carlo methods
        are suggested. If the underlying asset is not of type LognormalAsset, the method also
        returns ``None``, indicating unsupported asset dynamics.
        """
        if type(self.underlying) is list:
            print("Method not available for baskets; try Monte Carlo instead.")
            return None
        if isinstance(self.underlying, LognormalAsset):
            fpx, d1, d2, discount, vol = Vanilla.get_px(
                self, dates=dates, discount_curve=discount_curve
            )
            dates = fpx.index
            ultima = (
                -Vanilla.get_vega(self, dates=dates, discount_curve=discount_curve)
                / self.nominal
                / (vol**2)
                * (d1 * d2 * (1 - d1 * d2) + d1**2 + d2**2)
            )
            return self.nominal * ultima
        else:
            raise TypeError("Underlying dynamics not supported")

    def get_dual_gamma(self, dates, discount_curve):
        """
        Calculate the dual gamma of an option. Dual gamma is a second-order derivative of the option value to strike.

        Parameters
        ----------
        dates : pandas.DatetimeIndex, list of strings, pandas.Timestamp, or string
            Dates at which dual-gamma is calculated. It can be a single date (as a pandas.Timestamp or its string representation) or an array-like object (as a pandas.DatetimeIndex or a
            list of its string representation) containing dates.
        discount_curve : pricing.discount_curves.DiscountCurve
            Provides the discounts for the computation of the dual gamma.

        Returns
        -------
        pandas.Series
            Returns ultima for the given dates if the underlying asset is of type LognormalAsset. For a basket of assets or other unsupported types, a message is printed
            and ``None`` is returned.

        Notes
        -----
        This method is not applicable for a basket of assets; in such cases, Monte Carlo methods
        are suggested. If the underlying asset is not of type LognormalAsset, the method also
        returns ``None``, indicating unsupported asset dynamics.
        """
        if type(self.underlying) is list:
            print("Method not available for baskets; try Monte Carlo instead.")
            return None
        if isinstance(self.underlying, LognormalAsset):
            strike = self.strike
            fpx, d1, d2, discount, vol = Vanilla.get_px(
                self, dates=dates, discount_curve=discount_curve
            )
            dates = fpx.index
            tau = self.calendar.interval(dates=dates, future_dates=self.maturity)
            dual_gamma = discount * self.f(d2) / (strike * vol * np.sqrt(tau))
            return self.nominal * dual_gamma
        else:
            raise TypeError("Underlying dynamics not supported")

    def get_greek_table(self, dates, discount_curve):
        """
        Calculate the premium and greeks of an option. Time greeks are given per day,
        and volatility one per percentage point.

        Parameters
        ----------
        dates : pandas.DatetimeIndex, list of strings, pandas.Timestamp, or string
            Dates at which greeks are calculated. It can be a single date (as a pandas.Timestamp or its string representation)
            or an array-like object (as a pandas.DatetimeIndex or a list of its string representation) containing dates.
        discount_curve : pricing.discount_curves.DiscountCurve
            Provides the discounts for the computation of the greeks.

        Returns
        -------
        pandas.Dataframe
            Returns the premium and greeks for the given dates if the underlying asset is of type LognormalAsset.
            For a basket of assets or other unsupported types, a message is printed and ``None`` is returned.

        Notes
        -----
        If the underlying asset is not of type LognormalAsset, the method also
        returns ``None``, indicating unsupported asset dynamics.
        """
        premium = self.get_px(dates, discount_curve)
        delta = self.get_delta(dates, discount_curve)
        gamma = self.get_gamma(dates, discount_curve)
        vega = self.get_vega(dates, discount_curve)
        theta = self.get_theta(dates, discount_curve)
        vanna = self.get_vanna(dates, discount_curve)
        vomma = self.get_vomma(dates, discount_curve)
        charm = self.get_charm(dates, discount_curve)
        rho = self.get_rho(dates, discount_curve)
        veta = self.get_veta(dates, discount_curve)
        speed = self.get_speed(dates, discount_curve)
        zomma = self.get_zomma(dates, discount_curve)
        color = self.get_color(dates, discount_curve)
        ultima = self.get_color(dates, discount_curve)
        nominal = pd.DataFrame(
            data=self.nominal, columns=["nominal"], index=delta.index
        )

        df_greeks = pd.DataFrame(
            data=[],
            columns=[
                "nominal",
                "premium",
                "delta",
                "gamma",
                "theta (day)",
                "vega (%)",
                "rho (%)",
                "volga (%)",
                "vanna (%)",
                "charm (day)",
                "color (day)",
                "speed",
                "veta (day, %)",
                "zomma (%)",
                "ultima (%)",
            ],
        )
        list_columns = [
            nominal,
            premium,
            delta,
            gamma,
            theta / self.calendar.days_in_year,
            vega / 100,
            rho / 100,
            vomma / 10000,
            vanna / 100,
            charm / self.calendar.days_in_year,
            color / self.calendar.days_in_year,
            speed,
            veta / self.calendar.days_in_year / 100,
            zomma / 100,
            ultima / 1000000,
        ]

        for column_name, column_value in zip(df_greeks.columns, list_columns):
            df_greeks[column_name] = column_value

        return df_greeks

    def get_risk_matrix(
        self,
        present_date,
        discount_curve,
        scenarios,
        complete_scenarios=False,
        return_scenarios_greeks=True,
    ):
        """
        Calculate the PnL (total and decomposed per greeks) of an option in different scenarios given a initial one.

        Parameters
        ----------
        present_date : pandas.Timestamp, or string
            Date for the initial scenario. It must be a single date (as a pandas.Timestamp or its string representation).
        discount_curve : pricing.discount_curves.DiscountCurve
            Provides the discounts for the computation of the greeks and prices.
        scenarios : dictionary
            Dictionary containing the different scenarios. Values must be lists or numpy.arrays of the same length (and equal
            to the number of scenarios considering). The ith scenario is defined by the ith element in each value. The keys
            represent the final value in the final scenarios for spot, implied volatility and tenor. It is also possible to
            specify these values by changes instead of final values. We can include dividends as an optional key in the
            scenarios, if so, these values are used in the scenarios, if not, we assume the same interests.
        complete_scenarios : boolean, default False.
            If True it extends the scenarios by making the cartesian product of all the possibilities.

        Returns
        -------
        pandas.Dataframe
            Dataframe containing the total PnL of the option and decomposed by greeks for each scenario.
        dictionary
            Returns in a dictionary the premium and greeks of the strategy in each scenario as returned by the method
            ``get_greek_table``. The key for the ith scenario is "Scenario i"

        Notes
        -----
        The only variables to change in each scenario are the spot, the tenor and the implied volatility (and optionally
        the dividends). Dividend rates are assumed to be constant.
        """
        # Check that all scenarios values are lists or arrays of the same length
        have_same_length = len(set(len(value) for value in scenarios.values())) == 1

        if not have_same_length:
            raise ValueError(
                "All values in scenarios dictionary must be lists or arrays of the same length."
            )

        spot_init = self.underlying.get_value(present_date)
        ivol_init = self.implied_volatility

        if not (("spot" in scenarios.keys()) ^ ("spot_change" in scenarios.keys())):
            raise ValueError(
                "You must provide either final spot or relative change, but not both at the same time"
            )
        if not (("future_date" in scenarios.keys()) ^ ("tenor" in scenarios.keys())):
            raise ValueError(
                "You must provide either final date or tenor, but not both at the same time"
            )
        if not (
            ("implied_vol" in scenarios.keys())
            ^ ("implied_vol_change" in scenarios.keys())
        ):
            raise ValueError(
                "You must provide either final implied volatility or change, but not both at the same time"
            )

        if complete_scenarios:
            combinations = list(
                product(*[np.unique(np.array(x)) for x in scenarios.values()])
            )
            values = [list(t) for t in zip(*combinations)]
            scenarios = {key: value for key, value in zip(scenarios.keys(), values)}

        # Compute scenarios final values from relative changes
        if "spot" in scenarios.keys():
            spots = scenarios["spot"]
        else:
            spots = spot_init * scenarios["spot_change"]

        if "future_date" in scenarios.keys():
            future_dates = [pd.to_datetime(x) for x in scenarios["future_date"]]
        else:
            future_dates = [
                pd.to_datetime(present_date)
                + timedelta(days=int(tau * self.calendar.days_in_year))
                for tau in scenarios["tenor"]
            ]

        if "implied_vol" in scenarios.keys():
            final_implied_volatilities = np.array(scenarios["implied_vol"])
        else:
            final_implied_volatilities = ivol_init + scenarios["implied_vol_change"]

        if "future_dividends" in scenarios.keys():
            final_dividends = np.array(scenarios["future_dividends"])
        else:
            final_dividends = None

        # Compute present Greeks
        df_present_greeks = self.get_greek_table(present_date, discount_curve)
        prem_init = df_present_greeks.iloc[-1]["premium"]
        delta = df_present_greeks.iloc[-1]["delta"]
        gamma = df_present_greeks.iloc[-1]["gamma"]
        theta = df_present_greeks.iloc[-1]["theta (day)"]
        vega = df_present_greeks.iloc[-1]["vega (%)"]
        volga = df_present_greeks.iloc[-1]["volga (%)"]
        vanna = df_present_greeks.iloc[-1]["vanna (%)"]
        charm = df_present_greeks.iloc[-1]["charm (day)"]
        color = df_present_greeks.iloc[-1]["color (day)"]
        speed = df_present_greeks.iloc[-1]["speed"]
        veta = df_present_greeks.iloc[-1]["veta (day, %)"]
        zomma = df_present_greeks.iloc[-1]["zomma (%)"]
        ultima = df_present_greeks.iloc[-1]["ultima (%)"]
        rho = df_present_greeks.iloc[-1]["rho (%)"]

        # Compute scenarios and scenario greeks
        df_risk = pd.DataFrame(
            data=[],
            columns=[
                "Total PnL",
                "delta PnL",
                "gamma PnL",
                "theta PnL",
                "vega PnL",
                "volga PnL",
                "vanna PnL",
                "charm PnL",
                "color PnL",
                "speed PnL",
                "Veta PnL",
                "Zomma PnL",
                "Ultima PnL",
                "Discrete PnL",
                "Discrete PnL Error",
                "Discrete PnL Error (%)",
            ],
        )

        dict_greeks_scenarios = {}
        for i, (spot, future_date, final_implied_volatility) in enumerate(
            zip(spots, future_dates, final_implied_volatilities)
        ):
            # Set new option
            option_scenario = copy.deepcopy(self)
            option_scenario.implied_volatility = final_implied_volatility
            new_underlying = copy.deepcopy(self.underlying)

            # We assume interest rates do not change
            if final_dividends is not None:
                new_div = final_dividends[i]
            else:
                new_div = self.underlying.get_divrate(dates=present_date).iloc[0]

            scenario_underlying = pd.DataFrame(
                data=[
                    np.array(
                        [
                            spot,
                            final_implied_volatility,
                            new_div,
                        ]
                    )
                ],
                index=[future_date],
                columns=["Price", "Volatility", "Dividend Rate"],
            )
            new_underlying.set_data(scenario_underlying)
            option_scenario.underlying = new_underlying
            spot_change = (spot - spot_init).iloc[0]
            ivol_change = final_implied_volatility - ivol_init
            tau = self.calendar.interval(dates=present_date, future_dates=future_date)[
                0
            ]

            # We consider the same discount curve but shifted
            displaced_discount_curve = ShiftedCurve(
                discount_curve, date_0=afsfun.dates_formatting(present_date)
            )

            if return_scenarios_greeks:
                df_scene = option_scenario.get_greek_table(
                    future_date, displaced_discount_curve
                )
                dict_greeks_scenarios[f"Scenario {i}"] = df_scene
                scenario_premium = df_scene.iloc[-1]["premium"]
            else:
                scenario_premium = option_scenario.get_px(
                    future_date, displaced_discount_curve
                ).iloc[0]

            # Compute PnL and divided by greeks
            total_pnl = scenario_premium - prem_init
            delta_pnl = delta * spot_change
            gamma_pnl = 0.5 * gamma * spot_change**2
            theta_pnl = theta * tau * self.calendar.days_in_year
            vega_pnl = vega * ivol_change * 100
            volga_pnl = 0.5 * volga * (ivol_change * 100) ** 2
            vanna_pnl = 2 * 0.5 * vanna * ivol_change * 100 * spot_change
            charm_pnl = 2 * 0.5 * charm * spot_change * tau * self.calendar.days_in_year
            color_pnl = (
                3 * 1 / 6 * color * spot_change**2 * tau * self.calendar.days_in_year
            )
            speed_pnl = 1 / 6 * speed * spot_change**3
            veta_pnl = (
                2 * 0.5 * veta * tau * self.calendar.days_in_year * ivol_change * 100
            )
            zomma_pnl = 3 * 1 / 6 * zomma * spot_change**2 * ivol_change * 100
            ultima_pnl = 1 / 6 * ultima * (ivol_change * 100) ** 3
            discrete_pnl = (
                delta_pnl
                + gamma_pnl
                + theta_pnl
                + vega_pnl
                + volga_pnl
                + vanna_pnl
                + charm_pnl
                + color_pnl
                + speed_pnl
                + veta_pnl
                + zomma_pnl
                + ultima_pnl
            )
            discrete_pnl_error = total_pnl - discrete_pnl
            discrete_pnl_error_per = (total_pnl - discrete_pnl) / total_pnl * 100

            # Final dataframe with all information
            df_risk.loc[
                f"Spot: {spot}. Tau: {int(tau*self.calendar.days_in_year)} days. IVOL changes: {ivol_change*100:.2f} %."
            ] = [
                total_pnl,
                delta_pnl,
                gamma_pnl,
                theta_pnl,
                vega_pnl,
                volga_pnl,
                vanna_pnl,
                charm_pnl,
                color_pnl,
                speed_pnl,
                veta_pnl,
                zomma_pnl,
                ultima_pnl,
                discrete_pnl,
                discrete_pnl_error,
                discrete_pnl_error_per,
            ]

        if return_scenarios_greeks:
            return df_risk, dict_greeks_scenarios
        else:
            return df_risk

    def create_strategy(self, vanilla_list):
        """
        Return a pricing.structured.VanillaStrategy strategy combining this vanilla and the ones in vanilla_list.

        Parameters
        ----------
        vanilla_list : list of pricing.structured.VanillaStrategy
           List of vanilla options to add in order to create the total strategy.

        Returns
        -------
        pricing.structured.VanillaStrategy
            Global strategy combining this option with all the options in vanilla_list.
        """
        strategy = VanillaStrategy([self] + vanilla_list)
        return strategy


class VanillaStrategy:
    """
    This class is a wrapper for a collection of Vanilla Option.

    Parameters
    ----------
    vanillas : list of pricing.structured.Vanilla instances
        List of the vanilla options making up the strategy.

    Attributes
    ----------
    vanillas : list of pricing.structured.Vanilla instances
        List of the vanilla options making up the strategy.
    """

    def __init__(self, vanillas):
        self.vanillas = vanillas

    def append(self, other):
        """
        Append a new vanilla option to the strategy.

        Parameters
        ----------
        other : structured.Vanilla
            The new option to be added to the strategy.

        Returns
        -------
        None

        Notes
        -----
        This method updates the attribute ``vanillas``.
        """
        if isinstance(other, Vanilla):
            self.vanillas.append(other)
        else:
            raise ValueError("You can only append Vanilla Options.")

    def concat(self, other):
        """
        Create a new strategy with the vanilla options in both strategies.

        Parameters
        ----------
        other : structured.VanillaStrategy
            The strategy to add the options from.

        Returns
        -------
        structured.VanillaStrategy
            New strategy with the options from the two initial ones.
        """
        if isinstance(other, VanillaStrategy):
            new_strategy = VanillaStrategy(self.vanillas + other.vanillas)
            return new_strategy
        else:
            raise ValueError("You can only concat VanillaStrategy.")

    def map_method(self, method, *args):
        """
        Apply a method from the Vanilla class to the strategy. In order
        to do so, it applies the method to each option in the strategy and
        then it combines the results.

        Parameters
        ----------
        method : string
            Name of the original method from structured.Vanilla class.
        *args : arguments
            Arguments to pass to the method.

        Returns
        -------
        pandas.Dataframe or dict.
            The results for each individual option and their aggregation.
        """
        if method == "get_risk_matrix":
            raise ValueError(
                "Use the method get_risk_matrix from VanillaStrategy class directly."
            )

        if len(self.vanillas) == 0:
            raise ValueError("The strategy is empty.")
        results = [getattr(x, method)(*args) for x in self.vanillas]
        if isinstance(results[0], tuple):
            results_list = list(map(list, zip(*results)))
            final_results = []
            for results in results_list:
                final_results.append(self._join_results(results))
            return tuple(final_results)
        else:
            return self._join_results(results)

    def get_risk_matrix(
        self,
        date,
        discount_curve,
        scenarios,
        complete_scenarios=False,
        return_scenarios_greeks=True,
    ):
        """
        Calculate the PnL (total and decomposed per greeks) of an option in different scenarios given a initial one.

        Parameters
        ----------
        present_date : pandas.Timestamp, or string
            Date for the initial scenario. It must be a single date (as a pandas.Timestamp or its string representation).
        discount_curve : pricing.discount_curves.DiscountCurve
            Provides the discounts for the computation of the greeks and prices.
        scenarios : dictionary
            Dictionary containing the different scenarios. Values must be lists or numpy.arrays of the same length (and equal
            to the number of scenarios considering). The ith scenario is defined by the ith element in each value. The keys
            represent the final value in the final scenarios for spot, implied volatilities and tenor. It is also possible to
            specify these values by changes instead of final values. We can include dividends as an optional key in the
            scenarios, if so, these values are used in the scenarios, if not, we assume the same interests.
        complete_scenarios : boolean, default False.
            If True it extends the scenarios by making the cartesian product of all the possibilities.

        Returns
        -------
        pandas.Dataframe
            Dataframe containing the total PnL of the option and decomposed by greeks for each scenario.
        dictionary
            Returns in a dictionary the premium and greeks of the strategy in each scenario as returned by the method
            ``get_greek_table``. The key for the ith scenario is "Scenario i"

        Notes
        -----
        The only variables to change in each scenario are the spot, the tenor and the implied volatility (and optionally
        the dividends). Dividend rates are assumed to be constant.
        This method is analogous to the one of the class pricing.structured.Vanilla, but for the strategy you must provide
        the implied volatility for each option.
        """
        # Check that all scenarios values are lists of the same length
        values_lengths = set(len(value) for value in scenarios.values())
        have_same_length = len(values_lengths) == 1

        if not have_same_length:
            raise ValueError(
                "All values in scenarios dictionary must be lists of the same length."
            )
        num_scenarios = list(values_lengths)[0]

        if complete_scenarios:
            combinations = list(
                product(*[np.unique(np.array(x)) for x in scenarios.values()])
            )
            values = [list(t) for t in zip(*combinations)]
            scenarios = {key: value for key, value in zip(scenarios.keys(), values)}
            num_scenarios = list(set(len(value) for value in scenarios.values()))[0]

        results = []

        # Create individual scenarios for each option with individual volatilities
        for i, x in enumerate(self.vanillas):
            individual_scenario = copy.deepcopy(scenarios)
            for key, value in individual_scenario.items():
                if isinstance(value[0], (list, np.ndarray)):
                    i_volatilities = [implied_scenario[i] for implied_scenario in value]
                    individual_scenario[key] = i_volatilities
            sol = x.get_risk_matrix(
                date,
                discount_curve,
                individual_scenario,
                return_scenarios_greeks=return_scenarios_greeks,
            )
            results.append(sol)

        if return_scenarios_greeks:
            results_list = list(map(list, zip(*results)))
            risk_tables = results_list[0]
            dict_greeks = results_list[1]
        else:
            risk_tables = results

        # Aggregate results of individual scenarios
        risk_agg = pd.concat(
            risk_tables,
            keys=[f"Option {i}" for i in range(len(results))],
            names=["Option", "Scenario"],
        )
        risk_agg_2 = copy.deepcopy(risk_agg)
        new_sublevel = [f"Scenario {i}" for i in range(num_scenarios)]
        new_level = [f"Option {i}" for i in range(len(self.vanillas))]
        new_index = pd.MultiIndex.from_tuples(list(product(new_level, new_sublevel)))
        risk_agg_2.index = new_index
        summed_rows = risk_agg_2.groupby(level=1).sum()
        total_row = (
            summed_rows.assign(DataFrame="Total Strategy")
            .set_index("DataFrame", append=True)
            .reorder_levels([1, 0])
        )
        risk_agg = pd.concat([risk_agg, total_row])
        final_results = [risk_agg]

        if return_scenarios_greeks:
            final_results.append(self._join_results(dict_greeks))
            return tuple(final_results)
        else:
            return risk_agg

    def _join_results(self, results):
        """
        Aggregate the results of each option. If pandas.Series or pandas.Dataframes
        it concats them, if a dictionary where values are pandas.Series or
        pandas.Dataframes, we aggregate the value for corresponding keys.

        Parameters
        ----------
        results : list of pandas.Dataframe or list of dict.
            List of results for each option.

        Returns
        -------
        pandas.Dataframe or dict.
            The results for each individual option and their aggregation.
        """
        if isinstance(results[0], pd.Series):
            df_results = pd.concat(results, axis=1)
            df_results.columns = [f"Option {i}" for i in range(len(self.vanillas))]
            df_results["Total Strategy"] = df_results.sum(axis=1)
            return df_results
        if isinstance(results[0], pd.DataFrame):
            df_results = pd.concat(
                results,
                keys=[f"Option {i}" for i in range(len(results))],
                names=["Option", None],
            )
            summed_rows = df_results.groupby(level=1).sum()
            total_row = (
                summed_rows.assign(DataFrame="Total Strategy")
                .set_index("DataFrame", append=True)
                .reorder_levels([1, 0])
            )
            df_results = pd.concat([df_results, total_row])
            return df_results
        if isinstance(results[0], dict):
            return {
                key: self._join_results(list(map(lambda x: x[key], results)))
                for key in results[0]
            }

    def update_options(
        self, date, underlying, implied_volatilities, dividend_yield=None
    ):
        if len(self.vanillas) != len(implied_volatilities):
            raise ValueError(
                "The length of implied volatilities must match the number of options."
            )
        for i, option in enumerate(self.vanillas):
            option.implied_volatility = implied_volatilities[i]

            # We assume dividend is the same from the closest date if dividend_yield is None
            if dividend_yield is None:
                closest_index = pd.Series(
                    (option.underlying.get_dates() - date).total_seconds().map(abs)
                ).idxmin()
                div = option.underlying.get_divrate(
                    dates=option.underlying.get_dates()[closest_index]
                ).iloc[0]
            else:
                div = dividend_yield[i]
            new_underlying = pd.DataFrame(
                data=[
                    np.array(
                        [
                            underlying,
                            implied_volatilities[i],
                            div,
                        ]
                    )
                ],
                index=[date],
                columns=["Price", "Volatility", "Dividend Rate"],
            )
            option.underlying.set_data(new_underlying)


class ImpVolObject:
    def impvol(self, px, sigma_est, discount_curve):
        """
        Calculate implied volatility.

        Parameters
        -----------
        px : panda.Series
            The first inner list contains dates as strings, and the second inner list contains corresponding price values.
        sigma_est : float
            Initial estimate of implied volatility.
        discount_curve : pricing.discount_curves.DiscountCurve
            Gives the discounts for the computations.

        Returns
        --------
        pandas.Series
            A Pandas Series containing the calculated implied volatilities indexed by dates.

        Notes
        ------
        This method calculates the implied volatility using an optimization method based on the observed prices, estimated volatility, and discount curve. It returns a Pandas
        Series of implied volatilities with dates as the index.
        """
        # separated from Vanilla because it conflicts with the get_px, but it's uniform for both put and call
        # px=[[array dates (str)],[array values]]

        temp_under = copy.deepcopy(self.underlying)
        d, p = pd.to_datetime(copy.copy(px[0])), px[1]
        if np.asarray(px).shape == (2,):
            d, p = np.array([d]), np.array([p])

        def difference(sigma):
            vol = np.array([d, sigma])
            # the following is for when vol changes to surface:
            # vol = pd.DataFrame([sigma, sigma], index=d, columns=[0, 1/4])
            temp_under.set_vol(vol)
            f = self.get_px(px[0], discount_curve).to_numpy() - np.asarray(p)
            df = np.diag(self.get_vega(px[0], discount_curve).to_numpy())
            return f.astype("float64"), df.astype("float64")

        iv = optimize.root(difference, sigma_est, jac=True, method="lm").x
        return pd.Series(iv, index=d, name="Implied Volatility")


class Call(Vanilla, ImpVolObject):
    """
    Class for instantiating a Call Option. This class is inherited from two other classes, Vanilla and ImpVolObject

    Parameters
    ----------
    underlying : str
        The underlying asset's symbol.
    strike : float
        The strike price of the option.
    maturity : pandas.Timestamp
        The expiration date of the option.
    calendar : data.calendars.DayCountCalendar
        Calendar convention for counting days.
    nominal : float, optional
        Number of underlying shares for each call option contract (default is 100.0).
    credit_curve : pricing.discount_curves.CRDC
        The credit spread curve (default is zero_credit_risk = CRDC(0), namely there is no credit spread).
    alpha : float, optional
        Quantity added to the strike price (default is 0).
    phi : str, optional
        If phi = "min" or None, self.phi is set to numpy.min; if phi = "max", self.phi is set to numpy.max.
        Otherwise it raises a ValueError "Value of self.phi must be 'min' or 'max'".
        This attribute is used when the price depends on several underlyings (default is None).
    implied_volatility : float, optional
        If None, the method get_vol is exploited to obtain the value of implied volatility from data.
        Otherwise, the input value is accepted as implied volatility for the option (default is None).
    """

    def __init__(
        self,
        underlying,
        strike,
        maturity,
        calendar,
        nominal=100.0,
        credit_curve=zero_credit_risk,
        alpha=0,
        phi=None,
        implied_volatility=None,
    ):
        Vanilla.__init__(
            self,
            underlying=underlying,
            strike=strike,
            maturity=maturity,
            kind="call",
            calendar=calendar,
            nominal=nominal,
            credit_curve=credit_curve,
            alpha=alpha,
            phi=phi,
            implied_volatility=implied_volatility,
            monotonicity_price_function="increasing",
        )

    def get_px(self, dates, discount_curve, no_sims=10**5, no_calcs=1):
        # TODO: Add parameters no_sims and no_calcs in docstring.
        """
        Compute the price of a call option for different underlying dynamics.

        If the underlying has lognormal dynamics, the following analytic formula is applied

        .. math::
            C(S_t,t) = N \\cdot \\left( S_t \\mathrm{e}^{- q(t) (T - t)}
            N(d_1) - K \\mathrm{e}^{- \\int^T_t r(s) \, \\mathrm{d}s} N(d_2) \\right)\,,

        where:

        - :math:`C(S_t,t)` is the price of the call option,
        - :math:`N` is the nominal,
        - :math:`S_t` is the value of the underlying at time `t`,
        - :math:`K` is the strike price of the option,
        - :math:`T` is the maturity of the option,
        - :math:`r(t)` is the risk-free rate at time `t`,
        - :math:`\\sigma(t)` is the implied volatility of the underlying at time `t`,
        - :math:`q(t)` is the dividend yield payed by the underlying at time `t`,
        - :math:`D(t,T) = \\mathrm{e}^{- \\int^T_t r(s) \\, \\mathrm{d}s}` is the discount factor,
        - :math:`F_t = S_t \\, \\mathrm{e}^{\\int^T_t r(s) \\, \\mathrm{d}s}` is the value of future price of the underlying at time `t`,
        - :math:`d_1 \\equiv \\frac{\\ln\\frac{F_t}{K} + \\left( \\frac{1}{2}\\sigma(t)^2 - q(t) \\right)(T-t)}{\\sigma(t)\\sqrt{T-t}}` and :math:`d_2 \\equiv d_1 - \\sigma(t) \\sqrt{T-t}`,
        - :math:`N(x)` is the standard normal cumulative distribution function evaluated at :math:`x`.

        If the underlying has normal dynamics and zero dividend yield, the following analytic formula is applied

        .. math::
            C(S_t,t) = N \\cdot \\mathrm{e}^{- \\int^T_t r(s) \, \\mathrm{d}s}
            \\left[ (F_t-K) N(d_1) +  \\frac{\\sigma(t)\\sqrt{T-t}}{\\sqrt{2\\pi}}\\mathrm{e}^{-d_1^2/2} \\right]\,,

        where:

        - :math:`C(S_t,t)` is the price of the call option,
        - :math:`N` is the nominal,
        - :math:`K` is the strike price of the option,
        - :math:`T` is the maturity of the option,
        - :math:`r(t)` is the risk-free rate at time `t`,
        - :math:`\\sigma(t)` is the implied volatility of the underlying at time `t`,
        - :math:`D(t,T) = \\mathrm{e}^{- \\int^T_t r(s) \\, \\mathrm{d}s}` is the discount factor,
        - :math:`F_t = S_t \\, \\mathrm{e}^{\\int^T_t r(s) \\, \\mathrm{d}s}` is the value of future price of the underlying at time `t`,
        - :math:`d_1 \\equiv \\frac{F_t-K}{\\sigma(t) \\sqrt{T-t}}`,
        - :math:`N(x)` is the standard normal cumulative distribution function evaluated at :math:`x`.

        In the special case :math:`N=K`, the usual expressions for the call price are recovered.

        For other implemented underlying dynamics (NormalAsset with dividend yield, MultiAsset,
        ExposureIndex, Heston, SABR, MultiAssetHeston), the price of the call is generated
        via Montecarlo simulation.

        If an underlying with a non implemented dynamics is used, a ValueError raises.

        Parameters
        ----------
        dates : pandas.DatetimeIndex, list of strings, pandas.Timestamp, or string
            Dates at which forward price is computed. It can be a single date (as a pandas.Timestamp or its string representation) or an array-like object (as a
            pandas.DatetimeIndex or a list of its string representation) containing dates.
        discount_curve : pricing.discount_curves.DiscountCurve
            Gives the discounts for the computations.

        Returns
        -------
        float
            The value of the call option for implemented underlying dynamics.

        Raises
        ----------
        ValueError
            If the underlying dynamics is not implemented.

        References
        ----------
        - [J. C. Hull, 2021] Hull, J. C. (2021). Options, Futures, and Other Derivatives, 11th Edition.

        - [K. Iwasawa, 2001] Iwasawa, K. (2001) Analytic Formula for the European Normal Black-Scholes Formula.

        See Also
        ----------
        structured.Put.get_px
        """
        dates = afsfun.dates_formatting(dates)
        valid_dates = dates[
            dates <= self.maturity
        ]  # Valid valuation dates for computing the price
        nonvalid_dates = dates[dates > self.maturity]
        if nonvalid_dates.size != 0:
            print(
                f"For {nonvalid_dates.strftime('%Y-%m-%d').tolist()} the price is zero since this valuation dates are after the maturity date."
            )
            # TODO: This price should also be returned with the prices (pd.Series defined below) for valid dates?
        if valid_dates.size != 0:
            if type(self.underlying) is list:
                print("Method not available for baskets; try Monte Carlo instead.")
                return None
            if isinstance(self.underlying, LognormalAsset):
                alpha = self.alpha
                fpx, d1, d2, discount, vol = Vanilla.get_px(
                    self, dates=valid_dates, discount_curve=discount_curve
                )
                strike = alpha + self.strike
                # Vanilla.get_px already adjusts data by alpha
                prices = (
                    self.nominal * discount * (fpx * self.N(d1) - strike * self.N(d2))
                )
                return prices
            elif (
                isinstance(self.underlying, NormalAsset)
                and not self.underlying.yieldsdiv
            ):
                # Normal is only used for forward and swap rates, so we use the Bachelier
                # formula for zero interest rates
                fpx, vol, d, discount = Vanilla.get_px(
                    self, dates=dates, discount_curve=discount_curve
                )
                return (
                    self.nominal
                    * discount
                    * ((fpx - self.strike) * self.N(d) + vol * self.f(d))
                )
            elif isinstance(self.underlying, tuple(underlying_dynamics_classes)):
                return MCProduct.get_px(
                    self,
                    dates=valid_dates,
                    discount_curve=discount_curve,
                    no_sims=no_sims,
                    no_calcs=no_calcs,
                )
            else:  # This else is more important when import_underlying indata_factory_bd.py is used.
                raise ValueError(
                    f"Underlying dynamics not supported. Try with {underlying_dynamics}"
                )  # List the right dynamics.

    def get_delta(
        self, dates, discount_curve, no_sims=None, no_calcs=None, monte_carlo=None
    ):
        """
        Compute the delta of a call option for different underlying dynamics.
        Delta is the derivative of the option value to the underlying.
        It is multiplied by the nominal value to calculate the hedge ratio.

        If the underlying has lognormal dynamics, the following analytic formula is applied

        .. math::
            \\Delta(S_t,t) = N \\cdot \\mathrm{e}^{- q(t) (T - t)} N(d_1) \,,

        where:

        - :math:`\\Delta(S_t,t)` is the delta of the call option,
        - :math:`N` is the nominal,
        - :math:`S_t` is the value of the underlying at time `t`,
        - :math:`K` is the strike price of the option,
        - :math:`T` is the maturity of the option,
        - :math:`r(t)` is the risk-free rate at time `t`,
        - :math:`\\sigma(t)` is the implied volatility of the underlying at time `t`,
        - :math:`q(t)` is the dividend yield payed by the underlying at time `t`,
        - :math:`F_t = S_t \\, \\mathrm{e}^{\\int^T_t r(s) \\, \\mathrm{d}s}` is the value of future price of the underlying at time `t`,
        - :math:`d_1 \\equiv \\frac{\\ln\\frac{F_t}{K} + \\left( \\frac{1}{2}\\sigma(t)^2 - q(t) \\right) (T-t)}{\\sigma(t)\\sqrt{T-t}}`,
        - :math:`N(x)` is the standard normal cumulative distribution function evaluated at :math:`x`.

        If the underlying has normal dynamics and zero dividend yield, the following analytic formula is applied

        .. math::
            \\Delta(S_t,t) = N \\cdot N(d_1) \,,

        where:

        - :math:`\\Delta(S_t,t)` is the delta of the call option,
        - :math:`N` is the nominal,
        - :math:`S_t` is the value of the underlying at time `t`,
        - :math:`K` is the strike price of the option,
        - :math:`T` is the maturity of the option,
        - :math:`r(t)` is the risk-free rate at time `t`,
        - :math:`\\sigma(t)` is the implied volatility of the underlying at time `t`,
        - :math:`F_t = S_t \\, \\mathrm{e}^{\\int^T_t r(s) \\, \\mathrm{d}s}` is the value of future price of the underlying at time `t`,
        - :math:`d_1 \\equiv \\frac{F_t-K}{\\sigma(t)\\sqrt{T-t}}`,
        - :math:`N(x)` is the standard normal cumulative distribution function evaluated at :math:`x`.

        In the special case :math:`N=K`, the usual expressions for the delta are recovered.

        For other implemented underlying dynamics (NormalAsset with dividend yield, MultiAsset,
        ExposureIndex, Heston, SABR, MultiAssetHeston), the delta of the call is generated
        via Montecarlo simulation.

        If an underlying with a non implemented dynamics is used, a ValueError raises.

        This method overrides the parent class's one, but ignores the ``no_sims``, ``no_calcs``, and ``monte_carlo`` parameters.

        Parameters
        ----------
        dates : pandas.DatetimeIndex, list of strings, pandas.Timestamp, or string
            Dates at which delta is calculated. It can be a single date (as a pandas.Timestamp or its string representation) or an array-like object (as a pandas.DatetimeIndex or a
            list of its string representation) containing dates.
        discount_curve : pricing.discount_curves.DiscountCurve
            Provides the discounts for the computation of the delta.
        no_sims : int, optional
            Ignored in this implementation (defaults to None).
        no_calcs : int, optional
            Ignored in this implementation (defaults to None).
        monte_carlo : bool, optional
            Ignored in this implementation (defaults to None).

        Returns
        -------
        float
            The delta of the call option for implemented underlying dynamics.

        Raises
        ----------
        ValueError
            If the underlying dynamics is not implemented.

        References
        ----------
        - [J. C. Hull, 2021] Hull, J. C. (2021). Options, Futures, and Other Derivatives, 11th Edition.

        - [K. Iwasawa, 2001] Iwasawa, K. (2001) Analytic Formula for the European Normal Black-Scholes Formula.

        See Also
        ----------
        structured.Put.get_delta
        """
        if type(self.underlying) is list:
            print("Method not available for baskets; try Monte Carlo instead.")
            return None
        if isinstance(self.underlying, LognormalAsset):
            fpx, d1, d2, discount, vol = Vanilla.get_px(
                self, dates=dates, discount_curve=discount_curve
            )
            q = self.underlying.get_divrate(dates).values
            tau = self.calendar.interval(dates=dates, future_dates=self.maturity)
            delta = np.exp(-q * tau) * self.N(d1)
            delta = pd.Series(delta, index=fpx.index, name="Price")
            return self.nominal * delta
        elif isinstance(self.underlying, NormalAsset) and not self.underlying.yieldsdiv:
            fpx, vol, d, discount = Vanilla.get_px(
                self, dates=dates, discount_curve=discount_curve
            )
            # This can be found in "Analytic Formula for the European Normal Black-Scholes Formula" by K. Iwasawa.
            delta = self.N(d)
            delta = pd.Series(delta, index=fpx.index, name="Price")
            return self.nominal * delta
        elif isinstance(self.underlying, tuple(underlying_dynamics_classes)):
            return MCProduct.get_delta(
                self,
                dates=dates,
                discount_curve=discount_curve,
                no_sims=100000,
                no_calcs=1,
            )
        else:  # This else is more important when import_underlying indata_factory_bd.py is used.
            raise ValueError(
                f"Underlying dynamics not supported. Try with {underlying_dynamics}"
            )  # List the right dynamics.

    def get_gamma(
        self, dates, discount_curve, no_sims=None, no_calcs=None, monte_carlo=None
    ):
        """
        Compute the gamma of a call option for different underlying dynamics.
        Gamma is the second derivative of the option value to the underlying.

        If the underlying has lognormal dynamics, the following analytic formula is applied

        .. math::
            \\Gamma(S_t,t) = N \\cdot \\frac{\\mathrm{e}^{- q(t) (T - t)}}
            {S_t \\sigma(t) \\sqrt{T-t}} N'(d_1) \,,

        where:

        - :math:`\\Gamma(S_t,t)` is the gamma of the call option,
        - :math:`N` is the nominal,
        - :math:`S_t` is the value of the underlying at time `t`,
        - :math:`K` is the strike price of the option,
        - :math:`T` is the maturity of the option,
        - :math:`r(t)` is the risk-free rate at time `t`,
        - :math:`\\sigma(t)` is the implied volatility of the underlying at time `t`,
        - :math:`q(t)` is the dividend yield payed by the underlying at time `t`,
        - :math:`D(t,T) = \\mathrm{e}^{- \\int^T_t r(s) \\, \\mathrm{d}s}` is the discount factor,
        - :math:`F_t = S_t \\, \\mathrm{e}^{\\int^T_t r(s) \\, \\mathrm{d}s}` is the value of future price of the underlying at time `t`,
        - :math:`d_1 \\equiv \\frac{\\ln\\frac{F_t}{K} + \\left( \\frac{1}{2}\\sigma(t)^2 - q(t) \\right) (T-t)}{\\sigma(t)\\sqrt{T-t}}`,
        - :math:`N(x)` is the standard normal cumulative distribution function evaluated at :math:`x`,
        - :math:`N'(x) = \\frac{\\mathrm{e}^{-d_1^2/2}}{\\sqrt{2\\pi}}` is the standard normal probability distribution function evaluated at :math:`x`.

        If the underlying has normal dynamics and zero dividend yield, the following analytic formula is applied

        .. math::
            \\Gamma(S_t,t) = N \\cdot \\frac{\\mathrm{e}^{\\int^T_t r(s) \, \\mathrm{d}s}}
            {\\sigma(t) \\sqrt{T-t}} N'(d_1) \,,

        where:

        - :math:`\\Gamma(S_t,t)` is the gamma of the call option,
        - :math:`N` is the nominal,
        - :math:`S_t` is the value of the underlying at time `t`,
        - :math:`K` is the strike price of the option,
        - :math:`T` is the maturity of the option,
        - :math:`r(t)` is the risk-free rate at time `t`,
        - :math:`\\sigma(t)` is the implied volatility of the underlying at time `t`,
        - :math:`F_t = S_t \\, \\mathrm{e}^{\\int^T_t r(s) \\, \\mathrm{d}s}` is the value of future price of the underlying at time `t`,
        - :math:`d_1 \\equiv \\frac{F_t-K}{\\sigma(t)\\sqrt{T-t}}`,
        - :math:`N(x)` is the standard normal cumulative distribution function evaluated at :math:`x`.
        - :math:`N'(x) = \\frac{\\mathrm{e}^{-d_1^2/2}}{\\sqrt{2\\pi}}` is the standard normal probability distribution function evaluated at :math:`x`.

        In the special case :math:`N=K`, the usual expressions for the gamma are recovered.

        For other implemented underlying dynamics (NormalAsset with dividend yield, MultiAsset,
        ExposureIndex, Heston, SABR, MultiAssetHeston), the gamma of the call is generated
        via Montecarlo simulation.

        If an underlying with a non implemented dynamics is used, a ValueError raises.

        This method overrides the parent class's one, but ignores the ``no_sims``, ``no_calcs``, and ``monte_carlo`` parameters.

        Parameters
        ----------
        dates : pandas.DatetimeIndex, list of strings, pandas.Timestamp, or string
            Dates at which gamma is calculated. It can be a single date (as a pandas.Timestamp or its string representation) or an array-like object (as a pandas.DatetimeIndex or a
            list of its string representation) containing dates.
        discount_curve : pricing.discount_curves.DiscountCurve
            Provides the discounts for the computation of the gamma.
        no_sims : int, optional
            Ignored in this implementation (defaults to None).
        no_calcs : int, optional
            Ignored in this implementation (defaults to None).
        monte_carlo : bool, optional
            Ignored in this implementation (defaults to None).

        Returns
        -------
        float
            The gamma of the call option for implemented underlying dynamics.

        Raises
        ----------
        ValueError
            If the underlying dynamics is not implemented.

        References
        ----------
        - [J. C. Hull, 2021] Hull, J. C. (2021). Options, Futures, and Other Derivatives, 11th Edition.

        - [K. Iwasawa, 2001] Iwasawa, K. (2001) Analytic Formula for the European Normal Black-Scholes Formula.

        See Also
        ----------
        structured.Put.get_gamma
        """
        if type(self.underlying) is list:
            print("Method not available for baskets; try Monte Carlo instead.")
            return None
        if isinstance(self.underlying, LognormalAsset):
            fpx, d1, d2, discount, vol = Vanilla.get_px(
                self, dates=dates, discount_curve=discount_curve
            )
            q = self.underlying.get_divrate(dates).values
            tau = self.calendar.interval(dates=dates, future_dates=self.maturity)
            px = self.underlying.get_value(dates=dates)
            gamma = np.exp(-q * tau) * self.f(d1) / (px * vol * np.sqrt(tau))
            gamma = pd.Series(gamma, index=fpx.index, name="Price")
            return self.nominal * gamma
        elif isinstance(self.underlying, NormalAsset) and not self.underlying.yieldsdiv:
            fpx, vol, d, discount = Vanilla.get_px(
                self, dates=dates, discount_curve=discount_curve
            )
            # This can be found in "Analytic Formula for the European Normal Black-Scholes Formula" by K. Iwasawa.
            gamma = self.f(d) / (discount * vol)
            gamma = pd.Series(gamma, index=fpx.index, name="Price")
            return self.nominal * gamma
        elif isinstance(self.underlying, tuple(underlying_dynamics_classes)):
            return MCProduct.get_gamma(
                self,
                dates=dates,
                discount_curve=discount_curve,
                no_sims=100000,
                no_calcs=1,
            )
        else:  # This else is more important when import_underlying indata_factory_bd.py is used.
            raise ValueError(
                f"Underlying dynamics not supported. Try with {underlying_dynamics}"
            )  # List the right dynamics.

    def get_vega(
        self, dates, discount_curve, no_sims=None, no_calcs=None, monte_carlo=None
    ):
        """
        Compute the vega of a call option for different underlying dynamics.
        Vega is the derivative of the option value to the underlying volatility.
        It is multiplied by the nominal value to calculate the hedge ratio.

        If the underlying has lognormal dynamics, the following analytic formula is applied

        .. math::
            \\mathcal{V}(S_t,t) = N \\cdot S_t \, \\mathrm{e}^{- q(t) (T - t)}
            \\sqrt{T-t} \, N'(d_1) \,,

        where:

        - :math:`\\mathcal{V}(S_t,t)` is the vega of the call option,
        - :math:`N` is the nominal,
        - :math:`S_t` is the value of the underlying at time `t`,
        - :math:`K` is the strike price of the option,
        - :math:`T` is the maturity of the option,
        - :math:`r(t)` is the risk-free rate at time `t`,
        - :math:`\\sigma(t)` is the implied volatility of the underlying at time `t`,
        - :math:`q(t)` is the dividend yield payed by the underlying at time `t`,
        - :math:`D(t,T) = \\mathrm{e}^{- \\int^T_t r(s) \\, \\mathrm{d}s}` is the discount factor,
        - :math:`F_t = S_t \\, \\mathrm{e}^{\\int^T_t r(s) \\, \\mathrm{d}s}` is the value of future price of the underlying at time `t`,
        - :math:`d_1 \\equiv \\frac{\\ln\\frac{F_t}{K} + \\left( \\frac{1}{2}\\sigma(t)^2 - q(t) \\right) (T-t)}{\\sigma(t)\\sqrt{T-t}}`,
        - :math:`N(x)` is the standard normal cumulative distribution function evaluated at :math:`x`,
        - :math:`N'(x) = \\frac{\\mathrm{e}^{-d_1^2/2}}{\\sqrt{2\\pi}}` is the standard normal probability distribution function evaluated at :math:`x`.

        If the underlying has normal dynamics and zero dividend yield, the following analytic formula is applied

        .. math::
            \\mathcal{V}(S_t,t) = N \\cdot \\mathrm{e}^{- \\int^T_t r(s) \, \\mathrm{d}s}
            \\sqrt{T-t} \, N'(d_1) \,,

        where:

        - :math:`\\mathcal{V}(S_t,t)` is the vega of the call option,
        - :math:`N` is the nominal,
        - :math:`S_t` is the value of the underlying at time `t`,
        - :math:`K` is the strike price of the option,
        - :math:`T` is the maturity of the option,
        - :math:`r(t)` is the risk-free rate at time `t`,
        - :math:`\\sigma(t)` is the implied volatility of the underlying at time `t`,
        - :math:`F_t = S_t \\, \\mathrm{e}^{\\int^T_t r(s) \\, \\mathrm{d}s}` is the value of future price of the underlying at time `t`,
        - :math:`d_1 \\equiv \\frac{F_t-K}{\\sigma(t)\\sqrt{T-t}}`,
        - :math:`N(x)` is the standard normal cumulative distribution function evaluated at :math:`x`.
        - :math:`N'(x) = \\frac{\\mathrm{e}^{-d_1^2/2}}{\\sqrt{2\\pi}}` is the standard normal probability distribution function evaluated at :math:`x`.

        In the special case :math:`N=K`, the usual expressions for the vega are recovered.

        For other implemented underlying dynamics (NormalAsset with dividend yield, MultiAsset,
        ExposureIndex, Heston, SABR, MultiAssetHeston), the vega of the call is generated
        via Montecarlo simulation.

        If an underlying with a non implemented dynamics is used, a ValueError raises.

        This method overrides the parent class's one, but ignores the ``no_sims``, ``no_calcs``, and ``monte_carlo`` parameters.

        Parameters
        ----------
        dates : pandas.DatetimeIndex, list of strings, pandas.Timestamp, or string
            Dates at which vega is calculated. It can be a single date (as a pandas.Timestamp or its string representation) or an array-like object (as a pandas.DatetimeIndex or a
            list of its string representation) containing dates.
        discount_curve : pricing.discount_curves.DiscountCurve
            Provides the discounts for the computation of the vega.
        no_sims : int, optional
            Ignored in this implementation (defaults to None).
        no_calcs : int, optional
            Ignored in this implementation (defaults to None).
        monte_carlo : bool, optional
            Ignored in this implementation (defaults to None).

        Returns
        -------
        float
            The vega of the call option for implemented underlying dynamics.

        Raises
        ----------
        ValueError
            If the underlying dynamics is not implemented.

        References
        ----------
        - [J. C. Hull, 2021] Hull, J. C. (2021). Options, Futures, and Other Derivatives, 11th Edition.

        - [K. Iwasawa, 2001] Iwasawa, K. (2001) Analytic Formula for the European Normal Black-Scholes Formula.

        See Also
        ----------
        structured.Put.get_vega
        """
        if type(self.underlying) is list:
            print("Method not available for baskets; try Monte Carlo instead.")
            return None
        if isinstance(self.underlying, LognormalAsset):
            fpx, d1, d2, discount, vol = Vanilla.get_px(
                self, dates=dates, discount_curve=discount_curve
            )
            q = self.underlying.get_divrate(dates).values
            tau = self.calendar.interval(dates=dates, future_dates=self.maturity)
            px = self.underlying.get_value(dates=dates)
            vega = px * np.exp(-q * tau) * np.sqrt(tau) * self.f(d1)
            vega = pd.Series(vega, index=fpx.index, name="Price")
            return self.nominal * vega
        elif isinstance(self.underlying, NormalAsset) and not self.underlying.yieldsdiv:
            fpx, vol, d, discount = Vanilla.get_px(
                self, dates=dates, discount_curve=discount_curve
            )
            tau = self.calendar.interval(dates=dates, future_dates=self.maturity)
            # This can be found in "Analytic Formula for the European Normal Black-Scholes Formula" by K. Iwasawa.
            vega = discount * np.sqrt(tau) * self.f(d)
            vega = pd.Series(vega, index=fpx.index, name="Price")
            return self.nominal * vega
        elif isinstance(self.underlying, tuple(underlying_dynamics_classes)):
            return MCProduct.get_vega(
                self,
                dates=dates,
                discount_curve=discount_curve,
                no_sims=100000,
                no_calcs=1,
            )
        else:  # This else is more important when import_underlying indata_factory_bd.py is used.
            raise ValueError(
                f"Underlying dynamics not supported. Try with {underlying_dynamics}"
            )  # List the right dynamics.

    def get_theta(
        self,
        dates,
        discount_curve,
        is_interest_rate_constant=False,
        no_sims=None,
        no_calcs=None,
        maturity_method=None,
        monte_carlo=None,
        constant_curve=None,
        refit=None,
    ):
        """
        Compute the theta of a call option for different underlying dynamics.
        Theta is the derivative of the option value to time.

        If the underlying has lognormal dynamics, the following analytic formula is applied

        .. math::
            \\Theta(S_t,t) = N \\cdot \\mathrm{e}^{- \\int^T_t r(s) \, \\mathrm{d}s}
            \\left( - \\frac{\\sigma(t) F_t \, \\mathrm{e}^{- q(t) (T - t)}}{2 \\sqrt{T-t}} N'(d_1) -
            r(t)KN(d_2) + q(t)F_t \, \\mathrm{e}^{- q(t) (T - t)} N(d_1) \\right)\,,

        where:

        - :math:`\\Theta(S_t,t)` is the theta of the call option,
        - :math:`N` is the nominal,
        - :math:`S_t` is the value of the underlying at time `t`,
        - :math:`K` is the strike price of the option,
        - :math:`T` is the maturity of the option,
        - :math:`r(t)` is the risk-free rate at time `t`,
        - :math:`\\sigma(t)` is the implied volatility of the underlying at time `t`,
        - :math:`q(t)` is the dividend yield payed by the underlying at time `t`,
        - :math:`D(t,T) = \\mathrm{e}^{- \\int^T_t r(s) \\, \\mathrm{d}s}` is the discount factor,
        - :math:`F_t = S_t \\, \\mathrm{e}^{\\int^T_t r(s) \\, \\mathrm{d}s}` is the value of future price of the underlying at time `t`,
        - :math:`d_1 \\equiv \\frac{\\ln\\frac{F_t}{K} + \\left( \\frac{1}{2}\\sigma(t)^2 - q(t) \\right)(T-t)}{\\sigma(t)\\sqrt{T-t}}` and :math:`d_2 \\equiv d_1 - \\sigma(t) \\sqrt{T-t}`,
        - :math:`N(x)` is the standard normal cumulative distribution function evaluated at :math:`x`,
        - :math:`N'(x) = \\frac{\\mathrm{e}^{-d_1^2/2}}{\\sqrt{2\\pi}}` is the standard normal probability distribution function evaluated at :math:`x`.

        If the underlying has normal dynamics and zero dividend yield, the following analytic formula is applied

        .. math::
            \\Theta(S_t,t) = N \\cdot \\mathrm{e}^{- \\int^T_t r(s) \, \\mathrm{d}s}
            \\left[ - \\frac{N'(x) (1 - 2r(t) (T-t))}{2 \\sqrt{T-t}} - K r(t) N(d_1) \\right] \,,

        where:

        - :math:`\\Theta(S_t,t)` is the theta of the call option,
        - :math:`N` is the nominal,
        - :math:`S_t` is the value of the underlying at time `t`,
        - :math:`K` is the strike price of the option,
        - :math:`T` is the maturity of the option,
        - :math:`r(t)` is the risk-free rate at time `t`,
        - :math:`\\sigma(t)` is the implied volatility of the underlying at time `t`,
        - :math:`D(t,T) = \\mathrm{e}^{- \\int^T_t r(s) \\, \\mathrm{d}s}` is the discount factor,
        - :math:`F_t = S_t \\, \\mathrm{e}^{\\int^T_t r(s) \\, \\mathrm{d}s}` is the value of future price of the underlying at time `t`,
        - :math:`d_1 \\equiv \\frac{F_t-K}{\\sigma\\sqrt{T-t}}`,
        - :math:`N(x)` is the standard normal cumulative distribution function evaluated at :math:`x`.
        - :math:`N'(x) = \\frac{\\mathrm{e}^{-d_1^2/2}}{\\sqrt{2\\pi}}` is the standard normal probability distribution function evaluated at :math:`x`.

        In the special case :math:`N=K`, the usual expressions for the theta are recovered.

        For other implemented underlying dynamics (NormalAsset with dividend yield, MultiAsset,
        ExposureIndex, Heston, SABR, MultiAssetHeston), the theta of the call is generated
        via Montecarlo simulation.

        If an underlying with a non implemented dynamics is used, a ValueError raises.

        This method overrides the parent class's one, but ignores the ``no_sims``, ``no_calcs``, and ``monte_carlo`` parameters.

        Parameters
        ----------
        dates : pandas.DatetimeIndex, list of strings, pandas.Timestamp, or string
            Dates at which theta is calculated. It can be a single date (as a pandas.Timestamp or its string representation) or an array-like object (as a pandas.DatetimeIndex or a
            list of its string representation) containing dates.
        discount_curve : pricing.discount_curves.DiscountCurve
            Provides the discounts for the computation of the theta.
        is_interest_rate_constant : bool
            True if interest rate is constant, False otherwise (default is False). When the boolean is True,
            the interest rate is computed as :math:`r = -\\frac{\\ln D(t,T)}{T-t}`.
            Otherwise, :math:`r(t) = \\frac{1}{D(t,T)} \\frac{\\partial D(t,T)}{\\partial t}`
            and the finite difference approximation is exploited to compute the derivative of `D(t,T)`.
        no_sims : int, optional
            Ignored in this implementation (defaults to None).
        no_calcs : int, optional
            Ignored in this implementation (defaults to None).
        monte_carlo : bool, optional
            Ignored in this implementation (defaults to None).

        Returns
        -------
        float
            The theta of the call option for implemented underlying dynamics.

        Raises
        ----------
        ValueError
            If the underlying dynamics is not implemented.

        References
        ----------
        - [J. C. Hull, 2021] Hull, J. C. (2021). Options, Futures, and Other Derivatives, 11th Edition.

        - [K. Iwasawa, 2001] Iwasawa, K. (2001) Analytic Formula for the European Normal Black-Scholes Formula.

        See Also
        ----------
        structured.Put.get_theta
        """
        if isinstance(self.underlying, LognormalAsset):
            strike = self.strike
            fpx, d1, d2, discount, vol = Vanilla.get_px(
                self, dates=dates, discount_curve=discount_curve
            )
            original_dates = dates
            dates = fpx.index
            q = self.underlying.get_divrate(dates=dates).values
            px = self.underlying.get_value(dates=dates)
            tau = self.calendar.interval(dates=dates, future_dates=self.maturity)
            if is_interest_rate_constant:
                r = -np.log(discount) / tau
            else:
                dates_p = pd.to_datetime(original_dates) + pd.tseries.offsets.BDay()
                fpx_p, d1_p, d2_p, discount_p, vol_p = Vanilla.get_px(
                    self, dates=dates_p, discount_curve=discount_curve
                )
                dates_m = pd.to_datetime(original_dates) - pd.tseries.offsets.BDay()
                fpx_m, d1_m, d2_m, discount_m, vol_m = Vanilla.get_px(
                    self, dates=dates_m, discount_curve=discount_curve
                )
                time_int_mat = self.calendar.interval(dates_m, dates_p)
                deriv_df = (discount_p - discount_m) / time_int_mat
                r = deriv_df / discount

            # Vanilla.get_px already adjusts data by alpha
            theta = (
                -np.exp(-q * tau) * px * self.f(d1) * vol / (2 * np.sqrt(tau))
                - strike * discount * self.N(d2) * r
                + q * px * np.exp(-q * tau) * self.N(d1)
            )
            return self.nominal * theta
        elif isinstance(self.underlying, NormalAsset) and not self.underlying.yieldsdiv:
            fpx, vol, d, discount = Vanilla.get_px(
                self, dates=dates, discount_curve=discount_curve
            )
            # This can be found in "Analytic Formula for the European Normal Black-Scholes Formula" by K. Iwasawa.
            original_dates = dates
            dates = fpx.index
            tau = self.calendar.interval(dates=dates, future_dates=self.maturity)

            if is_interest_rate_constant:
                r = -np.log(discount) / tau
            else:
                dates_p = pd.to_datetime(original_dates) + pd.tseries.offsets.BDay()
                fpx_p, vol_p, d_p, discount_p = Vanilla.get_px(
                    self, dates=dates_p, discount_curve=discount_curve
                )

                dates_m = pd.to_datetime(original_dates) - pd.tseries.offsets.BDay()
                fpx_m, vol_m, d_m, discount_m = Vanilla.get_px(
                    self, dates=dates_m, discount_curve=discount_curve
                )

                time_int_mat = self.calendar.interval(dates_m, dates_p)
                deriv_df = (discount_p - discount_m) / time_int_mat
                r = deriv_df / discount

            theta = discount * (
                r * ((-self.strike) * self.N(d) + vol * self.f(d))
                - vol / (2 * tau) * self.f(d)
            )
            return self.nominal * theta
        elif isinstance(self.underlying, tuple(underlying_dynamics_classes)):
            return MCProduct.get_theta(
                self,
                dates=dates,
                discount_curve=discount_curve,
                no_sims=100000,
                no_calcs=1,
            )
        else:  # This else is more important when import_underlying indata_factory_bd.py is used.
            raise ValueError(
                f"Underlying dynamics not supported. Try with {underlying_dynamics}"
            )  # List the right dynamics.

    def get_rho(
        self, dates, discount_curve, no_sims=None, no_calcs=None, monte_carlo=None
    ):
        """
        Compute the rho of a call option for different underlying dynamics.
        Rho is the derivative of the option value to the interest rate.

        If the underlying has lognormal dynamics, the following analytic formula is applied

        .. math::
            \\rho(S_t,t) = N \\cdot K(T-t) \\mathrm{e}^{- \\int^T_t r(s) \, \\mathrm{d}s} N(d_2) \,,

        where:

        - :math:`\\rho(S_t,t)` is the rho of the call option,
        - :math:`N` is the nominal,
        - :math:`S_t` is the value of the underlying at time `t`,
        - :math:`K` is the strike price of the option,
        - :math:`T` is the maturity of the option,
        - :math:`r(t)` is the risk-free rate at time `t`,
        - :math:`\\sigma(t)` is the implied volatility of the underlying at time `t`,
        - :math:`D(t,T) = \\mathrm{e}^{- \\int^T_t r(s) \\, \\mathrm{d}s}` is the discount factor,
        - :math:`q(t)` is the dividend yield payed by the underlying at time `t`,
        - :math:`F_t = S_t \\, \\mathrm{e}^{\\int^T_t r(s) \\, \\mathrm{d}s}` is the value of future price of the underlying at time `t`,
        - :math:`d_2 \\equiv \\frac{\\ln\\frac{F_t}{K} - \\left( \\frac{1}{2}\\sigma(t)^2 + q(t) \\right)(T-t)}{\\sigma(t)\\sqrt{T-t}}`,
        - :math:`N(x)` is the standard normal cumulative distribution function evaluated at :math:`x`.

        If the underlying has normal dynamics and zero dividend yield, the following analytic formula is applied

        .. math::
            \\rho(S_t,t) = N \\cdot (T-t) \\mathrm{e}^{- \\int^T_t r(s) \, \\mathrm{d}s}
            \\left( K N(d_1) - \\sigma(t) \\sqrt{T-t} \, N'(d_1) \\right) \,,

        where:

        - :math:`\\rho(S_t,t)` is the rho of the call option,
        - :math:`N` is the nominal,
        - :math:`S_t` is the value of the underlying at time `t`,
        - :math:`K` is the strike price of the option,
        - :math:`T` is the maturity of the option,
        - :math:`r(t)` is the risk-free rate at time `t`,
        - :math:`\\sigma(t)` is the implied volatility of the underlying at time `t`,
        - :math:`D(t,T) = \\mathrm{e}^{- \\int^T_t r(s) \\, \\mathrm{d}s}` is the discount factor,
        - :math:`F_t = S_t \\, \\mathrm{e}^{\\int^T_t r(s) \\, \\mathrm{d}s}` is the value of future price of the underlying at time `t`,
        - :math:`d_1 \\equiv \\frac{F_t-K}{\\sigma(t)\\sqrt{T-t}}`,
        - :math:`N(x)` is the standard normal cumulative distribution function evaluated at :math:`x`,
        - :math:`N'(x) = \\frac{\\mathrm{e}^{-d_1^2/2}}{\\sqrt{2\\pi}}` is the standard normal probability distribution function evaluated at :math:`x`.

        In the special case :math:`N=K`, the usual expressions for the rho are recovered.

        For other implemented underlying dynamics (NormalAsset with dividend yield, MultiAsset,
        ExposureIndex, Heston, SABR, MultiAssetHeston), the rho of the call is generated
        via Montecarlo simulation.

        If an underlying with a non implemented dynamics is used, a ValueError raises.

        This method overrides the parent class's one, but ignores the ``no_sims``, ``no_calcs``, and ``monte_carlo`` parameters.

        Parameters
        ----------
        dates : pandas.DatetimeIndex, list of strings, pandas.Timestamp, or string
            Dates at which rho is calculated. It can be a single date (as a pandas.Timestamp or its string representation) or an array-like object (as a pandas.DatetimeIndex or a
            list of its string representation) containing dates.
        discount_curve : pricing.discount_curves.DiscountCurve
            Provides the discounts for the computation of the rho.
        no_sims : int, optional
            Ignored in this implementation (defaults to None).
        no_calcs : int, optional
            Ignored in this implementation (defaults to None).
        monte_carlo : bool, optional
            Ignored in this implementation (defaults to None).

        Returns
        -------
        float
            The rho of the call option for implemented underlying dynamics.

        Raises
        ----------
        ValueError
            If the underlying dynamics is not implemented.

        References
        ----------
        - [J. C. Hull, 2021] Hull, J. C. (2021). Options, Futures, and Other Derivatives, 11th Edition.

        - [K. Iwasawa, 2001] Iwasawa, K. (2001) Analytic Formula for the European Normal Black-Scholes Formula.

        See Also
        ----------
        structured.Put.get_rho
        """
        if type(self.underlying) is list:
            print("Method not available for baskets; try Monte Carlo instead.")
            return None
        if isinstance(self.underlying, LognormalAsset):
            fpx, d1, d2, discount, vol = Vanilla.get_px(
                self, dates=dates, discount_curve=discount_curve
            )
            tau = self.calendar.interval(dates=dates, future_dates=self.maturity)
            rho = self.strike * discount * tau * self.N(d2)
            rho = pd.Series(rho, index=fpx.index, name="Price")
            return self.nominal * rho
        elif isinstance(self.underlying, NormalAsset) and not self.underlying.yieldsdiv:
            fpx, vol, d, discount = Vanilla.get_px(
                self, dates=dates, discount_curve=discount_curve
            )
            tau = self.calendar.interval(dates=dates, future_dates=self.maturity)
            # This can be found in "Analytic Formula for the European Normal Black-Scholes Formula" by K. Iwasawa.
            rho = discount * tau * self.strike * self.N(
                d
            ) - discount * tau * vol * self.f(d)
            rho = pd.Series(rho, index=fpx.index, name="Price")
            return self.nominal * rho
        elif isinstance(self.underlying, tuple(underlying_dynamics_classes)):
            return MCProduct.get_vega(
                self,
                dates=dates,
                discount_curve=discount_curve,
                no_sims=100000,
                no_calcs=1,
            )
        else:  # This else is more important when import_underlying indata_factory_bd.py is used.
            raise ValueError(
                f"Underlying dynamics not supported. Try with {underlying_dynamics}"
            )  # List the right dynamics.

    def get_omega(self, dates, discount_curve):
        """
        Calculate the omega of an option. It s the percentage change in option value per percentage change in the underlying price.

        Parameters
        ----------
        dates : pandas.DatetimeIndex, list of strings, pandas.Timestamp, or string
            Dates at which omega is calculated. It can be a single date (as a pandas.Timestamp or its string representation) or an array-like object (as a pandas.DatetimeIndex or a
            list of its string representation) containing dates.
        discount_curve : pricing.discount_curves.DiscountCurve
            Provides the discounts for the computation of the omega.

        Returns
        -------
        pandas.Series
            Returns omega for the given dates if the underlying asset is of type LognormalAsset. For a basket of assets or other unsupported types, a message is printed
            and ``None`` is returned.

        Notes
        -----
        This method is not applicable for a basket of assets; in such cases, Monte Carlo methods
        are suggested. If the underlying asset is not of type LognormalAsset, the method also
        returns ``None``, indicating unsupported asset dynamics.
        """

        if type(self.underlying) is list:
            print("Method not available for baskets; try Monte Carlo instead.")
            return None
        if isinstance(self.underlying, LognormalAsset):
            fpx, d1, d2, discount, vol = Vanilla.get_px(
                self, dates=dates, discount_curve=discount_curve
            )
            dates = fpx.index
            px = self.underlying.get_value(dates=dates)
            # Vanilla.get_px already adjusts data by alpha
            return (
                self.N(d1)
                * px
                / Call.get_px(self, dates=dates, discount_curve=discount_curve)
            )
        else:
            raise TypeError("Underlying dynamics not supported")

    def get_charm(self, dates, discount_curve):
        """
        Calculate the charm of an option. Charm is the second order derivative of the option value, once to the underlying spot price and once to time.

        Parameters
        ----------
        dates : pandas.DatetimeIndex, list of strings, pandas.Timestamp, or string
            Dates at which charm is calculated. It can be a single date (as a pandas.Timestamp or its string representation) or an array-like object (as a pandas.DatetimeIndex or a
            list of its string representation) containing dates.
        discount_curve : pricing.discount_curves.DiscountCurve
            Provides the discounts for the computation of the charm.

        Returns
        -------
        pandas.Series
            Returns charm for the given dates if the underlying asset is of type LognormalAsset. For a basket of assets or other unsupported types, a message is printed
            and ``None`` is returned.

        Notes
        -----
        This method is not applicable for a basket of assets; in such cases, Monte Carlo methods
        are suggested. If the underlying asset is not of type LognormalAsset, the method also
        returns ``None``, indicating unsupported asset dynamics.
        """
        if type(self.underlying) is list:
            print("Method not available for baskets; try Monte Carlo instead.")
            return None
        if isinstance(self.underlying, LognormalAsset):
            fpx, d1, d2, discount, vol = Vanilla.get_px(
                self, dates=dates, discount_curve=discount_curve
            )
            dates = fpx.index
            q = self.underlying.get_divrate(dates=dates).values
            px = self.underlying.get_value(dates=dates)
            tau = self.calendar.interval(dates=dates, future_dates=self.maturity)
            r = -np.log(discount) / tau
            charm = q * np.exp(-q * tau) * self.N(d1) - np.exp(-q * tau) * self.f(
                d1
            ) * (2 * (r - q) * tau - d2 * vol * np.sqrt(tau)) / (
                2 * tau * vol * np.sqrt(tau)
            )
            return self.nominal * charm
        else:
            raise TypeError("Underlying dynamics not supported")

    def get_dual_delta(self, dates, discount_curve):
        """
        Calculate the dual delta of an option. Dual delta  is the derivative of the option value to strike.

        Parameters
        ----------
        dates : pandas.DatetimeIndex, list of strings, pandas.Timestamp, or string
            Dates at which dual-delta is calculated. It can be a single date (as a pandas.Timestamp or its string representation) or an array-like object (as a pandas.DatetimeIndex or a
            list of its string representation) containing dates.
        discount_curve : pricing.discount_curves.DiscountCurve
            Provides the discounts for the computation of the dual delta.

        Returns
        -------
        pandas.Series
            Returns dual delta for the given dates if the underlying asset is of type LognormalAsset. For a basket of assets or other unsupported types, a message is printed
            and ``None`` is returned.

        Notes
        -----
        This method is not applicable for a basket of assets; in such cases, Monte Carlo methods
        are suggested. If the underlying asset is not of type LognormalAsset, the method also
        returns ``None``, indicating unsupported asset dynamics.
        """
        if type(self.underlying) is list:
            print("Method not available for baskets; try Monte Carlo instead.")
            return None
        if isinstance(self.underlying, LognormalAsset):
            fpx, d1, d2, discount, vol = Vanilla.get_px(
                self, dates=dates, discount_curve=discount_curve
            )
            dual_delta = -discount * self.N(d2)
            return self.nominal * dual_delta
        else:
            raise TypeError("Underlying dynamics not supported")


class Put(Vanilla, ImpVolObject):
    """
    Class for instantiating a Put Option.
    This class is inherited from two other classes, Vanilla and ImpVolObject

    Parameters
    ----------
    underlying : str
        The underlying asset's symbol.
    strike : float
        The strike price of the  option.
    maturity : pandas.Timestamp
        The expiration date of the option.
    calendar : data.calendars.DayCountCalendar
        Calendar convention for counting days.
    nominal : float, optional
        Number of underlying shares for each call option contract (default is 100.0).
    credit_curve : pricing.discount_curves.CRDC
        The credit spread curve (default is zero_credit_risk = CRDC(0), namely there is no credit spread).
    alpha : float, optional
        Quantity added to the strike price (default is 0).
    phi : str, optional
        If phi = "min" or None, self.phi is set to numpy.min; if phi = "max", self.phi is set to numpy.max.
        Otherwise it raises a ValueError "Value of self.phi must be 'min' or 'max'".
        This attribute is used when the price depends on several underlyings (default is None).
    implied_volatility : float, optional
        If None, the method get_vol is exploited to obtain the value of implied volatility from data.
        Otherwise, the input value is accepted as implied volatility for the option (default is None).
    """

    def __init__(
        self,
        underlying,
        strike,
        maturity,
        calendar,
        nominal=100,
        credit_curve=zero_credit_risk,
        alpha=0,
        phi=None,
        implied_volatility=None,
    ):
        Vanilla.__init__(
            self,
            underlying=underlying,
            strike=strike,
            maturity=maturity,
            kind="put",
            calendar=calendar,
            nominal=nominal,
            credit_curve=credit_curve,
            alpha=alpha,
            phi=phi,
            implied_volatility=implied_volatility,
            monotonicity_price_function="decreasing",
        )

    def get_px(self, dates, discount_curve, no_sims=10**5, no_calcs=1):
        # TODO: Add parameters no_sims and no_calcs in docstring.
        """
        Compute the price of a put option for different underlying dynamics.

        If the underlying has lognormal dynamics, the following analytic formula is applied

        .. math::
            P(S_t,t) = N \\cdot \\left( K \\mathrm{e}^{- \\int^T_t r(s) \, \\mathrm{d}s}
            N(-d_2) - S_t \\mathrm{e}^{- q(t) (T - t)} N(-d_1) \\right)\,,

        where:

        - :math:`P(S_t,t)` is the price of the put option,
        - :math:`N` is the nominal,
        - :math:`S_t` is the value of the underlying at time `t`,
        - :math:`K` is the strike price of the option,
        - :math:`T` is the maturity of the option,
        - :math:`r(t)` is the risk-free rate at time `t`,
        - :math:`\\sigma(t)` is the implied volatility of the underlying at time `t`,
        - :math:`q(t)` is the dividend yield payed by the underlying at time `t`,
        - :math:`D(t,T) = \\mathrm{e}^{- \\int^T_t r(s) \\, \\mathrm{d}s}` is the discount factor,
        - :math:`F_t = S_t \\, \\mathrm{e}^{\\int^T_t r(s) \\, \\mathrm{d}s}` is the value of future price of the underlying at time `t`,
        - :math:`d_1 \\equiv \\frac{\\ln\\frac{F_t}{K} + \\left( \\frac{1}{2}\\sigma(t)^2 - q(t) \\right)(T-t)}{\\sigma(t)\\sqrt{T-t}}` and :math:`d_2 \\equiv d_1 - \\sigma(t) \\sqrt{T-t}`,
        - :math:`N(x)` is the standard normal cumulative distribution function evaluated at :math:`x`.

        If the underlying has normal dynamics and zero dividend yield, the following analytic formula is applied

        .. math::
            P(S_t,t) = N \\cdot \\mathrm{e}^{- \\int^T_t r(s) \, \\mathrm{d}s}
            \\left[ (K-F_t) N(-d_1) +  \\frac{\\sigma(t)\\sqrt{T-t}}{\\sqrt{2\\pi}}\\mathrm{e}^{-d_1^2/2} \\right]\,,

        where:

        - :math:`P(S_t,t)` is the price of the put option,
        - :math:`N` is the nominal,
        - :math:`K` is the strike price of the option,
        - :math:`T` is the maturity of the option,
        - :math:`r(t)` is the risk-free rate at time `t`,
        - :math:`\\sigma(t)` is the implied volatility of the underlying at time `t`,
        - :math:`D(t,T) = \\mathrm{e}^{- \\int^T_t r(s) \\, \\mathrm{d}s}` is the discount factor,
        - :math:`F_t = S_t \\, \\mathrm{e}^{\\int^T_t r(s) \\, \\mathrm{d}s}` is the value of future price of the underlying at time `t`,
        - :math:`d_1 \\equiv \\frac{F_t-K}{\\sigma(t)\\sqrt{T-t}}`,
        - :math:`N(x)` is the standard normal cumulative distribution function evaluated at :math:`x`.

        In the special case :math:`N=K`, the usual expressions for the put price are recovered.

        For other implemented underlying dynamics (NormalAsset with dividend yield, MultiAsset,
        ExposureIndex, Heston, SABR, MultiAssetHeston), the price of the put is generated
        via Montecarlo simulation.

        If an underlying with a non implemented dynamics is used, a ValueError raises.

        Parameters
        ----------
        dates : pandas.DatetimeIndex, list of strings, pandas.Timestamp, or string
            Dates at which forward price is computed. It can be a single date (as a pandas.Timestamp or its string representation) or an array-like object (as a
            pandas.DatetimeIndex or a list of its string representation) containing dates.
        discount_curve : pricing.discount_curves.DiscountCurve
            Gives the discounts for the computations.

        Returns
        -------
        float
            The value of the put option for implemented underlying dynamics.

        Raises
        ----------
        ValueError
            If the underlying dynamics is not implemented.

        References
        ----------
        - [J. C. Hull, 2021] Hull, J. C. (2021). Options, Futures, and Other Derivatives, 11th Edition.

        - [K. Iwasawa, 2001] Iwasawa, K. (2001) Analytic Formula for the European Normal Black-Scholes Formula.

        See Also
        ----------
        structured.Call.get_px
        """

        if type(self.underlying) is list:
            print("Method not available for baskets; try Monte Carlo instead.")
            return None
        # add loop checking that underlying has prices for all dates
        if isinstance(self.underlying, LognormalAsset):
            alpha = self.alpha
            fpx, d1, d2, discount, vol = Vanilla.get_px(
                self, dates=dates, discount_curve=discount_curve
            )
            strike = alpha + self.strike
            # Vanilla.get_px already adjusts data by alpha
            prices = (
                -self.nominal * discount * (fpx * self.N(-d1) - strike * self.N(-d2))
            )
            return prices
        elif isinstance(self.underlying, NormalAsset) and not self.underlying.yieldsdiv:
            # Normal is only used for forward and swap rates, so we use the Bachelier formula for zero interest rates
            fpx, vol, d, discount = Vanilla.get_px(
                self, dates=dates, discount_curve=discount_curve
            )
            return (
                self.nominal
                * discount
                * ((self.strike - fpx) * self.N(-d) + vol * self.f(d))
            )
        elif isinstance(self.underlying, tuple(underlying_dynamics_classes)):
            return MCProduct.get_px(
                self,
                dates=dates,
                discount_curve=discount_curve,
                no_sims=no_sims,
                no_calcs=no_calcs,
            )
        else:  # This else is more important when import_underlying indata_factory_bd.py is used.
            raise ValueError(
                f"Underlying dynamics not supported. Try with {underlying_dynamics}"
            )  # List the right dynamics.

    def get_delta(
        self, dates, discount_curve, no_sims=None, no_calcs=None, monte_carlo=None
    ):
        """
        Compute the delta of a put option for different underlying dynamics.
        Delta is the derivative of the option value to the underlying.
        It is multiplied by the nominal value to calculate the hedge ratio.

        If the underlying has lognormal dynamics, the following analytic formula is applied

        .. math::
            \\Delta(S_t,t) = N \\cdot \\mathrm{e}^{- q(t) (T - t)} (N(d_1) - 1) \,,

        where:

        - :math:`\\Delta(S_t,t)` is the delta of the put option,
        - :math:`N` is the nominal,
        - :math:`S_t` is the value of the underlying at time `t`,
        - :math:`K` is the strike price of the option,
        - :math:`T` is the maturity of the option,
        - :math:`r(t)` is the risk-free rate at time `t`,
        - :math:`\\sigma(t)` is the implied volatility of the underlying at time `t`,
        - :math:`q(t)` is the dividend yield payed by the underlying at time `t`,
        - :math:`F_t = S_t \\, \\mathrm{e}^{\\int^T_t r(s) \\, \\mathrm{d}s}` is the value of future price of the underlying at time `t`,
        - :math:`d_1 \\equiv \\frac{\\ln\\frac{F_t}{K} + \\left( \\frac{1}{2}\\sigma(t)^2 - q(t) \\right)(T-t)}{\\sigma(t)\\sqrt{T-t}}`,
        - :math:`N(x)` is the standard normal cumulative distribution function evaluated at :math:`x`.

        If the underlying has normal dynamics and zero dividend yield, the following analytic formula is applied

        .. math::
            \\Delta(S_t,t) = N \\cdot N(-d_1) \,,

        where:

        - :math:`\\Delta(S_t,t)` is the delta of the put option,
        - :math:`N` is the nominal,
        - :math:`S_t` is the value of the underlying at time `t`,
        - :math:`K` is the strike price of the option,
        - :math:`T` is the maturity of the option,
        - :math:`r(t)` is the risk-free rate at time `t`,
        - :math:`\\sigma(t)` is the implied volatility of the underlying at time `t`,
        - :math:`F_t = S_t \\, \\mathrm{e}^{\\int^T_t r(s) \\, \\mathrm{d}s}` is the value of future price of the underlying at time `t`,
        - :math:`d_1 \\equiv \\frac{F_t-K}{\\sigma(t)\\sqrt{T-t}}`,
        - :math:`N(x)` is the standard normal cumulative distribution function evaluated at :math:`x`.

        In the special case :math:`N=K`, the usual expressions for the delta are recovered.

        For other implemented underlying dynamics (NormalAsset with dividend yield, MultiAsset,
        ExposureIndex, Heston, SABR, MultiAssetHeston), the delta of the put is generated
        via Montecarlo simulation.

        If an underlying with a non implemented dynamics is used, a ValueError raises.

        This method overrides the parent class's one, but ignores the ``no_sims``, ``no_calcs``, and ``monte_carlo`` parameters.

        Parameters
        ----------
        dates : pandas.DatetimeIndex, list of strings, pandas.Timestamp, or string
            Dates at which delta is calculated. It can be a single date (as a pandas.Timestamp or its string representation) or an array-like object (as a pandas.DatetimeIndex or a
            list of its string representation) containing dates.
        discount_curve : pricing.discount_curves.DiscountCurve
            Provides the discounts for the computation of the delta.
        no_sims : int, optional
            Ignored in this implementation (defaults to None).
        no_calcs : int, optional
            Ignored in this implementation (defaults to None).
        monte_carlo : bool, optional
            Ignored in this implementation (defaults to None).

        Returns
        -------
        float
            The delta of the put option for implemented underlying dynamics.

        Raises
        ----------
        ValueError
            If the underlying dynamics is not implemented.

        References
        ----------
        - [J. C. Hull, 2021] Hull, J. C. (2021). Options, Futures, and Other Derivatives, 11th Edition.

        - [K. Iwasawa, 2001] Iwasawa, K. (2001) Analytic Formula for the European Normal Black-Scholes Formula.

        See Also
        ----------
        structured.Call.get_delta
        """
        if type(self.underlying) is list:
            print("Method not available for baskets; try Monte Carlo instead.")
            return None
        if isinstance(self.underlying, LognormalAsset):
            fpx, d1, d2, discount, vol = Vanilla.get_px(
                self, dates=dates, discount_curve=discount_curve
            )
            q = self.underlying.get_divrate(dates=dates).values
            tau = self.calendar.interval(dates=dates, future_dates=self.maturity)
            delta = np.exp(-q * tau) * (self.N(d1) - 1)
            delta = pd.Series(delta, index=fpx.index, name="Price")
            return self.nominal * delta
        elif isinstance(self.underlying, NormalAsset) and not self.underlying.yieldsdiv:
            fpx, vol, d, discount = Vanilla.get_px(
                self, dates=dates, discount_curve=discount_curve
            )
            # This can be found in "Analytic Formula for the European Normal Black-Scholes Formula" by K. Iwasawa.
            delta = -self.N(-d)
            return self.nominal * delta
        elif isinstance(self.underlying, tuple(underlying_dynamics_classes)):
            return MCProduct.get_delta(
                self,
                dates=dates,
                discount_curve=discount_curve,
                no_sims=100000,
                no_calcs=1,
            )
        else:  # This else is more important when import_underlying indata_factory_bd.py is used.
            raise ValueError(
                f"Underlying dynamics not supported. Try with {underlying_dynamics}"
            )  # List the right dynamics.

    def get_gamma(
        self, dates, discount_curve, no_sims=None, no_calcs=None, monte_carlo=None
    ):
        """
        Compute the gamma of a put option for different underlying dynamics.
        Gamma is the second derivative of the option value to the underlying.

        If the underlying has lognormal dynamics, the following analytic formula is applied

        .. math::
            \\Gamma(S_t,t) = N \\cdot \\frac{\\mathrm{e}^{- q(t) (T - t)}}
            {S_t \\sigma(t) \\sqrt{T-t}} N'(d_1) \,,

        where:

        - :math:`\\Gamma(S_t,t)` is the gamma of the put option,
        - :math:`N` is the nominal,
        - :math:`S_t` is the value of the underlying at time `t`,
        - :math:`K` is the strike price of the option,
        - :math:`T` is the maturity of the option,
        - :math:`r(t)` is the risk-free rate at time `t`,
        - :math:`\\sigma(t)` is the implied volatility of the underlying at time `t`,
        - :math:`q(t)` is the dividend yield payed by the underlying at time `t`,
        - :math:`D(t,T) = \\mathrm{e}^{- \\int^T_t r(s) \\, \\mathrm{d}s}` is the discount factor,
        - :math:`F_t = S_t \\, \\mathrm{e}^{\\int^T_t r(s) \\, \\mathrm{d}s}` is the value of future price of the underlying at time `t`,
        - :math:`d_1 \\equiv \\frac{\\ln\\frac{F_t}{K} + \\left( \\frac{1}{2}\\sigma(t)^2 - q(t) \\right)(T-t)}{\\sigma(t)\\sqrt{T-t}}`,
        - :math:`N(x)` is the standard normal cumulative distribution function evaluated at :math:`x`,
        - :math:`N'(x) = \\frac{\\mathrm{e}^{-d_1^2/2}}{\\sqrt{2\\pi}}` is the standard normal probability distribution function evaluated at :math:`x`.

        If the underlying has normal dynamics and zero dividend yield, the following analytic formula is applied

        .. math::
            \\Gamma(S_t,t) = N \\cdot \\frac{\\mathrm{e}^{\\int^T_t r(s) \, \\mathrm{d}s}}
            {\\sigma(t) \\sqrt{T-t}}N'(d_1) \,,

        where:

        - :math:`\\Gamma(S_t,t)` is the gamma of the put option,
        - :math:`N` is the nominal,
        - :math:`S_t` is the value of the underlying at time `t`,
        - :math:`K` is the strike price of the option,
        - :math:`T` is the maturity of the option,
        - :math:`r(t)` is the risk-free rate at time `t`,
        - :math:`\\sigma(t)` is the implied volatility of the underlying at time `t`,
        - :math:`F_t = S_t \\, \\mathrm{e}^{\\int^T_t r(s) \\, \\mathrm{d}s}` is the value of future price of the underlying at time `t`,
        - :math:`d_1 \\equiv \\frac{F_t-K}{\\sigma(t)\\sqrt{T-t}}`,
        - :math:`N(x)` is the standard normal cumulative distribution function evaluated at :math:`x`.
        - :math:`N'(x) = \\frac{\\mathrm{e}^{-d_1^2/2}}{\\sqrt{2\\pi}}` is the standard normal probability distribution function evaluated at :math:`x`.

        In the special case :math:`N=K`, the usual expressions for the gamma are recovered.

        For other implemented underlying dynamics (NormalAsset with dividend yield, MultiAsset,
        ExposureIndex, Heston, SABR, MultiAssetHeston), the gamma of the put is generated
        via Montecarlo simulation.

        If an underlying with a non implemented dynamics is used, a ValueError raises.

        This method overrides the parent class's one, but ignores the ``no_sims``, ``no_calcs``, and ``monte_carlo`` parameters.

        Parameters
        ----------
        dates : pandas.DatetimeIndex, list of strings, pandas.Timestamp, or string
            Dates at which gamma is calculated. It can be a single date (as a pandas.Timestamp or its string representation) or an array-like object (as a pandas.DatetimeIndex or a
            list of its string representation) containing dates.
        discount_curve : pricing.discount_curves.DiscountCurve
            Provides the discounts for the computation of the gamma.
        no_sims : int, optional
            Ignored in this implementation (defaults to None).
        no_calcs : int, optional
            Ignored in this implementation (defaults to None).
        monte_carlo : bool, optional
            Ignored in this implementation (defaults to None).

        Returns
        -------
        float
            The gamma of the put option for implemented underlying dynamics.

        Raises
        ----------
        ValueError
            If the underlying dynamics is not implemented.

        References
        ----------
        - [J. C. Hull, 2021] Hull, J. C. (2021). Options, Futures, and Other Derivatives, 11th Edition.

        - [K. Iwasawa, 2001] Iwasawa, K. (2001) Analytic Formula for the European Normal Black-Scholes Formula.

        See Also
        ----------
        structured.Call.get_gamma
        """
        if type(self.underlying) is list:
            print("Method not available for baskets; try Monte Carlo instead.")
            return None
        if isinstance(self.underlying, LognormalAsset):
            fpx, d1, d2, discount, vol = Vanilla.get_px(
                self, dates=dates, discount_curve=discount_curve
            )
            q = self.underlying.get_divrate(dates).values
            tau = self.calendar.interval(dates=dates, future_dates=self.maturity)
            px = self.underlying.get_value(dates=dates)
            gamma = np.exp(-q * tau) * self.f(d1) / (px * vol * np.sqrt(tau))
            gamma = pd.Series(gamma, index=fpx.index, name="Price")
            return self.nominal * gamma
        elif isinstance(self.underlying, NormalAsset) and not self.underlying.yieldsdiv:
            fpx, vol, d, discount = Vanilla.get_px(
                self, dates=dates, discount_curve=discount_curve
            )
            # This can be found in "Analytic Formula for the European Normal Black-Scholes Formula" by K. Iwasawa.
            gamma = self.f(d) / (discount * vol)
            gamma = pd.Series(gamma, index=fpx.index, name="Price")
            return self.nominal * gamma
        elif isinstance(self.underlying, tuple(underlying_dynamics_classes)):
            return MCProduct.get_gamma(
                self,
                dates=dates,
                discount_curve=discount_curve,
                no_sims=100000,
                no_calcs=1,
            )
        else:  # This else is more important when import_underlying indata_factory_bd.py is used.
            raise ValueError(
                f"Underlying dynamics not supported. Try with {underlying_dynamics}"
            )  # List the right dynamics.

    def get_vega(
        self, dates, discount_curve, no_sims=None, no_calcs=None, monte_carlo=None
    ):
        """
        Compute the vega of a put option for different underlying dynamics.
        Vega is the derivative of the option value to the underlying volatility.
        It is multiplied by the nominal value to calculate the hedge ratio.

        If the underlying has lognormal dynamics, the following analytic formula is applied

        .. math::
            \\mathcal{V}(S_t,t) = N \\cdot S_t \\mathrm{e}^{- q(t) (T - t)}
            \\sqrt{T-t} \, N'(d_1) \,,

        where:

        - :math:`\\mathcal{V}(S_t,t)` is the vega of the put option,
        - :math:`N` is the nominal,
        - :math:`S_t` is the value of the underlying at time `t`,
        - :math:`K` is the strike price of the option,
        - :math:`T` is the maturity of the option,
        - :math:`r(t)` is the risk-free rate at time `t`,
        - :math:`\\sigma(t)` is the implied volatility of the underlying at time `t`,
        - :math:`q(t)` is the dividend yield payed by the underlying at time `t`,
        - :math:`D(t,T) = \\mathrm{e}^{- \\int^T_t r(s) \\, \\mathrm{d}s}` is the discount factor,
        - :math:`F_t = S_t \\, \\mathrm{e}^{\\int^T_t r(s) \\, \\mathrm{d}s}` is the value of future price of the underlying at time `t`,
        - :math:`d_1 \\equiv \\frac{\\ln\\frac{F_t}{K} + \\left( \\frac{1}{2}\\sigma(t)^2 - q(t) \\right)(T-t)}{\\sigma(t)\\sqrt{T-t}}`,
        - :math:`N(x)` is the standard normal cumulative distribution function evaluated at :math:`x`,
        - :math:`N'(x) = \\frac{\\mathrm{e}^{-d_1^2/2}}{\\sqrt{2\\pi}}` is the standard normal probability distribution function evaluated at :math:`x`.

        If the underlying has normal dynamics and zero dividend yield, the following analytic formula is applied

        .. math::
            \\mathcal{V}(S_t,t) = N \\cdot \\mathrm{e}^{- \\int^T_t r(s) \, \\mathrm{d}s}
            \\sqrt{T-t} \, N'(d_1) \,,

        where:

        - :math:`\\mathcal{V}(S_t,t)` is the vega of the put option,
        - :math:`N` is the nominal,
        - :math:`S_t` is the value of the underlying at time `t`,
        - :math:`K` is the strike price of the option,
        - :math:`T` is the maturity of the option,
        - :math:`r(t)` is the risk-free rate at time `t`,
        - :math:`\\sigma(t)` is the implied volatility of the underlying at time `t`,
        - :math:`F_t = S_t \\, \\mathrm{e}^{\\int^T_t r(s) \\, \\mathrm{d}s}` is the value of future price of the underlying at time `t`,
        - :math:`d_1 \\equiv \\frac{F_t-K}{\\sigma(t)\\sqrt{T-t}}`,
        - :math:`N(x)` is the standard normal cumulative distribution function evaluated at :math:`x`.
        - :math:`N'(x) = \\frac{\\mathrm{e}^{-d_1^2/2}}{\\sqrt{2\\pi}}` is the standard normal probability distribution function evaluated at :math:`x`.

        In the special case :math:`N=K`, the usual expressions for the vega are recovered.

        For other implemented underlying dynamics (NormalAsset with dividend yield, MultiAsset,
        ExposureIndex, Heston, SABR, MultiAssetHeston), the vega of the put is generated
        via Montecarlo simulation.

        If an underlying with a non implemented dynamics is used, a ValueError raises.

        This method overrides the parent class's one, but ignores the ``no_sims``, ``no_calcs``, and ``monte_carlo`` parameters.

        Parameters
        ----------
        dates : pandas.DatetimeIndex, list of strings, pandas.Timestamp, or string
            Dates at which vega is calculated. It can be a single date (as a pandas.Timestamp or its string representation) or an array-like object (as a pandas.DatetimeIndex or a
            list of its string representation) containing dates.
        discount_curve : pricing.discount_curves.DiscountCurve
            Provides the discounts for the computation of the vega.
        no_sims : int, optional
            Ignored in this implementation (defaults to None).
        no_calcs : int, optional
            Ignored in this implementation (defaults to None).
        monte_carlo : bool, optional
            Ignored in this implementation (defaults to None).

        Returns
        -------
        float
            The vega of the put option for implemented underlying dynamics.

        Raises
        ----------
        ValueError
            If the underlying dynamics is not implemented.

        References
        ----------
        - [J. C. Hull, 2021] Hull, J. C. (2021). Options, Futures, and Other Derivatives, 11th Edition.

        - [K. Iwasawa, 2001] Iwasawa, K. (2001) Analytic Formula for the European Normal Black-Scholes Formula.

        See Also
        ----------
        structured.Call.get_vega
        """
        if type(self.underlying) is list:
            print("Method not available for baskets; try Monte Carlo instead.")
            return None
        if isinstance(self.underlying, LognormalAsset):
            fpx, d1, d2, discount, vol = Vanilla.get_px(
                self, dates=dates, discount_curve=discount_curve
            )
            q = self.underlying.get_divrate(dates).values
            tau = self.calendar.interval(dates=dates, future_dates=self.maturity)
            px = self.underlying.get_value(dates=dates)
            vega = px * np.exp(-q * tau) * np.sqrt(tau) * self.f(d1)
            vega = pd.Series(vega, index=fpx.index, name="Price")
            return self.nominal * vega
        elif isinstance(self.underlying, NormalAsset) and not self.underlying.yieldsdiv:
            fpx, vol, d, discount = Vanilla.get_px(
                self, dates=dates, discount_curve=discount_curve
            )
            tau = self.calendar.interval(dates=dates, future_dates=self.maturity)
            # This can be found in "Analytic Formula for the European Normal Black-Scholes Formula" by K. Iwasawa.
            vega = discount * np.sqrt(tau) * self.f(d)
            vega = pd.Series(vega, index=fpx.index, name="Price")
            return self.nominal * vega
        elif isinstance(self.underlying, tuple(underlying_dynamics_classes)):
            return MCProduct.get_vega(
                self,
                dates=dates,
                discount_curve=discount_curve,
                no_sims=100000,
                no_calcs=1,
            )
        else:  # This else is more important when import_underlying indata_factory_bd.py is used.
            raise ValueError(
                f"Underlying dynamics not supported. Try with {underlying_dynamics}"
            )  # List the right dynamics.

    def get_theta(
        self,
        dates,
        discount_curve,
        is_interest_rate_constant=False,
        no_sims=None,
        no_calcs=None,
        maturity_method=None,
        monte_carlo=None,
        constant_curve=None,
        refit=None,
    ):
        """
        Compute the theta of a put option for different underlying dynamics.
        Theta is the derivative of the option value to time.

        If the underlying has lognormal dynamics, the following analytic formula is applied

        .. math::
            \\Theta(S_t,t) = N \\cdot \\mathrm{e}^{- \\int^T_t r(s) \, \\mathrm{d}s}
            \\left( - \\frac{\\sigma(t) F_t \, \\mathrm{e}^{- q(t) (T - t)}}{2 \\sqrt{T-t}} N'(d_1) +
            K r(t) N(-d_2) - q(t) F_t \, \\mathrm{e}^{- q(t) (T - t)} N(-d_1) \\right)\,,

        where:

        - :math:`\\Theta(S_t,t)` is the theta of the put option,
        - :math:`N` is the nominal,
        - :math:`S_t` is the value of the underlying at time `t`,
        - :math:`K` is the strike price of the option,
        - :math:`T` is the maturity of the option,
        - :math:`r(t)` is the risk-free rate at time `t`,
        - :math:`\\sigma(t)` is the implied volatility of the underlying at time `t`,
        - :math:`q(t)` is the dividend yield payed by the underlying at time `t`,
        - :math:`D(t,T) = \\mathrm{e}^{- \\int^T_t r(s) \\, \\mathrm{d}s}` is the discount factor,
        - :math:`F = S_t \\, \\mathrm{e}^{\\int^T_t r(s) \\, \\mathrm{d}s}` is the value of future price of the underlying at time `t`,
        - :math:`d_1 \\equiv \\frac{\\ln\\frac{F_t}{K} + \\left( \\frac{1}{2}\\sigma(t)^2 - q(t) \\right)(T-t)}{\\sigma(t)\\sqrt{T-t}}` and :math:`d_2 \\equiv d_1 - \\sigma(t) \\sqrt{T-t}`,
        - :math:`N(x)` is the standard normal cumulative distribution function evaluated at :math:`x`,
        - :math:`N'(x) = \\frac{\\mathrm{e}^{-d_1^2/2}}{\\sqrt{2\\pi}}` is the standard normal probability distribution function evaluated at :math:`x`.

        If the underlying has normal dynamics and zero dividend yield, the following analytic formula is applied

        .. math::
            \\Theta(S_t,t) = N \\cdot \\mathrm{e}^{- \\int^T_t r(s) \, \\mathrm{d}s}
            \\left[ - \\frac{N'(x) (1 - 2r(t) (T-t))}{2 \\sqrt{T-t}} + K r(t) N(- d_1) \\right] \,,

        where:

        - :math:`\\Theta(S_t,t)` is the theta of the put option,
        - :math:`N` is the nominal,
        - :math:`S_t` is the value of the underlying at time `t`,
        - :math:`K` is the strike price of the option,
        - :math:`T` is the maturity of the option,
        - :math:`r(t)` is the risk-free rate at time `t`,
        - :math:`\\sigma(t)` is the implied volatility of the underlying at time `t`,
        - :math:`D(t,T) = \\mathrm{e}^{- \\int^T_t r(s) \\, \\mathrm{d}s}` is the discount factor,
        - :math:`F_t = S_t \\, \\mathrm{e}^{\\int^T_t r(s) \\, \\mathrm{d}s}` is the value of future price of the underlying at time `t`,
        - :math:`d_1 \\equiv \\frac{F_t-K}{\\sigma(t) \\sqrt{T-t}}`,
        - :math:`N(x)` is the standard normal cumulative distribution function evaluated at :math:`x`.
        - :math:`N'(x) = \\frac{\\mathrm{e}^{-d_1^2/2}}{\\sqrt{2\\pi}}` is the standard normal probability distribution function evaluated at :math:`x`.

        In the special case :math:`N=K`, the usual expressions for the theta are recovered.

        For other implemented underlying dynamics (NormalAsset with dividend yield, MultiAsset,
        ExposureIndex, Heston, SABR, MultiAssetHeston), the theta of the put is generated
        via Montecarlo simulation.

        If an underlying with a non implemented dynamics is used, a ValueError raises.

        This method overrides the parent class's one, but ignores the ``no_sims``, ``no_calcs``, and ``monte_carlo`` parameters.

        Parameters
        ----------
        dates : pandas.DatetimeIndex, list of strings, pandas.Timestamp, or string
            Dates at which theta is calculated. It can be a single date (as a pandas.Timestamp or its string representation) or an array-like object (as a pandas.DatetimeIndex or a
            list of its string representation) containing dates.
        discount_curve : pricing.discount_curves.DiscountCurve
            Provides the discounts for the computation of the theta.
        is_interest_rate_constant : bool
            True if interest rate is constant, False otherwise (default is False). When the boolean is True,
            the interest rate is computed as :math:`r = -\\frac{\\ln D(t,T)}{T-t}`.
            Otherwise, :math:`r(t) = \\frac{1}{D(t,T)} \\frac{\\partial D(t,T)}{\\partial t}`
            and the finite difference approximation is exploited to compute the derivative of `D(t,T)`.
        no_sims : int, optional
            Ignored in this implementation (defaults to None).
        no_calcs : int, optional
            Ignored in this implementation (defaults to None).
        monte_carlo : bool, optional
            Ignored in this implementation (defaults to None).

        Returns
        -------
        float
            The theta of the put option for implemented underlying dynamics.

        Raises
        ----------
        ValueError
            If the underlying dynamics is not implemented.

        References
        ----------
        - [J. C. Hull, 2021] Hull, J. C. (2021). Options, Futures, and Other Derivatives, 11th Edition.

        - [K. Iwasawa, 2001] Iwasawa, K. (2001) Analytic Formula for the European Normal Black-Scholes Formula.

        See Also
        ----------
        structured.Call.get_theta
        """
        if type(self.underlying) is list:
            print("Method not available for baskets; try Monte Carlo instead.")
            return None
        if isinstance(self.underlying, LognormalAsset):
            strike = self.strike
            fpx, d1, d2, discount, vol = Vanilla.get_px(
                self, dates=dates, discount_curve=discount_curve
            )
            original_dates = dates
            dates = fpx.index
            q = self.underlying.get_divrate(dates=dates).values
            px = self.underlying.get_value(dates=dates)
            tau = self.calendar.interval(dates=dates, future_dates=self.maturity)

            if is_interest_rate_constant:
                r = -np.log(discount) / tau
            else:
                dates_p = pd.to_datetime(original_dates) + pd.tseries.offsets.BDay()
                fpx_p, d1_p, d2_p, discount_p, vol_p = Vanilla.get_px(
                    self, dates=dates_p, discount_curve=discount_curve
                )

                dates_m = pd.to_datetime(original_dates) - pd.tseries.offsets.BDay()
                fpx_m, d1_m, d2_m, discount_m, vol_m = Vanilla.get_px(
                    self, dates=dates_m, discount_curve=discount_curve
                )

                time_int_mat = self.calendar.interval(dates_m, dates_p)
                deriv_df = (discount_p - discount_m) / time_int_mat
                r = deriv_df / discount

            # Vanilla.get_px already adjusts data by alpha
            theta = (
                -np.exp(-q * tau) * px * self.f(d1) * vol / (2 * np.sqrt(tau))
                + strike * discount * self.N(-d2) * r
                - q * px * np.exp(-q * tau) * self.N(-d1)
            )
            return self.nominal * theta
        elif isinstance(self.underlying, NormalAsset) and not self.underlying.yieldsdiv:
            fpx, vol, d, discount = Vanilla.get_px(
                self, dates=dates, discount_curve=discount_curve
            )
            # This can be found in "Analytic Formula for the European Normal Black-Scholes Formula" by K. Iwasawa.
            original_dates = dates
            dates = fpx.index
            tau = self.calendar.interval(dates=dates, future_dates=self.maturity)

            if is_interest_rate_constant:
                r = -np.log(discount) / tau
            else:
                dates_p = pd.to_datetime(original_dates) + pd.tseries.offsets.BDay()
                fpx_p, vol_p, d_p, discount_p = Vanilla.get_px(
                    self, dates=dates_p, discount_curve=discount_curve
                )

                dates_m = pd.to_datetime(original_dates) - pd.tseries.offsets.BDay()
                fpx_m, vol_m, d_m, discount_m = Vanilla.get_px(
                    self, dates=dates_m, discount_curve=discount_curve
                )

                time_int_mat = self.calendar.interval(dates_m, dates_p)
                deriv_df = (discount_p - discount_m) / time_int_mat
                r = deriv_df / discount

            theta = discount * (
                -self.f(d) * (1 - 2 * r * tau) * vol / (2 * tau)
                + self.strike * r * self.N(-d)
            )
            return self.nominal * theta
        elif isinstance(self.underlying, tuple(underlying_dynamics_classes)):
            return MCProduct.get_theta(
                self,
                dates=dates,
                discount_curve=discount_curve,
                no_sims=100000,
                no_calcs=1,
            )
        else:  # This else is more important when import_underlying indata_factory_bd.py is used.
            raise ValueError(
                f"Underlying dynamics not supported. Try with {underlying_dynamics}"
            )  # List the right dynamics.

    def get_rho(
        self, dates, discount_curve, no_sims=None, no_calcs=None, monte_carlo=None
    ):
        """
        Compute the rho of a put option for different underlying dynamics.
        Rho is the derivative of the option value to the interest rate.

        If the underlying has lognormal dynamics, the following analytic formula is applied

        .. math::
            \\rho(S_t,t) = - N \\cdot K(T-t) \\mathrm{e}^{- \\int^T_t r(s) \, \\mathrm{d}s} N(- d_2) \,,

        where:

        - :math:`\\rho(S_t,t)` is the rho of the put option,
        - :math:`N` is the nominal,
        - :math:`S_t` is the value of the underlying at time `t`,
        - :math:`K` is the strike price of the option,
        - :math:`T` is the maturity of the option,
        - :math:`r(t)` is the risk-free rate at time `t`,
        - :math:`\\sigma(t)` is the implied volatility of the underlying at time `t`,
        - :math:`D(t,T) = \\mathrm{e}^{- \\int^T_t r(s) \\, \\mathrm{d}s}` is the discount factor,
        - :math:`q(t)` is the dividend yield payed by the underlying at time `t`,
        - :math:`F_t = S_t \\, \\mathrm{e}^{\\int^T_t r(s) \\, \\mathrm{d}s}` is the value of future price of the underlying at time `t`,
        - :math:`d_2 \\equiv \\frac{\\ln\\frac{F_t}{K} - \\left( \\frac{1}{2}\\sigma(t)^2 + q(t) \\right) (T-t)}{\\sigma(t)\\sqrt{T-t}}`,
        - :math:`N(x)` is the standard normal cumulative distribution function evaluated at :math:`x`.

        If the underlying has normal dynamics and zero dividend yield, the following analytic formula is applied

        .. math::
            \\rho(S_t,t) = - N \\cdot (T-t) \\mathrm{e}^{- \\int^T_t r(s) \, \\mathrm{d}s}
            \\left( K N(- d_1) + \\sigma(t) \\sqrt{T-t} \, N'(d_1) \\right) \,,

        where:

        - :math:`\\rho(S_t,t)` is the rho of the put option,
        - :math:`N` is the nominal,
        - :math:`S_t` is the value of the underlying at time `t`,
        - :math:`K` is the strike price of the option,
        - :math:`T` is the maturity of the option,
        - :math:`r(t)` is the risk-free rate at time `t`,
        - :math:`\\sigma(t)` is the implied volatility of the underlying at time `t`,
        - :math:`D(t,T) = \\mathrm{e}^{- \\int^T_t r(s) \\, \\mathrm{d}s}` is the discount factor,
        - :math:`F_t = S_t \\, \\mathrm{e}^{\\int^T_t r(s) \\, \\mathrm{d}s}` is the value of future price of the underlying at time `t`,
        - :math:`d_1 \\equiv \\frac{F_t-K}{\\sigma(t)\\sqrt{T-t}}`,
        - :math:`N(x)` is the standard normal cumulative distribution function evaluated at :math:`x`,
        - :math:`N'(x) = \\frac{\\mathrm{e}^{-d_1^2/2}}{\\sqrt{2\\pi}}` is the standard normal probability distribution function evaluated at :math:`x`.

        In the special case :math:`N=K`, the usual expressions for the rho are recovered.

        For other implemented underlying dynamics (NormalAsset with dividend yield, MultiAsset,
        ExposureIndex, Heston, SABR, MultiAssetHeston), the rho of the put is generated
        via Montecarlo simulation.

        If an underlying with a non implemented dynamics is used, a ValueError raises.

        This method overrides the parent class's one, but ignores the ``no_sims``, ``no_calcs``, and ``monte_carlo`` parameters.

        Parameters
        ----------
        dates : pandas.DatetimeIndex, list of strings, pandas.Timestamp, or string
            Dates at which rho is calculated. It can be a single date (as a pandas.Timestamp or its string representation) or an array-like object (as a pandas.DatetimeIndex or a
            list of its string representation) containing dates.
        discount_curve : pricing.discount_curves.DiscountCurve
            Provides the discounts for the computation of the rho.
        no_sims : int, optional
            Ignored in this implementation (defaults to None).
        no_calcs : int, optional
            Ignored in this implementation (defaults to None).
        monte_carlo : bool, optional
            Ignored in this implementation (defaults to None).

        Returns
        -------
        float
            The rho of the put option for implemented underlying dynamics.

        Raises
        ----------
        ValueError
            If the underlying dynamics is not implemented.

        References
        ----------
        - [J. C. Hull, 2021] Hull, J. C. (2021). Options, Futures, and Other Derivatives, 11th Edition.

        - [K. Iwasawa, 2001] Iwasawa, K. (2001) Analytic Formula for the European Normal Black-Scholes Formula.

        See Also
        ----------
        structured.Call.get_rho
        """
        if type(self.underlying) is list:
            print("Method not available for baskets; try Monte Carlo instead.")
            return None
        if isinstance(self.underlying, LognormalAsset):
            fpx, d1, d2, discount, vol = Vanilla.get_px(
                self, dates=dates, discount_curve=discount_curve
            )
            tau = self.calendar.interval(dates=dates, future_dates=self.maturity)
            rho = -self.strike * discount * tau * self.N(-d2)
            rho = pd.Series(rho, index=fpx.index, name="Price")
            return self.nominal * rho
        elif isinstance(self.underlying, NormalAsset) and not self.underlying.yieldsdiv:
            fpx, vol, d, discount = Vanilla.get_px(
                self, dates=dates, discount_curve=discount_curve
            )
            tau = self.calendar.interval(dates=dates, future_dates=self.maturity)
            # This can be found in "Analytic Formula for the European Normal Black-Scholes Formula" by K. Iwasawa.
            rho = -discount * tau * (self.strike * self.N(-d) + vol * self.f(d))
            rho = pd.Series(rho, index=fpx.index, name="Price")
            return self.nominal * rho
        elif isinstance(self.underlying, tuple(underlying_dynamics_classes)):
            return MCProduct.get_vega(
                self,
                dates=dates,
                discount_curve=discount_curve,
                no_sims=100000,
                no_calcs=1,
            )
        else:  # This else is more important when import_underlying indata_factory_bd.py is used.
            raise ValueError(
                f"Underlying dynamics not supported. Try with {underlying_dynamics}"
            )  # List the right dynamics.

    def get_omega(self, dates, discount_curve):
        """
        See Also
        ----------
        structured.Call.get_omega
        """
        if type(self.underlying) is list:
            print("Method not available for baskets; try Monte Carlo instead.")
            return None
        if isinstance(self.underlying, LognormalAsset):
            px = self.underlying.get_value(dates)
            fpx, d1, d2, discount, vol = Vanilla.get_px(
                self, dates=dates, discount_curve=discount_curve
            )
            dates = fpx.index
            # Vanilla.get_px already adjusts data by alpha
            return (
                self.N(d1)
                * px
                / Put.get_px(self, dates=dates, discount_curve=discount_curve)
            )
        else:
            raise TypeError("Underlying dynamics not supported")

    def get_charm(self, dates, discount_curve):
        """
        See Also
        ----------
        structured.Call.get_charm
        """
        if type(self.underlying) is list:
            print("Method not available for baskets; try Monte Carlo instead.")
            return None
        if isinstance(self.underlying, LognormalAsset):
            fpx, d1, d2, discount, vol = Vanilla.get_px(
                self, dates=dates, discount_curve=discount_curve
            )
            dates = fpx.index
            q = self.underlying.get_divrate(dates=dates).values
            px = self.underlying.get_value(dates=dates)
            tau = self.calendar.interval(dates=dates, future_dates=self.maturity)
            r = -np.log(discount) / tau
            charm = -q * np.exp(-q * tau) * self.N(-d1) - np.exp(-q * tau) * self.f(
                d1
            ) * (2 * (r - q) * tau - d2 * vol * np.sqrt(tau)) / (
                2 * tau * vol * np.sqrt(tau)
            )
            return self.nominal * charm
        else:
            raise TypeError("Underlying dynamics not supported")

    def get_dual_delta(self, dates, discount_curve):
        """
        See Also
        ----------
        structured.Call.get_dual_delta
        """
        if type(self.underlying) is list:
            print("Method not available for baskets; try Monte Carlo instead.")
            return None
        if isinstance(self.underlying, LognormalAsset):
            fpx, d1, d2, discount, vol = Vanilla.get_px(
                self, dates=dates, discount_curve=discount_curve
            )
            dual_delta = discount * (1 - self.N(d2))
            return self.nominal * dual_delta
        else:
            raise TypeError("Underlying dynamics not supported")


class Digital(MCProduct, Derivative):
    # TODO: Write docstring
    def __init__(
        self,
        underlying,
        strike,
        maturity,
        kind,
        calendar,
        border=True,
        nominal=100,
        credit_curve=zero_credit_risk,
        implied_volatility=None,
    ):
        MCProduct.__init__(self)
        self.strike = np.array(strike)
        self.nominal = nominal
        maturity = pd.to_datetime(maturity)
        if kind == "call":
            Derivative.__init__(
                self,
                underlying=underlying,
                obsdates=[maturity],
                pastmatters=False,
                calendar=calendar,
                credit_curve=credit_curve,
                monotonicity_price_function="increasing",
            )
        elif kind == "put":
            Derivative.__init__(
                self,
                underlying=underlying,
                obsdates=[maturity],
                pastmatters=False,
                calendar=calendar,
                credit_curve=credit_curve,
                monotonicity_price_function="decreasing",
            )

        self.kind = kind
        self.border = border
        self.N = np.vectorize(norm.cdf)
        self.f = np.vectorize(norm.pdf)
        self.implied_volatility = implied_volatility

    def payoff(self, prices, dates_dic, n):
        payoffs_matrix = np.zeros(
            (prices.shape[0], prices.shape[1] - n, prices.shape[2])
        )
        index = dates_dic[self.maturity]
        prices = prices[:, index]
        if self.border:
            boolean = prices >= self.strike
        else:
            boolean = prices > self.strike
        payoffs = (boolean - 1 * (self.kind == "put")) * (-1) ** (self.kind == "put")
        if payoffs.ndim == 3:
            if type(self.underlying) is not list:
                payoffs = payoffs[:, :, 0]
            else:
                payoffs = np.min(payoffs, axis=-1)
        payoffs_matrix[:, index - n] = payoffs
        return self.nominal * payoffs_matrix


class Airbag(Call, Put, ZCBond):
    # Inheritance added so it appears nicely in the inheritance diagram
    """
    The airbag is constructed by taking the zero-coupon bond, adding the call option, and subtracting the put option.
    An additional cap call option may be considered.

    Parameters
    ----------
    underlying : str
        The underlying asset of the option.
    maturity : pandas.Timestamp
        The maturity date of the option.
    low_strike : float
        The strike price of the put option.
    main_strike : float
        The strike price of the call option.
    calendar : data.calendars.DayCountCalendar
        The calendar used for date calculations.
    cap_strike : float, optional
        The strike price of the cap call option (default is None).
    nominal : float, optional
        The nominal value of the option (default is 100).
    implied_volatility : float, optional
        The implied volatility to be assigned to the airbag.
        This attribute is necessary to compute the VaR. If None, the value of
        implied volatility is obtained from underlying's data.
        Otherwise, the input value is accepted as implied volatility of the option (default is None).

    Returns
    -------
    numpy.array
        An airbag option position.
        
    Notes
    -------
    The price function of an airbag is an increasing one, hence the attribute \
    `monotonicity_price_function` is set to be `increasing`. \
    Both `monotonicity_price_function` and `ìmplied_volatility` attributes are beneficial \
    for calculating risk metrics, such as VaR.
    """

    def __new__(
        cls,
        underlying,
        maturity,
        low_strike,
        main_strike,
        calendar,
        cap_strike=None,
        nominal=100,
        implied_volatility=None,
    ):
        b = ZCBond(maturity=maturity, calendar=calendar, nominal=nominal)
        c = Call(
            underlying=underlying,
            strike=main_strike,
            maturity=maturity,
            calendar=calendar,
            nominal=nominal,
        )
        p = Put(
            underlying=underlying,
            strike=low_strike,
            maturity=maturity,
            calendar=calendar,
            nominal=nominal,
        )
        airbag = (
            b + c - (1 / low_strike) * p
        )  # TODO: payoff does not admit, e.g., prices=prices
        if cap_strike is not None:
            c_cap = Call(
                underlying=underlying,
                strike=cap_strike,
                maturity=maturity,
                calendar=calendar,
                nominal=nominal,
            )
            airbag -= c_cap
        airbag.monotonicity_price_function = "increasing"
        # TODO: Find a smarter way to compute VaR
        airbag.implied_volatility = implied_volatility
        airbag.__class__.__name__ = "Airbag"
        return airbag


# ------------------------------------------------------------------------------------
# Quanto
# ------------------------------------------------------------------------------------

# This code is redundant: QuantoCall is the same as Call with QuantoEquity
# Keeping both versions because it is not clear which is preferable


class Quanto:
    def __init__(self, underlying, exchange_rate, maturity, calendar):
        # Works with multiple underlyings (as long as they come as MultiAsset) -- I think
        self.underlying = underlying
        self.exchange_rate = exchange_rate
        self.maturity = pd.to_datetime(maturity)
        self.calendar = calendar
        self.is_quanto = True
        # Derivative.__init__(self, underlying, [maturity], calendar, payoff)

    def compute_correction(self, dates):
        dates = afsfun.dates_formatting(dates)
        all_dates = self.underlying.get_dates()
        all_dates = all_dates.intersection(self.exchange_rate.get_dates())
        all_dates = all_dates[
            (all_dates >= dates[0] - pd.Timedelta(days=365)) & (all_dates < dates[-1])
        ]
        asset_returns = self.underlying.get_return(all_dates)
        xchange_returns = self.exchange_rate.get_return(all_dates)
        if hasattr(self.underlying, "components"):
            try:
                correlation = pd.DataFrame(
                    index=dates,
                    columns=[
                        self.underlying.components[i].ticker
                        for i in range(len(self.underlying.components))
                    ],
                    dtype="float64",
                )
            except:
                correlation = pd.DataFrame(
                    index=dates,
                    columns=[
                        self.underlying.components[i].name
                        for i in range(len(self.underlying.components))
                    ],
                    dtype="float64",
                )
        else:
            correlation = pd.Series(index=dates, dtype="float64")

        for date in dates:
            window = all_dates[:date][-250:]
            if window.size <= 125:
                print(
                    "Eliminating {}: not enough price data for historical correlations".format(
                        date.strftime("%Y-%m-%d")
                    )
                )
                return None
            temp_asset_returns = asset_returns.loc[window]
            temp_xchange_returns = xchange_returns.loc[window]
            if hasattr(self.underlying, "components"):
                correlation.loc[date] = temp_asset_returns.corrwith(
                    temp_xchange_returns
                )
        if hasattr(self, "obsdates"):
            tenors = np.full((self.obsdates.size, dates.size), np.nan)
            for i in range(dates.size):
                tenors[:, i] = self.calendar.interval(dates[i], self.obsdates)
            asset_vol = self.underlying.get_vol(dates=dates, maturities=self.obsdates)
            xchange_vol = self.exchange_rate.get_vol(
                dates=dates, maturities=self.obsdates
            )
        else:
            asset_vol = self.underlying.get_vol(dates=dates)
            xchange_vol = self.exchange_rate.get_vol(dates)
        correction = -correlation * asset_vol * xchange_vol
        return correction

    def get_underlying_fpx(self, dates, foreign_discount_curve):
        correction = self.compute_correction(dates)
        correction = np.exp(correction * self.calendar.interval(dates, self.maturity))
        dates = correction.index
        prices = self.underlying.get_value(dates)
        foreign_discount = foreign_discount_curve.get_value(
            dates, self.maturity, self.calendar
        )
        fpx = prices * correction / foreign_discount
        return fpx


class QuantoForward(Quanto, Forward):
    def __init__(
        self,
        underlying,
        exchange_rate,
        strike,
        maturity,
        calendar,
        nominal=100,
        credit_curve=zero_credit_risk,
    ):
        Quanto.__init__(self, underlying, exchange_rate, maturity, calendar)
        Forward.__init__(
            self,
            underlying,
            maturity,
            strike,
            calendar,
            nominal=nominal,
            credit_curve=credit_curve,
        )


class QuantoCall(Quanto, Call):
    def __init__(
        self,
        underlying,
        exchange_rate,
        strike,
        maturity,
        calendar,
        nominal=100,
        credit_curve=zero_credit_risk,
        alpha=0,
    ):
        Quanto.__init__(self, underlying, exchange_rate, maturity, calendar)
        Call.__init__(
            self,
            underlying,
            strike,
            maturity,
            calendar,
            nominal=nominal,
            credit_curve=credit_curve,
            alpha=alpha,
        )


class QuantoPut(Quanto, Put):
    def __init__(
        self,
        underlying,
        exchange_rate,
        strike,
        maturity,
        calendar,
        nominal=100,
        credit_curve=zero_credit_risk,
        alpha=0,
    ):
        Quanto.__init__(self, underlying, exchange_rate, maturity, calendar)
        Put.__init__(
            self,
            underlying,
            strike,
            maturity,
            calendar,
            nominal=nominal,
            credit_curve=credit_curve,
            alpha=alpha,
        )


class QuantoMultiAsset(
    Quanto
):  # Inheritance added so it appears nicely in the inheritance diagram
    def __init__(self, underlying, exchange_rate):
        self.underlying = underlying
        self.exchange_rate = exchange_rate
        self.is_quanto = True
        # Derivative.__init__(self, underlying, [maturity], calendar, payoff)

    def compute_correction(self, dates):
        dates = afsfun.dates_formatting(dates)
        correlation = pd.DataFrame(
            columns=np.arange(len(self.underlying)), index=dates, dtype="float64"
        )
        for date in dates:
            start_date = date - pd.Timedelta("360 days")
            window = pd.date_range(start=start_date, end=date, freq="1D")
            window = pd.DatetimeIndex.intersection(
                window, self.exchange_rate.get_dates()
            )
            for i in range(len(self.underlying.components)):
                underlying = self.underlying.components[i]
                temp_window = pd.DatetimeIndex.intersection(
                    window, underlying.get_dates()
                )
                if temp_window.size <= 60:
                    print(
                        "Eliminating {}: not enough price data for historical correlations".format(
                            date.strftime("%Y-%m-%d")
                        )
                    )
                    return None
                asset_prices = underlying.get_value(temp_window)
                xchange_prices = self.exchange_rate.get_value(temp_window)
                correlation[i].loc[date] = asset_prices.corr(xchange_prices)
        asset_vol = pd.DataFrame(
            columns=np.arange(len(self.underlying)), index=dates, dtype="float64"
        )
        for i in range(len(self.underlying)):
            underlying = self.underlying[i]
            asset_vol[i] = underlying.get_vol(dates)
        xchange_vol = self.exchange_rate.get_vol(dates)
        correction = (
            -correlation.values
            * asset_vol.values
            * xchange_vol.values.reshape((xchange_vol.values.size, 1))
        )
        return correction


# ------------------------------------------------------------------------------------
# Path dependents
# ------------------------------------------------------------------------------------

# When defining path dependent options, set self.pastmatters=True
# Monte Carlo will then provide the number of past dates and the discount factors at obs dates (in addition to prices)
# Write payoff accordingly

# ------------------------------------------------------------------------------------
# Asians
# ------------------------------------------------------------------------------------


class ArAsian(MCProduct, Derivative):
    def __init__(
        self,
        underlying,
        obsdates,
        strike,
        kind,
        calendar,
        nominal=100,
        credit_curve=zero_credit_risk,
        implied_volatility=None,
    ):
        MCProduct.__init__(self)
        self.strike = strike
        self.nominal = nominal
        obsdates = pd.to_datetime(obsdates).sort_values()
        if kind == "call":
            Derivative.__init__(
                self,
                underlying=underlying,
                obsdates=obsdates,
                pastmatters=True,
                calendar=calendar,
                credit_curve=credit_curve,
                monotonicity_price_function="increasing",
            )
        elif kind == "put":
            Derivative.__init__(
                self,
                underlying=underlying,
                obsdates=obsdates,
                pastmatters=True,
                calendar=calendar,
                credit_curve=credit_curve,
                monotonicity_price_function="decreasing",
            )

        self.kind = kind
        self.implied_volatility = implied_volatility

    def payoff(self, prices, dates_dic, n):
        payoffs_matrix = np.zeros(
            (prices.shape[0], prices.shape[1] - n, prices.shape[2])
        )
        indices = [dates_dic[date] for date in self.obsdates]
        prices = prices[:, indices, :]
        averages = np.mean(prices, axis=1)
        payoffs = (averages - self.strike) * (averages > self.strike) - (
            averages - self.strike
        ) * (self.kind == "put")
        if payoffs.ndim == 3:  # For four-dimensional prices.
            payoffs = np.min(payoffs, axis=-1)  # The price is defined as the minimum.
        index = dates_dic[self.maturity]
        payoffs_matrix[:, index - n] = payoffs
        return self.nominal * payoffs_matrix


class GAsian(MCProduct, Derivative):
    def __init__(
        self,
        underlying,
        strike,
        obsdates,
        kind,
        calendar,
        nominal=100,
        credit_curve=zero_credit_risk,
    ):
        MCProduct.__init__(self)
        self.strike = strike
        self.nominal = nominal
        Derivative.__init__(
            self,
            underlying=underlying,
            obsdates=obsdates,
            pastmatters=True,
            calendar=calendar,
            credit_curve=credit_curve,
        )

        self.kind = kind

    def payoff(self, prices, dates_dic, n):
        payoffs_matrix = np.zeros(
            (prices.shape[0], prices.shape[1] - n, prices.shape[2])
        )
        indices = [dates_dic[date] for date in self.obsdates]
        prices = prices[:, indices, :]
        strike = self.strike
        averages = gmean(prices, axis=1)
        payoffs = (averages - strike) * (averages > strike) - (averages - strike) * (
            self.kind == "put"
        )
        if payoffs.ndim == 3:
            payoffs = np.min(payoffs, axis=-1)
        index = dates_dic[self.maturity]
        payoffs_matrix[:, index - n] = payoffs
        return self.nominal * payoffs_matrix

    # TODO: implement analytic formulas


class Lookback(MCProduct, Derivative):
    """
    Class for instantiating an object which represents a financial product known as a Lookback Option.
    This class is inherited from two other classes, MCProduct and Derivative.

    Parameters
    ----------
    underlying : str
        The underlying asset's symbol.
    obsdates : pandas.DatetimeIndex or list
        Observation dates. If ``pastmatters = False`` we define maturity as the last element of observation dates.
    strike : float
        The strike price of the option.
    kind : str
        The type of option, either 'call' or 'put'.
    calendar : data.calendars.DayCountCalendar
        Calendar convention for counting days.
    nominal : float, optional
        Number of underlying shares for each call option contract (default is 100.0).
    credit_curve : pricing.discount_curves.CRDC
        The credit spread curve (default is zero_credit_risk = CRDC(0), namely there is no credit spread).
    alpha : float, optional
        Quantity added to the strike price (default is 0).
    """

    def __init__(
        self,
        underlying,
        obsdates,
        strike,
        kind,
        calendar,
        nominal=100,
        credit_curve=zero_credit_risk,
        alpha=0,
    ):
        MCProduct.__init__(self)
        self.strike = strike
        self.nominal = nominal
        Derivative.__init__(
            self,
            underlying=underlying,
            obsdates=obsdates,
            pastmatters=True,
            calendar=calendar,
            credit_curve=credit_curve,
        )
        self.alpha = alpha
        self.kind = kind
        self.N = np.vectorize(norm.cdf)
        self.f = np.vectorize(norm.pdf)

    def payoff(self, prices, dates_dic, n):
        """
        Payoff method.

        Warnings
        --------
        - Nominal convention is different in this class, the payoff is divided by strike.
        """
        payoffs_matrix = np.zeros(
            (prices.shape[0], prices.shape[1] - n, prices.shape[2])
        )
        indices = [dates_dic[date] for date in self.obsdates]
        index = dates_dic[self.maturity]
        prices = prices[:, indices, :]
        max_prices = np.max(prices, axis=1)
        min_prices = np.min(prices, axis=1)
        price = prices[:, index]
        if self.strike is not None:
            if self.kind == "call":
                payoffs = (max_prices / self.strike - 1) * (max_prices > self.strike)
            elif self.kind == "put":
                payoffs = -(min_prices / self.strike - 1) * (self.strike > min_prices)
            else:
                raise AttributeError("kind must be either call or put")
        else:
            if self.kind == "call":
                payoffs = price / min_prices - 1
            elif self.kind == "put":
                payoffs = -(price / max_prices - 1)
            else:
                raise AttributeError("kind must be either call or put")

        if payoffs.ndim == 3:  # For four-dimensional prices.
            payoffs = np.min(payoffs, axis=-1)  # The price is defined as the minimum.

        payoffs_matrix[:, index - n, :] = payoffs
        return self.nominal * payoffs_matrix

    @staticmethod
    def _delta(vol, tau, s, r):
        """
        Calculates the delta for a given volatility, time to maturity, stock price, and risk-free rate. These values represent the standardized distances of the current spot price and are used in the Black-Scholes formula to calculate the theoretical price of the lookback option.

        Parameters
        ----------
        vol : float
            The implied volatility of the underlying asset.
        tau : float
            The time to maturity of the option.
        s : float
            The stock price.
        r : float
            The risk-free interest rate.

        Returns
        ----------
        float
            The delta1 value calculated using the Black-Scholes formula.
        float
            The delta2 value calculated using the Black-Scholes formula.

        References
        ----------
        - [Privault, 2022] Privault, N. (2022). Notes on Stochastic Finance. Chapter 10: 388.
        """
        # Chapter 10: 388 of [Privault, 2022]
        delta1 = 1 / (vol * np.sqrt(tau)) * (np.log(s) + (r + 0.5 * vol**2) * tau)
        delta2 = 1 / (vol * np.sqrt(tau)) * (np.log(s) + (r - 0.5 * vol**2) * tau)
        return delta1, delta2

    def get_px(self, dates, discount_curve, no_sims=None, no_calcs=None):
        """
        Calculates the prices of European lookback call and put options based on the Black-Scholes formula. Overrides parent's method. This implementation ignores the
        ``no_sims``, ``no_calcs``, and ``monte_carlo`` parameters from the parent class.


        Parameters
        ----------
        dates : pandas.DatetimeIndex, list of strings, pandas.Timestamp, or string
            Valuation date.It can be a single date (as a pandas.Timestamp or its string representation) or an array-like object (as a pandas.DatetimeIndex or a list of its string
            representation) containing dates.
        discount_curve : pricing.discount_curves.DiscountCurve
            Gives the discounts for the computations.
        no_sims : int, optional
            Ignored in this implementation (defaults to None).
        no_calcs : int, optional
            Ignored in this implementation (defaults to None).

        Returns
        ----------
        list
            The prices of the options.

        Raises
        ----------
        - AttributeError: If the kind attribute is neither 'call' nor 'put' or if the option has a fixed strike.
        - NameError: If the underlying dynamics are not supported.

        References
        ----------
        - [Privault, 2022] Privault, N. (2022). Notes on Stochastic Finance.

        """
        # Chapter 12: 429–440 of [Privault, 2022]
        obs_dates = self.obsdates
        # noinspection PyShadowingNames
        px = self.underlying.get_value(dates)  # TODO: check this
        tau = self.calendar.interval(dates, self.maturity)
        vol = self.underlying.get_vol(dates=dates, tenors=tau, strike=self.strike)
        discount = discount_curve.get_value(dates, self.maturity, self.calendar)
        dates = px.index  # dates = pd.to_datetime(dates)
        # We need to set price = 0 by hand when tau = 0 (equivalently discount = 1)
        # mask = discount != 1
        # prices = np.empty_like(discount)
        # r = np.empty_like(discount)
        # r[mask] = -np.log(discount[mask]) / tau[mask]
        # prices[~mask] = 0
        # prices[mask] = ...
        r = -np.log(discount) / tau
        prices = []
        date_dict = {date: index for index, date in enumerate(dates)}
        for date in dates:
            i = date_dict[date]
            past_dates = obs_dates[obs_dates <= date]
            past_prices = self.underlying.get_value(past_dates)
            max_prices = np.max(past_prices)
            min_prices = np.min(past_prices)
            if type(self.underlying) is list:
                print("Method not available for baskets; try Monte Carlo instead.")
                return None
            elif isinstance(self.underlying, LognormalAsset):
                if self.strike is None:
                    if self.kind == "call":
                        delta1, delta2 = Lookback._delta(vol, tau, px / min_prices, r)
                        delta3, delta4 = Lookback._delta(vol, tau, min_prices / px, r)
                        price = (
                            self.nominal
                            / min_prices
                            * (
                                px[date] * self.N(delta1[date])
                                - min_prices * discount[i] * self.N(delta2[date])
                                + discount[i]
                                * px[date]
                                * vol[date] ** 2
                                / (2 * r[i])
                                * (min_prices / px[date]) ** (2 * r[i] / vol[date] ** 2)
                                * self.N(delta4[date])
                                - px[date]
                                * vol[date] ** 2
                                / (2 * r[i])
                                * self.N(-delta1[date])
                            )
                        )
                        prices.append(price)
                    elif self.kind == "put":
                        delta1, delta2 = Lookback._delta(vol, tau, px / max_prices, r)
                        delta3, delta4 = Lookback._delta(vol, tau, max_prices / px, r)
                        price = (
                            self.nominal
                            / max_prices
                            * (
                                max_prices * discount[i] * self.N(-delta2[date])
                                + px[date]
                                * (1 + vol[date] ** 2 / (2 * r[i]))
                                * self.N(delta1[date])
                                - px[date]
                                * discount[i]
                                * vol[date] ** 2
                                / (2 * r[i])
                                * (max_prices / px[date]) ** (2 * r[i] / vol[date] ** 2)
                                * self.N(-delta4[date])
                                - px[date]
                            )
                        )
                        prices.append(price)
                    else:
                        raise AttributeError("Kind must be either call or put")
                else:
                    raise AttributeError(
                        "Method not available for fixed strike; try Monte Carlo instead."
                    )
            else:
                raise NameError("Underlying dynamics not supported")
        prices = pd.Series(prices, index=dates)

        return prices


class BarrierOption(MCProduct, Derivative):
    def __init__(
        self,
        underlying,
        obsdates,
        strike,
        kind,
        barrier_kind,
        barrier,
        calendar,
        nominal=100,
        credit_curve=zero_credit_risk,
        multiassets_extremum_kind="max",
    ):
        obsdates = pd.to_datetime(obsdates).sort_values()
        MCProduct.__init__(self)
        Derivative.__init__(
            self,
            underlying=underlying,
            obsdates=obsdates,
            pastmatters=True,
            calendar=calendar,
            credit_curve=credit_curve,
        )

        self.strike = strike
        self.barrier = barrier
        self.barrier_kind = barrier_kind
        self.nominal = nominal
        self.kind = kind
        self.extremum_kind = multiassets_extremum_kind  # Function determining the payoff when there are several underlyings.

    def payoff(self, prices, dates_dic, n, *others):
        """
        Payoff method.

        Warnings
        --------
        - Nominal convention is different in this class, the payoff is divided by strike.
        """
        payoffs_matrix = np.zeros(
            (prices.shape[0], prices.shape[1] - n, prices.shape[2])
        )
        indices = [dates_dic[date] for date in self.obsdates]

        fin_index = dates_dic[self.maturity]  # Final date, index
        fin_prices = prices[:, fin_index]  # Final price array
        ind_bol = [
            i for i in indices if i != fin_index
        ]  # Relevant indices for boolean arguments
        prices_bol = prices[:, ind_bol]
        # Taking the extrema of multiple assets
        if self.extremum_kind == "max":
            fin_prices = np.max(prices[:, fin_index], axis=-1)
            prices_bol = np.max(prices[:, ind_bol], axis=-1)
        elif self.extremum_kind == "min":
            fin_prices = np.min(prices[:, fin_index], axis=-1)
            prices_bol = np.min(prices[:, ind_bol], axis=-1)
        else:
            raise NameError(f"The function {self.extremum_kind} is not implemented.")

        mins = np.min(prices_bol, axis=1)
        maxs = np.max(prices_bol, axis=1)

        prices = prices[:, indices]

        boolean = None  # Let us declare the boolean variable depending on the given attributes. Obviously, there is some redundancy, e.g., d-a-o is equivalent to u-a-i
        if self.barrier_kind == "down-and-out":
            boolean = mins > self.barrier
        if self.barrier_kind == "down-and-in":
            boolean = 1 - (mins > self.barrier)
        if self.barrier_kind == "up-and-in":
            boolean = maxs >= self.barrier
        if self.barrier_kind == "up-and-out":
            boolean = 1 - (maxs >= self.barrier)

        payoffs = (
            (fin_prices / self.strike - 1) * (fin_prices > self.strike)
            - (fin_prices / self.strike - 1) * (self.kind == "put")
        ) * boolean
        payoffs_matrix[:, fin_index - n, :] = payoffs.reshape(
            prices.shape[0], prices.shape[2]
        )  # Reshaping is needed

        return self.nominal * payoffs_matrix


# -----------------------------------------------------------------------------------
# Contingent payments
# -----------------------------------------------------------------------------------


class KnockOutContingentPayment(MCProduct, Derivative):
    def __init__(
        self,
        underlying,
        obsdates,
        coupon_rate,
        pay_barrier,
        calendar,
        ko_barrier=np.inf,
        min_time_pay=0,
        nominal=100,
        credit_curve=zero_credit_risk,
        min_time_redemption=0,
        pay_extremum_multi_asset="min",
        ko_extremum_multi_asset="max",
    ):
        self.coupon_rate = coupon_rate
        self.nominal = nominal
        self.pay_barrier = pay_barrier
        self.ko_barrier = ko_barrier  # As we see by default this is infinity, so the associated conditionals are never activated
        self.min_time_pay = min_time_pay  # min time payment can be made
        self.min_time_red = min_time_redemption  # min time for early redemption
        self.pay_ext = pay_extremum_multi_asset
        self.ko_ext = ko_extremum_multi_asset

        Derivative.__init__(
            self,
            underlying=underlying,
            obsdates=obsdates,
            pastmatters=True,
            calendar=calendar,
            credit_curve=credit_curve,
        )

    def payoff(self, prices, dates_dic, n, *others):
        payoff_matrix = np.zeros(
            (prices.shape[0], prices.shape[1] - n, prices.shape[2])
        )
        indices = [
            dates_dic[date] for date in self.obsdates
        ]  # Indices for the observation dates
        # We do not nec. have that n_small = n, dates_dic and obsdates might differ for some dates, being dates_dic larger
        previous_dates = pd.to_datetime(
            [date for date in dates_dic.keys()]
        ).sort_values()[:n]
        n_small = np.sum(
            self.obsdates.isin(previous_dates)
        )  # number of obs dates AND previous dates
        prices = prices[
            :, indices
        ]  # Now prices has the appropriate prices for our observation dates
        min_index = self.min_time_pay
        min_index_small = np.max(
            (min_index - n_small, 0)
        )  # Possibly negative, so it imposes no condition
        min_red_index = self.min_time_red
        min_red_index_small = np.max(
            (min_red_index, n_small)
        )  # If min_red_index is less than n_small, it imposes no condition
        payoff = np.full(
            (prices.shape[0], prices.shape[1] - n_small, prices.shape[2]), 0
        )

        def np_max(a, b):
            """
            Return numpy.max(a) unless a=[], when it returns b. Useful for empty boolean list that should be treated as a true or false. Otherwise numpy.max([]) raises a
            ValueError. Empty boolean arrays might appear while slicing. Equivalent syntax: numpy.max([] or [b]).
            ----------
            Parameters
            ----------
            a : ndarray
            b : float
            """
            try:
                result = np.max(a, axis=-1)
            except ValueError:
                result = b  # Result in case of empty a

            return result

        def np_ext(a, ext):
            """
            Returns numpy.extremum(a, axis=-1) where extremum is either max or min.
            ----------
            Parameters
            ----------
            a : ndarray
            ext : string. Either "max" or "min"
            """
            if ext == "max":
                result = np.max(a, axis=-1)
            elif ext == "min":
                result = np.min(a, axis=-1)
            else:
                raise ValueError

            return result

        for k in range(prices.shape[2]):
            pay_prices_prev = np_ext(prices[:, min_index:n_small, k], self.pay_ext)
            ko_prices_prev = np_ext(prices[:, min_red_index:n_small, k], self.ko_ext)
            boolean_pay_prev = np.array(
                (pay_prices_prev >= self.pay_barrier)
            )  # The payment was in the past
            boolean_ko_prev = np.array(
                (ko_prices_prev >= self.ko_barrier)
            )  # The knock-out was in the past
            boolean_ko_prev = np_max(
                boolean_ko_prev, 0
            )  # Maximum over the last axis (obs dates). If for one of the obs_dates boolean is True then the payoff=0.
            boolean_pay_prev = np_max(
                boolean_pay_prev, 0
            )  # Maximum over the last axis (obs dates). If for one of the obs_dates boolean is True then the payoff=0.
            if (
                pay_prices_prev.size == 0 and ko_prices_prev.size == 0
            ):  # if both are empty arrays.
                indices_false = np.arange(
                    prices.shape[0]
                )  # Simulations for which the payoff != 0
            elif pay_prices_prev.size == 0:
                indices_true = np.where(boolean_ko_prev == True)[
                    0
                ]  # Simulations for which the payoff = 0
                indices_false = np.where(boolean_ko_prev == False)[
                    0
                ]  # Simulations for which the payoff != 0
                payoff[indices_true, :, k] = 0
            elif ko_prices_prev.size == 0:
                indices_true = np.where(boolean_pay_prev == True)[
                    0
                ]  # Simulations for which the payoff = 0
                indices_false = np.where(boolean_pay_prev == False)[
                    0
                ]  # Simulations for which the payoff != 0
                payoff[indices_true, :, k] = 0
            else:
                # Note that if the boolean is empty, conditional should be false, 0
                boolean_total = np.logical_or(
                    boolean_ko_prev, boolean_pay_prev
                )  # For each simulation both (boolean_ko_prev and boolean_pay_prev) should be false for payoff != 0
                indices_true = np.where(boolean_total == True)[
                    0
                ]  # Simulations for which the payoff = 0
                indices_false = np.where(boolean_total == False)[
                    0
                ]  # Simulations for which the payoff != 0
            # Now we can forget about the past, so we subtract n_small
            pay_prices = np_ext(
                prices[indices_false, n_small:, k], self.pay_ext
            )  # Prices for future observation dates
            boolean = np.array(
                (pay_prices >= self.pay_barrier)
            )  # boolean for possible payments (surpass the payment barrier)
            boolean[:, :min_index_small] = (
                False  # All positions must be >= than the first date the payment can be made
            )
            position = boolean.argmax(
                -1
            )  # Position of the possible firsts future dates of payment
            # We select the prices for which the knock-out barrier could be surpassed before the first payment (given by position)
            ix = np.arange(prices.shape[1])[
                np.newaxis, :, np.newaxis
            ]  # Instead of a for loop we use advanced indexing.
            pos = position[:, np.newaxis, np.newaxis]
            ko_prices = np_ext(
                np.where(
                    (ix >= min_red_index_small) & (ix < pos + n_small),
                    prices[indices_false, :, k, :],
                    0,
                ),
                self.ko_ext,
            )  # Values not satisfying the condition
            # are set to 0, so they do not affect the following part of the code.
            max_boolean = np.max(boolean, axis=1)[:, np.newaxis]
            max_ko_prices = np.max(ko_prices, axis=1)[:, np.newaxis]
            mask = (max_boolean == 1) & (max_ko_prices < self.ko_barrier)
            payoff[indices_false, :, k] = np.where(
                mask, np.identity(pay_prices.shape[1])[position], 0
            )
            # Payment if we are in the proper corridor, 0 otherwise
        ind = np.array(indices[n_small:]) - n
        payoff_matrix[:, ind] = payoff
        # The slicing is such that the LHS has shape (payoff_matrix.shape(0), len(ind), payoff_matrix.shape(2))
        payoff = self.nominal * self.coupon_rate * payoff_matrix

        return payoff


# -----------------------------------------------------------------------------------
# Autocallable
# -----------------------------------------------------------------------------------


class AutocallableComp(
    KnockOutContingentPayment, BarrierOption
):  # Inheritance added so it appears nicely in the inheritance diagram
    # As the composition of simpler classes
    def __new__(
        cls,
        underlying,
        obsdates,
        coupon_rate,
        initial_level,
        early_barrier,
        capital_barrier,
        pay_barrier,
        calendar,
        nominal=100,
        credit_curve=zero_credit_risk,
        min_time_redemption=0,
        implied_volatility=None,
    ):
        # Coupons
        coupons = {
            0: KnockOutContingentPayment(
                underlying=underlying,
                obsdates=obsdates,
                coupon_rate=coupon_rate,
                pay_barrier=pay_barrier,
                calendar=calendar,
                ko_barrier=early_barrier,
                min_time_pay=0,
                nominal=nominal,
                credit_curve=credit_curve,
                min_time_redemption=min_time_redemption,
                pay_extremum_multi_asset="min",
                ko_extremum_multi_asset="min",
            )
        }  # Intermediate coupons as a dictionary
        total_coupon = coupons[0]
        for i in range(1, len(obsdates)):
            coupons[i] = KnockOutContingentPayment(
                underlying=underlying,
                obsdates=obsdates,
                coupon_rate=coupon_rate,
                pay_barrier=pay_barrier,
                calendar=calendar,
                ko_barrier=early_barrier,
                min_time_pay=i,
                nominal=nominal,
                credit_curve=credit_curve,
                min_time_redemption=min_time_redemption,
                pay_extremum_multi_asset="min",
                ko_extremum_multi_asset="min",
            )
            total_coupon += coupons[i]

        # Nominal
        cp = capital_barrier / initial_level
        cont_payment1 = KnockOutContingentPayment(
            underlying=underlying,
            obsdates=obsdates[:-1],
            coupon_rate=1,
            pay_barrier=early_barrier,
            calendar=calendar,
            ko_barrier=early_barrier,
            min_time_pay=min_time_redemption,
            nominal=nominal,
            credit_curve=credit_curve,
            pay_extremum_multi_asset="min",
            ko_extremum_multi_asset="min",
        )
        cont_payment2 = KnockOutContingentPayment(
            underlying=underlying,
            obsdates=obsdates,
            coupon_rate=1 - cp,
            pay_barrier=capital_barrier,
            calendar=calendar,
            ko_barrier=early_barrier,
            min_time_pay=len(obsdates) - 1,
            nominal=nominal,
            credit_curve=credit_curve,
            min_time_redemption=min_time_redemption,
            pay_extremum_multi_asset="min",
            ko_extremum_multi_asset="min",
        )
        cont_payment3 = KnockOutContingentPayment(
            underlying=underlying,
            obsdates=obsdates,
            coupon_rate=cp,
            pay_barrier=0,
            calendar=calendar,
            ko_barrier=early_barrier,
            min_time_pay=len(obsdates) - 1,
            nominal=nominal,
            credit_curve=credit_curve,
            min_time_redemption=min_time_redemption,
            pay_extremum_multi_asset="min",
            ko_extremum_multi_asset="min",
        )
        put_uao_barrier = BarrierOption(
            underlying=underlying,
            obsdates=obsdates[min_time_redemption:],
            strike=initial_level * cp,
            kind="put",
            barrier_kind="up-and-out",
            barrier=early_barrier,
            calendar=calendar,
            nominal=nominal * cp,
            credit_curve=credit_curve,
            multiassets_extremum_kind="min",
        )

        nominal = cont_payment1 + cont_payment2 + cont_payment3 - put_uao_barrier

        total = nominal + total_coupon
        total.monotonicity_price_function = "increasing"
        total.implied_volatility = implied_volatility
        total.__class__.__name__ = "Autocallable"
        return total


class AutocallableSingleAsset(MCProduct, Derivative):
    def __init__(
        self,
        underlying,
        pay_dates,
        coupon_rate,
        autocall,
        interest_strike,
        pay_strike,
        knock_in,
        calendar,
        nominal=100,
        credit_curve=zero_credit_risk,
    ):
        MCProduct.__init__(self)
        # saving specific data
        # We save pay_dates as its own attribute because AveraginAutocallable changes it below
        self.pay_dates = afsfun.dates_formatting(pay_dates)
        Derivative.__init__(
            self,
            underlying=underlying,
            obsdates=self.pay_dates,
            pastmatters=True,
            calendar=calendar,
            credit_curve=credit_curve,
        )

        if autocall is None:
            self.autocall = np.full(self.pay_dates.size, np.inf)
        elif np.asarray(autocall).shape == ():
            self.autocall = autocall * np.ones(self.pay_dates.size)
        else:
            self.autocall = np.array(autocall)
        if np.asarray(interest_strike).shape == ():
            self.interest = interest_strike * np.ones(self.pay_dates.size)
        else:
            self.interest = np.array(interest_strike)
        if np.asarray(coupon_rate).shape == ():
            self.coupon_rate = coupon_rate * np.ones(self.pay_dates.size)
        else:
            self.coupon_rate = np.array(coupon_rate)
        self.strike = pay_strike
        self.knock_in = knock_in
        self.nominal = nominal

        # define payoff

    def payoff(self, matrix_simulations, dates_dic, n):
        """
        Payoff method.
        
        Warnings
        --------
        - Nominal convention is different in this class, the payoff is divided by strike.
        """
        payoffs_matrix = np.zeros(
            (
                matrix_simulations.shape[0],
                matrix_simulations.shape[1] - n,
                matrix_simulations.shape[2],
            )
        )
        if matrix_simulations.ndim == 4:
            if matrix_simulations.shape[3] > 1:
                print("Simulations for multiple arrays passed: use AutocallableBasket")
                return np.nan
            else:
                matrix_simulations = matrix_simulations[:, :, :, 0]
        # MUST USE SELF.PAY_DATES; SELF.OBSDATES DOESN'T WORK BECAUSE AVERAGINGAUTOCALLABLE CHANGES IT BELOW
        indices = [dates_dic[date] for date in self.pay_dates]
        matrix_simulations = matrix_simulations[:, indices, :]
        # las proximas lineas necesarias porque podemos aplicar MC directamente a sumas de productos
        previous_dates = pd.to_datetime(
            [date for date in dates_dic.keys()]
        ).sort_values()[:n]
        n_small = np.sum(self.obsdates.isin(previous_dates))

        no_dates = matrix_simulations.shape[2]
        no_sims = matrix_simulations.shape[0]
        payoffs = np.full(
            (
                matrix_simulations.shape[0],
                matrix_simulations.shape[1] - n_small,
                matrix_simulations.shape[2],
            ),
            np.nan,
        )
        for d in range(no_dates):
            # the following just checks if the simulation gave nans (for lack of data)
            no_null = np.sum(matrix_simulations[0, :, d]) == np.sum(
                matrix_simulations[0, :, d]
            )
            if not no_null:
                continue
            # boolean checks if it was autocalled:
            if n_small != 0:
                past_obs = matrix_simulations[1, :n_small, d]
                boolean = np.max(past_obs >= self.autocall[:n_small])
            else:
                boolean = False

            if boolean:
                continue
            else:
                for m in range(no_sims):
                    sim = matrix_simulations[m, :, d]
                    if np.max(sim >= self.autocall):
                        ab = np.argmax(sim >= self.autocall)
                        sim = sim[: ab + 1]
                    else:
                        ab = sim.size - 1
                    if n_small != 0:
                        past = sim[:n_small]
                        if np.max((past >= self.interest[:n_small])):
                            open_periods = np.argmax(
                                np.flip((past >= self.interest[:n_small]))
                            )
                        else:
                            open_periods = past.size
                        outstanding_coupon = np.sum(
                            self.coupon_rate[n_small - open_periods : n_small]
                        )
                        future = sim[n_small:]
                    else:
                        outstanding_coupon = 0
                        future = sim
                    pays_interest = (
                        future >= self.interest[n_small : n_small + future.shape[0]]
                    )
                    positions = np.arange(future.size)
                    positions = positions[pays_interest]
                    if positions.size == 0:
                        coupons = 0
                    else:
                        coupons = np.zeros(payoffs.shape[1])
                        coupon_rate = self.coupon_rate[n_small:]
                        coupons[positions[0]] = outstanding_coupon + np.sum(
                            coupon_rate[: positions[0] + 1]
                        )
                        for i in range(positions.size - 1):
                            # the +1 below is because python starts indexing at 0
                            coupons[positions[i + 1]] = np.sum(
                                coupon_rate[positions[i] + 1 : positions[i + 1] + 1]
                            )
                    # computing final amount
                    final = sim[-1]
                    final = 1 - (1 - final / self.strike) * (final <= self.knock_in)
                    final = final * np.identity(payoffs.shape[1])[ab - n_small]
                    payoffs[m, :, d] = coupons + final
        payoffs_matrix[:, np.array(indices[n_small:]) - n, :] = payoffs
        return self.nominal * payoffs_matrix

    def get_px_analytic(self, dates, discount_curve):
        # CHECK IF IT WAS AUTOCALLED
        # CREDIT
        dates = afsfun.dates_formatting(dates)
        for date in dates:
            print(date)
            obs_past = self.obsdates[self.obsdates <= date]
            px_max_past = np.max(self.underlying.get_value(obs_past))
            if px_max_past < self.autocall:
                continue
            else:
                eliminated_dates = dates[dates >= date]
                if eliminated_dates.size != 0:
                    print("Eliminating dates past autocall event:", eliminated_dates)
                dates = dates[dates < date]
                break
        if dates.size == 0:
            print("All dates past autocall event.")
            return None
        prices = self.underlying.get_value(dates=dates)
        volatilities = self.underlying.get_vol(dates=dates)
        divrates = self.underlying.get_divrate(dates=dates)
        for date in dates:
            px = prices.loc[date]
            vol = volatilities.loc[date]
            div = divrates.loc[date]
            obsdates = self.obsdates[self.obsdates > date]
            discounts = discount_curve.get_value(date, obsdates, self.calendar)
            taus = self.calendar.interval(date, obsdates)
            covariance = np.zeros((obsdates.size, obsdates.size))
            for i in range(obsdates.size):
                covariance[i:, i:] = taus[i]
            covariance = covariance / np.sqrt(np.einsum("i,j", taus, taus))
            # print(covariance)
            rates = -np.log(discounts) / taus
            d_ac = (
                np.log(self.autocall / px) - (rates - div - vol**2 / 2) * taus
            ) / (vol * np.sqrt(taus))
            d_int = (
                np.log(self.interest / px) - (rates - div - vol**2 / 2) * taus
            ) / (vol * np.sqrt(taus))
            # PENDING COUPONS!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            probabilities = []
            autocall_amounts = 0
            coupons = 0
            for i in range(1, obsdates.size + 1):
                print(i, "-------------------------")
                x1 = np.ones(i)
                x1[-1] = -1
                x2 = x1.reshape((x1.size, 1))
                cov1 = x1 * x2 * np.copy(covariance[:i, :i])
                # print(cov1)
                d = np.copy(d_int[:i])
                d[-1] = -d[-1]  # -d_int[i-1]
                y = multivariate_normal(cov=cov1)
                prob = y.cdf(d)
                # print(prob)
                probabilities.append(y.cdf(d))
                weights = i * y.cdf(d)
                d = np.copy(d_ac[:i])
                d[-1] = -d[-1]
                autocall_amounts = autocall_amounts + discounts[i - 1] * y.cdf(d)
                # print(autocall_amounts)
                for j in range(i - 1):
                    # print(j)
                    d1 = np.zeros(i)
                    d2 = np.zeros(i)
                    d1[: j + 1] = d_ac[: j + 1]
                    d1[j + 1 : -1] = d_int[j + 1 : i - 1]
                    d1[-1] = -d_int[i - 1]
                    d2[:j] = d_ac[:j]
                    d2[j:] = d_int[j:i]
                    d2[-1] = -d2[-1]
                    d2[j] = -d2[j]
                    # print(d_ac)
                    # print(d_int)
                    # print(d1)
                    # print(d2)
                    # x1 = np.ones(i)
                    # x1[j] = -1
                    # x2 = x1.reshape((x1.size, 1))
                    # cov2 = x1*x2*cov1
                    # print(cov2)
                    # y1 = multivariate_normal(cov=cov1)
                    # print(y.cdf(d1)-y.cdf(d2))
                    # probabilities.append(y.cdf(d1)-y1.cdf(d2))
                    # weights = weights + (i-j)*(y.cdf(d1)-y.cdf(d2))
                coupons = coupons + discounts[i - 1] * self.coupon_rate * weights
            print("Loop done")
            print("Coupons:", coupons)
            print("Autocall:", autocall_amounts)
            y = multivariate_normal(cov=covariance)
            d = np.copy(d_ac)
            d[-1] = d_int[-1]
            mid_range = y.cdf(d_ac) - y.cdf(d)
            d = (np.log(self.autocall / px) - (rates - div + vol**2 / 2) * taus) / (
                vol * np.sqrt(taus)
            )
            d[-1] = (
                (np.log(self.interest / px) - (rates - div + vol**2 / 2) * taus)
                / (vol * np.sqrt(taus))
            )[-1]
            low_range = np.exp((rates - div) * taus)[-1] * px / self.strike * y.cdf(d)
            final_nominal = discounts[-1] * (mid_range + low_range)
            print("Final nominal", final_nominal)
            prices = coupons + autocall_amounts + final_nominal
            print("Price:", prices)
            # print(probabilities)
            # print(prices)
            # print(np.sum(np.array(probabilities)))
        return None


class AveragingAutocallable(AutocallableSingleAsset):
    def __init__(
        self,
        underlying,
        pay_dates,
        no_averaging_days,
        coupon_rate,
        autocall,
        interest_strike,
        pay_strike,
        knock_in,
        calendar,
        nominal=100,
        credit_curve=zero_credit_risk,
    ):
        self.no_averaging_days = no_averaging_days
        AutocallableSingleAsset.__init__(
            self,
            underlying,
            pay_dates,
            coupon_rate,
            autocall,
            interest_strike,
            pay_strike,
            knock_in,
            calendar,
            nominal=nominal,
            credit_curve=credit_curve,
        )
        obsdates_full = pay_dates[0] - pd.timedelta_range(
            start="0", periods=self.no_averaging_days, freq="1D"
        )
        for date in pay_dates[1:]:
            other = date - pd.timedelta_range(
                start="0", periods=self.no_averaging_days, freq="1D"
            )
            obsdates_full = pd.DatetimeIndex.union(obsdates_full, other)
        self.obsdates = obsdates_full

        # changing payoff functions
        self.short_payoff = self.payoff

        def payoff(matrix_simulations, dates_dic, n):
            if matrix_simulations.ndim == 4:
                matrix_simulations = matrix_simulations[:, :, :, 0]
            indices = [dates_dic[date] for date in self.obsdates]
            matrix_simulations = matrix_simulations[:, indices, :]
            shape = list(matrix_simulations.shape)
            shape[1] = int(shape[1] / self.no_averaging_days)
            averaged_simulations = np.zeros(shape)
            for i in range(shape[1]):
                l = i * self.no_averaging_days
                averaged_simulations[:, i, :] = np.mean(
                    matrix_simulations[:, l : l + self.no_averaging_days, :], axis=1
                )

            disc_indices = np.array([dates_dic[date] for date in self.pay_dates]) - n
            discounts_reduced = discounts[disc_indices]
            dates_dic_reduced = {}
            for date in self.pay_dates:
                dates_dic_reduced[date] = int(dates_dic[date] / self.no_averaging_days)
            payoffs = self.short_payoff(
                averaged_simulations, int(n / 3), discounts_reduced, dates_dic_reduced
            )
            return self.nominal * payoffs

        self.payoff = payoff


class Autocallable(MCProduct, Derivative):
    def __init__(
        self,
        underlyings,
        pay_dates,
        coupon_rate,
        autocall,
        interest_strike,
        pay_strike,
        knock_in,
        calendar,
        nominal=100,
        credit_curve=zero_credit_risk,
    ):
        MCProduct.__init__(self)
        # saving specific data
        if np.asarray(pay_dates).shape == ():
            pay_dates = [pay_dates]
        # We save pay_dates as its own attribute because AveraginAutocallable changes it below
        self.pay_dates = pd.to_datetime(pay_dates)

        # initialize derivatives class
        Derivative.__init__(
            self,
            underlying=underlyings,
            obsdates=self.pay_dates,
            pastmatters=True,
            calendar=calendar,
            credit_curve=credit_curve,
        )

        if hasattr(self.underlying, "components"):
            no_underlyings = len(underlyings.components)
        else:
            no_underlyings = 1
        if autocall is None:
            autocall = np.inf
        autocall = np.asarray(autocall)
        if autocall.ndim == 0 or autocall.shape == (no_underlyings,):
            self.autocall = autocall * np.ones((self.pay_dates.size, no_underlyings))
        elif autocall.shape == (self.pay_dates.size,):
            self.autocall = autocall.reshape(autocall.size, 1) * np.ones(
                (self.pay_dates.size, no_underlyings)
            )
        elif autocall.shape == (self.pay_dates.size, no_underlyings):
            self.autocall = np.array(autocall)
        else:
            print("Autocall barrier array dimensions do not make sense; aborting")
            raise ValueError

        interest_strike = np.asarray(interest_strike)
        if interest_strike.ndim == 0 or interest_strike.shape == (no_underlyings,):
            self.interest = interest_strike * np.ones(
                (self.pay_dates.size, no_underlyings)
            )
        elif interest_strike.shape == (self.pay_dates.size,):
            self.interest = interest_strike.reshape(interest_strike.size, 1) * np.ones(
                (self.pay_dates.size, no_underlyings)
            )
        elif autocall.shape == (self.pay_dates.size, no_underlyings):
            self.interest = np.array(interest_strike)
        else:
            print("Interest barrier array dimensions do not make sense; aborting")
            raise ValueError

        coupon_rate = np.asarray(coupon_rate)
        if coupon_rate.ndim == 0:
            self.coupon_rate = coupon_rate * np.ones((self.pay_dates.size,))
        elif coupon_rate.shape == (self.pay_dates.size,):
            self.coupon_rate = coupon_rate
        else:
            print("Coupon rate array dimensions do not make sense; aborting")
            raise ValueError

        pay_strike = np.asarray(pay_strike)
        if pay_strike.ndim == 0:
            self.strike = pay_strike * np.ones(no_underlyings)
        elif pay_strike.shape == (no_underlyings,):
            self.strike = pay_strike
        else:
            print("Put strike not conforming; aborting")
            raise ValueError

        knock_in = np.asarray(knock_in)
        if knock_in.ndim == 0:
            self.knock_in = knock_in * np.ones(no_underlyings)
        elif pay_strike.shape == (no_underlyings,):
            self.knock_in = knock_in
        else:
            print("Knock-in level array not conforming; aborting")
            raise ValueError
        self.nominal = nominal

        # define payoff

    def payoff(self, matrix_simulations, dates_dic, n):
        """
        Payoff method.
        
        Warnings
        --------
        - Nominal convention is different in this class, the payoff is divided by strike.
        """
        payoffs_matrix = np.zeros(
            (
                matrix_simulations.shape[0],
                matrix_simulations.shape[1] - n,
                matrix_simulations.shape[2],
            )
        )
        # MUST USE SELF.PAY_DATES; SELF.OBSDATES DOESN'T WORK BECAUSE AVERAGINGAUTOCALLABLE CHANGES IT BELOW
        indices = [dates_dic[date] for date in self.pay_dates]
        matrix_simulations = matrix_simulations[:, indices, :]
        # las proximas lineas necesarias porque podemos aplicar MC directamente a sumas de productos
        previous_dates = pd.to_datetime(
            [date for date in dates_dic.keys()]
        ).sort_values()[:n]
        n_small = np.sum(self.obsdates.isin(previous_dates))

        no_dates = matrix_simulations.shape[2]
        no_sims = matrix_simulations.shape[0]
        payoffs = np.full(
            (
                matrix_simulations.shape[0],
                matrix_simulations.shape[1] - n_small,
                matrix_simulations.shape[2],
            ),
            np.nan,
        )
        for d in range(no_dates):
            # the following just checks if the simulation gave nans (for lack of data)
            no_null = np.sum(matrix_simulations[0, :, d]) == np.sum(
                matrix_simulations[0, :, d]
            )
            if not no_null:
                continue
            # boolean checks if it was autocalled:
            if n_small != 0:
                past_obs = matrix_simulations[1, :n_small, d]
                boolean = np.max(past_obs >= self.autocall[:n_small])
            else:
                boolean = False

            if boolean:
                continue
            else:
                for m in range(no_sims):
                    sim = matrix_simulations[m, :, d]
                    if np.max(sim >= self.autocall):
                        ab = np.argmax(np.max(sim >= self.autocall, axis=1))
                        sim = sim[: ab + 1]
                    else:
                        ab = sim.shape[0] - 1
                    # computing coupons
                    if n_small != 0:
                        past = sim[:n_small]
                        future = sim[n_small:]
                        boolean = past >= self.interest[:n_small]
                        if np.max(boolean):
                            temp = [
                                np.argmax(np.flip(boolean[:, i]))
                                for i in range(sim.shape[1])
                            ]
                            open_periods = min(temp)
                        else:
                            open_periods = n_small
                        outstanding_coupon = np.sum(
                            self.coupon_rate[n_small - open_periods : n_small]
                        )
                    else:
                        outstanding_coupon = 0
                        future = sim
                    pays_interest = (
                        future >= self.interest[n_small : n_small + future.shape[0]]
                    )
                    pays_interest = np.max(pays_interest, axis=1)
                    positions = np.arange(future.shape[0])
                    positions = positions[pays_interest]
                    if positions.size == 0:
                        coupons = 0
                    else:
                        coupons = np.zeros(payoffs.shape[1])
                        coupon_rate = self.coupon_rate[n_small:]
                        coupons[positions[0]] = outstanding_coupon + np.sum(
                            coupon_rate[: positions[0] + 1]
                        )
                        for i in range(positions.size - 1):
                            # the +1 below is because python starts indexing at 0
                            coupons[positions[i + 1]] = np.sum(
                                coupon_rate[positions[i] + 1 : positions[i + 1] + 1]
                            )
                    # computing final amount
                    final = sim[-1]
                    final = 1 - (1 - np.min(final / self.strike)) * np.max(
                        final <= self.knock_in
                    )
                    final = final * np.identity(payoffs.shape[1])[ab - n_small]
                    payoffs[m, :, d] = coupons + final
        payoffs_matrix[:, np.array(indices[n_small:]) - n, :] = payoffs
        return self.nominal * payoffs_matrix


# -----------------------------------------------------------------------------------
# Volatility strategies
# -----------------------------------------------------------------------------------


class Butterfly(
    Call, Put
):  # Inheritance added so it appears nicely in the inheritance diagram
    """
    A financial derivative representing a butterfly option strategy, which involves selling two options at a middle strike price and buying one option each at a lower and
    higher strike price.

    Parameters
    ----------
    underlying : str
        The underlying asset of the option.
    maturity : pandas.Timestamp
        The maturity date of the option.
    low_strike : float
        The strike price of the lower option.
    main_strike : float
        The strike price of the main (middle) option.
    cap_strike : float
        The strike price of the higher option.
    kind : str
        The type of option, either 'call' or 'put'.
    calendar : data.calendars.DayCountCalendar
        The calendar used for date calculations.
    nominal : float
        The nominal value of the option.

    Returns
    -------
    numpy.array
        A Butterfly option position.

    Raises
    ------
    AttributeError
        If ``kind`` is not ``call`` or ``put``.
    """

    # noinspection PyInitNewSignature
    def __new__(
        cls,
        underlying,
        maturity,
        low_strike,
        main_strike,
        cap_strike,
        kind,
        calendar,
        nominal=100,
        implied_volatility=None,
    ):
        if kind == "call":
            c1 = Call(
                underlying=underlying,
                strike=low_strike,
                maturity=maturity,
                calendar=calendar,
                nominal=nominal,
            )
            c2 = Call(
                underlying=underlying,
                strike=main_strike,
                maturity=maturity,
                calendar=calendar,
                nominal=nominal,
            )
            c3 = Call(
                underlying=underlying,
                strike=cap_strike,
                maturity=maturity,
                calendar=calendar,
                nominal=nominal,
            )
            butterfly = c1 - 2 * c2 + c3
            butterfly.monotonicity_price_function = None
            butterfly.implied_volatility = implied_volatility

        elif kind == "put":
            p1 = Put(
                underlying=underlying,
                strike=low_strike,
                maturity=maturity,
                calendar=calendar,
                nominal=nominal,
            )
            p2 = Put(
                underlying=underlying,
                strike=main_strike,
                maturity=maturity,
                calendar=calendar,
                nominal=nominal,
            )
            p3 = Put(
                underlying=underlying,
                strike=cap_strike,
                maturity=maturity,
                calendar=calendar,
                nominal=nominal,
            )
            butterfly = p1 - 2 * p2 + p3
            butterfly.monotonicity_price_function = None
            butterfly.implied_volatility = implied_volatility

        else:
            raise AttributeError("kind must be either put or call.")
        butterfly.__class__.__name__ = "Butterfly"
        return butterfly


class Straddle(Call, Put):
    """
    A Straddle is an options trading strategy that involves buying both a call option and a put option with same strike prices, expiration date and underlying asset.

    Parameters
    ----------
    underlying : str
        The underlying asset of the option.
    maturity : pandas.Timestamp
        The expiration date of the option.
    strike : float
        The strike price of the option.
    calendar : data.calendars.DayCountCalendar
        The calendar used to calculate dates.
    nominal : float
        The nominal value of the option.

    Returns
    -------
    numpy.array
        A  Straddle option.

    Raises
    ------
    AttributeError
        If ``kind`` is not ``call`` or ``put``.
    """

    def __new__(cls, underlying, maturity, strike, calendar, nominal=100):
        c = Call(
            underlying=underlying,
            strike=strike,
            maturity=maturity,
            calendar=calendar,
            nominal=nominal,
        )
        p = Put(
            underlying=underlying,
            strike=strike,
            maturity=maturity,
            calendar=calendar,
            nominal=nominal,
        )
        straddle = c + p
        straddle.__class__.__name__ = "Straddle"
        return straddle


class Strangle(Call, Put):
    """
    A Strangle is an options trading strategy that involves buying both a call option and a put option with different strike prices, but with the same expiration date and
    underlying asset.

    Parameters
    ----------
    underlying : str
        The underlying asset's symbol.
    maturity : datetime.date or str
        The maturity date of the options.
    low_strike : float
        The strike price of the put option.
    cap_strike : float
        The strike price of the call option.
    calendar : data.calendars.DayCountCalendar
        Calendar.
    nominal : float
        The nominal value of the option contract. (default is 100)

    Returns
    -------
    numpy.array
        A Strangle option position.

    Raises
    ------
    AttributeError
        If ``kind`` is not ``call`` or ``put``.
    """

    def __new__(
        cls, underlying, maturity, low_strike, cap_strike, calendar, nominal=100
    ):
        c = Call(
            underlying=underlying,
            strike=cap_strike,
            maturity=maturity,
            calendar=calendar,
            nominal=nominal,
        )
        p = Put(
            underlying=underlying,
            strike=low_strike,
            maturity=maturity,
            calendar=calendar,
            nominal=nominal,
        )
        strangle = c + p
        strangle.__class__.__name__ = "Strangle"
        return strangle


class Condor(
    Call, Put
):  # Inheritance added so it appears nicely in the inheritance diagram
    """
    A condor option strategy involves selling two options at a middle different strike price and buying one option each at a lower and higher strike price.

    Parameters
    ----------
    underlying : str
        The underlying asset of the option.
    maturity : pandas.Timestamp
        The maturity date of the option.
    low_strike : float
        The strike price of the lower option.
    main_strike : float
        The strike price of the main (middle) option.
    high_strike : float
        The strike price of the second main (middle) option.
    cap_strike : float
        The strike price of the higher option.
    kind : str
        The type of option, either 'call' or 'put'.
    calendar : data.calendars.DayCountCalendar
        The calendar used for date calculations.
    nominal : float, optional
        The nominal value of the option (default is 100).

    Returns
    -------
    numpy.array
        A condor option position.

    Raises
    ------
    AttributeError
        If ``kind`` is not ``call`` or ``put``.

    """

    # noinspection PyInitNewSignature
    def __new__(
        cls,
        underlying,
        maturity,
        low_strike,
        main_strike,
        high_strike,
        cap_strike,
        kind,
        calendar,
        nominal=100,
    ):
        """
        Calculate the value of a condor option position.

        Returns
        -------
        numpy.array
            A condor option position.
        """

        if kind == "call":
            c1 = Call(
                underlying=underlying,
                strike=low_strike,
                maturity=maturity,
                calendar=calendar,
                nominal=nominal,
            )
            c2 = Call(
                underlying=underlying,
                strike=main_strike,
                maturity=maturity,
                calendar=calendar,
                nominal=nominal,
            )
            c3 = Call(
                underlying=underlying,
                strike=high_strike,
                maturity=maturity,
                calendar=calendar,
                nominal=nominal,
            )
            c4 = Call(
                underlying=underlying,
                strike=cap_strike,
                maturity=maturity,
                calendar=calendar,
                nominal=nominal,
            )
            condor = c1 - c2 - c3 + c4

        elif kind == "put":
            p1 = Put(
                underlying=underlying,
                strike=low_strike,
                maturity=maturity,
                calendar=calendar,
                nominal=nominal,
            )
            p2 = Put(
                underlying=underlying,
                strike=main_strike,
                maturity=maturity,
                calendar=calendar,
                nominal=nominal,
            )
            p3 = Put(
                underlying=underlying,
                strike=high_strike,
                maturity=maturity,
                calendar=calendar,
                nominal=nominal,
            )
            p4 = Put(
                underlying=underlying,
                strike=cap_strike,
                maturity=maturity,
                calendar=calendar,
                nominal=nominal,
            )
            condor = p1 - p2 - p3 + p4

        else:
            raise AttributeError("kind must be either put or call.")
        condor.__class__.__name__ = "Condor"
        return condor
