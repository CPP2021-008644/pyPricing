import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from scipy.stats import norm, ncx2, multivariate_normal
from scipy.optimize import (
    minimize_scalar,
    minimize,
    root_scalar,
    dual_annealing,
    bisect,
)
from scipy.special._ufuncs import gammainc
import pyfeng as pf

# from numba import njit, prange
import plotly.graph_objs as go
from scipy.stats import norm
import py_vollib_vectorized

try:
    from ..pricing import (
        functions as afsfun,
    )  # (Relative) import needed for the workspace. In this case __package__ is 'pypricing.data'
except (ImportError, ModuleNotFoundError, ValueError):
    import pricing.functions as afsfun  # (Absolute) local import. In this case __package__ is 'data'


class Underlying:
    """
    A parent class representing an Underlying asset.

    Parameters
    ----------
    ticker : str
        The ticker symbol of the asset.
    name : str, optional
        The name of the asset. Defaults to ``None``.
    tenor : float, optional
        The tenor of the asset. Defaults to ``None``.

    Attributes
    ----------
    ticker : str
        The ticker symbol of the asset.
    name : str
        The name of the asset.
    tenor : float
        The tenor of the asset.
    data : pandas.DataFrame
        DataFrame containing the historical data of the asset.
    """

    def __init__(self, ticker, name=None, tenor=None):
        self.ticker = ticker
        self.name = name
        self.tenor = tenor
        self.data = pd.DataFrame()

    def generate_paths(
        self, start_dates, step_dates, draws, intervals, forward_measure
    ):
        """
        Generate paths for the specific underlying.

        Parameters
        ----------
        start_dates : pandas.DatetimeIndex
            The start dates for which to generate the paths.
        step_dates : pandas.DatetimeIndex
            The step dates for which to generate the paths.
        draws : numpy.ndarray
            The draws for which to compute the paths.
        intervals : numpy.ndarray
            Time intervals for the simulation.
        discount_curve : discount_curves.DiscountCurve
            Gives the discounts for the computation of the T-forward measure numeráires.
        forward_measure : bool,
            Flag indicating if forward measure is used.

        Returns
        -------
        numpy.ndarray
            Paths of the underlying.
        """
        pass

    def _get_ewma_vol_abstract(
        self,
        returns,
        l=1,
        window=None,
        percentage=False,
        annualized=True,
        mean=True,
        biased=True,
        logs=False,
    ):
        """Compute the volatility as the exponentially weighted standard deviation of the given returns on the given window.

        Parameters
        ----------
        returns : numpy.ndarray
            Return values for which to compute the exponentially weighted standard deviation. Must be one dimensional with shape $(J+1,)$ or four dimensional with shape :math:`(I+1,J+1,1,1)`. Represented as :math:`\\{R_{i,j}\\}_{i=0, j=0}^{I,J}` where $I=0$ in the one dimensional case.
        l : float, optional
            Positive parameter for computing the weights. Represented as $l$. Defaults to 1.
        window : int, optional
            Parameter that sets how many elements of ``returns`` go into the computation. Represented as $w$. Defaults to ``None``.
        percentage : bool, optional
            Flag indicating if the volatility is given a a percentage. Defaults to ``False``.
        annualized : bool, optional
            Flag indicating if the annual volatility is given. Defaults to ``True``.
        mean : bool, optional
            Flag indicating if a centered average is used. Defaults to ``True``.
        biased : bool, optional
            If ``True`` the bias (Bessel's) correction factor is not used. If ``False`` the bias correction factor is used. Defaults to ``True``.
        logs : bool, optional
            Flag indicating if the logarithm is applied to ``returns`` before computing the exponentially weighted standard deviation. Defaults to ``False``.

        Returns
        -------
        float, numpy.ndarray
            The computed volatility. It is a float if ``returns`` is one dimensional, or an numpy.ndarray of shape $(I+1,)$ if  ``returns`` is four dimensional.

        Warnings
        --------
        - (An array of) ``numpy.nan`` will be returned if ``window = None`` and ``biased = False``.
        - (An array of) ``numpy.nan`` will be returned if ``logs = True`` and any of the elements of ``returns`` negative.

        Notes
        -----
        Set :math:`w=\\infty` whenever ``window = None`` and define :math:`J'=\\min(J,w-1)`.

        - If ``logs = False`` the volatility :math:`\\sigma_i` is computed as

        .. math::
            \\sigma_{i} = 100^p \\left[ 250^{a}\\left(\\frac{w}{w-1}\\right)^{(1-b)}\\frac{l-1}{l^{J'+1}-1}\\sum_{j=0}^{J'} l^j\\left(R_{i,J-j}-m M\\right)^2\\right]^{1/2}\,,

        where $M$ is the arithmetic mean

        .. math::
            M=\\frac{1}{(I+1) (J'+1)}\\sum_{i=0}^{I}\\sum_{j=0}^{J'} R_{i,J-j}\,,

        and :math:`p,a,m,b\\in\\{0,1\\}` are binary variables corresponding to ``percentage``, ``annualized``, ``mean``, ``biased``.

        - If ``logs = True``, $R_{i,j}$ is replaced by :math:`\\log(R_{i,j})` in the formulas above.
        """
        # Returns must be np.arrays. In theory, this method should be a function as it doesn't use self. (static method)
        if logs:  # TODO: returns could be negative. Do it properly
            returns = np.log(returns)
            returns = returns[returns != 0]
        if window is not None:
            if returns.ndim == 4:
                returns = returns[:, -window:]  # First index are simulations
            else:
                returns = returns[-window:]
        if mean:
            mean_value = np.mean(returns)
        else:
            mean_value = 0
        if returns.ndim == 4:
            weights = l ** np.flip(np.arange(returns.shape[1])).reshape(
                (1, returns.shape[1], 1, 1)
            )
            sigma = np.sqrt(
                np.sum(weights * (returns - mean_value) ** 2, axis=1) / np.sum(weights)
            )
        # elif returns.ndim == 3:  # This made sense for the old method generate_paths in ExposureIndex.
        #     weights = l ** np.flip(np.arange(returns.shape[0])).reshape((returns.shape[0], 1, 1))
        #     sigma = np.sqrt(np.sum(weights * (returns - mean_value) ** 2, axis=0) / np.sum(weights))
        elif returns.ndim == 1:
            weights = l ** np.flip(np.arange(returns.shape[0]))
            sigma = np.sqrt(
                np.sum(weights * (returns - mean_value) ** 2) / np.sum(weights)
            )
        if annualized:
            sigma = np.sqrt(252) * sigma
        if not biased:
            sigma = np.sqrt(window / (window - 1)) * sigma
        if percentage:
            sigma = 100 * sigma

        if returns.ndim <= 3:
            sigma = sigma.item()
        elif returns.ndim == 4:
            sigma = sigma.reshape(returns.shape[0])

        return sigma

    def set_data(self, data):
        """
        Adds ``data`` to ``self.data``. Used in import_underlying from ``data_factory.py`` to assign to the asset the (historical) data from the Data Base.

        Parameters
        ----------
        data : pandas.DataFrame
            Data to be added to ``self.data``.
        """
        temp = data.merge(
            self.data,
            left_index=True,
            right_index=True,
            suffixes=("", "_old"),
            how="outer",
        )
        for column in [column for column in temp.columns if column[-4:] == "_old"]:
            dates_missing_from_new = temp.index[temp[column[:-4]].isna()]
            temp.loc[dates_missing_from_new, column[:-4]] = temp.loc[
                dates_missing_from_new, column
            ]
        self.data = temp[
            [column for column in temp.columns if column[-4:] != "_old"]
        ].sort_index()

    def fillna(self):
        """
        Fill ``numpy.nan`` values in ``self.data`` using linear interpolation.
        """
        self.data = self.data.interpolate("linear", limit_area="inside", axis=0)

    def get_data(self, field, dates):
        """
        Get data for given field at given dates.

        Parameters
        ----------
        field : str
            Name of the field.

        dates : pandas.DatetimeIndex, list of strings, pandas.Timestamp, or string
            Dates at which to get data.

        Returns
        -------
        pandas.Series
            Series indexed by ``dates`` containing the data of the given field . When a data is not available ``numpy.nan`` is given.

        Raises
        ------
        ValueError
            If ``self.data`` does not have a column named ``field``.
        """

        dates = afsfun.dates_formatting(dates)
        if field in self.data.columns:
            df = pd.Series(np.nan, index=dates)
            existing_dates = pd.DatetimeIndex.intersection(dates, self.data.index)
            df.loc[dates] = self.data.loc[existing_dates, field]
            df.name = self.ticker + " " + field
            return df
        else:
            raise ValueError("Non existing field")

    def get_value(self, dates):
        """
        Get prices for this underlying at the given dates.

        Parameters
        ----------
        dates : pandas.DatetimeIndex, list of strings, pandas.Timestamp, or string
            Dates at which to get the prices.

        Returns
        -------
        pandas.Series
            Series indexed by ``dates`` containing the prices. When a price is not available ``numpy.nan`` is given.
        """

        return self.get_data("Price", dates)

    def get_dates(self):
        """
        Get dates where there are available prices for this underlying.

        Returns
        -------
        pandas.DatetimeIndex
            The dates obtained as index labels of ``self.data["Price"]``.
        """
        return self.data["Price"].dropna().index

    def get_arithmetic_return(self, dates, percentage=False, annualized=False):
        """Get the arithmetic return of the underlying at the given dates.

        Parameters
        ----------
        dates :  pandas.DatetimeIndex, list of strings, pandas.Timestamp, or string
            Dates at which to get the return. Represented as :math:`\\{t_i\\}`.
        percentage : bool, optional
            Flag indicating if the returns are given as a percentage. Defaults to ``False``.
        annualized : bool, optional
            Flag indicating if the annual returns are given. Defaults to ``False``.

        Returns
        -------
        pandas.Series
            Series of returns indexed by the subset of ``dates`` at which returns could be computed.

        Notes
        -----
        For each $t_i$ at which the price of the underling $S(t_i)$ is available, let $t'_i$ be the immediately previous date at which the price of the underlying $S(t'_i)$ is also available. Then the return at $t_i$ is given by

        .. math::
            R(t_i) = 100^p\\left[\\left(\\frac{S(t_i)}{S(t'_i)}\\right)^{\\left(365/(t_i-t_i')\\right)^a}-1\\right],

        where :math:`p,a\\in\\{0,1\\}` are binary variables corresponding to ``percentage`` and ``annualized``.
        """
        dates = afsfun.dates_formatting(dates)
        dates = dates[dates.isin(self.get_dates())]
        all_dates = pd.to_datetime(list(self.get_dates())).sort_values()
        if dates[0] == all_dates[0]:
            dates = dates[1:]
        previous_dates = [all_dates[all_dates < date][-1] for date in dates]
        previous_dates = pd.to_datetime(previous_dates)
        prices = self.get_value(dates)
        previous_prices = self.get_value(previous_dates)
        taus = (dates - previous_dates).days
        if annualized:
            returns = (prices / previous_prices.values) ** (365 / taus)
        else:
            returns = prices / previous_prices.values  # **(1/taus)
        returns = returns - 1
        if percentage:
            returns = returns * 100
        returns.title = "{} return".format(self.name)
        return returns

    def get_return(self, dates, percentage=False, annualized=False, raw=False):
        """Get the log or raw return of the underlying at the given dates.

        Parameters
        ----------
        dates :  pandas.DatetimeIndex, list of strings, pandas.Timestamp, or string
            Dates at which to get the return. Represented as :math:`\\{t_i\\}`.
        percentage : bool, optional
            Flag indicating if the returns are given as a percentage. Defaults to ``False``.
        annualized : bool, optional
            Flag indicating if the annual returns are given. Defaults to ``False``.
        raw : bool, optional
            Flag indicating if raw returns are given. Defaults to ``False``.

        Returns
        -------
        pandas.Series
            Series of returns indexed by the subset of ``dates`` at which returns could be computed.

        Notes
        -----
        For each $t_i$ at which the price of the underling $S(t_i)$ is available, let $t'_i$ be the immediately previous date at which the price of the underlying $S(t'_i)$ is also available.

        - If ``raw = False``, the return at time $t_i$ is given by

        .. math::
            R(t_i) = 100^p\\left(\\frac{360}{t_i-t'_i}\\right)^a\\log\\left(\\frac{S(t_i)}{S(t'_i)}\\right),

        - if ``raw = True``, the return at time $t_i$ is given by

        .. math::
            R(t_i) = 100^p\\left(\\frac{360}{t_i-t'_i}\\right)^a\\left(S(t_i)-S(t'_i)\\right),

        where in both cases :math:`p,a\\in\\{0,1\\}` are binary variables corresponding to ``percentage`` and ``annualized``.
        """
        dates = afsfun.dates_formatting(dates)
        dates = dates[dates.isin(self.get_dates())]
        all_dates = pd.to_datetime(list(self.get_dates())).sort_values()
        if dates[0] == all_dates[0]:
            dates = dates[1:]
        previous_dates = [all_dates[all_dates < date][-1] for date in dates]
        previous_dates = pd.to_datetime(previous_dates)
        prices = self.get_value(dates)
        previous_prices = self.get_value(previous_dates)
        if raw:
            returns = prices - previous_prices.values
        else:
            returns = np.log(prices / previous_prices.values)
        if annualized:
            taus = (dates - previous_dates).days
            returns = returns * (365 / taus)
        if percentage:
            returns = returns * 100
        return returns

    def get_ewma_vol(
        self,
        dates,
        l=0.94,
        window=None,
        percentage=False,
        annualized=False,
        mean=True,
        biased=True,
        arithmetic=False,
        logs=False,
    ):
        """Compute the volatility at the given dates as the exponentially weighted standard deviation of the returns on the given window.

        Parameters
        ----------
        dates :  pandas.DatetimeIndex, list of strings, pandas.Timestamp, or string
            Dates at which to compute the volatility.
        l : float, optional
            Positive parameter for computing the weights. Defaults to 0.94.
        window : int, optional
            Parameter that sets how many returns into the past are used in the computation. Defaults to ``None``.
        percentage : bool, optional
            Flag indicating if the volatility is given a a percentage. Defaults to ``False``.
        annualized : bool, optional
            Flag indicating if the annual volatility is given. Defaults to ``True``.
        mean : bool, optional
            Flag indicating if a centered average is used. Defaults to ``True``.
        biased : bool, optional
            If ``True`` the bias (Bessel's) correction factor is not used. If ``False`` the bias correction factor is used. Defaults to ``True``.
        arithmetic : bool, optional
            If ``True``, arithmetic returns are used. If ``False``, log returns are used. Defaults to ``False``.
        logs : bool, optional
            Flag indicating if the logarithm is applied to the returns.

        Returns
        -------
        pandas.Series
            Series of volatilities indexed by the subset of ``dates`` at which the are price of the underlying is available.

        Warnings
        --------
        - ``logs = True`` should never be used because returns, both arithmetic or log, can be negative.

        Notes
        -----
        For each date in ``dates`` at which the are price of the underlying is available, the volatility is computed with :py:meth:`Underlying._get_ewma_vol_abstract <Underlying._get_ewma_vol_abstract>`.
        """
        dates = afsfun.dates_formatting(dates)
        all_dates_total = pd.to_datetime(list(self.get_dates()))
        if window is not None:
            min_date = dates[0] - pd.Timedelta(
                2 * window, "d"
            )  # min date needed for the calculations, it can reduce computation time considerably
        else:
            min_date = all_dates_total[0]
        all_dates = all_dates_total[all_dates_total >= min_date]
        if arithmetic:
            all_returns = self.get_arithmetic_return(all_dates)
        else:
            all_returns = self.get_return(all_dates, percentage=False)
        dates = dates[dates.isin(all_returns.index)]
        sigma = pd.Series(np.nan, index=dates)
        for date in dates:
            returns = all_returns.loc[:date]
            returns = returns[returns != 0]
            sigma.loc[date] = self._get_ewma_vol_abstract(
                returns=returns.values,
                l=l,
                window=window,
                percentage=percentage,
                annualized=annualized,
                mean=mean,
                biased=biased,
                logs=logs,
            )

        #     if window is not None:
        #         returns = returns[-window:]
        #     mean = np.mean(returns)
        #     weights = l ** np.flip(np.arange(len(returns)))
        #     sigma.loc[date] = np.sqrt(np.sum(weights * (returns - mean) ** 2) / np.sum(weights))
        # if annualized:
        #     sigma = np.sqrt(252) * sigma
        # if percentage:
        #     sigma = 100 * sigma

        return sigma

    # def get_standard_absolute_return(self, dates, l = 1, window = none):
    #    if np.asarray(dates).shape == (): dates = [dates]
    #    dates = pd.to_datetime(dates).sort_values()
    #    all_dates = pd.to_datetime(list(self.get_dates()))
    #    all_returns = self.get_absolute_return(all_dates)
    #    dates = dates[dates.isin(all_returns.index)]
    #    sigma = pd.series(np.nan, index=dates)
    #    for date in dates:
    #        returns = all_returns.loc[:date]
    #        returns = returns[returns != 0]
    #        if window is not none:
    #            returns = returns[-window:]
    #        mean = np.mean(returns)
    #        weights = l ** np.flip(np.arange(len(returns)))
    #        sigma.loc[date] = np.sqrt(np.sum(weights*(returns-mean)**2)/np.sum(weights))
    #    return sigma


class Basis:
    """Deprecated class."""

    def __init__(self, underlying1, underlying2):
        self.underlying = [underlying1, underlying2]
        self.name = "{}-{} basis".format(underlying1.name, underlying2.name)

    def get_value(self, dates):
        prices = [self.underlying[i].get_value(dates) for i in range(2)]
        dates = pd.DatetimeIndex.intersection(prices[0].index, prices[1].index)
        return prices[0].loc[dates] - prices[1].loc[dates]

    def get_dates(self):
        return pd.DatetimeIndex.intersection(
            self.underlying[0].get_dates(), self.underlying[1].get_dates()
        ).sort_values()

    def get_return(self, dates, percentage=False, annualized=False, raw=False):
        returns = [
            self.underlying[i].get_return(
                dates, percentage=percentage, annualized=annualized, raw=raw
            )
            for i in range(2)
        ]
        dates = pd.DatetimeIndex.intersection(returns[0].index, returns[1].index)
        return returns[0].loc[dates] - returns[1].loc[dates]

    def get_absolute_return(self, dates):
        returns = [self.underlying[i].get_absolute_return(dates) for i in range(2)]
        dates = pd.DatetimeIndex.intersection(returns[0].index, returns[1].index)
        return returns[0].loc[dates] - returns[1].loc[dates]

    def get_log_return(self, dates, percentage=False, annualized=False):
        return np.log(1 + self.get_return(dates))


class DeterministicVolAsset(Underlying):
    """
    date format: %Y%m%d
    dynamics should be a string, e.g. "lognormal"
    to set values, must feed numpy array; sort of inconsistent with then returning pandas
    """

    def __init__(self, ticker, name=None, tenor=None, yieldsdiv=False):
        Underlying.__init__(self, ticker=ticker, name=name, tenor=tenor)
        self.yieldsdiv = yieldsdiv
        # defining dictionaries that store values
        # self.vol = {}
        # self.div = {}

    def get_fpx(self, dates, fdate, discountcurve, calendar):
        """
        Computes the forward price, :math:`F_t^T := \\frac{S_t}{D(t, T)}`

        Parameters
        ----------
        dates : pandas.DatetimeIndex
            Valuation dates, the first argument (:math:`t`) of :math:`F_t^T`.
        fdate : str or pandas.Timestamp
            Future date or time horizon of the forward price.
        discountcurve : pricing.discount_curves.DiscountCurve , default = None
            Gives the discounts for the computations.
        calendar : data.calendars.DayCountCalendar
            Calendar

        Returns
        -------
        pandas.Series
            Series with the forward prices.

        Notes
        -----
            If the valuation date is after the future date, the function returns the growth at the risk-free rate.
        """
        fd = pd.to_datetime(fdate)
        dates = pd.to_datetime(dates)
        if not np.min(dates <= fd):
            missing = dates[dates > fd]
            dates = dates[dates <= fd]
            print("Eliminating valuation dates past future date:")
            for date in missing:
                print(date.strftime("%Y-%m-%d"))
        px = self.get_value(dates=dates)
        dates = px.index
        px = px.values
        if px.shape == (1,):
            px = px[0]
        disc = discountcurve.get_value(dates=dates, future_dates=fd, calendar=calendar)
        x = px / disc
        if self.yieldsdiv and isinstance(self, LognormalAsset):
            div = self.get_divrate(dates)
            dates = div.index
            div = div.values
            if div.size == 1:
                div = div[0]
            tau = calendar.interval(dates, fd)
            x = x * np.exp(-div * tau)
        if dates.size >= np.asarray(fd).size:
            x = pd.Series(x, index=dates, name="Forward price")
        else:
            x = pd.Series(x, index=fd, name="Forward price")
        return x

    def get_vol(self, dates, tenors=None, strike=None):
        """
        Get volatilities for this asset at the given dates.

        Parameters
        ----------
        dates : pandas.DatetimeIndex, list of strings, pandas.Timestamp, or string
            Dates at which to get the volatilities.
        tenors : optional
            Not used. Defaults to ``None``.
        strike : optional
            Not used. Defaults to ``None``.

        Returns
        -------
        pandas.Series
            Series indexed by ``dates`` containing the volatilities. When a volatility is not available ``numpy.nan`` is given.
        """
        return self.get_data(field="Volatility", dates=dates)

    def get_vol_dates(self):
        """
        Get dates where there are available volatilities for this asset.

        Returns
        -------
        pandas.DatetimeIndex
            The dates obtained as index labels of ``self.data["Volatility"]``.
        """
        return self.data["Volatility"].dropna().index

    def get_divrate(self, dates):
        """
        Get dividend rates for this asset at the given dates.

        Parameters
        ----------
        dates : pandas.DatetimeIndex, list of strings, pandas.Timestamp, or string
            Dates at which to get the dividend rates.

        Returns
        -------
        pandas.Series
            Series indexed by ``dates`` containing the dividend rates. When a volatility is not available ``numpy.nan`` is given. If ``self.yieldsdiv = False`` the series is filled with 0.
        """
        if self.yieldsdiv:
            divrate = self.get_data(field="Dividend Rate", dates=dates)
        else:
            dates = afsfun.dates_formatting(dates)
            df = pd.Series(0, index=dates)
            divrate = df
        return divrate

    def get_div_dates(self):
        """
        Get dates where there are available dividend rates for this asset.

        Returns
        -------
        pandas.DatetimeIndex
            The dates obtained as index labels of ``self.data["Dividend Rate"]``.
        """
        return self.data["Dividend Rate"].dropna().index


class NormalAsset(DeterministicVolAsset):
    """
    A class representing a Normal Asset, Bachelier's model, derived from the DeterministicVolAsset class.

    Parameters
    ----------
    ticker : str
        The ticker symbol of the asset.
    name : str, optional
        The name of the asset. Defaults to ``None``.
    tenor : float, optional
        The tenor of the asset. Defaults to ``None``.
    yieldsdiv : bool, optional
        Flag indicating of the asset yields dividends. Defaults to ``False``.
    drift : float, optional
        Drift of the asset. Defaults to ``numpy.nan``.

    Attributes
    ----------
    ticker : str
        The ticker symbol of the asset.
    name : str
        The name of the asset.
    tenor : float
        The tenor of the asset.
    data : pandas.DataFrame
        DataFrame containing the historical data of the asset.
    yieldsdiv : bool
        Flag indicating of the asset yields dividends.
    drift : float
        Drift of the asset.
    """

    def __init__(self, ticker, name=None, tenor=None, yieldsdiv=False, drift=np.nan):
        DeterministicVolAsset.__init__(
            self, ticker=ticker, name=name, tenor=tenor, yieldsdiv=yieldsdiv
        )
        self.drift = drift

    def generate_draws(
        self, no_obsdates, no_valuation_dates, no_sims, start_dates=None
    ):
        """
        For Normal and Lognormal we only need one normal (so ``start_dates`` is not used).
        This method is needed for the new class :py:meth:`DiffusionMC <pricing.mc_engines.DiffusionMC>`.

        Parameters
        ----------
        no_obsdates : int
            Number of observation dates.
        no_valuation_dates : int
            Number of valuation dates.
        no_sims : int
            Number of simulations.
        start_dates :  pandas.DatetimeIndex, list of strings, pandas.Timestamp, or string, optional
            Starting dates. Not used in this method. Defaults to ``None``.

        Returns
        -------
        numpy.ndarray, shape (no_sims, no_obsdates, no_valuation_dates, 1)
            Uncorrelated random draws of the standard Gaussian distribution.
            For consistency, this is a four-dimensional array, although the last index is not not used.
        """
        draws = np.random.randn(
            no_sims, no_obsdates, no_valuation_dates, 1
        )  # We use np.random.randn instead of multivariate_normal.rvs (used in mc_engines).
        return draws

    def _generate_base(
        self, start_dates, step_dates, draws, intervals, forward_measure=True
    ):
        """
        Generate the base value for the underlying dynamics.
        It is implemented only for Normal Assets with zero dividend yield and Lognormal assets.

        Parameters
        ----------
        start_dates : pandas.DatetimeIndex
            The start dates for which to compute the paths. Represented as :math:`\\{t^0_k\\}_{k=0}^K` .
        step_dates : pandas.DatetimeIndex
            The step dates for which to compute the paths. Not used in this method. Represented as :math:`\\{t^1_j\\}_{j=0}^J` .
        draws : numpy.ndarray
            The draws for which to compute the paths. For consistency considerations with the case of several assets,
            this is a four-dimensional array, although the last index is unused here. Represented as :math:`\\textrm{draws}[i,j,k]` where :math:`i` indexes the simulation number.
        intervals : numpy.ndarray
            Time intervals for the simulation. Represented as :math:`\\textrm{intervals}[j,k]=\\textrm{DCC}(t^1_{j-1,k}, t^1_{j,k})` where

            .. math::
                t^1_{j,k}:= \\begin{cases} t^1_{j} \qquad \\text{ for }j=0,\ldots,J \quad \\text{ (independent of } k\\text{)}\,,\\\\t^0_{k} \qquad \\text{ for }j=-1\,.\end{cases}

            The function :math:`\\textrm{DCC}` represents the day count convention and will depend on the calendar used. For simplicity, it can be thought of as :math:`\\textrm{DCC}(t,t')=t'-t` as a year fraction.
        forward_measure: bool, optional
            Flag indicating if forward measure is used. Defaults to ``True``.

        Returns
        -------
        numpy.ndarray
            The base value given the inputs and underlying dynamics.
            It is a three-dimensional array represented as :math:`\\textrm{base}[i,j,k]=\\Delta_j{Y}_{i,k}` .

        Raises
        ------
        TypeError
            If the underlying dynamics is not implemented.

        See also
        --------
        Underlyings_Jupyter: See that documentation for more details.

        Notes
        -----
        For **NormalAsset** with zero dividends:

        Let's assume that  :math:`Y_t` follows a standard **arithmetic** Brownian motion

        .. math::
            \\textrm{d} Y_t = \mu \\textrm{d} t + \sigma \\textrm{d} W_t\,.

        which can be easily integrated over the interval $[s,t)$ to obtain

        .. math::
            Y_t - Y_s = \\mu (t-s) + \\sigma Z \\sqrt{t-s}\,,

        with :math:`Z\sim\mathcal{N}(0,1)`. Then the :math:`\\mathrm{base}[i,j,k]` of the **NormalAsset** $Y_t$ is given by

        .. math::
            \Delta_j{Y}_{i,k} := Y_{i,k}(t^1_{j,k})-Y_{i,k}(t^1_{j-1,k}) = \mu_k \cdot \\textrm{intervals}_{jk} + \sigma_k \cdot \\textrm{draws}_{ijk} \cdot \sqrt{\\textrm{intervals}_{jk}}\,,

        where

        -:math:`\mu_k=0` if ``forward_measure = True``,

        -:math:`\mu_k=\\textrm{self.drift}(t^0_k)` if ``forward_measure = False``,

        -:math:`\sigma_k=\\textrm{volatility}(t^0_k)` .

        For **LognormalAsset**:

        Let's assume that  $S_t$ follows a standard **geometric** Brownian motion

        .. math::
            \\mathrm{d} S_t = \mu S_t \\mathrm{d} t + \sigma S_t \\mathrm{d} W_t\,,

        so :math:`Y_t := \log(S_t)` follows a standard **arithmetic** Brownian motion

        .. math::
            \\mathrm{d} Y_t =  \left(\mu -\\frac12 \sigma^2\\right) \mathrm{d} t + \sigma \mathrm{d} W_t\,.

        Then the :math:`\\mathrm{base}[i,j,k]` of the **LognormalAsset** $S_t$ is given by

        .. math::
            \Delta_j{Y}_{i,k} := Y_{i,k}(t^1_{j,k})-Y_{i,k}(t^1_{j-1,k}) = \left(\mu_k -\\frac12 \sigma_k^2\\right) \\textrm{intervals}_{jk} + \sigma_k \cdot \\textrm{draws}_{ijk} \cdot \sqrt{\\textrm{intervals}_{jk}}\,,

        where

        -:math:`\mu_k=-q(t^0_k)` if ``forward_measure = True``,

        -:math:`\mu_k=\\textrm{self.drift}(t^0_k)-q(t^0_k)` if ``forward_measure = False``,

        -:math:`\sigma_k=\\textrm{volatility}(t^0_k)` .

        and :math:`q` stands for the dividend rate.
        """
        divrate = self.get_divrate(dates=start_dates).values
        divrate = divrate.reshape((1, divrate.size))
        vols = self.get_vol(dates=start_dates).values
        vols = vols.reshape((1, vols.size))

        vols = vols.transpose((1, 0))
        divrate = divrate.transpose((1, 0))
        intervals = intervals.reshape(intervals.shape + (1,))

        if isinstance(self, LognormalAsset):
            if forward_measure:
                drift = -divrate
            else:
                drift = self.drift - divrate

            base = (drift - vols**2 / 2) * intervals + vols * draws * np.sqrt(intervals)
        elif isinstance(self, NormalAsset) and not self.yieldsdiv:
            # TODO: implement Normal with dividends using a discretization scheme
            if forward_measure:
                drift = 0
            else:
                drift = self.drift

            base = drift * intervals + vols * draws * np.sqrt(intervals)
        else:
            raise TypeError("Underlying dynamics not supported")

        return base

    def _generate_prices(self, start_dates, final_date, discount_curve):
        """
        Generate the prices for the Normal Asset at the given start dates divided by the numeráire.
        In our case the numeráire is the ``final_date``-forward measure.

        Parameters
        ----------
        start_dates : pandas.DatetimeIndex
            The start dates for which to compute the price $S$. Represented as :math:`\\{t^0_k\\}_{k=0}^K` .
        final_date : pandas.Timestamp
            The final date for which to compute the numeráires. Represented as :math:`T`.
        discount_curve : discount_curves.DiscountCurve
            Gives the discounts for the computation of the $T$-forward measure numeráires. Represented as :math:`t\\mapsto D(t, T)` .

        Returns
        -------
        numpy.ndarray
            The prices of the Normal Asset at the given start dates divided by the $T$-forward measure numeráires. Represented as

            .. math::
                FS(t^0_k,T)=\\frac{S(t^0_k)}{D(t^0_k, T)}\,.
        """
        prices = self.get_value(dates=start_dates).values  # Prices at the first date
        numeraire = discount_curve.get_value(dates=start_dates, future_dates=final_date)
        fwd_prices = prices / numeraire
        fwd_prices = fwd_prices.reshape((1, fwd_prices.size))
        fwd_prices = fwd_prices.transpose((1, 0))

        return fwd_prices

    def generate_paths(
        self,
        start_dates,
        step_dates,
        draws,
        intervals,
        discount_curve,
        forward_measure=True,
    ):
        """
        Generate paths for the Normal Asset based on the provided inputs.

        Parameters
        ----------
        start_dates : pandas.DatetimeIndex
            The start dates for which to generate the paths. Represented as :math:`\\{t^0_k\\}_{k=0}^K` .
        step_dates : pandas.DatetimeIndex
            The step dates for which to generate the paths. Represented as :math:`\\{t^1_j\\}_{j=0}^J` .
        draws : numpy.ndarray
            The draws for which to compute the paths. For consistency considerations with the case of several assets,
            this is a four-dimensional array, although the last index is unused here. Represented as :math:`\\textrm{draws}[i,j,k]` where :math:`i` indexes the simulation number.
        intervals : numpy.ndarray
            Time intervals for the simulation. Represented as :math:`\\textrm{intervals}[j,k]=\\textrm{DCC}(t^1_{j-1,k}, t^1_{j,k})` where

            .. math::
                t^1_{j,k}:= \\begin{cases} t^1_{j} \qquad \\text{ for }j=0,\ldots,J \quad \\text{ (independent of } k\\text{)}\,,\\\\t^0_{k} \qquad \\text{ for }j=-1\,.\end{cases}

            The function :math:`\\textrm{DCC}` represents the day count convention and will depend on the calendar used. For simplicity, it can be thought of as :math:`\\textrm{DCC}(t,t')=t'-t` as a year fraction.
        discount_curve : discount_curves.DiscountCurve
            Gives the discounts for the computation of the T-forward measure numeráires.
        forward_measure : bool, optional
            Flag indicating if forward measure is used. Defaults to ``True``.

        Returns
        -------
        numpy.ndarray
            It returns the simulation paths for the asset price. This is a four-dimensional array, although the last index is unused here.
            Represented as

            .. math::
                \\textrm{paths}[i,j,k] = Y_{i,k}(t^1_{j-1,k})\,, \quad j\in\{0, \ldots, J+1\}\,.
        """
        base = self._generate_base(
            start_dates, step_dates, draws, intervals, forward_measure=forward_measure
        )
        final_date = step_dates[-1]
        prices = self._generate_prices(start_dates, final_date, discount_curve)

        paths = np.zeros(
            (draws.shape[0], intervals.shape[0] + 1, prices.shape[0], draws.shape[3])
        )
        paths[:, 1:, :] = np.add.accumulate(base, 1)
        paths = prices + paths

        return paths


class LognormalAsset(NormalAsset, DeterministicVolAsset):
    """
    A class representing a Lognormal Asset, derived from both the NormalAsset and DeterministicVolAsset classes.

    Parameters
    ----------
    ticker : str
        The ticker symbol of the asset.
    name : str, optional
        The name of the asset. Defaults to ``None``.
    tenor : float, optional
        The tenor of the asset. Defaults to ``None``.
    yieldsdiv : bool, optional
        Flag indicating of the asset yields dividends. Defaults to ``False``.
    drift : float, optional
        Drift of the asset. Defaults to ``numpy.nan``.

    Attributes
    ----------
    ticker : str
        The ticker symbol of the asset.
    name : str
        The name of the asset.
    tenor : float
        The tenor of the asset.
    data : pandas.DataFrame
        DataFrame containing the historical data of the asset.
    yieldsdiv : bool
        Flag indicating of the asset yields dividends.
    drift : float
        Drift of the asset.
    """

    def __init__(self, ticker, name=None, tenor=None, yieldsdiv=False, drift=np.nan):
        DeterministicVolAsset.__init__(
            self, ticker=ticker, name=name, tenor=tenor, yieldsdiv=yieldsdiv
        )
        self.drift = drift

    def _generate_prices(
        self, start_dates, final_date, discount_curve
    ):  # This method is overridden
        """
        Generate the logarithm of the ``final_date``-forward prices for the Lognormal Asset at the given start dates using the given discount curve.

        Parameters
        ----------
        start_dates : pandas.DatetimeIndex
            The start dates for which to compute the price $S$. Represented as :math:`\\{t^0_k\\}_{k=0}^K` .
        final_date : pandas.Timestamp
            The final date for which to compute the forward prices. Represented as :math:`T`.
        discount_curve : discount_curves.DiscountCurve
            Gives the discounts for the computation of the $T$-forward prices. Represented as :math:`t\\mapsto D(t, T)` .

        Returns
        -------
        numpy.ndarray
            The logarithm of the $T$-forward prices of the Lognormal Asset at the given start dates. Represented as

            .. math::
                \\log\\left(FS(t^0_k,T)\\right)=\\log\\left(\\frac{S(t^0_k)}{D(t^0_k, T)}\\right)\,.

        Notes
        -----
            This method calls :py:meth:`NormalAsset._generate_prices <NormalAsset._generate_prices>` and then applies the logarithm to ts output.
        """
        fwd_prices = NormalAsset._generate_prices(
            self, start_dates, final_date, discount_curve
        )
        return np.log(fwd_prices)

    def generate_paths(
        self,
        start_dates,
        step_dates,
        draws,
        intervals,
        discount_curve,
        forward_measure=True,
    ):
        """
        Generate paths for the Lognormal Asset based on the provided inputs.

        Parameters
        ----------
        start_dates : pandas.DatetimeIndex
            The start dates for which to generate the paths. Represented as :math:`\\{t^0_k\\}_{k=0}^K` .
        step_dates : pandas.DatetimeIndex
            The step dates for which to generate the paths. Represented as :math:`\\{t^1_j\\}_{j=0}^J` .
        draws : numpy.ndarray
            The draws for which to compute the paths. For consistency considerations with the case of several assets,
            this is a four-dimensional array, although the last index is unused here. Represented as :math:`\\textrm{draws}[i,j,k]` where :math:`i` indexes the simulation number.
        intervals : numpy.ndarray
            Time intervals for the simulation. Represented as :math:`\\textrm{intervals}[j,k]=\\textrm{DCC}(t^1_{j-1,k}, t^1_{j,k})` where

            .. math::
                t^1_{j,k}:= \\begin{cases} t^1_{j} \qquad \\text{ for }j=0,\ldots,J \quad \\text{ (independent of } k\\text{)}\,,\\\\t^0_{k} \qquad \\text{ for }j=-1.\,\end{cases}

            The function :math:`\\textrm{DCC}` represents the day count convention and will depend on the calendar used. For simplicity, it can be thought of as :math:`\\textrm{DCC}(t,t')=t'-t` as a year fraction.
        discount_curve : discount_curves.DiscountCurve
            Gives the discounts for the computation of the T-forward measure numeráires.
        forward_measure : bool, optional, optional
            Flag indicating if forward measure is used. Defaults to ``True``.

        Returns
        -------
        numpy.ndarray
            It returns the simulation paths for the asset price. This is a four-dimensional array, although the last index is unused here.
            Represented as

            .. math::
                \\textrm{paths}[i,j,k] = S_{i,k}(t^1_{j-1,k})=\\exp(Y_{i,k}(t^1_{j-1,k}))\,, \quad j\in\{0, \ldots, J+1\}\,.

        Notes
        -----
            This method calls :py:meth:`NormalAsset.generate_paths <NormalAsset.generate_paths>` to generate paths for the NormalAsset :math:`Y_t=\\log (S_t)` and then applies the exponential to its output giving

            .. math::
                S_t = e^{Y_t}= e^{Y_0+ \sum_{i=1}^t \Delta Y_i} = e^{\log(S_0)+ \sum_{i=1}^t \Delta Y_i}= S_0 e^{\sum_{i=1}^t \Delta Y_i}\,.

        """
        paths = NormalAsset.generate_paths(
            self,
            start_dates,
            step_dates,
            draws,
            intervals,
            discount_curve,
            forward_measure,
        )
        # generate_paths for LognormalAssets uses the overridden method,
        # not the original one, so it gives exp(log(P_0) + sum)=P_0 * exp(sum),
        # what we want
        return np.exp(paths)


class MultiAsset(Underlying):
    """
    A class representing a collection of DeterministicVolAsset objects, derived from the Underlying class.

    Parameters
    ----------
    *equity_objects : DeterministicVolAsset
        Collection of DeterministicVolAsset objects that constitute the MultiAsset.

    Attributes
    ----------
    components : list
        List of DeterministicVolAsset objects that constitute the MultiAsset.
    yieldsdiv : bool
        Flag indicating of the MultiAsset yields dividends.
    corr: dict
        Dictionary that assigns to a date, given as a pandas.Timestamp, the correlation matrix of the prices of the components at such date.
    """

    def __init__(self, *equity_objects):
        self.components = equity_objects
        self.yieldsdiv = True
        self.corr = None

    def get_dates(self):
        """
        Get dates where there are available prices for all components.

        Returns
        -------
        pandas.DatetimeIndex
            List of dates at which there are available prices for all components.
        """
        dates = self.components[0].get_dates()
        for component in self.components[1:]:
            dates = dates.intersection(component.get_dates())
        return dates

    def get_vol_dates(self):
        """
        Get dates where there are available volatilities for all components.

        Returns
        -------
        pandas.DatetimeIndex
            List of dates at which there are available volatilities for all components.
        """
        dates = self.components[0].get_vol_dates()
        for component in self.components[1:]:
            dates = dates.intersection(component.get_vol_dates())
        return dates

    def get_div_dates(self):
        """
        Get dates where there are available dividend rates for all components.

        Returns
        -------
        pandas.DatetimeIndex
            List of dates at which there are available dividend rates for all components.
        """

        dates = self.components[0].get_div_dates()
        for component in self.components[1:]:
            dates = dates.intersection(component.get_div_dates())
        return dates

    def get_component_values(self, dates):
        """
        Get prices for each component at all given dates.

        Parameters
        ----------
        dates :  pandas.DatetimeIndex, list of strings, pandas.Timestamp, or string
            Dates for which to get the prices.

        Returns
        -------
        pandas.DataFrame
            DataFrame of prices, columns are components and index is ``dates``. When a price is not available ``numpy.nan`` is given.

        """
        return pd.concat(
            (component.get_value(dates) for component in self.components), axis=1
        )

    def get_value(self, dates):
        """
        Get prices for each component at all given dates.

        Parameters
        ----------
        dates :  pandas.DatetimeIndex, list of strings, pandas.Timestamp, or string
            Dates for which to get the prices.

        Returns
        -------
        pandas.DataFrame
            DataFrame of prices, columns are components and index is ``dates``. When a price is not available ``numpy.nan`` is given.
        """
        return self.get_component_values(dates=dates)

    def get_return(self, dates, percentage=False, annualized=False, raw=False):
        """Get the log or raw return of each component at the given dates.

        Parameters
        ----------
        dates :  pandas.DatetimeIndex, list of strings, pandas.Timestamp, or string
            Dates at which to get the return.
        percentage : bool, optional
            Flag indicating if the returns are given as a percentage. Defaults to ``False``.
        annualized : bool, optional
            Flag indicating if the annual returns are given. Defaults to ``False``.
        raw : bool, optional
            Flag indicating if raw returns are given. Defaults to ``False``.

        Returns
        -------
        pandas.DataFrame
            DataFrame of returns, columns are components and index is ``dates``. When a return is not available ``numpy.nan`` is given.

        Notes
        -----
            The return of each component is calculated using :py:meth:`Underlying.get_return <Underlying.get_return>`.
        """
        return pd.concat(
            (
                component.get_return(
                    dates, percentage=percentage, annualized=annualized, raw=raw
                )
                for component in self.components
            ),
            axis=1,
        )

    def get_fpx(self, dates, fdate, discountcurve, calendar):
        """
        Compute, at each of the dates, the forward price of each component given the final date, discount curve and calendar.

        Parameters
        ----------
        dates : pandas.DatetimeIndex
            Dates at which to compute the prices.
        fdate : str or pandas.Timestamp
            Future date or time horizon of the forward price.
        discountcurve : discount_curves.DiscountCurve
            Gives the discounts for the computation of the forward prices.
        calendar : calendars.DayCountCalendar
            Calendar used in the computation.

        Returns
        -------
        pandas.DataFrame
            DataFrame of forward prices, columns are components and index is ``dates``. When a forward price is not available ``numpy.nan`` is given.

        Notes
        -----
            The forward price of each component is calculated using :py:meth:`DeterministicVolAsset.get_fpx <DeterministicVolAsset.get_fpx>`.
        """
        return pd.concat(
            (
                component.get_fpx(dates, fdate, discountcurve, calendar)
                for component in self.components
            ),
            axis=1,
        )

    def get_vol(
        self, dates, tenors=None, strike=None
    ):  # TODO: tenors and strike are never used.
        """
        Get volatilities for each component at all given dates.

        Parameters
        ----------
        dates :  pandas.DatetimeIndex, list of strings, pandas.Timestamp, or string
            Dates for which to get the volatilities.
        tenors : optional
            Never used. Defaults to ``None``.
        strike : optional
            Never used. Defaults to ``None``.

        Returns
        -------
        pandas.DataFrame
             DataFrame of volatilities, columns are components and index is ``dates``. When a volatility is not available ``numpy.nan`` is given.
        """
        return pd.concat(
            (
                component.get_vol(dates, tenors=tenors, strike=strike)
                for component in self.components
            ),
            axis=1,
        )

    def get_divrate(self, dates):
        """
        Get dividend rates for each component at all given dates.

        Parameters
        ----------
        dates :  pandas.DatetimeIndex, list of strings, pandas.Timestamp, or string
            Dates for which to get the dividend rates.

        Returns
        -------
        pandas.DataFrame
            DataFrame of dividend rates, columns are components and index is ``dates``. When a dividend rate is not available ``numpy.nan`` is given.
        """
        return pd.concat(
            (component.get_divrate(dates) for component in self.components), axis=1
        )

    def get_correlation_matrix(self, dates):
        """Compute the correlation matrix of the component prices at the given dates.

        Parameters
        ----------
        dates : pandas.DatetimeIndex
            Dates at which to compute the correlation matrix.

        Returns
        -------
        numpy.ndarray, shape(dates.size, len(self.components), len(self.components))
            Empirical correlation matrix computed form the (recent) past component prices at all given dates. If at a given date less than 125 past prices are available, the corelation matrix for such date will be filled with ``numpy.nan``.
        """
        corr = np.full(
            (dates.size, len(self.components), len(self.components)), np.nan
        )  # TODO: formatting of dates is needed
        if self.corr is None:
            all_dates = self.get_dates()
            all_dates = all_dates[
                (all_dates <= dates[-1])
                & (all_dates >= dates[0] - pd.Timedelta(days=365))
            ]
            all_values = self.get_component_values(all_dates)
            for i in range(dates.size):
                temp_dates = all_dates[all_dates <= dates[i]][-250:]
                if temp_dates.size >= 125:  # TODO: improve this
                    corr[i] = all_values.loc[temp_dates].corr()
        else:  # TODO: self.corr is never assigned something different than None. It will never go into this condition.
            for i in range(dates.size):
                corr[i] = self.corr[dates[i]]

        return corr

    # def generate_draws_old(self, no_obsdates, no_valuation_dates, no_sims, start_dates):
    #     """
    #     Generate correlated random draws based on a provided correlation matrix.
    #     This method is needed for the new class :py:meth:`DiffusionMC <pricing.mc_engines.DiffusionMC>`.
    #
    #
    #     Parameters
    #     ----------
    #     no_obsdates : int
    #         Number of observation dates.
    #     no_valuation_dates : int
    #         Number of valuation dates.
    #     no_sims : int
    #         Number of simulations.
    #     start_dates : array_like
    #         # TODO: start_dates is not needed for Lognormal, VolModel or MultiAssetHeston. Do it properly.
    #
    #     Returns
    #     -------
    #     b_draw : np.ndarray
    #         A 4D array containing correlated random draws. The dimensions are
    #         (number of simulations, number of observation dates, number of valuation dates, number of assets).
    #
    #     Notes
    #     -----
    #     This method generates random draws :math:`d` from a standard normal distribution with
    #     shape `(no_sims, no_obsdates)` and covariance equal to the identity matrix. The method then
    #     reshapes the `draws_small` array to include the number of assets in the last dimension.
    #
    #     The new `draws` array is created by replicating `draws_small` along a new axis for the
    #     number of valuation dates. The `draws` array is then transposed to put the valuation dates
    #     as the third dimension.
    #
    #     The correlation structure is applied by multiplying the Cholesky decomposition of the correlation
    #     matrix :math:`B` with the corresponding slice of `draws`. In mathematical terms, if the
    #     correlation matrix is :math:`C`, the Cholesky decomposition is given by :math:`B = Chol(C)`.
    #     The correlated draws :math:`d_c` are then computed by :math:`d_c = Bd`.
    #
    #     If the `corr` size is 1, the correlation structure does not need to be applied, so `b_draw`
    #     is set to be equal to `draws`.
    #
    #     References
    #     ----------
    #     See Section "generate_draws" in "MC engines Documentation.ipynb" for details.
    #     """
    #     corr = self.get_correlation_matrix(start_dates)
    #     no_assets = corr.shape[-1]
    #     draws_small = multivariate_normal.rvs(cov=np.identity(no_assets), size=(no_sims, no_obsdates))
    #     draws_small = draws_small.reshape((no_sims, no_obsdates) + (no_assets,))
    #     draws = np.full((no_valuation_dates,)+draws_small.shape, 1)*draws_small
    #     draws = draws.transpose((1, 2, 0, 3))
    #     if corr.size > 1:
    #         if corr.ndim == 2:
    #             b = np.linalg.cholesky(corr)*np.ones((no_valuation_dates,)+corr.shape)
    #         elif corr.ndim == 3 and corr.shape[2] == 1:
    #             b = np.linalg.cholesky(corr[0]) * np.ones((no_valuation_dates,) + corr[0].shape)
    #         else:
    #             b = np.full(corr.shape, np.nan)
    #             for i in range(corr.shape[0]):
    #                 b[i] = np.linalg.cholesky(corr[i])
    #         b_draw = np.full(draws.shape, np.nan)
    #         for i in range(draws.shape[0]):
    #             for j in range(draws.shape[1]):
    #                 for k in range(draws.shape[2]):
    #                     b_draw[i, j, k] = np.matmul(b[k], draws[i, j, k])
    #     else:
    #         b_draw = draws
    #     return b_draws

    def generate_draws(self, no_obsdates, no_valuation_dates, no_sims, start_dates):
        """
        Generate correlated standard Gaussian random draws using the correlation matrix of the components at the start dates.

        This method generates random draws based on the Cholesky decomposition of the provided correlation matrix.
        It then applies the decomposition to raw random draws to generate correlated random numbers.

        Parameters
        ----------
        no_obsdates : int
            Number of observation dates.
        no_valuation_dates : int
            Number of valuation dates.
        no_sims : int
            Number of simulations.
        start_dates :  pandas.DatetimeIndex, list of strings, pandas.Timestamp, or string
            Starting dates.

        Returns
        -------
        numpy.ndarray, shape (no_sims, no_obsdates, no_valuation_dates, no_assets)
            Correlated random draws of the standard Gaussian distribution.

        Warnings
        --------
        - ``start_dates`` must be of size ``no_valuation_dates``.

        Notes
        -----
        The method follows the following mathematical steps:

        1. Raw (uncorrelated) draws are generated as:

           .. math::
                d^\\text{raw}_{ijkl} \\sim N(0, 1)\,,
           where :math:`i` indexes simulations, :math:`j` indexes observation dates, :math:`k` indexes valuation dates,
           and :math:`l` indexes assets.
        2. Cholesky decomposition is applied on the correlation matrix of the k-th valuation date :math:`\Sigma_k`, to obtain a lower triangular matrix :math:`L_k` that satisfies

           .. math:: \\Sigma_k = L_k L_k^T\,.

        3. The raw draws are then transformed using the Cholesky matrix to get the correlated random draws:

           .. math:: d_{ijkl} = \\sum_m L_{klm} d^\\text{raw}_{ijkm}\,.
        """
        start_dates = afsfun.dates_formatting(start_dates)
        corr = self.get_correlation_matrix(start_dates)
        no_assets = corr.shape[-1]
        raw_draws = np.random.randn(no_sims, no_obsdates, no_valuation_dates, no_assets)
        L = np.empty((no_valuation_dates, no_assets, no_assets))
        for k in range(no_valuation_dates):
            L[k] = np.linalg.cholesky(corr[k])

        draws = np.einsum("klm,ijkm->ijkl", L, raw_draws)

        return draws

    def generate_paths(
        self,
        start_dates,
        step_dates,
        draws,
        intervals,
        discount_curve,
        forward_measure=True,
    ):
        """
        Generate paths for the MultiAsset based on the provided inputs.

        Parameters
        ----------
        start_dates : pandas.DatetimeIndex
            The start dates for which to generate the paths. Represented as :math:`\\{t^0_k\\}_{k=0}^K` .
        step_dates : pandas.DatetimeIndex
            The step dates for which to generate the paths. Represented as :math:`\\{t^1_j\\}_{j=0}^J` .
        draws : numpy.ndarray
            The draws for which to compute the paths. Represented as :math:`\\textrm{draws}[i,j,k,l]` where :math:`i` indexes the simulation number and $l$ indexes the component.
        intervals : numpy.ndarray
            Time intervals for the simulation. Represented as :math:`\\textrm{intervals}[j,k]=\\textrm{DCC}(t^1_{j-1,k}, t^1_{j,k})` where

            .. math::
                t^1_{j,k}:= \\begin{cases} t^1_{j} \qquad \\text{ for }j=0,\ldots,J \quad \\text{ (independent of } k\\text{)}\,,\\\\t^0_{k} \qquad \\text{ for }j=-1\,.\end{cases}

            The function :math:`\\textrm{DCC}` represents the day count convention and will depend on the calendar used. For simplicity, it can be thought of as :math:`\\textrm{DCC}(t,t')=t'-t` as a year fraction.
        discount_curve : discount_curves.DiscountCurve
            Gives the discounts for the computation of the T-forward measure numeráires.
        forward_measure : bool, optional
            Flag indicating if forward measure is used. Defaults to ``True``.

        Returns
        -------
        numpy.ndarray
            It returns the simulation paths for the component prices. Represented as

            .. math::
                \\textrm{paths}[i,j,k,l] = S^l_{i,k}(t^1_{j-1,k})\,, \quad j\in\{0, \ldots, J+1\}\,.

        Notes
        -----
        The path of each component is generated by :py:meth:`NormalAsset.generate_paths <NormalAsset.generate_paths>` or :py:meth:`LognormalAsset.generate_paths <LognormalAsset.generate_paths>` according to the component type.
        """
        paths = np.full(
            (draws.shape[0], draws.shape[1] + 1, draws.shape[2], draws.shape[3]), np.nan
        )
        for l in range(draws.shape[3]):
            asset_draws = draws[:, :, :, l].reshape(draws.shape[:3] + (1,))
            component_path = self.components[l].generate_paths(
                start_dates=start_dates,
                step_dates=step_dates,
                draws=asset_draws,
                intervals=intervals,
                discount_curve=discount_curve,
                forward_measure=forward_measure,
            )
            paths[:, :, :, l] = component_path[:, :, :, 0]
        return paths


class Basket(MultiAsset):
    def __init__(
        self,
        equity_objects,
        weights,
        calendar=None,
        funding_rate_basket=0,
        reinvestment=False,
        withholding_tax=0,
    ):
        MultiAsset.__init__(self, *equity_objects)
        self.weights = np.array(weights)
        self.yieldsdiv = True
        self.calendar = calendar
        self.funding_rate_basket = funding_rate_basket
        self.reinvestment = reinvestment
        self.withholding_tax = withholding_tax

    def get_value(self, dates):
        return np.sum(self.weights * MultiAsset.get_value(self, dates), axis=1)

    def generate_paths(
        self,
        start_dates,
        step_dates,
        draws,
        intervals,
        discount_curve,
        forward_measure=True,
    ):
        separate_paths = MultiAsset.generate_paths(
            self,
            start_dates=start_dates,
            step_dates=step_dates,
            draws=draws,
            intervals=intervals,
            discount_curve=discount_curve,
            forward_measure=forward_measure,
        )
        paths = np.sum(
            separate_paths * self.weights, axis=-1
        )  # Last axis, the corresponding to set of underlyings
        return paths.reshape(paths.shape + (1,))

    def get_arithmetic_return(self, dates, percentage=False):
        dates = afsfun.dates_formatting(dates)
        # Raw is False, so it takes the log. Then we exponentiate so it gives R_t/R_{t-1}
        raw = np.sum(
            self.weights
            * np.exp(
                MultiAsset.get_return(
                    self, dates, percentage=False, annualized=False, raw=False
                )
            ),
            axis=1,
        )
        all_dates_total = (
            self.get_dates()
        )  # get_dates is a method inherited from MultiAsset
        # if dates[0] == all_dates_total[0]:  # If there were no previous dates
        #     dates = dates[1:]  # Dates is shifted
        prev_dates = [all_dates_total[all_dates_total < date][-1] for date in dates]
        prev_dates = pd.to_datetime(prev_dates).sort_values()
        returns = raw
        if self.calendar is not None:
            calendar_int = self.calendar.interval(
                prev_dates, dates
            )  # Vector of interval between dates
            calendar_interval_series = pd.Series(
                calendar_int, dates
            )  # Transform to a series
            if self.reinvestment:  # Basket with reinvestment
                divrate_annual = MultiAsset.get_divrate(self, dates)
                divrate_int = divrate_annual.mul(calendar_interval_series, axis=0)
                returns = returns + np.sum(
                    self.weights * divrate_int * (1 - self.withholding_tax), axis=1
                )
            if self.funding_rate_basket < 0:
                returns = returns * (1 + self.funding_rate_basket * calendar_int)
        if percentage:
            returns = returns * 100
        return returns


class ExposureIndex(Basket):
    def __init__(
        self,
        underlyings,
        weights,
        calendar,
        strike_date,
        funding_rate,
        overnight_rate,
        max_exposure,
        target_vol,
        window,
        biased=True,
        mean=True,
        tolerance=0,
        initial_index=1,
        initial_exposure=None,
        reinvestment=False,
        withholding_tax=0,
        bool_funding_rate_basket=False,
        exp_vol_delay=1,
    ):
        self.strike_date = strike_date
        self.bool_funding_rate_basket = bool_funding_rate_basket
        if self.bool_funding_rate_basket:
            self.funding_rate_basket = funding_rate
            self.funding_rate = 0
        else:
            self.funding_rate = funding_rate
            self.funding_rate_basket = 0
        self.overnight_rate = overnight_rate
        self.max_exposure = max_exposure
        self.target_vol = target_vol
        self.tolerance = tolerance
        self.window = window
        self.exp_vol_delay = exp_vol_delay
        self.initial_index = initial_index
        self.biased = biased
        self.mean = mean
        if initial_exposure is None:
            self.initial_exposure = self.max_exposure
        else:
            self.initial_exposure = initial_exposure
        Basket.__init__(
            self,
            equity_objects=underlyings,
            weights=weights,
            calendar=calendar,
            funding_rate_basket=self.funding_rate_basket,
            reinvestment=reinvestment,
            withholding_tax=withholding_tax,
        )

    def _get_exposures(self, realized_vol_prev):
        # Exp=min(maxExp,targetVol/RealizedVol). realized_vol_prev must be a np.array

        realized_vol_prev = realized_vol_prev[
            realized_vol_prev != 0
        ]  # Gives NaN if it happens
        exposures = np.minimum(self.max_exposure, self.target_vol / realized_vol_prev)
        # exposure = pd.Series(exposure, dates)
        if self.tolerance > 0:
            target_exposures = exposures
            exposures = target_exposures
            exposures[0] = self.initial_exposure  # Exp(0) = Exp_initial
            if len(exposures) > 1:
                exposures[1] = self.initial_exposure  # Exp(1) = Exp_initial
            for i in range(len(exposures)):
                if (
                    i > 1
                    and np.abs(exposures[i - 1] / target_exposures[i - 1] - 1)
                    <= self.tolerance
                ):
                    exposures[i] = exposures[i - 1]

        return exposures

    def _get_index(
        self,
        returns,
        exposures_previous,
        short_rates_previous,
        calendar_int,
        initial_index=None,
    ):
        # All inputs must be np.arrays
        if initial_index is None:
            initial_index = self.initial_index
        bond_return = 1 + short_rates_previous * calendar_int
        prod = (
            exposures_previous * returns
            + (1 - exposures_previous) * bond_return
            + self.funding_rate * calendar_int
        )  # Recursive formula
        if np.ndim(prod) == 2:  # When it is called within generate_paths (at the end).
            prod_final = np.full((prod.shape[0], prod.shape[1] + 1), np.nan)
            prod_final[:, 0] = initial_index
            # The first element of the second index corresponds to the start_date
            prod_final[:, 1:] = prod
            # By definition, returns[:, 0] corresponds to the first "future" (simulated) date, so when make the cumprod it is the correct formula
            # I_t= \prod_{i=0}^{t-1} (I_{t-i}/I_{t-i-1}) * I_0
            prod = prod_final
        else:
            prod[0] = initial_index
            # When this is called within get_value, prod[0]=Index[t_0]/Index[t_{-1}], but for Index to be the cumprod, this should be Index[t_0]
        index = np.cumprod(prod, axis=-1)

        return index

    def _get_prev_dates(self, delay, max_date):
        total_dates = Basket.get_dates(self)  # Dates of the basket
        dates = self.get_dates_index(max_date)
        prev_dates = [
            total_dates[total_dates < date][-delay] for date in dates
        ]  # Slicing properly, we can obtain the dates shifted one day
        prev_dates = pd.to_datetime(prev_dates)  # Otherwise is a list of Timestamps

        return prev_dates

    def get_dates_index(self, max_date=None):
        total_dates = Basket.get_dates(self)  # Dates of the basket
        strike_date = pd.to_datetime(self.strike_date)  # To a panda object
        dates = total_dates[
            total_dates >= strike_date
        ]  # Dates needed for the index calculation
        if not (max_date is None):
            dates = dates[dates <= max_date]  # Dates needed given our valuation dates
        if dates[0] == total_dates[0]:  # If there were no previous dates
            dates = dates[1:]  # Dates is shifted

        return dates

    def get_prev_ewma_vol(self, max_date=None):
        dates = self.get_dates_index(max_date=max_date)
        previous_dates_vol = self._get_prev_dates(self.exp_vol_delay + 1, max_date)
        realized_vol_prev = Basket.get_ewma_vol(
            self,
            previous_dates_vol,
            l=1,
            window=self.window,
            biased=self.biased,
            mean=self.mean,
            arithmetic=True,
            logs=True,
            annualized=True,
        )

        realized_vol_prev = pd.Series(data=realized_vol_prev.values, index=dates)

        return realized_vol_prev

    def get_previous_exposures(self, max_date=None):
        dates = self.get_dates_index(max_date=max_date)
        previous_dates_vol = self._get_prev_dates(self.exp_vol_delay + 1, max_date)
        realized_vol_prev = Basket.get_ewma_vol(
            self,
            previous_dates_vol,
            l=1,
            window=self.window,
            biased=self.biased,
            mean=self.mean,
            arithmetic=True,
            logs=True,
            annualized=True,
        )
        exposures_previous = self._get_exposures(
            realized_vol_prev.values
        )  # Exposures at t-1 need realized volatilities at (t-exp_vol_delay)-1.

        exposures_previous = pd.Series(data=exposures_previous, index=dates)

        return exposures_previous

    def get_value(self, val_dates):
        # Valuation dates formatting and definitions
        val_dates = afsfun.dates_formatting(val_dates)
        max_val_date = val_dates[-1]

        # Dates considerations
        dates = self.get_dates_index(max_date=max_val_date)
        previous_dates1 = self._get_prev_dates(1, max_val_date)
        calendar_int = self.calendar.interval(
            previous_dates1, dates
        )  # Vector of interval between dates

        # Data for the index dynamics
        previous_dates_vol = self._get_prev_dates(self.exp_vol_delay + 1, max_val_date)
        realized_vol_prev = Basket.get_ewma_vol(
            self,
            previous_dates_vol,
            l=1,
            window=self.window,
            biased=self.biased,
            mean=self.mean,
            arithmetic=True,
            logs=True,
            annualized=True,
        )
        exposures_previous = self._get_exposures(
            realized_vol_prev.values
        )  # Exposures at t-1 need realized volatilities at (t-exp_vol_delay)-1.
        # Exposure_previous(t):=Exposure(t-1), Exposure_previous(t) is the one needed for the index at t exposures_previous = pd.Series(exposures_previous, dates)
        short_rates_previous = self.overnight_rate.get_value(
            previous_dates1
        )  # TODO This assumes that there are short rate values for the previous_dates1. This should not be done
        # like this. Use days_in_interval of BusinessCalendar.

        returns = Basket.get_arithmetic_return(self, dates)

        index = self._get_index(
            returns=returns.values,
            exposures_previous=exposures_previous,
            short_rates_previous=short_rates_previous.values,
            calendar_int=calendar_int,
        )
        index = pd.Series(data=index, index=dates)
        index_val = index.loc[val_dates]
        return index_val

    def generate_paths(
        self,
        start_dates,
        step_dates,
        draws,
        intervals,
        discount_curve,
        forward_measure=True,
    ):
        # Paths must be simulated only for the future, but we need data from the past
        # At this moment, we are assuming draws.shape(2)=1 and that step_dates has enough dates, see below
        dates = self.get_dates()  # All dates available
        start_date = pd.to_datetime(
            start_dates[0]
        )  # Let us assume start_dates has only one element
        last_date = dates[dates <= start_date][-1]  # Last date where we have data
        first_sim_date = pd.to_datetime(
            step_dates[step_dates > last_date][0]
        )  # This obviously assumes that step_dates has enough dates
        first_sim_date = pd.date_range(
            start=first_sim_date, end=first_sim_date
        )  # From a Timestamp to a DatetimeIndex
        previous_p_dates = dates[dates <= last_date]
        d = self.exp_vol_delay + 1
        previous_p_dates = previous_p_dates[
            -self.window - d :
        ]  # Dates from (last_date - w - d) days to last_date. We need it to calculate realized_vol

        initial_index = self.get_value(last_date.date())
        returns_p_values = self.get_arithmetic_return(
            previous_p_dates
        )  # We get the previous basket values (FB in term sheet ISIN: XS2278803171)
        returns_p_values = returns_p_values[
            :, np.newaxis, np.newaxis
        ]  # Let us add a nex axis
        # Equivalently, returns_p_values = returns_p_values.values.reshape(returns_p_values.values.shape + (1, 1,))

        separate_f_paths = MultiAsset.generate_paths(
            self,
            start_dates=start_dates,
            step_dates=step_dates,
            draws=draws,
            intervals=intervals,
            discount_curve=discount_curve,
            forward_measure=forward_measure,
        )
        # It gives an array where the second index corresponds to the dates such that j=0-> start_dates, j>=1-> step_dates. f stands for future.
        raw_separate_returns_f_paths = np.exp(
            np.diff(np.log(separate_f_paths), axis=1)
        )  # rsrfp[j]=sfp[j+1]/sfp[j], except for the last j, where j is the second index,
        # observation dates, i.e., rsrfp[0]=sfp(first_sim_date)/sfp(last date). This matches with
        # returns_p_values[-1]=Basket value[last_date]/Basket value[date previous to last_date]
        raw_returns_f_paths = np.sum(
            raw_separate_returns_f_paths * self.weights, axis=-1
        )
        raw_returns_f_paths = raw_returns_f_paths.reshape(
            raw_returns_f_paths.shape + (1,)
        )  # We add the four index, which represents the number of assets. Here just one
        total_dates = dates.union(step_dates)  # Total dates needed for the process
        previous_step_dates = total_dates[
            total_dates >= last_date
        ]  # Dates starting at last_date instead of first_sim_date
        previous_step_dates = previous_step_dates[
            :-1
        ]  # We remove the last element, so previous_step_dates and dates has the same length
        calendar_int = self.calendar.interval(
            previous_step_dates, step_dates
        )  # Vector of interval between dates
        calendar_int = np.array(calendar_int)
        if self.reinvestment:
            divrate_annual = MultiAsset.get_divrate(self, last_date.date())
            divrate_int = divrate_annual.values * np.min(
                calendar_int
            )  # At this moment, let us assume that divrate is a single number for each asset. We are not taking into
            # account weekends and holidays
            reinvestment_sum = np.sum(
                self.weights * divrate_int * (1 - self.withholding_tax), axis=0
            )[0]
            returns_f_paths = raw_returns_f_paths + reinvestment_sum
        else:
            returns_f_paths = raw_returns_f_paths
        if self.bool_funding_rate_basket:
            calendar_int_path = calendar_int.reshape(
                calendar_int.shape
                + (
                    1,
                    1,
                )
            )  # We add the third and fourth dimensions
            calendar_int_path = np.repeat(
                calendar_int_path[np.newaxis, :], returns_f_paths.shape[0], axis=0
            )  # We add the first dimension, simulations, repeating values
            returns_f_paths = returns_f_paths * (
                1 + self.funding_rate_basket * calendar_int_path
            )

        returns_p_paths = np.repeat(
            returns_p_values[np.newaxis, :], returns_f_paths.shape[0], axis=0
        )  # To an array suitable for paths, repeating values for simulations
        returns_paths = np.concatenate(
            (returns_p_paths, returns_f_paths), axis=1
        )  # We glue them together

        realized_vol_prev_paths = np.full(
            (
                separate_f_paths.shape[0],
                separate_f_paths.shape[1],
                separate_f_paths.shape[2],
                1,
            ),
            np.nan,
        )  # Realized volatility at previous time
        # needed for simulations (of the future). Previous time at t is t-1.
        exposures_previous_paths = np.full(
            (
                separate_f_paths.shape[0],
                separate_f_paths.shape[1],
                separate_f_paths.shape[2],
                1,
            ),
            np.nan,
        )  # Idem, previous, i.e., at t-1
        index_final_paths = np.zeros(
            (separate_f_paths.shape[0], step_dates.size + 1, 1, 1)
        )

        short_rates_previous = self.overnight_rate.get_fvalue(
            dates=last_date, future_dates=step_dates, calendar=self.calendar
        )  # Short rate from forward rates
        short_rates_previous = short_rates_previous[
            :-1
        ]  # Let us remove the last element, we need previous values, i.e., short rate at t_final is not needed.
        short_rates_previous = np.append(
            self.overnight_rate.get_value(last_date), short_rates_previous
        )  # but we need the value at last_date (t=t_0-1)
        for j in range(returns_p_paths.shape[1], returns_paths.shape[1] + 1):
            realized_vol_prev_paths[:, j - returns_p_paths.shape[1], 0, 0] = (
                self._get_ewma_vol_abstract(
                    returns=returns_paths[:, : j - self.exp_vol_delay - 1],
                    l=1,
                    window=self.window,
                    percentage=False,
                    annualized=True,
                    mean=self.mean,
                    biased=self.biased,
                )
            )
            # At time t, we need the exposure at t-1 and exp at t depends on realized volatility at t-exp_vol_delay
        exposures_previous_paths[:, :, 0, 0] = self._get_exposures(
            realized_vol_prev_paths[:, :, 0, 0]
        ).reshape(
            realized_vol_prev_paths[:, :, 0, 0].shape[0],
            realized_vol_prev_paths[:, :, 0, 0].shape[1],
        )

        index_final_paths[:, :, 0, 0] = self._get_index(
            returns=returns_paths[:, self.window + d :, 0, 0],
            exposures_previous=exposures_previous_paths[:, :-1, 0, 0],
            short_rates_previous=short_rates_previous,
            calendar_int=calendar_int,
            initial_index=initial_index[0],
        )
        # We introduce the future returns: returns_paths[:, self.window + d, 0, 0] == returns_f_paths[:, 0, 0, 0]

        return index_final_paths


# ----------------------------------------------------------------------------------------------
# Stochastic Volatility
# ----------------------------------------------------------------------------------------------


class VolModel(Underlying):
    """
    A parent class representing a volatility model over an underlying asset, derived from the Underlying class.

    Parameters
    ----------
    ticker : str
        The ticker symbol of the asset.
    name : str, optional
        The name of the asset. Defaults to ``None``.
    tenor : float, optional
        The tenor of the asset. Defaults to ``None``.
    yieldsdiv : bool, optional
        Flag indicating of the asset yields dividends. Defaults to ``False``.

    Attributes
    ----------
    ticker : str
        The ticker symbol of the asset.
    name : str
        The name of the asset.
    tenor : float
        The tenor of the asset.
    data : pandas.DataFrame
        DataFrame containing the historical data of the asset.
    yieldsdiv : bool
        Flag indicating of the asset yields dividends.
    calibration_data: pandas.DataFrame
        Data used for calibrating the volatility model.
    parameters: pandas.DataFrame
        DataFrame containing the parameters of the volatility model.

    Notes
    -----
    The parameters DataFrame should have the valuation dates included in the index. One way to do it is writing ``VolModel.parameters = VolModel.parameters.reindex(valuation_dates)`` before computing the price.
    """

    def __init__(self, ticker, name=None, tenor=None, yieldsdiv=False):
        Underlying.__init__(
            self, ticker=ticker, name=name, tenor=tenor
        )  # TODO: Think properly.
        self.yieldsdiv = yieldsdiv  # Not used right now.
        self.calibration_data = None
        self.parameters = None

    def compute_implied_vol(
        self,
        dates,
        discount_curve,
        option_prices_df,
        kind,
        calendar,
        und_type="lognormal",
    ):
        """
        For a given set of option prices at a given date, it returns the implied volatility according to
        the specified Black ('lognormal') or Bachelier's formula ('normal') assuming the underling price is 1.

        Parameters
        ----------
        dates :  pandas.TimeStamp
            Date of valuation. Represented as $t$.
        discount_curve : discount_curves.DiscountCurve
            Curve needed for the interest free rate.Represented as :math:`(t,T)\\mapsto D(t, T)` .
        option_prices_df : pandas.DataFrame
            The DataFrame containing options prices. The columns are the maturities and the index the strikes. Strikes, maturities and option prices are represented as :math:`K_i\,,T_j\,,M_{i,j}` respectively.
        kind : str
            ``"call"`` or ``"put"`` are admitted.
        calendar : calendars.DayCountCalendar
            Calendar for computation of time intervals.
        und_type : str, optional
            Dynamics of the underlying. Defaults to ``"lognormal"``.

        Returns
        -------
        pandas.DataFrame
            Implied volatilities. The columns are the maturities and the index the strikes. Represented as :math:`\\sigma_{i,j}` .

        Raises
        ------
        NameError
            If ``kind`` is not ``"call"`` or ``"put"``; or if ``und_type`` is not ``"lognormal"``.

        Notes
        -----
        For call options, :math:`\\sigma_{i,j}` is the solution, computed numerically, of the equation

        .. math::
            \mathrm{Bl}_c(S_t, K_i, \sigma, r_j, T_j-t) = M_{i,j}\,,

        where $S_t$ is the price of the underlying at time $t$ and the risk-free interest rate $r_j$ is determined by :math:`D(t,T_j)=\\exp(-r_j(T_j-t))`.

        For put options, :math:`\\sigma_{i,j}` is the solution, computed numerically, of the equation

        .. math::
            \mathrm{Bl}_p(S_t, K_i, \sigma, r_j, T_j-t) = M_{i,j}\,.
        """
        maturities = np.array(option_prices_df.columns)
        px = 1  # TODO: Import this value from historical data
        tau = calendar.interval(dates, maturities)
        discount = discount_curve.get_value(
            dates=dates, future_dates=maturities, tenors=None, calendar=calendar
        )

        def difference(
            S, K, T, discount, sigma, model_price
        ):  # noqa (ignore error N803)
            """
            Difference between option prices using discount factors.

            Parameters
            ----------
                S : float
                    The underlying stock price.
                K : float
                    The option's strike price.
                T : float
                    Time to expiration in years.
                discount : float
                    The discount factor (e^(-r*T)), where r is the risk-free interest rate.
                sigma : float
                    The implied volatility of the underlying stock.
                model_price : float
                    Option price for which we want to obtain the implied volatility.

                Returns
                -------
                float
                    Difference between the two option prices.
            """
            # d1 = (np.log(S / (discount * K)) + (0.5 * sigma**2) * T) / (
            #     sigma * np.sqrt(T)
            # )
            d1 = (np.log(S / K) + (0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
            d2 = d1 - sigma * np.sqrt(T)
            if und_type == "lognormal":
                if kind == "call":
                    price = S * norm.cdf(d1) - K * discount * norm.cdf(d2)
                elif kind == "put":
                    price = K * discount * norm.cdf(-d2) - S * norm.cdf(-d1)
                else:
                    raise NameError("kind not implemented.")
            else:  # TODO: Implement Bachelier model
                raise NameError("und_type not implemented.")
            return (price - model_price) ** 2

        implied_vols = pd.DataFrame(
            columns=option_prices_df.columns, index=option_prices_df.index
        )  # Empty dataframe with the same shape.
        for i, (index, row) in enumerate(option_prices_df.iterrows()):
            for j, column in enumerate(option_prices_df.columns):
                model_price = row[column]
                if tau.shape == ():
                    S, K, T, disc = (
                        px,
                        index,
                        tau[()],
                        discount,
                    )  # TODO: We assume a single spot date. Generalize.
                else:
                    S, K, T, disc = (
                        px,
                        index,
                        tau[j],
                        discount[j],
                    )  # TODO: We assume a single spot date. Generalize.

                def objective_func(sigma):
                    return difference(S, K, T, disc, sigma, model_price)

                sigma_init = 0.1  # Initial guess
                bounds = ((0, 3.5),)
                sigma_init = dual_annealing(func=objective_func, bounds=bounds).x
                # TODO: Use another method? The results of 2023_05_03_tst_compute_implied_vol are not good enough for some cases. See Notas ir_models for solutions (?).
                result = minimize(
                    objective_func,
                    sigma_init,
                    bounds=bounds,
                    method="powell",
                    options={"xatol": 1e-12, "fatol": 1e-12, "disp": False},
                )
                # disp=True can be used for printing Optimization terminated successfully, Current function value, Iterations and Function evaluations.
                implied_vols.loc[index, column] = result.x[0]

        return implied_vols

    def compute_implied_vol_py(
        self,
        dates,
        discount_curve,
        option_prices_dfs,
        kind,
        calendar,
        und_pxs,
        und_type="lognormal",
    ):
        """
        For a given set of option prices at given dates, it returns the implied volatility (only implemented for Black-Scholes model).

        Parameters
        ----------
        dates :  pandas.DatetimeIndex
            Dates of valuation. Represented as :math:`t_i` .
        discount_curve : discount_curves.DiscountCurve
            Curve needed for the interest free rate. Represented as :math:`(t,t')\\mapsto D(t, t')` .
        option_prices_dfs : dict
            Dictionary which assigns a pandas.DataFrame containing options prices to each date in ``dates``. For each DataFrame the columns are the times to expiry and the index and strikes.
            For each date $t_i$, the strikes, times to expiry and option prices are represented as :math:`K_{i,j}\,,T_{i,k}\,,M_{i,j,k}` respectively.
        kind : str
            ``"call"`` or ``"put"`` are admitted.
        calendar : calendars.DayCountCalendar
            Calendar for computation of time intervals.
        und_pxs : dict
            Dictionary which assigns a spot price to each date. Represented as :math:`t_i\\mapsto S(t_i)` .
        und_type : str, optional
            Dynamics of the underlying. Defaults to ``"lognormal"``.

        Returns
        -------
        dict
            Dictionary which assigns a pandas.DataFrame containing the implied volatilities to each date.
            For each DataFrame the columns are the times to expiry and the index and strikes. The implied volatilities are represented as represented as :math:`\\sigma_{i,j,k}` .

        Notes
        -----
        For call options, :math:`\\sigma_{i,j,k}` is the solution, computed numerically, of the equation

        .. math::
            \mathrm{Bl}_c\\left(S(t_i), K_{i,j}, \sigma, r_{i,k}, q_i, T_{i,k}\\right) = M_{i,j,k}\,,

        where the risk-free interest rate $r_{i,k}$ is determined by :math:`D(t_i,t_i+T_{i,k})=\\exp(-r_{i,k}T_{i,k})` and the dividend rate $q_i$ at time $t_i$ is obtained from ``self.get_divrate(dates)`` if available, if not $q_i$ is set to $0$.

        For put options, :math:`\\sigma_{i,j,k}` is the solution, computed numerically, of the equation

        .. math::
            \mathrm{Bl}_p\\left(S(t_i), K_{i,j}, \sigma, r_{i,k}, q_i, T_{i,k}\\right) = M_{i,j,k}\,.
        """
        dict_und = {"lognormal": "black_scholes_merton"}
        dict_kind = {"call": "c", "put": "p"}
        implied_vols_dict = {}
        dates_series = pd.Series(dates)
        for date in dates_series:
            option_prices_df = option_prices_dfs[date]
            und_px = und_pxs[date]
            if self.yieldsdiv:
                try:
                    q = self.get_data("Dividend", date)
                except:
                    q = 0
            else:
                q = 0
            taus = np.array(option_prices_df.columns)
            prices = []
            strikes = []
            for tau in taus:
                strikes.append(option_prices_df[tau].index.values)
                prices.append(option_prices_df[tau].values)
            tau = []
            discount = []
            for i in range(len(taus)):
                tau.append(taus[i] * np.ones(len(strikes[i])))
                discount.append(
                    discount_curve.get_value(
                        dates=date, tenors=taus[i], calendar=calendar
                    )
                    * np.ones(len(strikes[i]))
                )
            r = -np.log(discount) / tau
            # strikes = option_prices_df.index
            # option_prices_df = option_prices_df.to_series()
            implied_vols = np.empty((len(strikes[0]), len(taus)))
            for i in range(len(taus)):
                for j in range(len(strikes[i])):
                    implied_vol = py_vollib_vectorized.vectorized_implied_volatility(
                        price=prices[i][j],
                        S=und_px,
                        K=strikes[i][j],
                        t=tau[i][j],
                        r=r[i][j],
                        flag=dict_kind[kind],
                        q=q,
                        model=dict_und[und_type],
                        return_as="dataframe",
                    )
                    implied_vols[j, i] = implied_vol.values[0, 0]
            implied_vols_df = pd.DataFrame(
                data=implied_vols, index=strikes[0], columns=taus
            )
            implied_vols_dict[date] = implied_vols_df
        return implied_vols_dict

    def option_price_functions(self, params, spot, strikes, maturities, kind):
        """
        Return option prices according to the model.

        Parameters
        ----------
        params: numpy.ndarray
            Array of model parameters.
        strikes : numpy. ndarray
            Array of option strikes.
        maturities : list
            Array of option maturities.
        kind : str
            ``"call"`` or ``"put"`` are admitted.

        Returns
        -------
        numpy.ndarray
            Array of option prices.
        """
        pass

    def fit_to_options(self, date, options_df, kind):
        """
        Fits the model to a set of options data on a specified date.

        Parameters
        ----------
        date: pandas.Timestamp
            Date of valuation.
        options_df: pandas.DataFrame
            The DataFrame containing options prices. The columns are the times to expiry (in years) and the index the strikes.
        kind : str
            ``"call"`` or ``"put"`` are admitted.
        """
        pass

    def generate_paths(
        self,
        start_dates,
        step_dates,
        draws,
        intervals,
        discount_curve,
        forward_measure=True,
    ):
        """
        Return the paths of the asset whose dynamics is determined by a given volatility model.

        Parameters
        ----------
        start_dates : pandas.DatetimeIndex
            Starting dates for the simulation.
        step_dates : pandas.DatetimeIndex
            Step dates for the simulation.
        draws : numpy.ndarray
            Random draws generated for the simulation.
        intervals : numpy.ndarray
            Time intervals for the simulation.
        discount_curve : discount_curves.DiscountCurve
            Gives the discounts for the computation of the T-forward measure numeráires.
        forward_measure : bool, optional
            If ``True``, it uses the forward measure for the simulation. Defaults to ``True``.

        Returns
        -------
        numpy.ndarray
            Simulation paths.
        """
        pass


# ----------------------------------------------------------------------------------------------
# Volatility models
# ----------------------------------------------------------------------------------------------


class Heston(VolModel):
    """
    A class for simulating the Heston stochastic volatility model using Andersen's Quadratic Exponential (QE) method.

    The Heston model is a widely used model for the dynamics of asset prices and their volatilities. The QE method is an efficient discretization scheme for
    simulating the Heston model, particularly for cases with high mean reversion and low volatility of volatility.

    We will follow the notation used in [Andersen, 2008].

    Attributes
    ----------
    parameters : pandas.DataFrame
        Each column is numpy.ndarray of length equal to the number of valuation dates. Following Andersen's notation [Andersen, 2008]:

        kappa : numpy.ndarray
            The mean reversion rate of the volatility process.
        epsilon : numpy.ndarray
            The volatility of the volatility process.
        theta : numpy.ndarray
            The long-term mean of the volatility process.
        rho : numpy.ndarray
            The correlation between the asset price and its volatility.
        v0 : numpy.ndarray
            :math:`\hat{V}(t=0)`, asset price at the initial valuation dates.
        x0 : numpy.ndarray
            :math:`\hat{X}(t=0)`, asset price at the initial valuation dates. If x0 is not specified the market value of the underlying is used.

    References
    ----------
    - [Andersen, 2008] Andersen, L. (2008). Simple and efficient simulation of the Heston stochastic volatility model. Journal of Computational Finance, 11(3), 1-42.

    See also
    --------
    Underlyings_Jupyter: See that documentation for more details.

    Notes
    -----
    The parameters DataFrame should have the valuation dates included in the index. One way to do it is writing ``vol_model.parameters = vol_model.parameters.reindex(valuation_dates)`` before computing the price.

    """

    # TODO: Adapt this docstring and check that all of them are alright.
    def __init__(self, ticker, name=None, tenor=None, yieldsdiv=False):
        VolModel.__init__(self, ticker, name, tenor, yieldsdiv)
        self.parameters = pd.DataFrame(
            columns=["kappa", "epsilon", "theta", "rho", "v0", "x0"]
        )
        # x0 is obtained from the ticker if not explicitly indicated when instantiating an object of Heston class.

    def option_price_functions(self, params, spot, strikes, times_exp, kind="call"):
        """
        Return European option prices (Call/Put) using the Fast Fourier Transform (FFT)
        algorithm based on Lewis expression. We use the library pyfeng.

        Parameters
        ----------
        params: numpy.ndarray
            Array of model parameters ("kappa", "epsilon", "theta", "rho", "v0")
        spot: float
            Spot value :math:`x_0`
        strikes : numpy.ndarray
            Array of option strikes.
        times_exp : numpy.ndarray
            Array of times to expiry (in years).
        kind : string, default = 'call'
            'call' or 'put' are admitted.

        Returns
        -------
        out : ndarray
            Two dimensional array with the option prices for the given strikes and times_exp:
            prices(S,T). See "Underlyings_Jupyter"

        References
        ----------
        - Lewis AL (2000) Option valuation under stochastic volatility: with Mathematica code. Finance Press.

        See also
        --------
        Underlyings_Jupyter: See that documentation for more details.

        """
        kappa = params[0]
        epsilon = params[1]
        theta = params[2]
        rho = params[3]
        v0 = params[4]
        # Definitions in the docstring of class SvABC in pyfeng.sv_abc.py
        sigma = v0  # variance at t=0
        vov = epsilon  # volatility of volatility
        rho = rho  # correlation between price and volatility
        mr = kappa  # mean-reversion speed (kappa)
        theta = (
            theta  # long-term mean of volatility or variance. If None, same as sigma
        )
        # We instantiate a Heston object within pyfeng
        heston = pf.HestonFft(
            sigma, vov=vov, rho=rho, mr=mr, theta=theta, intr=0, divr=0.0, is_fwd=False
        )  # TODO: get intr from data.
        # We compute the price using FFT
        kind_map = {"call": 1, "put": -1}
        cp = kind_map.get(kind)
        if cp is None:
            raise ValueError("Invalid kind.")
        prices = np.column_stack(
            [heston.price_fft(strikes, spot, texp, cp=cp) for texp in times_exp]
        )  # There is an annoying print(self.theta) in the method mgf_logprice of
        # class HestonFft(heston.HestonABC, FftABC). TODO: Remove since this is a problem for the optimitation procedure.
        return prices

    def fit_to_options(self, date, options_df, kind="call", recalc=True):
        """
        Adjust parameters to option prices.

        Parameters
        ----------
        date : pandas.Timestamp
            Valuation date.
        options_df : pandas.DataFrame
            The DataFrame containing options information. The columns are the times to expiry (in years) and the index the strikes.
        kind : string, default = 'call'
            'call' or 'put' are admitted.
        recalc : boolean
            If ``True`` the result of the stochastic optimization is used as an initial guess for optimize.minimize.

        Returns
        -------
        None
            It assigns the parameters that minimize the difference of prices divided by the vega.

        References
        ----------
        - [Cui, 2017] "Full and fast calibration of the Heston stochastic volatility model" Y. Cui, S. del Baño Rollin, G. Germano. https://doi.org/10.1016/j.ejor.2017.05.018
        """
        # TODO: WRITE EXAMPLE using the tst_calibration
        self.calibration_data = options_df  # TODO: CHECK
        times_exp = np.array(options_df.columns)
        strikes = np.array(options_df.index)
        market_prices = options_df.values
        vega = 1  # TODO: Method for computing BS Vega
        spot = 1  # TODO: Get x0 from historical data.

        def difference(params):
            """
            Returns difference of model prices and market prices divided by the vega, which tries
            to approximate the difference of volatilities, up to first term.

            Parameters
            ----------
            params: numpy.ndarray
                Array of model parameters ("kappa", "epsilon", "theta", "rho", "v0")
            """
            model_prices = self.option_price_functions(
                params, spot, strikes, times_exp, kind
            )
            differences = np.sum(((market_prices - model_prices) / vega) ** 2)
            return differences

        bounds = [
            (0.5, 5),
            (0.05, 0.95),
            (0.05, 0.95),
            (-0.9, -0.1),
            (0.05, 0.95),
        ]  # From Table 5 of [Cui, 2017] ("kappa", "epsilon", "theta", "rho", "v0",)
        minimum = dual_annealing(func=difference, bounds=bounds).x
        if recalc:
            minimum = minimize(
                difference,
                minimum,
                bounds=bounds,
                method="powell",
                options={"xatol": 1e-12, "fatol": 1e-12, "disp": True},
            ).x
            # disp=False can be used for not printing Optimization terminated successfully, Current function value, Iterations and Function evaluations.

        return minimum
        # self.parameters.loc[date] = minimum  TODO: Use this.

    @staticmethod
    def generate_draws(no_obsdates, no_valuation_dates, no_sims, start_dates=None):
        """
        Normal random variables needed for generating the paths of volatilities and asset prices.

        Parameters
        ----------
        no_obsdates : int
            Number of observation dates.
        no_valuation_dates : int
            Number of valuation dates.
        no_sims : int
            Number of simulations.

        Returns
        -------
        numpy.array
            | draws[i,j,k,0] is :math:`Z_V` in [Andersen, 2008] for the i-th simulation and the k-th valuation date for the j-th step (simulation date).
            | draws[i,j,k,1] is :math:`Z` in [Andersen, 2008] for the i-th simulation and the k-th valuation date for the j-th step (simulation date).
        """
        draws = np.random.randn(
            no_sims, no_obsdates, no_valuation_dates, 2
        )  # We use np.random.randn instead of multivariate_normal.rvs (used in mc_enginces).
        return draws

    def generate_paths_vol(self, start_dates, draws, intervals):
        # TODO: We can use Numba here with @njit and prange to optimize the for loop (?).
        # TODO: Include step_dates although it is not necessary?
        """
        Compute the variance in the Heston model using Andersen's Quadratic Exponential (QE) method.

        Parameters
        ----------
        start_dates :  pandas.DatetimeIndex or pandas.Timestamp
            Starting dates for the simulation.

        intervals : numpy.ndarray
            Time intervals for the simulation.

        draws : numpy.ndarray
            4 dimensional array with the draws for the simulation following the standard conventions for the indexes.
            Vectors of normals :math:`Z_V` [Andersen, 2008].

        Returns
        -------
        numpy.ndarray
            :math:`V(t)` in [Andersen, 2008].

        References
        ----------
        See Section "compute_vol" in "Underlyings_Jupyter" for details.

        """
        kappa = self.parameters.loc[start_dates, "kappa"].values
        epsilon = self.parameters.loc[start_dates, "epsilon"].values
        theta = self.parameters.loc[start_dates, "theta"].values
        v0 = self.parameters.loc[start_dates, "v0"].values

        def compute_next_step_vol(vt, dt, kappa, epsilon, theta, draw):
            """
            Compute the next-step variance in the Heston model using Andersen's Quadratic
            Exponential (QE) method.

            These parameters are used to discretize the conditional distribution of the
            next-step variance, as described in [Andersen, 2008].

            Parameters
            ----------
            vt : numpy.ndarray
                 The initial volatility.
            dt : numpy.ndarray
                 The time step of the simulation.
            kappa : numpy.ndarray
                 The mean reversion rate of the volatility process. One value for each valuation date.
            epsilon : numpy.ndarray
                 The volatility of the volatility process. One value for each valuation date.
            theta : numpy.ndarray
                 The long-term mean of the volatility process. One value for each valuation date.
            draw : numpy.ndarray
                :math:`Z_V` in [Andersen, 2008].

            Returns
            -------
            float
                Next-step variance :math:`V(t + \Delta t)`.
            """
            # First two moments (m and s^2) of the conditional distribution of the next-step variance.
            m = theta + (vt - theta) * np.exp(-kappa * dt)
            s2_1 = ((vt * (epsilon**2) * np.exp(-kappa * dt)) / kappa) * (
                1 - np.exp(-kappa * dt)
            )
            s2_2 = ((theta * (epsilon**2)) / (2 * kappa)) * (
                1 - np.exp(-kappa * dt)
            ) ** 2
            s2 = s2_1 + s2_2

            psi = s2 / (m**2)
            mask = psi > 2
            # Define a mask to vectorize and avoid a loop in the simulations to consider the different cases possible.
            psi_i = 1 / psi
            psi_c = 1.5
            # The choice for ψc (in [1, 2]) appears to have relatively small effects [Andersen, 2008].
            b2 = np.empty_like(psi_i)
            b2[~mask] = (
                2 * psi_i[~mask]
                - 1
                + np.sqrt(2 * psi_i[~mask]) * np.sqrt(2 * psi_i[~mask] - 1)
            )
            b2[mask] = 0
            # The value 0 of b2[mask] = 0 is arbitrary and does not affect the computation of res.
            # In fact, when psi > 2 > psi_c, the value of res does not involve b2.
            # Reference: Section 3.2, Propositions 5 and 6 in [Andersen, 2008]
            b = np.sqrt(b2)
            a = m / (1 + b2)
            p = (psi - 1) / (psi + 1)
            beta = (1 - p) / m
            # TODO. We compute all a,b,p and beta (not efficient since for a given simulation we only need a and b or p and beta).
            u = norm.cdf(draw)
            res = np.where(
                psi <= psi_c,
                a * (b + draw) ** 2,
                np.where(u <= p, 0, beta ** (-1) * np.log((1 - p) / (1 - u))),
            )
            # Essentially a boolean determining the regime (Quadratic or Exponential)
            return res

        shape = (
            draws.shape[0],
            draws.shape[1] + 1,
            draws.shape[2],
            1,
        )  # We only have one asset, but we include the fourth index for convention considerations.
        # draws.shape[1] + 1 since we need to include the first valuation date and all the observation dates.
        vt = np.full(shape, np.nan)
        # Array to store all the values of V(t), for each valuation date.
        vt[:, 0, :, 0] = v0  # Initial values of V(t) for each simulation date.

        for j in range(shape[1] - 1):  # Recursive formula for each simulation date
            vt[:, j + 1, :, 0] = compute_next_step_vol(
                vt[:, j, :, 0],
                intervals[j, :],
                kappa,
                epsilon,
                theta,
                draws[:, j, :, 0],
            )
        return vt

    def generate_paths(
        self,
        start_dates,
        step_dates,
        draws,
        intervals,
        discount_curve,
        forward_measure=True,
    ):
        """
        Compute the paths in the Heston model using Andersen's Quadratic Exponential (QE) method.

        Parameters
        ----------
        start_dates :  pandas.DatetimeIndex or pandas.Timestamp
            Starting dates for the simulation.
        step_dates : pandas.DatetimeIndex
            The step dates for which to compute the paths.
        draws : numpy.ndarray
            The draws for which to compute the paths. This is a four-dimensional array .
        intervals : numpy.ndarray
            The intervals for which to compute the volatilities.
        discount_curve : pricing.discount_curves.DiscountCurve
            Gives the discounts for the computation of the T-forward measure numeráires.
        forward_measure : bool, default = True
            If ``True``, it uses the forward measure for the simulation.

        Returns
        -------
        numpy.ndarray
            :math:`\\hat{X}(t)` in [Andersen, 2008].

        See also
        --------
        Underlyings_Jupyter: See that documentation for more details.

        """
        # TODO: We can use Numba here with @njit and prange to optimize the for loop (?).
        kappa = self.parameters.loc[start_dates, "kappa"].values
        epsilon = self.parameters.loc[start_dates, "epsilon"].values
        theta = self.parameters.loc[start_dates, "theta"].values
        rho = self.parameters.loc[start_dates, "rho"].values
        if afsfun.check_all_nonnumeric(
            self.parameters.loc[start_dates, "x0"].values
        ):  # TODO: ADAPT for cases in which we specify the value for certain dates (but not all).
            x0 = (
                self.data["Price"].loc[start_dates].values
            )  # If we have not assigned a value to this parameter we use historical data.
        else:
            x0 = self.parameters.loc[start_dates, "x0"].values
        gamma_1 = (
            1 / 2
        )  # TODO: A more sophisticated approach could be based on moment-matching, see Eq. (32) of  [Andersen, 2008] (not crucial).
        gamma_2 = 1 / 2

        vt = self.generate_paths_vol(
            start_dates, draws[..., 0:1], intervals
        )  # the 0:1 syntax is used instead of just 0 to ensure
        # that the resulting array is also 4-dimensional.
        k0 = (
            -(rho * kappa * theta * intervals) / epsilon
        )  # TODO: Martingale correction could also be implemented (both methods, so we can compare) (not crucial).
        k1 = gamma_1 * intervals * ((kappa * rho) / epsilon - 1 / 2) - rho / epsilon
        k2 = gamma_2 * intervals * ((kappa * rho) / epsilon - 1 / 2) + rho / epsilon
        k3 = gamma_1 * intervals * (1 - rho**2)
        k4 = gamma_2 * intervals * (1 - rho**2)
        shape = (
            draws.shape[0],
            draws.shape[1] + 1,
            draws.shape[2],
            1,
        )
        # We only have one asset, but we include the fourth index for convention considerations.
        # draws.shape[1] + 1 since we need to include the first valuation date and all the observation dates.
        log_xt = np.full(shape, np.nan)  # Array to store all the values of ln X(t)`.
        log_xt[:, 0, :, 0] = np.log(x0)
        for j in range(shape[1] - 1):  # Recursive formula for each simulation date
            log_xt[:, j + 1, :, 0] = (
                log_xt[:, j, :, 0]
                + k0[j]
                + k1[j] * vt[:, j, :, 0]
                + k2[j] * vt[:, j + 1, :, 0]
                + np.sqrt(k3[j] * vt[:, j, :, 0] + k4[j] * vt[:, j + 1, :, 0])
                * draws[:, j, :, 1]
            )
        paths = np.exp(log_xt)
        return paths


class SABR(VolModel):
    """
    Initializes a SABR process simulation with absorption at 0 based on the Hagan et al. (2002) model.
    The simulation algorithm is taken from Chen et al. (2011).

    Attributes
    ----------
    ticker : str
        The ticker symbol of the underlying asset.
    name : str, optional
        The name of the underlying asset.
    tenor : str, optional
        The tenor of the underlying asset.
    parameters : pandas.DataFrame
        A DataFrame with columns ["alpha", "beta", "rho", "sigma0"] to store the SABR model parameters.
        :math:`\\sigma_0` is the initial stochastic volatility.

    References
    ----------
    - Hagan, P. S., Kumar, D., Lesniewski, A. S., & Woodward, D. E. (2002). Managing smile risk. \
    The Best of Wilmott, 1, 249-296.
    - Chen, B., Grzelak, L. A., & Oosterlee, C. W. (2011). Calibration and Monte Carlo pricing of the SABR-Hull-White model for long-maturity equity derivatives. \
    The Journal of Computational Finance (79–113) Volume, 15.

    Notes
    -----
    The parameters DataFrame should have the valuation dates included in the index.
    One way to do it is writing vol_model.parameters = vol_model.parameters.reindex(valuation_dates)
    before computing the price.
    """

    def __init__(self, ticker, name=None, tenor=None, yieldsdiv=False):
        """
        Initializes a SABR process simulation with absorption at 0 based on the Hagan et al. (2002) model.
        The simulation algorithm is taken from Chen et al. (2011).

        Attributes
        ----------
        ticker : str
            The ticker symbol of the underlying asset.
        name : str, optional
            The name of the underlying asset.
        tenor : str, optional
            The tenor of the underlying asset.

        Parameters
        ----------
        parameters : pandas.DataFrame
            A DataFrame with columns ["alpha", "beta", "rho", "sigma0"] to store the SABR model parameters.
            sigma_0 is the initial stochastic volatility.

        References
        ----------
        - Hagan, P. S., Kumar, D., Lesniewski, A. S., & Woodward, D. E. (2002). Managing smile risk.
        The Best of Wilmott, 1, 249-296.

        - Chen, B., Grzelak, L. A., & Oosterlee, C. W. (2011). Calibration and Monte Carlo pricing of
        the SABR-Hull-White model for long-maturity equity derivatives. The Journal of Computational
        Finance (79–113) Volume, 15.

        Notes
        -----
        The parameters DataFrame should have the valuation dates included in the index. One way to do it is writing ``vol_model.parameters = vol_model.parameters.reindex(valuation_dates)`` before computing the price.
        """
        VolModel.__init__(self, ticker, name, tenor, yieldsdiv)
        self.parameters = pd.DataFrame(
            columns=["alpha", "beta", "rho", "sigma0", "S0"]
        )  # sigma_0 is the initial stochastic volatility
        # S0 is obtained from the ticker if not explicitly indicated when instantiating an object of Heston class.

    @staticmethod
    def generate_draws(no_obsdates, no_valuation_dates, no_sims, start_dates=None):
        """
        Generate random draws from normal and uniform distributions for simulations.

        This function creates a set of random draws for each simulation, observation date, and valuation date.
        It generates draws from a normal distribution and a uniform distribution, then concatenates these draws.

        Parameters
        ----------
        no_obsdates : int
            The number of observation dates for which to generate random draws.

        no_valuation_dates : int
            The number of valuation dates for which to generate random draws.

        no_sims : int
            The number of simulation paths to generate.

        start_dates :  pandas.DatetimeIndex, list of strings, pandas.Timestamp, or string, optional
            Starting dates for the simulations. If provided, it can be used for additional context or calculations,
            but is not directly used in the current implementation of this function.

        Returns
        -------
        ndarray
            A 4-dimensional numpy.ndarray where the last dimension consists of random draws from a normal distribution
            (first 3 indices) and a uniform distribution (last index). The shape of the array is
            (no_sims, no_obsdates, no_valuation_dates, 4).

        Notes
        -----
        The random draws are generated as follows:

        1. Normal draws (:math:`d^\\text{norm}`) are sampled from the numpy random normal distribution function:

           .. math::
               d^\\text{norm}_{jkl} \sim N(0, 1)\,, d_{ijkl}=d^\\text{norm}_{jkl}(\omega_i)\,,

           where :math:`i` indexes simulations, :math:`j` indexes observation dates, :math:`k` indexes valuation dates,
           and :math:`l` indexes the first 3 channels (for normal distribution).

        2. Uniform draws (:math:`d^\\text{unif}`) are generated using the numpy random uniform distribution function:

           .. math::
               d^\\text{unif}_{jk} \sim U(0, 1)\,, d^\\text{unif}_{ijk}= d^\\text{unif}_{jk}(\omega_i)\,,
           where :math:`i` indexes simulations, :math:`j` indexes observation dates, and :math:`k` indexes valuation dates.

        3. The final draws are concatenated along the last axis to combine the normal and uniform draws:

           .. math::
               [d_{ijkl}]_{i,j,k,l} = [d^\\text{norm}_{ijk1}, d^\\text{norm}_{ijk2}, d^\\text{norm}_{ijk3}, d^\\text{unif}_{ijk}]\,.
        """
        draws_normal = np.random.randn(no_sims, no_obsdates, no_valuation_dates, 3)
        draws_unif = np.random.rand(no_sims, no_obsdates, no_valuation_dates, 1)
        draws = np.concatenate([draws_normal, draws_unif], axis=-1)
        return draws

    def generate_paths_vol(self, start_dates, draws, intervals):
        """
        Compute the volatility for given start dates, draws, and intervals using the model
        parameters following a GBM, p.18 of Chen et al.

        Parameters
        ----------
        start_dates : pandas.DatetimeIndex
            The start dates for which to compute the volatilities.
        draws : numpy.ndarray
            The draws for which to compute the volatilities. This is a three-dimensional array.
        intervals : numpy.ndarray
            The intervals for which to compute the volatilities.

        Returns
        -------
        numpy.ndarray
            The computed volatilities.

        See also
        --------
        Underlyings_Jupyter: See that documentation for more details.

        """
        alpha = self.parameters.loc[start_dates, "alpha"].values
        # beta = self.parameters.loc[start_dates, "beta"].values  # Unused
        # rho = self.parameters.loc[start_dates, "rho"].values
        sigma0 = self.parameters.loc[start_dates, "sigma0"].values

        # We do not use the Lognormal class because it would require to change how some
        # parameters are defined using get_vol, get_divrate...
        base = (-(alpha**2) / 2) * intervals + alpha * draws * np.sqrt(intervals)
        paths = np.zeros((draws.shape[0], intervals.shape[0] + 1, draws.shape[2]))
        paths[:, 1:, :] = np.add.accumulate(base, 1)  # Using telescopic cancellation
        paths = sigma0 * np.exp(paths)

        return paths

    def generate_paths(
        self,
        start_dates,
        step_dates,
        draws,
        intervals,
        discount_curve,
        forward_measure=True,
    ):
        """
        Compute the asset prices for given start dates, step_dates, draws, and intervals using the model parameters.

        Parameters
        ----------
        start_dates : pandas.DatetimeIndex
            The start dates for which to compute the paths.
        step_dates : pandas.DatetimeIndex
            The step dates for which to compute the paths.
        draws : numpy.ndarray
            The draws for which to compute the paths. This is a four-dimensional array .
        intervals : numpy.ndarray
            The intervals for which to compute the volatilities.
        discount_curve : pricing.discount_curves.DiscountCurve
            Gives the discounts for the computation of the T-forward measure numeráires.
        forward_measure : bool, default = True
            If ``True``, it uses the forward measure for the simulation.

        Returns
        -------
        numpy.ndarray
            The paths :math:`\\hat{S}`.

        References
        ----------
        - Chen, B., Grzelak, L. A., & Oosterlee, C. W. (2011). Calibration and Monte Carlo
        pricing of the SABR-Hull-White model for long-maturity equity derivatives.
        The Journal of Computational Finance (79–113) Volume, 15.

        - See also https://github.com/lionel75013/sabrMC/tree/f43e952d4417267b93a656a188f014a0375a9a51.

        See also
        --------
        Underlyings_Jupyter: See that documentation for more details.

        """
        alpha = self.parameters.loc[start_dates, "alpha"].values
        beta = self.parameters.loc[start_dates, "beta"].values
        rho = self.parameters.loc[start_dates, "rho"].values
        # sigma0 = self.parameters.loc[start_dates, "sigma0"].values

        # Step 1 : volatilities
        sigma = self.generate_paths_vol(start_dates, draws[..., 0], intervals)

        # Step 2 : integrated variance approximation moments. Using the small disturbance expansion Chen et al (2011), formula (3.18).
        # Broadcasting expressions
        W = intervals * draws[..., 0]
        W_2, W_3, W_4 = W**2, W**3, W**4
        m1 = alpha * W
        m2 = (1.0 / 3) * (alpha**2) * (2 * W_2 - intervals / 2)
        m3 = (1.0 / 3) * (alpha**3) * (W_3 - W * intervals)
        m4 = (
            (1.0 / 5)
            * (alpha**4)
            * ((2.0 / 3) * W_4 - (3.0 / 2) * W_2 * intervals + 2 * intervals**2)
        )
        m = (sigma[:, :-1] ** 2) * intervals * (1.0 + m1 + m2 + m3 + m4)

        v = (1.0 / 3) * (sigma[:, :-1] ** 4) * (alpha**2) * intervals**3

        # Step 3 : moment-matched log-normal distribution
        mu = np.log(m) - (1.0 / 2) * np.log(1.0 + v / m**2)
        sigma2 = np.log(1.0 + v / (m**2))

        # Step 4 : integrated variance
        A = np.exp(np.sqrt(sigma2) * draws[..., 1] + mu)
        v_t = (1.0 - rho**2) * A

        # Step 5 : Conditional CEV process
        b = 2.0 - (
            (1.0 - 2.0 * beta - (1.0 - beta) * (rho**2))
            / ((1.0 - beta) * (1.0 - rho**2))
        )
        shape = (
            draws.shape[0],
            draws.shape[1] + 1,
            draws.shape[2],
        )
        # draws.shape[1] + 1 since we need to include the first valuation date and all the observation dates.
        paths = np.full(shape, np.nan)
        if afsfun.check_all_nonnumeric(
            self.parameters.loc[start_dates, "S0"].values
        ):  # TODO: ADAPT for cases in which we specify the value for certain dates (but not all).
            s0 = (
                self.data["Price"].loc[start_dates].values
            )  # If we have not assigned a value to this parameter we use historical data.
        else:
            s0 = self.parameters.loc[start_dates, "S0"].values
        paths[:, 0] = s0
        counter = 0
        for j in range(shape[1] - 1):
            # Recursive formula for each simulation date, no broadcasting possible,
            # Step 1,2
            cond1 = paths[:, j] != 0
            a = (1.0 / v_t[:, j]) * (
                (paths[:, j] ** (1.0 - beta)) / (1.0 - beta)
                + (rho / alpha) * (sigma[:, j + 1] - sigma[:, j])
            ) ** 2

            # Step 3
            abs_prob = 1 - gammainc(b / 2, a / 2)
            cond2 = abs_prob <= draws[:, j, :, -1]
            zero_cond = ~(cond1 & cond2)
            paths[:, j + 1] = np.where(zero_cond, 0, paths[:, j + 1])

            # Step 4, 5 : Andersen's parameters
            k = 2.0 - b  # above (3.9)
            lambd = a  # above (3.9)
            s2 = 2 * (k + 2 * lambd)
            m_and = k + lambd
            psi = s2 / (m_and**2)
            psi_c = 2  # TODO: set as attribute?

            # Step 6 : Andersen's quadratic approximation
            cond3 = (0 < psi) & (psi <= psi_c) & (m_and >= 0)
            e2 = (2.0 / psi) - 1.0 + np.sqrt(2.0 / psi) * np.sqrt((2.0 / psi) - 1.0)
            d = m_and / (1.0 + e2)
            paths[:, j + 1] = np.where(
                ~zero_cond & cond3,
                (
                    ((1.0 - beta) ** 2)
                    * v_t[:, j]
                    * d
                    * ((np.sqrt(e2) + draws[:, j, :, 2]) ** 2)
                )
                ** (1.0 / (2.0 * (1.0 - beta))),
                paths[:, j + 1],
            )

            # Step 7 : direct inversion
            def equation(c, a, b, u):
                return 1 - ncx2.cdf(a, b, c) - u

            def equation_square(c, a, b, u):
                return np.square(1 - ncx2.cdf(a, b, c) - u)

            def c_star_func(
                a, b, u
            ):  # Vectorized. scipy.minimize does NOT admit vectorization. This part is the slow one. But these arrays, in general,
                # should have only a few simulations, see paper.
                bounds = [(0.0, None)]
                result = np.empty_like(u)

                for item in zip(
                    np.ndindex(a.shape), a.flatten(), b.flatten(), u.flatten()
                ):
                    idx = item[0]
                    ai = item[1]
                    bi = item[2]
                    ui = item[3]

                    # METHOD 1 (this seems faster)
                    res = minimize(
                        equation_square,
                        ai,
                        bounds=bounds,
                        args=(ai, bi, ui),
                        tol=equation_square(0, ai, bi, ui) * 10 ** (-9),
                    )  # Tolerance to avoid malfunctioning
                    root = res.x
                    # TODO: implement enhanced direct inversion for initial guess, much faster?

                    # METHOD 2 # TODO: implement Newton's method using derivative, faster?
                    # def function(x):
                    #     return equation(x, ai, bi, ui)
                    # lower_bound = 0
                    # upper_bound = ai
                    # # Ensure the function values at the bounds have opposite signs:
                    # while np.sign(function(lower_bound)) == np.sign(function(upper_bound)):
                    #     upper_bound *= 2
                    # root = bisect(function, lower_bound, upper_bound)

                    result[idx] = root
                    if np.abs(equation(root, ai, bi, ui)) > 0.01:
                        print(
                            "Numerical method malfunctioning",
                            root,
                            equation(root, ai, bi, ui),
                        )
                        print("Parameters (a, b, u) ", ai, bi, ui)
                        # Plot
                        x_vals = np.linspace(0, 2 * np.float(root), 100)
                        y_vals = equation_square(x_vals, ai, bi, ui)
                        trace = go.Scatter(x=x_vals, y=y_vals)
                        layout = go.Layout(
                            title="Numerical method malfunctioning",
                            xaxis=dict(title=f"x, sol={root}"),
                            yaxis=dict(title=f"y, sol={equation(root, ai, bi, ui)}"),
                        )
                        fig = go.Figure(data=[trace], layout=layout)
                        fig.show()

                return result

            mask = np.array(~zero_cond & ~cond3, bool)
            # To explicitly say that mask is an array, not an integer
            counter += mask[mask == 1].size
            # Calculate c_star only for the elements where the mask is True
            c_star_values = c_star_func(
                a[mask], np.broadcast_to(b, mask.shape)[mask], draws[:, j, :, -1][mask]
            )
            # Assign the c_star_values to the paths array only where the mask is True
            paths[:, j + 1][mask] = (
                c_star_values
                * np.broadcast_to(((1.0 - beta) ** 2) * v_t[:, j], mask.shape)[mask]
            ) ** np.broadcast_to(1.0 / (2.0 - 2.0 * beta), mask.shape)[mask]

        print("Number of direct inversions needed=", counter)
        paths = paths[..., np.newaxis]  # Convention considerations
        return paths

    def generate_paths_euler(
        self,
        start_dates,
        step_dates,
        draws,
        intervals,
        forward_measure=True,
        euler_vol=False,
    ):
        """
        Generate paths for the asset or forward rate using the SABR model and the Euler Full Truncation scheme
        for discretization following (3.1) of Chen et al., but with a truncation or full truncation Euler.

        Parameters
        ----------
        start_dates :  pandas.DatetimeIndex, list of strings, pandas.Timestamp, or string
            Starting dates.

        step_dates : array-like
            Array of time steps for the paths.
        draws : numpy.ndarray
            2D array containing the random draws for path simulation.
        intervals : numpy.ndarray
            Array containing the time intervals between steps.
        forward_measure : bool, optional, default: True
            Whether to use the forward measure.
        euler_vol : bool, optional, default: False
            Whether to use Euler's method for volatility.

        Returns
        -------
        numpy.ndarray
            A 2D NumPy array containing the simulated forward rates.

        Raises
        ------
        AttributeError
            If ``start_dates`` has more than one date.
        """

        # Let us define some parameters:
        #     T : float
        #         Time horizon.
        #     N : int
        #         Number of time steps.
        #     num_paths : int
        #         Number of paths to simulate.
        #     alpha0 : float
        #         Initial volatility (α).
        #     beta : float
        #         Exponent for the forward rate (β).
        #     rho : float
        #         Correlation between the Brownian motions (ρ).
        #     nu : float
        #         Volatility of volatility (ν).
        #     forward_rate : float
        #         Initial forward rate.

        start_dates, step_dates = afsfun.dates_formatting(start_dates, step_dates)
        if start_dates.size > 1:
            raise AttributeError("Start dates must be a single date.")

        num_paths = draws.shape[0]
        N = draws.shape[1]
        T = intervals[1, 0] * N  # TODO: We assume all the intervals are the same
        dt = T / N
        alpha0 = self.parameters.loc[
            start_dates, "sigma0"
        ].values  # Different notation here for nu and alpha
        nu = self.parameters.loc[start_dates, "alpha"].values
        beta = self.parameters.loc[start_dates, "beta"].values
        rho = self.parameters.loc[start_dates, "rho"].values

        # Initialize arrays for forward rates and volatilities
        forward_rates = np.zeros((N + 1, num_paths))
        volatilities = np.zeros((N + 1, num_paths))

        # Set initial forward rates and volatilities
        forward_rates[0, :] = self.parameters.loc[start_dates, "S0"].values
        volatilities[0, :] = alpha0

        # Brownian motions for the correlated Wiener processes.
        dW1 = np.sqrt(dt) * draws[:, :, 0, 0].T
        dW2 = np.sqrt(dt) * draws[:, :, 0, 1].T

        # Apply Cholesky decomposition for correlated Brownian motions
        dW2_corr = rho * dW1 + np.sqrt(1 - rho**2) * dW2

        if euler_vol:
            for i in range(1, N + 1):
                # Full truncation for negative volatilities. Better to use the other method (exact),
                #  this just for benchmark to check.
                volatilities_temp = volatilities[i - 1, :]
                d_alpha = nu * volatilities_temp * dW2_corr[i - 1, :]
                volatilities[i, :] = np.maximum(volatilities_temp + d_alpha, 0)

                d_forward = (
                    volatilities_temp * forward_rates[i - 1, :] ** beta * dW1[i - 1, :]
                )
                forward_rates[i, :] = np.maximum(forward_rates[i - 1, :] + d_forward, 0)

        else:
            for i in range(1, N + 1):
                # Update volatilities
                volatilities[i, :] = volatilities[i - 1, :] * np.exp(
                    -1 / 2 * nu**2 * dt + nu * dW2_corr[i - 1, :]
                )

                # Update forward rates
                d_forward = (
                    volatilities[i - 1, :]
                    * (forward_rates[i - 1, :] ** beta)
                    * dW1[i - 1, :]
                )
                forward_rates[i, :] = np.maximum(forward_rates[i - 1, :] + d_forward, 0)

        paths = forward_rates.T  # So the first index corresponds to simulations
        paths = paths[:, :, np.newaxis, np.newaxis]  # Convention considerations

        return paths


class MultiAssetHeston(MultiAsset):
    """
    Parsimonious Multiple assets Heston model.

    References
    ----------
    - [DLS, 2011] Dimitroff, G., Lorenz, S., & Szimayer, A. (2011). A parsimonious multi-asset
          Heston model: Calibration and derivative pricing. International Journal of Theoretical and Applied Finance, 14(08), 1299-1333.
    - [Mokone, 2022] Mokone, C. M. (2022). Gaussian process regression approach to pricing multi-asset American options (Master's thesis, Faculty of Commerce).
    - [Wadman, 2010] Wadman, Wander. (2010). An advanced Monte Carlo method for the multi-asset Heston model. 10.13140/RG.2.2.20537.26722.
    """

    def __init__(self, *equity_objects):
        MultiAsset.__init__(*equity_objects)
        self.components = equity_objects
        self.single_hestons = equity_objects  # TODO: What is the difference between components and single_hestons?
        self.no_assets = len(self.single_hestons)

        self.asset_correlation_structure = (
            {}
        )  # Defined after calibration, it is not an initialization argument. Dictionary for each date.

    def compute_corr_matrix(self, dates):
        """
        Compute the correlation matrix and its Cholesky decomposition for a given set of dates.

        Parameters
        ----------
        dates : pandas.DatetimeIndex
            An index of dates for which the correlation matrix and its Cholesky decomposition should be computed.

        Returns
        -------
        numpy.ndarray
            An array with the corresponding correlation matrices at each index.
        numpy.ndarray
            An array with the corresponding Cholesky decompositions of the correlation matrices at each index.
        numpy.ndarray
            An array with the corresponding Cholesky decomposition block of the correlation matrices at each index.

        Notes
        -----
        The correlation matrix is constructed using blocks of Proposition 2 of [Wadman, 2010].
        The Cholesky decomposition is computed using (A.17) of [Mokone, 2022].
        """
        no_assets = self.no_assets
        diagonal = np.full((no_assets, dates.size), np.nan)
        for i in np.arange(no_assets):
            diagonal[i] = self.single_hestons[i].parameters["rho"][dates]

        matrix_shape = (dates.size, 2 * no_assets, 2 * no_assets)
        matrix = np.empty(matrix_shape)
        L = np.empty(matrix_shape)
        L_star = np.empty((dates.size, no_assets, no_assets))

        for k, date in enumerate(dates):
            block_1 = np.identity(2)
            block_2 = np.diag(diagonal[:, k])
            block_3 = block_2
            block_4 = self.asset_correlation_structure[date]

            # Stack the blocks horizontally and vertically to form the final matrix
            top_row = np.hstack((block_1, block_2))
            bottom_row = np.hstack((block_3, block_4))
            matrix[k] = np.vstack((top_row, bottom_row))
            L[k] = np.linalg.cholesky(matrix[k])

            # (A.17) of [Mokone, 2022]
            L_star[k] = np.linalg.cholesky(block_4 - block_2**2)
            top_row = np.hstack((block_1, np.full((no_assets, no_assets), 0.0)))
            bottom_row = np.hstack((block_3, L_star[k]))
            Lp = np.vstack((top_row, bottom_row))

            assert np.allclose(L[k], Lp)
            assert np.allclose(np.dot(L[k], np.transpose(L[k])), matrix[k])

        return matrix, L, L_star

    def generate_draws(
        self, no_obsdates, no_valuation_dates, no_sims, start_dates=None
    ):
        """
        Normal random variables needed for generating the paths of volatilities and assets prices.
        This method is needed for the new class :py:meth:`DiffusionMC <pricing.mc_engines.DiffusionMC>`.

        Parameters
        ----------
        no_obsdates : int
            Number of observation dates.
        no_valuation_dates : int
            Number of valuation dates.
        no_sims : int
            Number of simulations.

        Returns
        -------
        numpy.array
            draws[i,j,k,l] is a four dimensional numpy.ndarray with all the normals.
        """
        no_assets = self.no_assets
        draws = np.random.randn(no_sims, no_obsdates, no_valuation_dates, 2 * no_assets)
        return draws

    def generate_paths_euler(
        self, start_dates, step_dates, draws, intervals, forward_measure=True
    ):
        """
        Generate paths for TWO assets using the (parsimonious) Heston model and Euler Full Truncation (EFT).

        Parameters
        ----------
        start_dates :  pandas.DatetimeIndex, list of strings, pandas.Timestamp, or string
            The starting date for the paths. We assume a SINGLE starting date.
        step_dates : pandas.DatetimeIndex, list of strings, pandas.Timestamp, or string
            The dates for each step in the paths.
        draws : ndarray
            The draws for each asset.
        intervals : ndarray
            The intervals for each step in the paths.
        forward_measure : bool, optional, default=True
            If ``True``, use the forward measure for the paths.

        Returns
        -------
        ndarray
            The generated paths for each asset.

        Notes
        -----
        - The single Heston model instances are stored in ``self.single_hestons``.
        - The :py:meth:`dates_formatting <pricing.ir_models.ShortRateModel.dates_formatting_cfb>` function is a helper function that formats the input dates.
        - The :py:meth:`compute_corr_matrix <data.underlyings.MultiAssetHeston.compute_corr_matrix>` function a method of the current class that computes the correlation matrix.
        """
        start_dates, step_dates = afsfun.dates_formatting(start_dates, step_dates)
        if start_dates.size > 1:
            raise AttributeError("Start dates must be a single date.")
        matrix, L, L_star = self.compute_corr_matrix(start_dates)
        correlation_matrix = matrix[0]  # Sigma of [Wadman, 2010]

        # Generate correlated random variables
        no_sims = draws.shape[0]
        n_time_steps = draws.shape[1]
        T = intervals[1, 0] * n_time_steps  # We assume all the intervals are the same
        dt = T / n_time_steps  # N is the Number of time steps
        sqrt_dt = np.sqrt(dt)

        z = np.random.multivariate_normal(
            [0, 0, 0, 0], correlation_matrix, size=(no_sims, n_time_steps)
        )  # Not optimal, draws should be used.
        z_v1 = z[:, :, 0]
        z_v2 = z[:, :, 1]
        z_S1 = z[:, :, 2]
        z_S2 = z[:, :, 3]

        S1 = np.zeros((no_sims, n_time_steps + 1))
        S2 = np.zeros((no_sims, n_time_steps + 1))
        v1 = np.zeros((no_sims, n_time_steps + 1))
        v2 = np.zeros((no_sims, n_time_steps + 1))

        kappa1 = self.single_hestons[0].parameters.loc[start_dates, "kappa"].values
        kappa2 = self.single_hestons[1].parameters.loc[start_dates, "kappa"].values
        epsilon1 = self.single_hestons[0].parameters.loc[start_dates, "epsilon"].values
        epsilon2 = self.single_hestons[1].parameters.loc[start_dates, "epsilon"].values
        theta1 = self.single_hestons[0].parameters.loc[start_dates, "theta"].values
        theta2 = self.single_hestons[1].parameters.loc[start_dates, "theta"].values
        v0_1 = self.single_hestons[0].parameters.loc[start_dates, "v0"].values
        v0_2 = self.single_hestons[1].parameters.loc[start_dates, "v0"].values
        x0_1 = self.single_hestons[0].parameters.loc[start_dates, "x0"].values
        x0_2 = self.single_hestons[1].parameters.loc[start_dates, "x0"].values

        S1[:, 0] = x0_1
        S2[:, 0] = x0_2
        v1[:, 0] = v0_1
        v2[:, 0] = v0_2

        for j in range(1, n_time_steps + 1):
            dv1 = (
                kappa1 * (theta1 - v1[:, j - 1]) * dt
                + epsilon1 * np.sqrt(v1[:, j - 1]) * z_v1[:, j - 1]
            )
            dv2 = (
                kappa2 * (theta2 - v2[:, j - 1]) * dt
                + epsilon2 * np.sqrt(v2[:, j - 1]) * z_v2[:, j - 1]
            )
            dS1 = S1[:, j - 1] * np.sqrt(v1[:, j - 1]) * sqrt_dt * z_S1[:, j - 1]
            dS2 = S2[:, j - 1] * np.sqrt(v2[:, j - 1]) * sqrt_dt * z_S2[:, j - 1]

            v1[:, j] = np.maximum(v1[:, j - 1] + dv1, 0)
            v2[:, j] = np.maximum(v2[:, j - 1] + dv2, 0)
            S1[:, j] = S1[:, j - 1] + dS1
            S2[:, j] = S2[:, j - 1] + dS2

        paths = np.full(
            (draws.shape[0], draws.shape[1] + 1, draws.shape[2], self.no_assets), np.nan
        )
        paths[:, :, 0, 0] = S1
        paths[:, :, 0, 1] = S2
        return paths

    def generate_paths(
        self,
        start_dates,
        step_dates,
        draws,
        intervals,
        discount_curve,
        forward_measure=True,
    ):
        """
        Generate paths for multiple assets using the (parsimonious) Heston model.

        Parameters
        ----------
        start_dates :  pandas.DatetimeIndex, list of strings, pandas.Timestamp, or string
            Starting dates.
        step_dates : pandas.DatetimeIndex, list of strings, pandas.Timestamp, or string
            The dates for each step in the paths.
        draws : ndarray
            The draws for each asset.
        intervals : ndarray
            The intervals for each step in the paths.
        discount_curve : pricing.discount_curves.DiscountCurve
            Gives the discounts for the computation of the T-forward measure numeráires.
        forward_measure : bool, default = True
            If ``True``, it uses the forward measure for the simulation.

        Returns
        -------
        ndarray
            The generated paths for each asset.

        Notes
        -----
        - The single Heston model instances are stored in ``self.single_hestons``.
        - The :py:meth:`dates_formatting <pricing.ir_models.ShortRateModel.dates_formatting_cfb>` function is a helper function that formats the input dates.
        - The :py:meth:`compute_corr_matrix <data.underlyings.MultiAssetHeston.compute_corr_matrix>` function a method of the current class that computes the correlation matrix.
        """

        start_dates, step_dates = afsfun.dates_formatting(start_dates, step_dates)
        matrix, L, L_star = self.compute_corr_matrix(start_dates)

        rho_values = np.array(
            [
                single_heston.parameters["rho"][start_dates]
                for single_heston in self.single_hestons
            ]
        )
        diagonal = 1 / np.sqrt(1 - rho_values**2).T
        diagonal_old = np.full((start_dates.size, self.no_assets), np.nan)
        # for l in np.arange(self.no_assets):
        #     diagonal_old[:, l] = (1/np.sqrt(1-self.single_hestons[l].parameters["rho"][start_dates]**2))
        # assert np.allclose(diagonal, diagonal_old)  # To check the syntax

        diagonal_matrix = np.full(
            (start_dates.size, self.no_assets, self.no_assets), np.nan
        )
        for k, date in enumerate(start_dates):
            diagonal_matrix[k] = np.diag(diagonal[k])

        corr_matrix = np.matmul(diagonal_matrix, L_star)
        raw_asset_draws = draws[
            ..., self.no_assets :
        ]  # self.no_assets independent normals
        asset_draws = np.einsum("klm,ijkm->ijkl", corr_matrix, raw_asset_draws)
        # for i in range(asset_draws.shape[0]):
        #     for j in range(asset_draws.shape[1]):
        #         for k in range(asset_draws.shape[2]):
        #             assert np.allclose(asset_draws[i, j, k, :], np.matmul(corr_matrix[k], raw_asset_draws[i, j, k, :]))  # To check the einsum notation
        paths = np.full(
            (draws.shape[0], draws.shape[1] + 1, draws.shape[2], self.no_assets), np.nan
        )
        paths_vol = np.full(
            (draws.shape[0], draws.shape[1] + 1, draws.shape[2], self.no_assets), np.nan
        )
        for i in range(self.no_assets):
            temp_draws = np.concatenate(
                [draws[..., i : i + 1], asset_draws[..., i : i + 1]], axis=-1
            )
            paths_vol[..., i] = self.single_hestons[i].generate_paths_vol(
                start_dates, temp_draws[..., 0:1], intervals
            )[
                ..., 0
            ]  # TODO: Not used
            paths[..., i] = self.single_hestons[i].generate_paths(
                start_dates,
                step_dates,
                temp_draws,
                intervals,
                discount_curve,
                forward_measure,
            )[..., 0]

        return paths
