import numpy as np
import pandas as pd
import itertools
import pmdarima as pm
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model  # Sheppard's library
from scipy.optimize import minimize, dual_annealing
from scipy.special import gamma
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from statsmodels.tools.numdiff import approx_hess, approx_fprime
from arch.univariate.base import constraint
import scipy.stats as stats


class EmpiricalRiskFactor:
    def __init__(self):
        self.underlying = None
        self.dates = None
        self.returns = None

    def get_dates(self):
        return self.dates

    def get_return(self, dates=None):
        if dates is None:
            dates = self.dates
        else:
            if np.asarray(dates).shape == ():
                dates = [dates]
            dates = pd.to_datetime(dates).sort_values()
            dates = dates[dates.isin(self.returns.index)]
        return self.returns.loc[dates]

    def get_mean(self, dates=None):
        if dates is None:
            dates = self.dates
        else:
            if np.asarray(dates).shape == (): dates = [dates]
            dates = pd.to_datetime(dates).sort_values()
            dates = dates[dates.isin(self.means.index)]
        return self.means.loc[dates]

    def get_vol(self, dates=None, annualized=False, percentage=False):
        if dates is None:
            dates = self.dates
        else:
            if np.asarray(dates).shape == (): dates = [dates]
            dates = pd.to_datetime(dates).sort_values()
            dates = dates[dates.isin(self.vol.index)]
        return self.vol.loc[dates] * (1 + (np.sqrt(252) - 1) * annualized) * (1 + 99 * percentage)

    def get_vol_bounds(self, dates=None):
        if dates is None:
            dates = self.dates
        else:
            if np.asarray(dates).shape == (): dates = [dates]
            dates = pd.to_datetime(dates).sort_values()
        bounds = pd.DataFrame(index=dates, columns=["Min", "Max"])
        for date in dates:
            vols = self.vol.loc[self.dates[self.dates < date]]
            bounds.loc[date] = [np.min(vols), np.max(vols)]
        return bounds

    def get_white_noise_quantile(self, dates, quantile, window=None):
        if dates is None:
            dates = self.dates
        else:
            if np.asarray(dates).shape == (): dates = [dates]
            dates = pd.to_datetime(dates).sort_values()
            dates = dates[dates.isin(self.vol.index)]
        dates = pd.to_datetime(dates).sort_values()
        dates = dates[dates.isin(self.dates)]
        normalized_returns = self.returns / self.vol
        # normalized_returns = self.residuals
        var = pd.Series(np.nan, index=dates)
        for date in dates:
            temp = normalized_returns.loc[:date - pd.Timedelta("1D")]
            temp = temp[temp !=0]
            if window is not None:
                temp = temp[-window:]
            var.loc[date] = temp.quantile(quantile)
        return var

    def get_var(self, dates, quantile, window=None, position="long", quantile_interpolation="linear"):
        """
        Calculate a VaR of the risk factor for the given dates. If the risk factor has drawdowns, VaR is computed on the adjusted drawdowns.
        Otherwise, VaR is computed on returns.

        Parameters
        ----------
        dates : array_like
            Dates on which we calculate the VaR
        quantile : float
            Quantile (between 0 and 1) used to calculate the VaR.
        window : int, str or None, optional
            window used in the range_percentile calculation previous to each date. If ``int``, it refers to number of days.
            If str, denotes 'unit' according to pandas.Timedelta.
            If ``None``, all previous available dates in dates parameter are considered.
        position : str
            A string representing the type position considered (either "long" or "short"). Default is "long".
        quantile_interpolation : str, optional
            Interpolation method used to calculate the empiric quantile, according to numpy.quantile 'method'. Default is 'linear'.

        Returns
        ----------
        var : pandas.Series
            Series with the calculated VaR for each date.

        """
        # print("Removing zero drawdowns!")
        if dates is None:
            dates = self.dates
        else:
            if np.asarray(dates).shape == ():
                dates = [dates]
            dates = pd.to_datetime(dates).sort_values()

        if hasattr(self, 'get_drawdowns'):
            drawdowns_dic = self.get_drawdowns(dates, position=position)
            liquidation_period = self.liquidation_period
        else:
            epsilon = 1 - 2*(position == "short")  # Either +1 or -1
            returns = epsilon * self.underlying.get_return(self.underlying.get_dates(), percentage=False, raw=True)
            drawdowns_dic = {date: returns[:date-pd.Timedelta("1D")] for date in dates}
            liquidation_period = 1
        adjustments_dic = self.get_adjustment(dates=dates)
        dates = dates[dates.isin(adjustments_dic.keys()) & dates.isin(drawdowns_dic.keys())]

        # We ignore dates which do not have enough drawdowns respect the liquidation period.
        adjusted_series = {date: drawdowns_dic[date] * adjustments_dic[date] 
                               for date in dates if drawdowns_dic[date].shape[0] >= liquidation_period}
        dates = pd.to_datetime(list(adjusted_series.keys())).sort_values()
        # Filter window
        if type(window) is int:
            adjusted_series = {date: adjusted_series[date].loc[adjusted_series[date].index[-liquidation_period] - pd.Timedelta(days = window): 
                                                               adjusted_series[date].index[-liquidation_period]] for date in dates}
        elif window == "1Y":
            adjusted_series = {date: adjusted_series[date].loc[adjusted_series[date].index[-liquidation_period] - pd.DateOffset(years = 1) + pd.Timedelta(days = 1): 
                                                               adjusted_series[date].index[-liquidation_period]] for date in dates}
        else:
            adjusted_series = {date: adjusted_series[date].loc[:adjusted_series[date].index[-liquidation_period]] for date in dates}
            # We always filter up to adjusted_series[date].index[-self.liquidation_period] since this drawdown uses the date price, while the next one will use data after date.

        # Filter nans. We filter nan after window to have a solid reference of the calculation date.
        adjusted_series = {date: adjusted_series[date].dropna() for date in dates}

        if len(adjusted_series[dates[0]].shape) == 1:  # Simple contract case
            var = [adjusted_series[date].quantile(quantile, interpolation=quantile_interpolation) for date in dates]
        elif adjusted_series[dates[0]].shape[1] == 2:  # Portfolio case
            var = [adjusted_series[date].sum(axis=1, skipna=False).quantile(quantile, interpolation=quantile_interpolation) for date in dates]
        var = pd.Series(var, index=dates)
        return var

    def get_bilateral_var(self, dates, quantile, window=None, position="long", quantile_interpolation="linear"):
        # print("Removing zero drawdowns!")
        if dates is None:
            dates = self.dates
        else:
            if np.asarray(dates).shape == (): dates = [dates]
            dates = pd.to_datetime(dates).sort_values()

        if hasattr(self, 'get_drawdowns'):
            drawdown_dic = self.get_drawdowns(dates, position=position)
        else:
            returns = self.underlying.get_return(self.underlying.get_dates(), percentage=False, raw=True)
            drawdown_dic = {date:returns[:date-pd.Timedelta("1D")] for date in dates}
        adjustment_dic = self.get_adjustment(dates)
        dates = dates[dates.isin(adjustment_dic.keys()) & dates.isin(drawdown_dic.keys())]
        adjusted_series = {date: drawdown_dic[date] * adjustment_dic[date] for date in dates}
        # adjusted_series = {date: adjusted_series[date][adjusted_series[date] != 0] for date in dates}
        if type(window) == int:
            adjusted_series = {date: adjusted_series[date].iloc[-window:] for date in dates}
        elif window == "1Y":
            adjusted_series = {date: adjusted_series[date].loc[date-pd.Timedelta(window):] for date in dates}
        try:
            var1 = [adjusted_series[date].sum(axis=1, skipna=False).quantile(quantile, interpolation=quantile_interpolation) for date in dates]
            var2 = [adjusted_series[date].sum(axis=1, skipna=False).quantile(1-quantile, interpolation=quantile_interpolation) for date in dates]
        except:
            var1 = [adjusted_series[date].quantile(quantile, interpolation=quantile_interpolation) for date in dates]
            var2 = [adjusted_series[date].quantile(1-quantile, interpolation=quantile_interpolation) for date in dates]
        var = pd.DataFrame(np.array([var1, var2]).transpose(), index=dates, columns=[quantile, 1-quantile])
        return var

    def get_expected_shortfall(self, dates, confidence_level=None, threshold=None, window=None, position="long",
                               return_no_points=False, raw=None):
        """
        Calculate the expected shortfall of a given dates, i.e., for each date, returns an average of
        the previous drawdowns (or returns if drawdowns are not defined) inside the window exceeding certain amount.
        The amount can be given by a confidence level (confidence_level) or by a value (threshold).

        Parameters
        ----------
        dates : array_like
            Dates on which we calculate the expected shortfall.
        confidence_level : float, optional
            Quantile (between 0 and 1) used to define the amount beyond which we consider drawdowns for the expected shortfall calculation.
            Default is None
        threshold : float
            Amount beyond which we consider drawdowns (returns) for the expected shortfall calculation.
            Default is None.
        window : int
            Number of previous dates considered in the calculation.
            If ``None``, all previous available dates in dates parameter are considered.
        position : str
            A string representing the type position considered (either "long" or "short"). Default is "long".
        return_no_points : bool
            If ``True``, function also returns the number of drawdowns (returns) used in the expected shortfall calculation.

        Returns
        ----------
        etl : pandas.Series
            Series with the expected shortfall calculated for each date.
        no_points : pandas.Series
            Series with the number of drawdowns (returns) used in the expected shortfall calculation for each date.
            Only returned if ``return_no_points = True``.

        """
        # print("Removing zero drawdowns!")
        if dates is None:
            dates = self.dates
        else:
            if np.asarray(dates).shape == (): dates = [dates]
            dates = pd.to_datetime(dates).sort_values()
        if hasattr(self, 'get_drawdowns'):
            drawdowns_dic = self.get_drawdowns(dates, position=position)
        else:
            returns = self.underlying.get_return(self.underlying.get_dates(), percentage=False, raw=True)
            drawdowns_dic = {date:returns[:date - pd.Timedelta("1D")] for date in dates}
        adjustments_dic = self.get_adjustment(dates, raw=raw)
        etl = pd.Series(index=dates, dtype='float64')
        if return_no_points:
            no_points = pd.Series(index=dates, dtype='float64')
        dates = dates[dates.isin(adjustments_dic.keys()) & dates.isin(drawdowns_dic.keys())]

        # We ignore dates which do not have enough drawdowns respect the liquidation period.
        adjusted_series = {date: drawdowns_dic[date] * adjustments_dic[date] 
                            for date in dates if drawdowns_dic[date].shape[0] >= self.liquidation_period}
        dates = pd.to_datetime(list(adjusted_series.keys())).sort_values()
        # Filter window
        if type(window) is int:
            adjusted_series = {date: adjusted_series[date].loc[adjusted_series[date].index[-(self.liquidation_period+1)] - pd.Timedelta(days = window)+ pd.Timedelta(days = 1): 
                                                               adjusted_series[date].index[-self.liquidation_period]] for date in dates}
        elif window == "1Y":
            adjusted_series = {date: adjusted_series[date].loc[adjusted_series[date].index[-self.liquidation_period] - pd.DateOffset(years = 1) + pd.Timedelta(days = 1): 
                                                               adjusted_series[date].index[-self.liquidation_period]] for date in dates}
        else:
            adjusted_series = {date: adjusted_series[date].loc[:adjusted_series[date].index[-self.liquidation_period]] for date in dates}
            # We always filter up to adjusted_series[date].index[-self.liquidation_period] since this drawdown uses the date price, while the next one will use data after date.
        
        # Filter nans. We filter nan after window to have a solid reference of the calculation date.
        adjusted_series = {date: adjusted_series[date].dropna() for date in dates}

        for date in dates:
            if len(adjusted_series[dates[0]].shape) == 1:  # Simple contract case
                adjusted_series_temp = adjusted_series[date]
            elif adjusted_series[dates[0]].shape[1] == 2:  # Portfolio case
                adjusted_series_temp = adjusted_series[date].sum(axis=1, skipna=False)

            if confidence_level is not None and threshold is None:
                q = adjusted_series_temp.quantile(confidence_level)
            elif confidence_level is None and threshold is not None:
                q = threshold.loc[date]
            elif confidence_level is None and threshold is None:  # Compute an average of the adjusted series
                q = np.inf
            else:
                raise ValueError("Incorrect prescription of quantile/threshold")
            
            extremes = adjusted_series_temp[adjusted_series_temp <= q]            
            etl.loc[date] = extremes.mean()
            if return_no_points:
                no_points.loc[date] = len(extremes)
        if return_no_points:
            return etl.fillna(0), no_points
        else:
            return etl.fillna(0)

    def get_bilateral_expected_shortfall(self, dates, confidence_level=None, threshold=None, window=None, position="long",
                                         return_no_points=False):
        # print("Removing zero drawdowns!")
        if dates is None:
            dates = self.dates
        else:
            if np.asarray(dates).shape == (): dates = [dates]
            dates = pd.to_datetime(dates).sort_values()
        if hasattr(self, 'get_drawdowns'):
            drawdowns_dic = self.get_drawdowns(dates, position=position)
        else:
            returns = self.underlying.get_return(self.underlying.get_dates(), percentage=False, raw=True)
            drawdowns_dic = {date:returns[:date - pd.Timedelta("1D")] for date in dates}
        adjustments_dic = self.get_adjustment(dates)
        etl = pd.Series(index=dates)
        if return_no_points:
            no_points = pd.Series(index=dates)
        dates = dates[dates.isin(adjustments_dic.keys()) & dates.isin(drawdowns_dic.keys())]
        for date in dates:
            try:
                adjusted_series = (drawdowns_dic[date]*adjustments_dic[date]).sum(axis=1, skipna=False)
            except:
                adjusted_series = drawdowns_dic[date]*adjustments_dic[date]
            # adjusted_series = adjusted_series[adjusted_series != 0]
            if window is not None:
                adjusted_series = adjusted_series[-window:]
            if confidence_level is not None and threshold is None:
                q1 = adjusted_series.quantile(confidence_level)
                q2 = adjusted_series.quantile(1-confidence_level)
                q = max(np.abs(q1), np.abs(q2))
            elif confidence_level is None and threshold is not None:
                q = threshold.loc[date]
            else:
                print("Incorrect prescription of quantile/threshold")
                pass
            extremes = adjusted_series[np.abs(adjusted_series) >= q]
            etl.loc[date] = np.abs(extremes).mean()
            if return_no_points:
                no_points.loc[date] = len(extremes)
        if return_no_points:
            return etl.fillna(0), no_points
        else:
            return etl.fillna(0)

    def get_adjustment(self, dates):
        """
        For each date, return a series of 1.0 indexed by all instrument dates strictly before that date.
        """
        if np.asarray(dates).shape == ():
            dates = [dates]
        dates = pd.to_datetime(dates)

        adjustment = {}
        name = getattr(self.underlying, "ticker",
                       getattr(self.underlying, "name", None))

        for date in dates:
            prior_idx = self.dates[self.dates < date]
            s = pd.Series(1.0, index=prior_idx, dtype="float64")
            if name is not None:
                s.name = name
            adjustment[date] = s

        return adjustment


class PlainEWMARiskFactor(EmpiricalRiskFactor):
    def __init__(self, underlying, memory=0.94, window=250, raw=False, tenor=None):
        self.underlying = underlying
        self.memory = memory
        self.window = window
        self.raw = raw
        self.tenor = tenor

    def compute_vol(self, returns_no_mean):
        sigma_init = np.std(returns_no_mean)
        # dates = self.underlying.get_dates()
        dates = returns_no_mean.index
        # means = pd.Series(np.nan, index=dates)
        sigma = pd.Series(np.nan, index=dates)
        for date in dates:
            returns = returns_no_mean.loc[:date - pd.Timedelta("1D")]
            returns = returns[returns != 0].sort_index()
            returns = returns.values
            returns = np.concatenate(([sigma_init], returns))
            if type(self.window) == int:
                returns = returns[-self.window:]
            elif type(self.window) == str:
                returns = returns[date-pd.Timedelta(self.window):]
            weights = self.memory ** np.flip(np.arange(len(returns)))
            # mean = np.sum(returns * weights) / np.sum(weights)
            # means.loc[date] = mean
            # mean = np.mean(returns)
            # sigma.loc[date] = np.sqrt(np.sum(weights * ((returns - mean) ** 2)) / np.sum(weights))
            sigma.loc[date] = np.sqrt(np.sum(weights * (returns ** 2)) / np.sum(weights))

        # self.means = means
        self.vol = sigma  # [2:]
        self.residuals = returns_no_mean / self.vol
        self.dates = self.vol.index.sort_values()

    def fit(self, beginning_date=None, end_date=None):
        # self.returns = self.underlying.get_return(self.underlying.get_dates(), percentage=False, raw=self.raw)
        dates = self.underlying.get_dates()
        if beginning_date is not None:
            dates = dates[dates >= beginning_date]
        if end_date is not None:
            dates = dates[dates <= end_date]
        if self.tenor is None:
            returns = self.underlying.get_return(self.underlying.get_dates(), percentage=False, raw=self.raw)
            raw_returns = self.underlying.get_return(self.underlying.get_dates(), percentage=False, raw=True)
        else:
            self.prices = self.underlying.get_value(dates)
            # index = self.prices.index[1:]
            raw_returns = pd.Series(index=dates)
            returns = pd.Series(index=dates)
            for date in dates:
                if self.tenor == "M":
                    array = self.prices[(dates <= date) & (dates.year == date.year) & (dates.month == date.month)]
                elif self.tenor == "Q":
                    array = self.prices[(dates <= date) & (dates.year == date.year) & (dates.quarter == date.quarter)]
                elif self.tenor == "Y":
                    array = self.prices[(dates <= date) & (dates.year == date.year)]
                else:
                    array = self.prices[(dates <= date)]
                if array.size <2: continue
                raw_returns.loc[date] = array.iloc[-1] - array.iloc[-2]
                if self.raw == True:
                    returns.loc[date] = array.iloc[-1] - array.iloc[-2]
                else:
                    returns.loc[date] = np.log(array.iloc[-1] / array.iloc[-2])
        self.raw_returns = raw_returns.dropna()
        self.returns = returns.dropna()
        self.compute_vol(self.returns)
        self.dates = self.vol.index.sort_values()

    def get_adjustment(self, dates):
        if np.asarray(dates).shape == (): dates = [dates]
        dates = pd.to_datetime(dates)
        all_dates = self.dates
        all_dates = all_dates[all_dates.isin(self.vol.index)]
        # we need to filter out the very first date, otherwise there is no past returns and vol to compute anything
        dates = dates[dates > all_dates[0]]
        adjustment = {}
        if not self.raw:
            # dates = dates[dates.isin(self.vol.index)]
            all_prices = self.underlying.get_value(all_dates)
            all_vols = self.vol.loc[all_dates]
            for date in dates:
                prices = all_prices[all_dates < date]
                vols = all_vols[all_dates < date]
                adjustment[date] = prices.iloc[-1]*vols.iloc[-1] / (prices * vols)
                try:
                    adjustment[date].name = self.underlying.ticker
                except:
                    adjustment[date].name = self.underlying.name
            return adjustment
        else:
            # dates = dates[dates.isin(self.vol.index)]
            all_vols = self.vol.loc[all_dates]
            for date in dates:
                vols = all_vols[all_dates < date]
                adjustment[date] = vols.iloc[-1] / vols
            return adjustment

    def get_currency_converter(self, dates):
        if np.asarray(dates).shape == (): dates = [dates]
        dates = pd.to_datetime(dates)
        all_dates = self.dates
        all_dates = all_dates[all_dates.isin(self.vol.index)]
        # we need to filter out the very first date, otherwise there is no past returns and vol to compute anything
        dates = dates[dates > all_dates[0]]
        converter = pd.Series(index=dates)
        if not self.raw:
            # dates = dates[dates.isin(self.vol.index)]
            all_prices = self.underlying.get_value(all_dates)
            all_vols = self.vol.loc[all_dates]
            for date in dates:
                prices = all_prices[all_dates < date]
                vols = all_vols[all_dates < date]
                converter.loc[date] = prices.iloc[-1]*vols.iloc[-1]
        else:
            # dates = dates[dates.isin(self.vol.index)]
            all_vols = self.vol.loc[all_dates]
            for date in dates:
                vols = all_vols[all_dates < date]
                converter.loc[date] = vols.iloc[-1]
        try:
            converter.name = self.underlying.ticker
        except:
            converter.name = self.underlying.name
        return converter


class EWMARiskFactor(PlainEWMARiskFactor):
    def __init__(self, underlying, memory=0.94, window=250, raw=False):
        PlainEWMARiskFactor.__init__(self, underlying, memory=memory, window=window, raw=raw)

    def fit(self, beginning_date=None, end_date=None, print_arima=False):
        self.returns = self.underlying.get_return(self.underlying.get_dates())
        if beginning_date is not None:
            self.returns = self.returns[beginning_date:]
        if end_date is not None:
            self.returns = self.returns[:end_date]
        if len(self.returns) < 10:
            raise ValueError("Not enough data to fit ARIMA after trimming.")
        self.compute_vol(self.returns)
        spec = pm.auto_arima(
            self.returns,
            d=0,
            start_p=1, start_q=1,
            max_p=4,  max_q=4,
            seasonal=False,
            information_criterion="bic",
            trace=print_arima,
            error_action="ignore",
            suppress_warnings=True,
            stepwise=True,
        )
        self.arima_orders = spec.order  # (p, d, q)
        model = ARIMA(
            self.returns,
            order=self.arima_orders,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        self.arima = model.fit()
        resid = self.arima.resid  # property on the fitted results
        if not isinstance(resid, pd.Series):
            resid = pd.Series(resid, index=self.returns.index[-len(resid):])
        else:
            # if statsmodels returned a Series with a RangeIndex, realign to returns
            if not resid.index.equals(self.returns.index):
                resid.index = self.returns.index[-len(resid):]
        self.arima_residuals = resid
        self.compute_vol(self.arima_residuals)
        self.dates = self.vol.index.sort_values()


class GARCHRiskFactor(EmpiricalRiskFactor):
    def __init__(self, underlying, p=1, q=1):
        self.underlying = underlying
        self.p = p
        self.q = q
        self.loglikelihood = []

    def fit(self, use_arima=False, print_arima=False, print_arch="off", scale_factor=1):
        self.returns = self.underlying.get_return(self.underlying.get_dates())
        if not isinstance(self.returns, pd.Series):
            self.returns = pd.Series(self.returns)
        self.returns = pd.to_numeric(self.returns, errors="coerce").dropna().sort_index()
        if use_arima:
            spec = pm.auto_arima(
                self.returns,
                d=0,
                start_p=1, start_q=1,
                max_p=4,  max_q=4,
                seasonal=False,
                information_criterion="bic",
                trace=print_arima,
                error_action="ignore",
                suppress_warnings=True,
                stepwise=True,
            )
            self.arima_orders = spec.order  # (p, d, q)
            model = ARIMA(
                self.returns,
                order=self.arima_orders,
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            # Keep the *fitted results* in self.arima so `.resid` works
            self.arima = model.fit()
            # Ensure residuals are a Series aligned to the last len(resid) dates
            resid = self.arima.resid
            if not isinstance(resid, pd.Series):
                resid = pd.Series(resid, index=self.returns.index[-len(resid):])
            elif not resid.index.equals(self.returns.index):
                resid.index = self.returns.index[-len(resid):]
            self.arima_residuals = resid
            s = self.arima_residuals
        else:
            s = self.returns
        #    Keep the same arch_model defaults (mean="Constant", vol="GARCH")
        #    Scale inside the fit to keep attributes consistent with a chosen unit.
        self.garch = arch_model(s * scale_factor, p=self.p, q=self.q).fit(disp=print_arch)
        self.residuals = self.garch.resid / scale_factor
        self.vol = pd.Series(self.garch.conditional_volatility, index=s.index) / scale_factor
        self.dates = self.vol.index.sort_values()


class drawdownsRiskFactor(EmpiricalRiskFactor):
    """
    This class represents a contract series with a predefined way of computing drawdowns. 
    A drawdown represents the worst price change in a n day window. we differentiate between long and short drawdowns-

    Long drawdowns are defined as:
    .. math::
    D_{\\text{long}, t} := \\text{min}(p_{t+1}, \dots, p_{t+H}) - p_t

    Short drawdowns are defined as:
    .. math::
    D_{\\text{short}, t} := \\text{max}(p_{t+1}, \dots, p_{t+H}) - p_t

    Parameters
    ----------
    underlying : Underlying
        Underlying with the contract series data.
    raw : bool, optional
        If True, drawdowns are adjusted using absolute returns.
        If False, drawdowns are adjusted using percentage returns.
        Default is False.
    tenor : str, optional
        Underlying tenor. Default is None.

    Attributes
    ----------
    underlying : underlyings.Underlying
        Underlying with the contract series data.
    raw : bool
        If ``True``, drawdowns are adjusted using absolute returns.
        If ``False``, drawdowns are adjusted using percentage returns.
    tenor : str
        Underlying tenor.
    liquidation_period : int
        Number of days used as liquidation period when computing drawdowns.
    prices : array-like
        Prices series of the Underlying.
    dates : array-like
        Dates of the prices' series.
    drawdowns : dict
        Dictionary of drawdowns. Keys are 'long' and 'short' and values are pandas.Series of drawdowns.

    """

    def __init__(self, underlying, raw=False, tenor="D"):
        super().__init__()
        self.vol = None
        self.drawdowns = None
        self.prices = None
        self.liquidation_period = None
        self.underlying = underlying
        self.raw = raw
        self.tenor = tenor

    def fit(self, liquidation_period=2, beginning_date=None, end_date=None):
        """
        Fit the drawdowns, for a given liquidation period and dates.

        Parameters
        ----------
        liquidation_period : int, optional
            Number of days used as liquidation period when computing drawdowns. Default is 2.
        beginning_date : int, float, str or datetime
            First date on which drawdowns are fitted. If None, drawdowns are fitted since the first date in self.underlying.
        end_date : int, float, str or datetime
            Last date on which drawdowns are fitted. If None, drawdowns are fitted until the last date in self.underlying.

        Notes
        -----
        The fitted drawdowns are stored in the ``drawdowns`` attribute of the object.
        It takes into consideration the ``self.tenor`` in order to deprecate drawdowns affected by the contract rolling.

        Note short drawdowns are defined as:

        .. math::
            D_{\\text{short}, t} := \\text{max}(p_{t+1}, \dots, p_{t+H}) - p_t,

        A change in sign is done in the ``get_drawdowns`` method, to represent properly PnL for short positions.

        We do not consider drawdowns for the last liquidation_period days before a roll. 
        Rolling depends on the contracts and date:
          - Monthly contracts roll on new months.
          - Quarterly contracts roll on new quarters.
          - Yearly contracts roll on new years.

        """

        self.liquidation_period = liquidation_period
        dates = self.underlying.get_dates().sort_values()
        if beginning_date is not None:
            dates = dates[dates >= pd.to_datetime(beginning_date)]
        if end_date is not None:
            dates = dates[dates >= pd.to_datetime(end_date)]
        self.prices = self.underlying.get_value(dates)
        self.dates = self.prices.index
        index = self.prices.index[:-1]  # drawdowns are tried to be computed until the date before.
        positions = ["long", "short"]
        self.drawdowns = {position: pd.Series(np.full(index.size, np.nan), index=index) for position in positions}
        if self.tenor is None:
            self.drawdowns["long"].iloc[-1] = self.prices[-2] - self.prices[-3]
            self.drawdowns["short"].iloc[-1] = self.prices[-2] - self.prices[-3]
            prices_matrix = np.full((self.liquidation_period, self.dates.size - self.liquidation_period - 1), np.nan)
            for i in range(self.liquidation_period):
                prices_matrix[i, :] = self.prices.iloc[i + 1:-self.liquidation_period + i].values
            self.drawdowns["long"].loc[index[:-1]] = np.amin(prices_matrix, axis=0) - self.prices.iloc[:-self.liquidation_period-1]
            self.drawdowns["short"].loc[index[:-1]] = np.amax(prices_matrix, axis=0) - self.prices.iloc[:-self.liquidation_period-1]
        else:
            for date in index:  # mind contract rolling
                if self.tenor == "M":
                    array = self.prices[(self.dates > date) & (self.dates.year == date.year) & (self.dates.month == date.month)]
                elif self.tenor == "Q":
                    array = self.prices[(self.dates > date) & (self.dates.year == date.year) & (self.dates.quarter == date.quarter)]
                elif self.tenor == "Y":
                    array = self.prices[(self.dates > date) & (self.dates.year == date.year)]
                else:
                    array = self.prices[(self.dates > date)]
                if len(array.values) < liquidation_period:
                    continue
                self.drawdowns["long"].loc[date] = array.iloc[:self.liquidation_period].min() - self.prices.loc[date]
                self.drawdowns["short"].loc[date] = array.iloc[:self.liquidation_period].max() - self.prices.loc[date]
            self.drawdowns["long"] = self.drawdowns["long"][~self.drawdowns["long"].isnull()]
            self.drawdowns["short"] = self.drawdowns["short"][~self.drawdowns["short"].isnull()]

        self.vol = pd.Series(np.full(self.dates.size,np.nan), index=self.dates)

    def get_adjustment(self, dates, raw=None):
        """
        Calculate the drawdowns adjustment.

        Parameters
        ----------
        dates : array_like
            Dates on which we calculate the adjustment

        Returns
        ----------
        adjustment : dict
            Dictionary of adjustments. Keys are dates and values are floats.

        Notes
        -----
        If ``self.raw`` is ``True``, adjustment is computed based on absolute returns.
        If ``self.raw`` is ``False``, adjustment is computed based on percentage returns.

        """
        if raw is None:
            raw = self.raw
        if raw:
            return super().get_adjustment(dates)
        
        if np.asarray(dates).shape == (): dates = [dates]
        dates = pd.to_datetime(dates)
        adjustment = {}

        prices = self.underlying.get_value(self.dates)
        for date in dates:
            temp_dates = prices.index[prices.index < date]
            if temp_dates.empty: 
                adjustment[date] = pd.Series([])
                continue
            temp_prices = prices.loc[temp_dates]
            adjustment_series = prices[prices.index <= date][-1] / temp_prices
            try:
                adjustment_series.name = self.underlying.ticker
            except:
                adjustment_series.name = self.underlying.name
            adjustment[date] = adjustment_series

        return adjustment

    def get_currency_converter(self, dates):
        """
        Returns a factor converter from drawdowns to currency for the given dates.

        Parameters
        ----------
        dates : array_like
            Dates on which we calculate the converter.

        Returns
        ----------
        converter : pandas.Series
            Series with the converter factor for each date.

        Notes
        -----
        If ``self.raw`` is ``True``, currency converter is always 1 since all the calculations (drawdowns, VaR, etc.) are already using currency terms (€).
        If ``self.raw`` is ``False``, currency converter is the previous date price, since this is the price it is used to get the scan range of the following day.

        """

        if np.asarray(dates).shape == (): dates = [dates]
        dates = pd.to_datetime(dates)
        converter = pd.Series(index=dates, dtype='float64')
        if self.raw:
            converter = pd.Series(1, index=dates)
        else:
            prices = self.underlying.get_value(self.dates)
            common_dates =  dates.intersection(self.underlying.get_value(self.dates).index)
            converter.loc[common_dates] = self.underlying.get_value(self.dates).loc[common_dates]
            converter.fillna(method='ffill', inplace=True)  # we fill the nans with the previois prices
        try:
            converter.name = self.underlying.ticker
        except:
            converter.name = self.underlying.name
        return converter

    def get_drawdowns(self, dates, position="long"):
        """
        Returns the required drawdowns for each date, i.e. for each date it returns all the possible drawdowns computed until that date.
        Drawdowns are already computed and are stored in ``self.drawdowns``.

        Parameters
        ----------
        dates : array_like
            Dates from which we return a drawdown.
        position : str
            A string representing the required drawdown position (either "long" or "short"). Default is "long".

        Returns
        ----------
        drawdowns_dic : dict
            Dictionary of required drawdowns for each date. Keys are dates and values are pandas.Series.

        Notes
        -----
        Note how ``self.tenor`` is used to compute accordingly the last drawdown, which is computed
        with a liquidation period of two although ``self.liquidation_period`` may be longer.

        Short drawdowns are changed the sign, in order to represent real PnL.

        """
        if np.asarray(dates).shape == ():
            dates = [dates]
        dates = pd.to_datetime(dates)
        drawdowns_dic = {}
        for date in dates:
            temp_dates = self.drawdowns["long"].index[self.drawdowns["long"].index < date]
            if temp_dates.empty: 
                drawdowns_dic[date]=pd.Series([], dtype='float64')
                continue
            drawdowns = pd.Series(index=temp_dates, dtype='float64')
            try:
                drawdowns.name = self.underlying.ticker
            except:
                drawdowns.name = self.underlying.name

            # in the following, we rewrite the last drawdown according to our date
            if self.tenor == "D":
                array = self.prices[(self.dates > temp_dates[-1]) & (self.dates <= date)]
            elif self.tenor == "M":
                array = self.prices[(self.dates > temp_dates[-1]) & (self.dates <= date) &
                                    (self.dates.year == temp_dates[-1].year) & (self.dates.month == temp_dates[-1].month)]
            elif self.tenor == "Q":
                array = self.prices[(self.dates > temp_dates[-1]) & (self.dates <= date)
                                    & (self.dates.year == temp_dates[-1].year) & (self.dates.quarter == temp_dates[-1].quarter)]
            elif self.tenor == "Y":
                array = self.prices[(self.dates > temp_dates[-1]) & (self.dates <= date)
                                    & (self.dates.year == temp_dates[-1].year)]
            else:
                raise ValueError("Incorrect tenor.")
            if position == "long":
                drawdowns.loc[temp_dates[-1]] = array[:2].min() - self.prices.loc[temp_dates[-1]]
            elif position == "short":
                drawdowns.loc[temp_dates[-1]] = array[:2].max() - self.prices.loc[temp_dates[-1]]

            drawdowns.loc[temp_dates[:-1]] = self.drawdowns[position].loc[temp_dates[:-1]]  # we assign the fitted drawdowns to the rest
            if position == "short":
                drawdowns = -drawdowns
            drawdowns_dic[date] = drawdowns.dropna()
        return drawdowns_dic


class drawdownsPortfolio(EmpiricalRiskFactor):
    """
    Portfolio of two drawdownsRiskFactor combined with user-defined weights.

    The Portfolio class represents a combination of two fitted risk factors,
    allowing for drawdown analysis and adjustment decomposition. Weights can be
    specified as scalars (automatically expanded to match factor dates) or as
    pandas Series indexed by dates. The class ensures that both risk factors
    share consistent raw parameters and aligns their dates with the provided weights.

    Parameters
    ----------
    riskfactor1 : EmpiricalRiskFactor
        First fitted risk factor instance.
    riskfactor2 : EmpiricalRiskFactor
        Second fitted risk factor instance.
    weights1 : float or pandas.Series
        Weight applied to the first risk factor. If scalar, it is expanded to
        a pandas Series aligned with `riskfactor1.dates`.
    weights2 : float or pandas.Series
        Weight applied to the second risk factor. If scalar, it is expanded to
        a pandas Series aligned with `riskfactor2.dates`.

    Attributes
    ----------
    risk_factors : tuple of EmpiricalRiskFactor
        The two risk factors included in the portfolio.
    weights : list of pandas.Series
        List containing the weights for each risk factor, indexed by dates.
    ticker : str, optional
        Concatenated ticker identifier if both risk factors expose a `ticker` attribute.
    dates : pandas.DatetimeIndex
        Common dates between both risk factors and their weights. Defined upon `fit`.

    """

    def __init__(self, riskfactor1, riskfactor2, weights1, weights2):
        if riskfactor1.raw != riskfactor2.raw:
            print("Inconsistent raw parameters")
            pass
        self.risk_factors = (riskfactor1, riskfactor2)
        self.weights = [weights1, weights2]
        for i in [0, 1]:
            if np.asarray(self.weights[i]).shape == ():
                self.weights[i] = pd.Series(self.weights[i], index=self.risk_factors[i].dates)
        if hasattr(riskfactor1, "ticker") and hasattr(riskfactor2, "ticker"):
            self.ticker = riskfactor1.ticker + "-" + riskfactor2.ticker

    def fit(self, liquidation_period=2, beginning_date=None, end_date=None):
        """
        Set value to self.liquidation_period and self.dates attributes. 
        self.dates is defined as dates which have an associate weight and are in both risk_factor dates. 
        In addition, beginning_date and end_date can limit self.dates.


        Parameters
        ----------
        liquidation_period : float
            liquidation_period to be used for drawdowns calculation.
        beginning_date : int, float, str or datetime
            Minimum date where self.dates starts. Default is None.
        end_date : int, float, str or datetime
            Maximum date where self.dates ends. Default is None.

        Returns
        ----------

        """
        self.liquidation_period = liquidation_period
        dates = pd.DatetimeIndex.intersection(*(self.risk_factors[i].dates for i in [0, 1]))
        if beginning_date is not None:
            dates = dates[dates >= pd.to_datetime(beginning_date)]
        if end_date is not None:
            dates = dates[dates >= pd.to_datetime(end_date)]
        weights_dates = pd.DatetimeIndex.intersection(*(self.weights[i].index for i in [0, 1]))
        self.dates = dates.intersection(weights_dates)

    def get_drawdowns(self, dates, position='long'):
        """
        Returns the required drawdowns for each date, i.e. for each date it returns all the possible drawdowns computed until that date.
        Drawdowns are not computed for the portfolio in this method, instead it returns a concat df with both risk factors drawdowns.

        Parameters
        ----------
        dates : array_like
            Dates from which we return a drawdown.
        position : str
            A string representing the required drawdown position (either "long" or "short"). Default is "long".

        Returns
        ----------
        drawdowns_dic : dict
            Dictionary of required drawdowns for each date. Keys are dates and values are pandas.Series.

        """
        if np.asarray(dates).shape == (): dates = [dates]
        dates = pd.to_datetime(dates)
        drawdowns_dic = {}
        for date in dates:
            temp_weights = [self.weights[i].loc[date] for i in [0, 1]]
            epsilon = 1 if position == "long" else -1
            positions = ["long" if (epsilon * temp_weights[i] >= 0) else "short" for i in [0,1]]
            individual_drawdowns = [self.risk_factors[i].get_drawdowns(date, position=positions[i]) for i in [0, 1]]
            # below, we multiply by abs(weight) because the drawdown series already comes negative for short
            drawdowns_dic[date] = pd.concat(
                (individual_drawdowns[i][date] * np.abs(self.weights[i]).loc[date] for i in [0, 1]), axis=1).dropna()
        return drawdowns_dic

    def get_adjustment(self, dates, raw=None):
        if np.asarray(dates).shape == (): dates = [dates]
        dates = pd.to_datetime(dates)
        individual_adjustments = [self.risk_factors[i].get_adjustment(dates) for i in [0, 1]]
        adjustment_dic = {date: pd.concat((individual_adjustments[i][date] for i in [0, 1]), axis=1).dropna() for date in dates}
        return adjustment_dic


class DCC_GARCHRiskFactor(EmpiricalRiskFactor):
    def __init__(self, underlying,  dist='norm', scale_factor=1):
        '''
        This library computes a DCC-GARCH(1,1) for the returns of a given underlying object

        To use it, we should give as input a underlying object, and the disrtibution for the DCC-GARCH (normal distribution, or t-student)

        First computes a standard GARCH(1,1) (with a N(0,1) as SWN(0,1))
        Next, it computes the DCC-GARCh using the paper of Eigel and Sheppard and saves the alfa and beta
        Finally, if we want to see the results, we can use the self.generate_plots()

        Example of use
        --------------
        dcc_garch = DCC_GARCHRiskFactor(underlying)
        dcc_garch.fit()
        dcc_garch.generate_plots()
        dcc_garch.information()

        Parameters
        ----------
        underlying -> underlying object with 2 instruments
        dist -> string (norm or t)
        scale_factor -> integer value to scale the returns to avoid numerical problems

        References
        ----------
        Theoretical and Empirical properties of Dynamic Conditional Correlation Multivariate GARCH - Engle, Sheppard
        '''
        returns = pd.DataFrame()
        for ticker in underlying.keys():
            returns[ticker] = underlying[ticker].get_return(underlying[ticker].get_dates())
        returns = returns.dropna()

        self.scale_factor = scale_factor

        self.index = returns.index
        self.df_rt = returns
        self.rt = np.matrix(returns.values) * self.scale_factor

        self.T = self.rt.shape[0]  # number of days
        self.N = self.rt.shape[1]  # Number of assets
        self.k = 2  # Assets on each DCC #Do not touch it, the code it is not ready
        if self.N == 1 or self.T < 4:
            return 'Invalidad data, it does not satisfy the required dimensions'

        if dist == 'norm' or dist == 't':
            self.dist = dist
        else:
            print("Takes pdf name as param: 'norm' or 't'.")

        self.garchs = {}
        self.dccgarchs = {}

    def garch_var(self, params, residuals):
        """
        Computes vector of \sigma_{t}^{2}. Definition 4.22 McNeil

        Parameters
        ----------
        params: ndarray
            Values of GARCH volatility model parameters (3)
        residuals: ndarray
            Products residuals
        
        Returns
        -------
        var_t: ndarray
            Vector of \sigma_{t}^{2}
        """
        T = len(residuals)
        omega = params[0]
        alpha = params[1]
        beta = params[2]
        var_t = np.zeros(T)
        for i in range(T):
            if i == 0:
                var_t[i] = residuals[i] ** 2
            else:
                var_t[i] = omega + alpha * (residuals[i - 1] ** 2) + beta * var_t[i - 1]
        return var_t

    def garch_loglike_sheppard(self, params, product, individual=False):
        """
        Computes the GARCH log-likelihood function using the entire model for a given product

        Parameters
        ----------
        params: ndarray
            Values for the GARCH parameters (1 mean parameter + 3 volatility parameters) 
        product: str
            Name of product
        individual : bool, optional
            Flag indicating whether to return the vector of individual log
            likelihoods (True) or the sum (False)

        Returns
        -------
        loglikelihood : float
            Negative of model loglikelihood
        """
        
        # Inicialize _loglikelihood (Arch Library Function) parameters.  
        resids = np.asarray(self.garchs[product]._model.resids(self.garchs[product]._model.starting_values()), dtype=float)
        sigma2 = np.zeros_like(resids)
        backcast = self.garchs[product]._model._backcast
        var_bounds = self.garchs[product]._model._var_bounds
        
        # Evaluate
        loglikelihood = -1.0 * self.garchs[product]._model._loglikelihood(params, sigma2, backcast, var_bounds,individual)

        return loglikelihood

    def mgarch_loglike(self, params, r_t, D_t, get_matrices=False, individual=False):
        """
        Computes the DCC QL_{2} log-likelihood function of Sheppard's paper for 2 products (k=2).

        Math expression:

        -1/2 \sum_{i=1}^{T}(k \log(2\pi) + 2*\log(|D_{t}|) + log(|R_{t}|) + r'_{t}D_{t}^{-1}R_{t}^{-1}D_{t}^{-1}r_{t})

        We do *(-1) to minimize instead of maximize.

        Parameters
        ----------
        params: ndarray
            Values for the DCC parameters (2)
        r_t: ndarray
            Products returns
        D_t: ndarray
            D_{t} matrix of Sheppard's paper
        get_matrices: bool, optional
            Flag indicating whether to return the likelihood and matrix R_t, H_t (True) or only likelihood (False)
        individual : bool, optional
            Flag indicating whether to return the vector of individual log
            likelihoods (True) or the sum (False)

        Returns
        -------
        loglike : float
            Negative of loglikelihood
        R_t: ndarray
            R_{t} matrix of Sheppard's paper
        H_t: ndarray
            H_{t} matrix of Sheppard's paper


        Notes
        -------
        Comentarios de Gonzalo:
        Creo que hay un typo en la fórmula 14.13 del libro de McNeil (Quantitative Risk Management) en el P_{t-j}

        He buscado por internet, y por ejemplo usando como referencia el paper original de Engle
        En la fórmula del log-likelihood (26) y es la misma que se ha programado.

        Aun así, he visto un error en el código:
        (et * et.T) ha sido modificado por (et_anterior * et_anterior.T)
        np.log(D_t[i].sum()) lo he sustituido por np.log(D_t[i]).sum()
        """
        # rho_21 = params[0]
        # a = params[1]
        # b = params[2]
        a = params[0]
        b = params[1]
        if a + b >= 1:
            print('Input parameters do not satisfy a+b<1. Returning 1e10')
            return 1e10  # Very high value

        # Q_bar = np.array([[1, rho_21], [rho_21, 1]])
        Q_bar = np.cov(r_t.T)

        Q_t = np.zeros((self.T, 2, 2))
        R_t = np.zeros((self.T, 2, 2))
        H_t = np.zeros((self.T, 2, 2))

        # Q_t[0] = np.matmul(r_t[0].T / 2, r_t[0] / 2)
        Q_t[0] = np.mat(np.eye(2))  # inicializo con la identidad

        if individual is False:
            loglike = 0
        else:
            loglike = []

        for i in range(1, self.T):
            dts = np.diag(D_t[i])
            # dtinv = np.linalg.inv(dts)
            # et = dtinv * r_t[i].T

            dts_anterior = np.diag(D_t[i-1])
            dtinv_anterior = np.linalg.inv(dts_anterior)
            et_anterior = dtinv_anterior * r_t[i-1].T

            # Q_t[i] = (1 - a - b) * Q_bar + a * (et * et.T) + b * Q_t[i - 1]
            Q_t[i] = (1 - a - b) * Q_bar + a * (et_anterior * et_anterior.T) + b * Q_t[i - 1]
            qts = np.linalg.inv(np.sqrt(np.diag(np.diag(Q_t[i]))))

            R_t[i] = np.matmul(qts, np.matmul(Q_t[i], qts))

            H_t[i] = np.matmul(dts, np.matmul(R_t[i], dts))

            if np.linalg.det(R_t[i]) <= 0:
                print('Negative determinant found')

            if individual is False:
                loglike = loglike + self.k * np.log(2 * np.pi) + 2 * np.log(D_t[i]).sum() + np.log(np.linalg.det(R_t[i])) + np.matmul(
                    r_t[i], (np.matmul(np.linalg.inv(H_t[i]), r_t[i].T)))  # Fórmula L del paper de Engle y Sheppard
            else:
                loglike.append((self.N * np.log(2 * np.pi) + 2 * np.log(D_t[i]).sum() + np.log(np.linalg.det(R_t[i])) + np.matmul(
                    self.rt[i], (np.matmul(np.linalg.inv(H_t[i]), self.rt[i].T)))).item())

        if get_matrices:
            if individual is False:
                return 1/2*loglike, R_t, H_t
            else: 
                return 1/2*np.array(loglike), R_t, H_t
        else:
            if individual is False:
                return 1/2*loglike
            else: 
                return 1/2*np.array(loglike)

    def mgarch_logliket(self, params, D_t):
        """
        Not implemented yet. Future t-student loglikelihood function
        """
        # No of assets
        a = params[0]
        b = params[1]
        dof = params[2]
        Q_bar = np.cov(self.rt.T)

        Q_t = np.zeros((self.T, self.N, self.N))
        R_t = np.zeros((self.T, self.N, self.N))
        H_t = np.zeros((self.T, self.N, self.N))

        Q_t[0] = np.matmul(self.rt[0].T / 2, self.rt[0] / 2)

        loglike = 0
        for i in range(1, self.T):
            dts = np.diag(D_t[i])
            dtinv = np.linalg.inv(dts)
            et = dtinv * self.rt[i].T
            Q_t[i] = (1 - a - b) * Q_bar + a * (et * et.T) + b * Q_t[i - 1]
            qts = np.linalg.inv(np.sqrt(np.diag(np.diag(Q_t[i]))))

            R_t[i] = np.matmul(qts, np.matmul(Q_t[i], qts))

            H_t[i] = np.matmul(dts, np.matmul(R_t[i], dts))

            loglike = loglike + np.log(gamma((self.N + dof) / 2.)) - np.log(gamma(dof / 2)) - (self.N / 2.) * np.log(
                np.pi * (dof - 2)) - np.log(np.linalg.det(H_t[i])) - ((dof + self.N) * (
                        ((np.matmul(self.rt[i], (np.matmul(np.linalg.inv(H_t[i]), self.rt[i].T)))) / (dof - 2.)) + 1) / 2.)

        return -loglike

    def QL1_sheppard(self, params, product1, product2, individual=False):
        """
        Computes the DCC QL_{1} log-likelihood function of Sheppard's paper for 2 products (k=2).

        Math expression:

        -1/2 \sum_{i=1}^{k}(T \log(2\pi) + \sum_{t=1}^{T}(\log(h_{it}) + \frac{r_{it}^{2}}{h_{it}}))

        which is simply the sum of the log-likelihoods of the individual GARCH models for each of the assets.

        Parameters
        ----------
        params: ndarray
            Values for the GARCHs parameters (4+4) and DCC parameters (2) 
        product1: str
            Name of product 1
        product2: str
            Name of product 2
        individual : bool, optional
            Flag indicating whether to return the vector of individual log
            likelihoods (True) or the sum (False)

        Returns
        -------
        ql1 : float
            Negative of loglikelihood
        """

        ql1 = self.garch_loglike_sheppard(params[:4], product1, individual) + self.garch_loglike_sheppard(params[4:8], product2, individual)
        return ql1
    
    def QL2_sheppard(self, params, r_t, individual):
        """
        Computes the DCC QL_{2} log-likelihood function of Sheppard's paper for 2 products (k=2) with 10 parameters.

        Math expression:

        -1/2 \sum_{i=1}^{T}(k \log(2\pi) + 2*\log(|D_{t}|) + log(|R_{t}|) + r'_{t}D_{t}^{-1}R_{t}^{-1}D_{t}^{-1}r_{t})

        We do *(-1) to minimize instead of maximize.

        Parameters
        ----------
        params: ndarray
            Values for the GARCHs parameters (4+4) and DCC parameters (2) 
        r_t: ndarray
            Residuals of products
        individual : bool, optional
            Flag indicating whether to return the vector of individual log
            likelihoods (True) or the sum (False)

        Returns
        -------
        ql2 : float
            Negative of loglikelihood

        Notes
        -----

        r_t should be calculated inside the function instead of be an input parameter because
        mu_product1 (params[0]) and mu_product2 (params[4]) are not parameters right now (never used)
        """

        D_t = np.column_stack((np.sqrt(self.garch_var(params[1:4],r_t[:,0])),
                                        np.sqrt(self.garch_var(params[5:8],r_t[:,0]))))
        
        ql2 = self.mgarch_loglike(params[8:],r_t, D_t, get_matrices=False, individual=individual)
        return ql2

    def errors_sheppard(self, r_t, D_t, product1, product2, name):
        """
        Computes significances following Sheppard's paper

        Parameters
        ----------
        r_t: ndarray
            Products returns
        D_t: ndarray
            D_{t} matrix of Sheppard's paper
        product1: str
            Name of product 1
        product2: str
            Name of product 2
        name: str
            Name of DCC model

        Returns
        ------- 
        output: str
            String to be printed with significances values.
        """

        params = np.hstack((np.array(self.garchs[product1].params),np.array(self.garchs[product2].params),self.dccgarchs[name].x))
    
        kwargs_2 = {"r_t": r_t,
                   "individual": False}

        hessian_lnf2 = approx_hess(params, self.QL2_sheppard, kwargs=kwargs_2)

        A_12 = hessian_lnf2[8:,:8]/self.T
        A_22 = hessian_lnf2[8:,8:]/self.T
        A_21 = np.zeros((8, 2))

        kwargs_1 = {"product1": product1, 
                    "product2": product2, 
                    "individual": False}

        hessian_lnf1 =  approx_hess(params[:8],self.QL1_sheppard,kwargs=kwargs_1)

        A_11 = hessian_lnf1/self.T

        A_0 = np.block([[A_11, A_21], [A_12, A_22]])

        kwargs_2g = {"r_t": r_t,
                   "individual": True}
        
        g_lnf2 = approx_fprime(
                params, self.QL2_sheppard, kwargs=kwargs_2g
                )

        g_lnf2 = g_lnf2[:,8:]

        kwargs_1g = {"product1": product1, 
                    "product2": product2, 
                    "individual": True}

        g_lnf1 = approx_fprime(
                params[:8], self.QL1_sheppard, kwargs=kwargs_1g
                )[1:,:]

        b = np.concatenate((g_lnf1, g_lnf2), axis=1)

        B_0 = np.cov(b.T)

        inv_A_0 = np.linalg.inv(A_0)

        param_cov = inv_A_0.dot(B_0).dot(inv_A_0) / self.T

        std_err = np.asarray(np.sqrt(np.diag(param_cov)), dtype=float)

        tvalues = params / std_err

        pvalues = np.asarray(stats.norm.sf(np.abs(tvalues)) * 2, dtype=float)

        filas = [
            f'mu_{product1}', f'alpha_{product1}', f'beta_{product1}', f'omega_{product1}',
            f'mu_{product2}', f'alpha_{product2}', f'beta_{product2}', f'omega_{product2}',
            'alpha_DCC', 'beta_DCC'
        ]
        columnas = ['Parameter','Coefficient', 'Std.Error', 't-value', 'p-value',]

        output = ""

        output += "\n"
        output += "="*90 + "\n"
        output += '{}{:<20}{:<20}{:<20}{:<20}{}'.format('', *columnas) + "\n"
        output += "-"*90 + "\n"

        for i, fila in enumerate(filas):
            values = [params[i], std_err[i], tvalues[i], pvalues[i]]
            output += '{:<20}'.format(fila)
            for value in values:
                output += '{:<20.4e}'.format(value)  # Formatear los valores con notación científica
            output += "\n"  # Nueva línea para la siguiente fila

        output += "="*90 + "\n"

        return output

    def fit(self):
        for i, product in enumerate(self.df_rt.columns):  # Compute the individual GARCH processes
            self.garchs[product] = arch_model(self.rt[:, i]).fit()  # GARCH(1,1)
            print(f'Garch for {product}')
            print(self.garchs[product].summary())

        if self.dist == 'norm':
            for product1, product2 in itertools.combinations(self.df_rt.columns, self.k):
                name = f'{product1}--{product2}'
                r_t = np.matrix(np.column_stack((self.garchs[product1].resid,
                                                self.garchs[product2].resid)))  # residuals = returns - mean
                D_t = np.column_stack((self.garchs[product1].conditional_volatility,
                                       self.garchs[product2].conditional_volatility))  # sigma (sqrt(h))
                # constraints
                # a = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, 0, 1], [0, -1, -1]])
                # b = np.array([-1, -1, 0, 0, -0.999])
                a = np.array([[1, 0], [0, 1], [-1, -1]])
                b = np.array([0, 0, -1 + 1e-6])
                res = minimize(self.mgarch_loglike, (0.01, 0.94),  # (0, 0.01, 0.94)
                               args=(r_t, D_t),  # calcula rho_21, alfa_1 y beta_1 del DCC-GARCH
                               bounds=((1e-6, 1), (1e-6, 1)),  # ((-1, 1), (1e-6, 1), (1e-6, 1))
                               method="SLSQP",
                               constraints=constraint(a, b),
                               options={'disp': True})
                # res = dual_annealing(self.mgarch_loglike,
                #                      args=(r_t, D_t),
                #                      bounds=((1e-6, 1), (1e-6, 1)),
                #                      maxiter=50)
                _, res.R_t, res.H_t = self.mgarch_loglike(params=res.x, r_t=r_t, D_t=D_t, get_matrices=True)  # Save the matrices
                print(f'DCC-GARCH between {product1} and {product2}.'
                      f'Min:{res.fun}, parameters:{res.x}, iterations:{res.nit}, success={res.success}')

                self.dccgarchs[name] = res

                significance = self.errors_sheppard(r_t, D_t, product1, product2, name)

                print(significance)

        elif self.dist == 't':
            print('We still have to code the t-student part')
            exit()
            res = minimize(self.mgarch_logliket, (0.01, 0.94, 3), args=D_t, bounds=((1e-6, 1), (1e-6, 1), (3, None)),
                           # options = {'maxiter':10000000, 'disp':True},
                           )

    def generate_plots(self):
        """
        Se debe haber hecho previamente el .fit()

        Hace un plot de retornos, Garch y DCC-Garch
        """
        number_subplots = 6
        row_heights = [400, 400, 400, 400, 600, 600]
        moving_average = 7

        for product1, product2 in itertools.combinations(self.df_rt.columns, self.k):
            name = f'{product1}--{product2}'
            fig = make_subplots(rows=number_subplots, cols=1,
                                subplot_titles=[f'Returns for {col}' for col in [product1, product2]] +
                                               [f'GARCH for {col}' for col in [product1, product2]] +
                                               [f'DCC-Garch for {product1} and {product2}'] +
                                               [f'DCC-Garch with Moving Average {moving_average} for {product1} and {product2}'],
                                row_heights=row_heights)

            for i, product in enumerate([product1, product2]):
                fig.add_trace(go.Scatter(x=self.index, y=self.df_rt[product], mode='lines', showlegend=False), row=i + 1, col=1)

            for i, product in enumerate([product1, product2]):
                con_vol = self.garchs[product].conditional_volatility / self.scale_factor
                fig.add_trace(go.Scatter(x=self.index, y=con_vol, mode='lines', showlegend=False), row=i + 3, col=1)

            fig.add_trace(go.Scatter(x=self.index, y=self.dccgarchs[name].R_t[:, 1, 0], mode='lines', showlegend=False), row=5, col=1)

            fig.add_trace(go.Scatter(x=self.index, y=pd.Series(self.dccgarchs[name].R_t[:, 1, 0]).rolling(window=moving_average).mean(),
                                     mode='lines', showlegend=False), row=6, col=1)
            fig.update_layout(height=sum(row_heights))

            fig.show()


class EWMAMultiDim:
    """
    This class implements the Exponentially Weighted Moving Average (EWMA) method
    for estimating the time-varying covariance matrix and correlation matrix of
    multivariate financial returns data, as described in Ch.9.2.2 of McNeil -
    Quantitative Risk Management, Concepts, Techniques and Tools.

    Parameters
    ----------
    selected_columns : pandas.DataFrame
        DataFrame containing the selected columns from specified sheets of an Excel file.
    theta : float, optional
        The smoothing parameter for the EWMA method, which controls the rate of decay of past observations. Defaults to 0.04.
    """

    def __init__(self, selected_columns, theta=0.04):
        self.selected_columns = selected_columns
        self.theta = theta

    def fit(self):
        """
        Fit the EWMA model to the selected financial data.

        Returns:
            pandas.DataFrame
                DataFrame containing the time-varying correlation values over the given period.
        """
        returns = self.selected_columns.pct_change().dropna()
        r1 = returns.iloc[:, 0]
        r2 = returns.iloc[:, 1]

        r1_standard = r1 - r1.mean()
        r2_standard = r2 - r2.mean()

        # covariance_matrix = np.cov(r1_standard, r2_standard)
        covariance_matrix = np.eye(2)

        # Calculate correlation matrix for the initial covariance matrix
        D_0 = np.diag(1 / np.sqrt(np.diag(covariance_matrix)))
        correlation_matrix_0 = np.dot(np.dot(D_0, covariance_matrix), D_0)

        covariance_matrices = [covariance_matrix]
        correlation_matrices = [correlation_matrix_0]
        dates = [returns.index[0]]

        for t in range(1, len(returns)):
            X_t = np.array([r1_standard.iloc[t], r2_standard.iloc[t]])
            covariance_matrix_t_plus_1 = (
                self.theta * np.outer(X_t, X_t.T)
                + (1 - self.theta) * covariance_matrices[-1]
            )
            covariance_matrices.append(covariance_matrix_t_plus_1)

            D_t = np.diag(1 / np.sqrt(np.diag(covariance_matrix_t_plus_1)))
            correlation_matrix_t = np.dot(np.dot(D_t, covariance_matrix_t_plus_1), D_t)
            correlation_matrices.append(correlation_matrix_t)

            dates.append(returns.index[t])

        # Extract off-diagonal elements as correlation values
        correlation_values = [
            correlation_matrix[0, 1] for correlation_matrix in correlation_matrices
        ]

        # Create DataFrame with correlation values and dates
        correlation_df = pd.DataFrame(
            correlation_values, index=dates, columns=["Correlation"]
        )
        correlation_df.index.name = "Date"

        return correlation_df

    def generate_plot(self):
        """
        Generate a plot of the time-varying correlation values.

        Returns:
            plotly.graph_objs._figure.Figure
                Plotly figure object displaying the time-series of correlation values over the given period.
        """
        correlation_df = self.fit()

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=correlation_df.index, y=correlation_df["Correlation"], mode="lines"
            )
        )
        fig.update_layout(
            title="Correlation over Time", xaxis_title="Date", yaxis_title="Correlation"
        )
        fig.show()

        return fig
