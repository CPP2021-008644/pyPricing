# import basics
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings("ignore")


class PlainCorrelation:
    def __init__(self, window=250):
        self.window = window

    def fit(self, underlying1, underlying2):
        self.underlying = [underlying1, underlying2]
        dates = pd.DatetimeIndex.intersection(
            self.underlying[0].get_dates(), self.underlying[1].get_dates()
        ).sort_values()
        returns = [self.underlying[i].get_return(dates) for i in [0, 1]]
        corr = pd.Series(index=dates)
        for date in dates[2:]:
            corr.loc[date] = returns[0][:date][-self.window :].corr(
                returns[1][:date][-self.window :]
            )
        self.correlation = corr
        self.dates = self.correlation.index

    def get_conditional_correlation(self, dates):
        dates = afsfun.dates_formatting(dates)
        dates = dates[dates.isin(self.dates)]
        return self.correlation.loc[dates]


class EWMA2d:
    def __init__(self, l=0.94, window=None):
        self.l = l
        self.window = window

    def fit(self, underlying1, underlying2, print_arima=False):
        import pmdarima as pm
        from statsmodels.tsa.arima.model import ARIMA

        self.underlyings = [underlying1, underlying2]

        dates = pd.DatetimeIndex.intersection(
            underlying1.get_dates().sort_values()[1:],
            underlying2.get_dates().sort_values()[1:],
        )
        self.series = [underlying1.get_return(dates), underlying2.get_return(dates)]

        models = [
            pm.auto_arima(
                self.series[i],
                d=0,  # non-seasonal difference order
                start_p=1,  # initial guess for p
                start_q=1,  # initial guess for q
                max_p=4,  # max value of p to test
                max_q=4,  # max value of q to test
                seasonal=False,  # is the time series seasonal
                information_criterion="bic",  # used to select best model
                trace=print_arima,  # print results whilst training
                error_action="ignore",  # ignore orders that don't work
                stepwise=True,  # apply intelligent order search
            )
            for i in [0, 1]
        ]
        self.arima_orders = [models[i].order for i in [0, 1]]
        self.arimas = [
            ARIMA(self.series[i], order=models[i].order).fit() for i in [0, 1]
        ]
        self.residuals = [self.arimas[i].resid for i in [0, 1]]

        #         self.residuals = self.series
        r = np.full((self.residuals[0].size + 1, 2, 2), np.nan)
        r[0] = np.cov(self.residuals[0], self.residuals[1])
        r[1:, 0, 0] = self.residuals[0] ** 2
        r[1:, 1, 0] = self.residuals[0] * self.residuals[1]
        r[1:, 0, 1] = self.residuals[0] * self.residuals[1]
        r[1:, 1, 1] = self.residuals[1] ** 2
        self.cov_matrix = np.full((self.residuals[0].size + 1, 2, 2), np.nan)
        self.cov_matrix[0] = r[0]
        if self.window is None:
            for i in range(len(self.series[0])):
                self.cov_matrix[i + 1] = (1 - self.l) * r[i] + self.l * self.cov_matrix[
                    i
                ]
        else:
            for i in range(len(self.series[0])):
                number = min(i + 1, self.window) if self.window is not None else i + 1
                weights = self.l ** np.flip(np.arange(number))
                self.cov_matrix[i + 1] = np.sum(
                    weights.reshape((number, 1, 1)) * r[i + 1 - number : i + 1], axis=0
                ) / np.sum(weights)

    def get_conditional_variance(self, series):
        if series == 1:
            return pd.Series(self.cov_matrix[1:, 0, 0], index=self.series[0].index)
        elif series == 2:
            return pd.Series(self.cov_matrix[1:, 1, 1], index=self.series[1].index)
        else:
            print("Series number must be 1 or 2")
            return None

    def get_conditional_volatility(self, series, annualized=False, percentage=False):
        if series == 1:
            var = pd.Series(self.cov_matrix[1:, 0, 0], index=self.series[0].index)
        elif series == 2:
            var = pd.Series(self.cov_matrix[1:, 1, 1], index=self.series[1].index)
        else:
            print("Series number must be 1 or 2")
            return None
        return np.sqrt((1 + 251 * annualized) * var) * (1 + 99 * percentage)

    def get_conditional_covariance(self, annualized=False):
        return pd.Series(
            (1 + 250 * annualized) * self.cov_matrix[1:, 1, 0],
            index=self.series[0].index,
        )

    def get_conditional_correlation(self, dates=None):
        # if np.asarray(dates).shape == (): dates = [dates]
        # dates = pd.to_datetime(dates).sort_values()
        corr = self.get_conditional_covariance() / np.sqrt(
            self.get_conditional_variance(1) * self.get_conditional_variance(2)
        )
        if dates is not None:
            corr = corr.loc[dates]
        return corr


class GARCH_computer:
    def __init__(self, series_1, series_2, p=1, q=1, print_arima=False):
        import pmdarima as pm
        from statsmodels.tsa.arima.model import ARIMA
        from arch import arch_model

        if len(series_1) != len(series_2):
            print("Length time series do not match")
            return None
        else:
            if np.min(series_1.index == series_2.index) == 0:
                print("Indices of given times series do not match")
                return None
            else:
                pass

        self.series = [series_1, series_2]

        models = [
            pm.auto_arima(
                self.series[i],
                d=0,  # non-seasonal difference order
                start_p=1,  # initial guess for p
                start_q=1,  # initial guess for q
                max_p=4,  # max value of p to test
                max_q=4,  # max value of q to test
                seasonal=False,  # is the time series seasonal
                information_criterion="bic",  # used to select best model
                trace=print_arima,  # print results whilst training
                error_action="ignore",  # ignore orders that don't work
                stepwise=True,  # apply intelligent order search
            )
            for i in [0, 1]
        ]
        self.arima_orders = [models[i].order for i in [0, 1]]
        self.arimas = [
            ARIMA(self.series[i], order=models[i].order).fit(disp=0) for i in [0, 1]
        ]
        self.residuals = [self.arimas[i].resid for i in [0, 1]]

        self.p = p
        self.q = q

        self.garch = [
            arch_model(self.residuals[i], p=self.p, q=self.q).fit(disp="off")
            for i in [0, 1]
        ]

        self.vol = [self.garch[i].conditional_volatility for i in [0, 1]]
        resid = [self.garch[i].resid / self.vol[i] for i in [0, 1]]
        self.corr = np.corrcoef(resid[0], resid[1])[0, 1]

    def get_conditional_volatility(self, series, annualized=False, percentage=False):
        return (
            (1 + (np.sqrt(252) - 1) * annualized)
            * (1 + 99 * percentage)
            * pd.Series(self.vol[series - 1], index=self.series[series - 1].index)
        )

    def get_conditional_covariance(self, annualized=False):
        return (1 + 251 * annualized) * self.corr * self.vol[0] * self.vol[1]
