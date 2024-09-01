import numpy as np
import pandas as pd

try:
    from . import (
        functions as afsfun,
    )  # (Relative) import needed for the workspace. In this case __package__ is 'pypricing.pricing'
except (ImportError, ModuleNotFoundError, ValueError):
    import pricing.functions as afsfun  # (Absolute) local import


class DayCountCalendar:
    """
    Class for instantiating a calendar.

    Parameters
    ----------
    daycount_function : function object
        Function used to count days in time intervals.
    days_in_year : float
        Number of days in a year for the instantiated calendar.
    """

    def __init__(self, daycount_function, days_in_year):
        self.daycount_function = daycount_function
        self.days_in_year = days_in_year

    def interval(self, dates, future_dates, sort_values=True):
        """
        Calculates the time interval (in years) between two dates or arrays of dates.

        Parameters
        ----------
        dates : pandas.DatetimeIndex, list of strings, pandas.Timestamp, or string
            The starting date(s) of the interval(s).
            It can be a single date (as a pandas.Timestamp or its string representation) or
            an array-like object (as a pandas.DatetimeIndex or a list of its string representation) containing dates.
            The number of elements in `dates` must equal that in `future_dates`.
        future_dates : pandas.DatetimeIndex, list of strings, pandas.Timestamp, or string
            The ending date(s) of the interval(s).
            It can be a single date (as a pandas.Timestamp or its string representation) or
            an array-like object (as a pandas.DatetimeIndex or a list of its string representation) containing dates.
            The number of elements in `future_dates` must equal that in `dates`.
        Returns
        -------
        tau : numpy.ndarray
            An array containing the time interval(s), in years, between the corresponding dates
            in `dates` and `future_dates`.
        """
        dates, future_dates = afsfun.dates_formatting(
            dates, future_dates, sort_values=sort_values
        )
        tau = self.daycount_function(dates, future_dates) / self.days_in_year
        if isinstance(tau, float):
            return np.array([tau])
        else:
            return np.array(tau)


class MonthYearCalendar:
    """
    Class for instantiating a calendar.

    Parameters
    ----------
    days_in_month : float
        Number of days in a month for the instantiated calendar.
    days_in_year : float
        Number of days in a year for the instantiated calendar.
    """

    def __init__(self, days_in_month, days_in_year):
        self.days_in_month = days_in_month
        self.days_in_year = days_in_year

    def interval(self, dates, future_dates, sort_values=True):
        """
        Calculates the time interval (in years) between two dates or arrays of dates.

        Parameters
        ----------
        dates : pandas.DatetimeIndex, list of strings, pandas.Timestamp, or string
            The starting date(s) of the interval(s).
            It can be a single date (as a pandas.Timestamp or its string representation) or
            an array-like object (as a pandas.DatetimeIndex or a list of its string representation) containing dates.
            The number of elements in `dates` must equal that in `future_dates`.
        future_dates : pandas.DatetimeIndex, list of strings, pandas.Timestamp, or string
            The ending date(s) of the interval(s).
            It can be a single date (as a pandas.Timestamp or its string representation) or
            an array-like object (as a pandas.DatetimeIndex or a list of its string representation) containing dates.
            The number of elements in `future_dates` must equal that in `dates`.
        Returns
        -------
        tau : numpy.ndarray
            An array containing the time interval(s), in years, between the corresponding dates in `dates` and `future_dates`.
        """
        dates, future_dates = afsfun.dates_formatting(
            dates, future_dates, sort_values=sort_values
        )
        years = future_dates.year - dates.year
        months = future_dates.month - dates.month
        days = future_dates.day - dates.day
        tau = years * self.days_in_year + months * self.days_in_month + days
        tau = np.array(tau) / self.days_in_year
        return tau


# counting functions
def seconds(dates, future_dates):
    """
    Counts the number of days (also fractional) between `future_dates` and `dates`
    through the number of seconds between them.

    Parameters
    ----------
    dates : pandas.DatetimeIndex
        The starting date(s) of the interval(s).
    future_dates : pandas.DatetimeIndex
        The ending date(s) of the interval(s).
    Returns
    -------
    days : pandas.Index
        Contains the number of days (also fractional) in the time interval(s).
    """
    if len(future_dates) == 1:
        future_dates = future_dates[0]
        seconds = (future_dates - dates).total_seconds()
    elif len(dates) == 1:
        dates = dates[0]
        seconds = (future_dates - dates).total_seconds()
    else:
        seconds = (future_dates - dates).total_seconds()
    days = seconds / (3600 * 24)
    return days


def actual(dates, future_dates):
    """
    Counts the number of days (can not be fractional) between `future_dates` and `dates`.
    If it is not integer, it is rounded to the lower integer.

    Parameters
    ----------
    dates : pandas.DatetimeIndex
        The starting date(s) of the interval(s).
    future_dates : pandas.DatetimeIndex
        The ending date(s) of the interval(s).
    Returns
    -------
    days : pandas.Index
        Contains the integer number of days in the time interval(s).
    """
    if len(future_dates) == 1:
        future_dates = future_dates[0]
        days = (future_dates - dates).days
    elif len(dates) == 1:
        dates = dates[0]
        days = (future_dates - dates).days
    else:
        days = (future_dates - dates).days
    return days


# --------------------------------------------------------------------------------------
# Business day calendars
# --------------------------------------------------------------------------------------


class BusinessCalendar:
    def __init__(self, business_days, days_in_year):
        self.business_days = pd.to_datetime(business_days)
        self.days_in_year = days_in_year
        # dataframe of day weights
        df = pd.DataFrame(self.business_days[:-1], columns=["Date"])
        df[1] = pd.to_datetime(self.business_days[1:])
        df["Weight"] = (df[1] - df["Date"]).apply(lambda x: x.days)
        self.all_weights = df.set_index("Date")["Weight"]

    def interval(self, dates, future_dates):
        dates = afsfun.dates_formatting(dates)
        future_dates = pd.to_datetime(future_dates)
        tau = np.zeros(dates.size)
        for i in range(dates.size):
            epsilon = (future_dates > dates[i]) - (future_dates < dates[i])
            tau[i] = self.business_days[
                (self.business_days > dates[i]) * (self.business_days <= future_dates)
            ].size
            tau[i] = tau[i] * epsilon / self.days_in_year
        return tau

    def weights(self, dates):
        return self.all_weights.loc[dates]

    def days_in_interval(self, date, interval):
        days = self.business_days
        if interval > 0:
            boolean = days > (date - pd.Timedelta("{} days".format(abs(interval))))
            days = days[(days <= date) * boolean]
        else:
            boolean = days < (date + pd.Timedelta("{} days".format(abs(interval))))
            days = days[(days >= date) * boolean]
        return days
