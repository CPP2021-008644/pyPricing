import sys
import os
import re

pypricing_directory = os.path.expanduser("~/ArfimaTools/pypricing")
sys.path.insert(1, pypricing_directory)
try:
    from .underlyings import *  # (Relative) import needed for the workspace. In this case __package__ is pypricing.data
except (ImportError, ModuleNotFoundError, ValueError):
    from data.underlyings import *  # (Absolute) local import
try:
    from ..pricing.discount_curves import *  # Import needed for the workspace. In this case we need the parent package since discount curves

    # is in a different subpackage (pricing)
except (ImportError, ModuleNotFoundError, ValueError):
    from pricing.discount_curves import *  # (Absolute) local import
try:
    from .calendars import *
except (ImportError, ModuleNotFoundError, ValueError):
    from data.calendars import *
try:
    from . import specs as specs  # (Change name?)
except (ImportError, ModuleNotFoundError, ValueError):
    import data.specs as specs
try:
    from ..pricing import (
        functions as afsfun,
    )  # (Relative) import needed for the workspace. In this case __package__ is 'pypricing.data'
except (ImportError, ModuleNotFoundError, ValueError):
    import pricing.functions as afsfun  # (Absolute) local import. In this case __package__ is 'data'
try:
    from ..pricing.implied_volatility import (
        VolatilitySmile,
        VolatilitySurface,
        VolatilitySurfaceDelta,
    )  # (Relative) import needed for the workspace. In this case __package__ is 'pypricing.data'
except (ImportError, ModuleNotFoundError, ValueError):
    from pricing.implied_volatility import (
        VolatilitySmile,
        VolatilitySurface,
        VolatilitySurfaceDelta,
    )  # (Absolute) local import. In this case __package__ is 'data'

databases_path = os.path.expanduser("~/ArfimaTools/Databases/")


ibor_names = {
    "EUR": "EURIBOR curve",  # https://strata.opengamma.io/indices/#:~:text=calendar%20data%20available.-,Ibor%20Indices%3A,-An%20Ibor%20index
    "USD": "USD LIBOR curve",
    "JPY": "Yen LIBOR curve",
    "GBP": "Sterling LIBOR curve",
    "CNY": "Chinese Yuan LIBOR curve",
    "CHF": "Swiss Franc LIBOR curve",
    "EUR ISDA": "ISDA Euro curve",
    "USD ISDA": "ISDA US Dollar curve",
}

ois_names = {
    "EONIA": "EONIA Overnight Interest rate Swap curve (Euro zone)",
    "SOFR": "SOFR Overnight Interest rate Swap curve (USA)",
    "SONIA": "SONIA Overnight Interest rate Swap curve (UK)",
}

swi_names = {
    "EUSWI": "Euroarea HICP inflation swap curve",
    "USSWI": "US HICP inflation swap curve",
    "BPSWIT": "UK HICP inflation swap curve",
}

equity_names = {
    "RTY": "Russell 2000 Index",
    "SX5E": "Eurostoxx 50 Index",
    "SD3E": "Eurostoxx Select Dividend 30",
    "SPX": "Standard & Poor 500",
    "TPX": "Tokyo Price Index",
    "IBEX": "IBEX 35",
    "MXEU": "Invesco MSCI Europe UCITS ETF",
    "MXASJ": "MSCI Asia Except Japan Index",
    "MT": "ArcelorMittal SA",
    "EURJPY": "Euro to Japanese Yen exchange rate",
    "EURUSD": "EUR-USD X-RATE",
    "EURBRL": "EUR-BRL X-RATE",
    "GOLD": "Gold Spot",
    "PLDM": "PLDMLNPM Index",
    "LBK": "Liberbank SA",
    "MXESSM": "MXESSM Index",
    "MXITSM": "MXITSM Index",
    "PLTM": "GraniteShares Platinum Trust",
    "SBE": "S&P BRIC 40 EURO Index",
    "NKY": "Nikkei 225",
    "SPGCCLP": "SPGCCLP Index",
    "USDIDR": "USD-IDR X-RATE",
    "USDJPY": "USD-JPY X-RATE",
    "SX7E": "iShares EURO STOXX Banks 30-15 UCITS ETF DE",
    "386 HK Equity": "China Petroleum & Chemical Ord Shs A",
}


class DataFactory:
    """
    Legacy class that reads local Excel files instead of accessing the database through ``beautifulData``.
    """

    def __init__(self, db_object):
        self.db_object = db_object

    def import_discount_curves(
        self,
        *tickers,
        start_date,
        end_date,
        interpolate_missing_dates=True,
        method="bond_spline",
    ):
        """
        Instantiate objects of child classes of DiscountCurve using data of the (Dropbox) DB.

        Parameters
        ----------
        *tickers :  str
            Tickers to be imported. Accepts multiple tickers as separate arguments.
            First date.
        end_date : pandas.DatetimeIndex
            Last date.
        interpolate_missing_dates :  bool, optional
            If ``True`` a linear interpolation is used for having values every day of the year. (default is True)
        method : str, optional
            Specify the method used to interpolate the data of the discount curves (default is "bond_spline").

        Returns
        -------
        dict
            Dictionary with the discount curves (objects of child class of DiscountCurve).

        See Also
        --------
        discount_curves.YieldCurve.fit
            For more details on how the the fit methods are performed.

        """
        curves = {}
        db = self.db_object
        data = db.load_curve(start_date=start_date, end_date=end_date, *tickers)
        # The data is imported from files such as irsw-curves.xlsx, where there is only one date per month (usually at the end of the month).
        # In consequence, if start_date is at the beginning of the month we miss all the dates between start_date and the date for that month in the .xlsx.
        for ticker in data.keys():
            temp_data_dic = data[ticker]
            temp_data = temp_data_dic["Data"]
            specs_temp = temp_data_dic[
                "Specs"
            ]  # We assign a tenor for each curve. This is specified in specs-irsw-curves.xlsx.
            temp_data = temp_data.rename(
                columns={
                    column: specs_temp.loc[column, "Tenor"]
                    for column in temp_data.columns
                }
            )  # We replace the names by the tenors (year fraction).
            calendar_str = temp_data_dic["Curve Specs"].loc[
                "Calendar"
            ]  # For each ticker, the calendar is determined in specs-curves.xlsx
            calendar = self.import_calendar(calendar_str)[calendar_str]
            if interpolate_missing_dates and len(temp_data) != 0:
                temp_data = temp_data.reindex(
                    pd.date_range(start=temp_data.index[0], end=temp_data.index[-1])
                )
                temp_data = temp_data.interpolate(
                    method="linear"
                )  # A linear interpolation for having values every date of the year.
            if temp_data_dic["Curve Specs"].loc["Type"] == "irsw":
                curves[ticker] = CubicSplineSwapCurve(
                    calendar=calendar
                )  # Interest rate Swap
            elif temp_data_dic["Curve Specs"].loc["Type"] == "swi":
                curves[ticker] = DepositCurve(calendar=calendar)  # Swap Index
            if len(temp_data_dic["Data"]) != 0:
                curves[ticker].fit(temp_data, method=method)
            setattr(curves[ticker], "ticker", ticker)
            setattr(curves[ticker], "type", temp_data_dic["Curve Specs"].loc["Type"])
        return curves

    def import_underlying(
        self,
        *tickers,
        start_date="19000101",
        end_date="21000101",
        fillna=True,
        asset_kind=LognormalAsset,
    ):
        """

        Parameters
        ----------
        tickers :  list
            Tickers to be imported.
        start_date : pandas.DatetimeIndex
            First date. By default, ``19000101``
        end_date : pandas.DatetimeIndex
            Last date. By default, ``21000101``
        fillna : bool, default = True
            If True, linear interpolation on asset.data DataFrame when certain values are missing (usually "Volatility" and/or "Dividend Rate").
        asset_kind : data.underlyings.Underlying
            Dynamics followed by the underlying. By default, it is assumed Lognormal dynamics (LognormalAsset).

        Returns
        -------
        dict
            Dictionary with the tickers.
        """
        db = self.db_object
        data = db.load_market_instrument(
            *tickers, start_date=start_date, end_date=end_date
        )
        missing_tickers = [ticker for ticker in tickers if ticker not in data.keys()]
        if len(missing_tickers) != 0:
            print("Missing data for", *missing_tickers)
        dic = {}
        for ticker in data.keys():
            if "Volatility" in data[ticker].columns:
                yieldsdiv = "Dividend Rate" in data[ticker].columns
                asset = asset_kind(
                    ticker=ticker, yieldsdiv=yieldsdiv
                )  # If yieldsdiv == True the corresponing changes are set in underlyings.py.
            else:
                asset = Underlying(ticker=ticker)
            asset.set_data(
                data[ticker]
            )  # The data from the DB is assigned to the asset.
            if fillna:
                asset.fillna()  # Linear interpolation when values are missing (usually "Volatility" and/or "Dividend Rate").
            dic[ticker] = asset
        return dic

    def import_calendar(self, *tickers):
        calendars = {}
        if "ActSecs" in tickers:
            calendars["ActSecs"] = DayCountCalendar(seconds, 365)
        if "Act360" in tickers:
            calendars["Act360"] = DayCountCalendar(actual, 360)
        if "Act365" in tickers:
            calendars["Act365"] = DayCountCalendar(actual, 365)
        if "Cal30360" in tickers:
            calendars["Cal30360"] = MonthYearCalendar(30, 360)

        if "Target2" in tickers:
            initial_year = 2000
            number_years = 100
            years = [initial_year + i for i in range(number_years)]
            target_dates = pd.date_range(
                str(initial_year) + "0101",
                str(initial_year + number_years - 1) + "1231",
            )
            # excluding weekends and fixed holidays
            target_dates = target_dates[target_dates.dayofweek < 5]
            target_dates = target_dates[
                (target_dates.day != 1) + (target_dates.month != 1)
            ]
            target_dates = target_dates[
                (target_dates.day != 1) + (target_dates.month != 5)
            ]
            target_dates = target_dates[
                (target_dates.day != 25) + (target_dates.month != 12)
            ]
            target_dates = target_dates[
                (target_dates.day != 26) + (target_dates.month != 12)
            ]

            def calc_easter(year):
                a = year % 19
                b = year % 4
                c = year % 7
                p = year // 100
                q = (13 + 8 * p) // 25
                m = (15 - q + p - p // 4) % 30
                n = (4 + p - p // 4) % 7
                d = (19 * a + m) % 30
                e = (n + 2 * b + 4 * c + 6 * d) % 7
                days = 22 + d + e
                if (d == 29) and (e == 6):
                    return 4, 19
                elif (d == 28) and (e == 6):
                    return 4, 18
                else:
                    if days > 31:
                        return 4, days - 31
                    else:
                        return 3, days

            for year in years:
                emonth, eday = calc_easter(year)
                if eday == 1:
                    fday = 30
                    fmonth = 3
                elif eday == 2:
                    fday = 31
                    fmonth = 3
                else:
                    fday = eday - 2
                    fmonth = emonth
                target_dates = target_dates[
                    (target_dates.day != fday)
                    + (target_dates.month != fmonth)
                    + (target_dates.year != year)
                ]
                if eday == 31:
                    mday = 1
                    mmonth = 4
                else:
                    mday = eday + 1
                    mmonth = emonth
                target_dates = target_dates[
                    (target_dates.day != mday)
                    + (target_dates.month != mmonth)
                    + (target_dates.year != year)
                ]

            calendars["Target2"] = BusinessCalendar(target_dates, 252)
        return calendars

    def list_discount_curves(self, return_result=False, print_result=True):
        db = self.db_object
        curves = db.list_curves(return_result=return_result, print_result=print_result)
        if return_result:
            return curves

    def list_underlying(self, return_result=False, print_result=True):
        db = self.db_object
        underlyiers = db.list_market_instruments(
            return_result=return_result, print_result=print_result
        )
        if return_result:
            return underlyiers

    def list_calendars(self, return_result=False, print_result=True):
        calendars = ["ActSecs", "Act360", "Act365", "Cal30360", "Target2"]
        if print_result:
            print(*calendars)
        if return_result:
            return calendars


class DataFactoryBeautifulData:
    def __init__(self):
        pass

    @staticmethod
    def list_calendars(return_result=False, print_result=True):
        """
        Return or print the calendars that can be (in principle) imported.

        Parameters
        ----------
        return_result : boolean
            If ``True`` the calendars are returned.
        print_result : boolean
            If ``True`` the calendars are printed.

        Returns
        -------
        list

        Notes
        -----
            Same as the old method :py:meth:`DataFactory.list_calendars<data_factory_bd.DataFactory.list_calendars>`, although now static.
        """
        calendars = ["ActSecs", "Act360", "Act365", "Cal30360", "Target2"]
        if print_result:
            print(*calendars)
        if return_result:
            return calendars

    @staticmethod
    def import_calendar(*tickers):
        """
        Import calendars for day counting.

        Parameters
        ----------
        tickers : list
            Calendars.
        Returns
        -------
        data.calendars.DayCountCalendar

        Notes
        -----
            Same as the old method :py:meth:`DataFactory.import_calendar<data_factory_bd.DataFactory.import_calendar>`, although now static.
        """
        calendars = {}
        if "ActSecs" in tickers:
            calendars["ActSecs"] = DayCountCalendar(seconds, 365)
        if "Act360" in tickers:
            calendars["Act360"] = DayCountCalendar(actual, 360)
        if "Act365" in tickers:
            calendars["Act365"] = DayCountCalendar(actual, 365)
        if "Cal30360" in tickers:
            calendars["Cal30360"] = MonthYearCalendar(30, 360)

        if "Target2" in tickers:
            initial_year = 2000
            number_years = 100
            years = [initial_year + i for i in range(number_years)]
            target_dates = pd.date_range(
                str(initial_year) + "0101",
                str(initial_year + number_years - 1) + "1231",
            )
            # excluding weekends and fixed holidays
            target_dates = target_dates[target_dates.dayofweek < 5]
            target_dates = target_dates[
                (target_dates.day != 1) + (target_dates.month != 1)
            ]
            target_dates = target_dates[
                (target_dates.day != 1) + (target_dates.month != 5)
            ]
            target_dates = target_dates[
                (target_dates.day != 25) + (target_dates.month != 12)
            ]
            target_dates = target_dates[
                (target_dates.day != 26) + (target_dates.month != 12)
            ]

            def calc_easter(year):
                a = year % 19
                b = year % 4
                c = year % 7
                p = year // 100
                q = (13 + 8 * p) // 25
                m = (15 - q + p - p // 4) % 30
                n = (4 + p - p // 4) % 7
                d = (19 * a + m) % 30
                e = (n + 2 * b + 4 * c + 6 * d) % 7
                days = 22 + d + e
                if (d == 29) and (e == 6):
                    return 4, 19
                elif (d == 28) and (e == 6):
                    return 4, 18
                else:
                    if days > 31:
                        return 4, days - 31
                    else:
                        return 3, days

            for year in years:
                emonth, eday = calc_easter(year)
                if eday == 1:
                    fday = 30
                    fmonth = 3
                elif eday == 2:
                    fday = 31
                    fmonth = 3
                else:
                    fday = eday - 2
                    fmonth = emonth
                target_dates = target_dates[
                    (target_dates.day != fday)
                    + (target_dates.month != fmonth)
                    + (target_dates.year != year)
                ]
                if eday == 31:
                    mday = 1
                    mmonth = 4
                else:
                    mday = eday + 1
                    mmonth = emonth
                target_dates = target_dates[
                    (target_dates.day != mday)
                    + (target_dates.month != mmonth)
                    + (target_dates.year != year)
                ]

            calendars["Target2"] = BusinessCalendar(target_dates, 252)
        return calendars

    @staticmethod
    def _read_tenor(
        tenor_index,
    ):  # TODO: We should introduce the calendar/day count for computing the tenors as an argument? For days and weeks.
        """
        Returns the tenor in years of a tenor index of the database.

        Parameters
        ----------
        tenor_index : str
            ``beautifulData`` ticker of the tenor index.

        Returns
        -------
        float
            Tenor in years.

        Notes
        ------
            Tenor Indices are products that refer to different tenors of the same curve.
            For instance the ICE Libor with tenor 6 Months: ``LIBORT6MDX``.

            We will store them as ``r'(?P<root>\w)T(?P<tenor>\d+[DWMY])DX'`` (in python re).
            That is to say:

            root + 'T' + tenor + unit + 'DX'

            where `root` is one or more word characters (which include letters, digits, or underscores),
            `tenor` is one or more digits and the `unit` is one of the letters 'D', 'W', 'M', or 'Y' (day, week, month and year).

            More details in `GitLab link <https://git.arfima.com/arfima/arfima/arfimabox/-/issues/1#note_5499>`_.


        Examples
        --------
        >>> DataFactoryBeautifulData._read_tenor('LIBORT6MDX')
        0.5
        >>> DataFactoryBeautifulData._read_tenor('ROOTT22YDX')
        22
        >>> DataFactoryBeautifulData._read_tenor('Word_1_Word2T48MDX')
        4.0
        """
        pattern = r"(?P<root>\w)T(?P<tenor>\d+[DWMY])DX"  # Pattern for tenor indices in the database.
        match = re.search(pattern, tenor_index)

        if match:
            tenor = match.group("tenor")
        else:
            raise NameError(
                f"{tenor_index} is not a tenor index. An example is LIBORT6MDX."
            )

        # Old version not using python re

        # numbers = ['1', '2', '3', '4', '5', '6', '7', '8', '9']
        # units = ['D', 'W', 'M', 'Y']
        # if 'TDX' in tenor_index or not all(char in tenor_index for char in ['T', 'DX'])\
        #         or not any(char in tenor_index for char in units) or not any(char in tenor_index for char in numbers):
        #     raise NameError(f'{tenor_index} is not a tenor index. An example is LIBORT6MDX.')
        #
        # tenor_info = tenor_index
        # while tenor_info[0] not in numbers:  # We need this loop for roots containing the letter 'T'.
        #     tenor_info = tenor_info.split("T", 1)[-1]
        # tenor = tenor_info[:-3]  # We remove the last three characters (unit and 'DX')
        # unit = tenor_info[-3:][0]

        number = int(tenor[:-1])
        unit = tenor[-1]

        if unit == "D":
            return (
                number / 365.0
            )  # days to years   TODO: We should introduce the calendar/day count for computing the tenors here?
        elif unit == "W":
            return (
                (number * 7) / 365.0
            )  # weeks to years   TODO: We should introduce the calendar/day count for computing the tenors here?
        elif unit == "M":
            return number / 12.0  # months to years
        else:  # years
            return number

    @staticmethod
    def list_discount_curves(fields=True):
        """
        Return discount curves that can be imported.

        Parameters
        ----------
        fields : bool, optional
             If ``True`` the details of the curve are also shown. The default is ``True``.

        Notes
        -------
            When fields is ``True``, this method prints a dict of tuples with the following format for each discount curve (key):

            - Discount curve name : {'kind of curve' (str), 'Day Count' (str), 'Tickers' (tuple)}
            - In this case 'Tickers' is a tuple of strings with the tickers from ``beautifulData`` used for constructing the curve.
        """
        if fields:
            print(specs.discount_curves)
        else:
            print(list(specs.discount_curves.keys()))

    def import_underlying(
        self,
        *tickers,
        start_date="19000101",
        end_date="21000101",
        fillna=True,
        asset_kind=LognormalAsset,
        volatility_kind="realized_volatility_30d",
        dividend_kind="forward_dividend_yield",
    ):
        """
        Import the underlyings specified by tickers.

        Parameters
        ----------
        *tickers :  str
            Tickers to be imported. Accepts multiple tickers as separate arguments.
            Use :py:meth:`list_underlyings<data_factory_bd.DataFactoryBeautifulData.list_underlyings>` to see the list of tickers that can be imported.
        start_date : str
            First date.
        end_date : str
            Last date.
        asset_kind : data.underlyings.Underlying
            Dynamics followed by the underlying. By default, it is assumed Lognormal dynamics (``LognormalAsset``).
        volatility_kind : str
            Kind of volatility needed. The ``beautifulData`` name (see Mimir) should be used. This method uses the same volatility for every ticker.
        dividend_kind : str
            Kind of dividend needed. The ``beautifulData`` name (see Mimir) should be used. This method uses the same dividend for every ticker.

        Returns
        -------
        dict
            Dictionary with the tickers.

        Examples
        -------
        >>> factory_bd = DataFactoryBeautifulData()
        >>> dict_und = factory_bd.import_underlying("SPXIDX", "SX5EIDX", start_date="20220101", end_date="20220301")
        """
        if not (asset_kind in tuple(specs.underlying_dynamics_classes)):
            raise ValueError(
                f"Underlying dynamics not supported. Try with {specs.underlying_dynamics}"
            )
        else:
            data = {
                ticker: self._load_underlying_bd(
                    ticker, start_date, end_date, volatility_kind, dividend_kind
                )
                for ticker in tickers
            }
            missing_tickers = [
                ticker for ticker in tickers if ticker not in data.keys()
            ]
            if len(missing_tickers) != 0:
                print("Missing data for", *missing_tickers)
            dic = {}
            for ticker in data.keys():
                if "Volatility" in data[ticker].columns:
                    yieldsdiv = "Dividend Rate" in data[ticker].columns
                    asset = asset_kind(
                        ticker=ticker, yieldsdiv=yieldsdiv
                    )  # If yieldsdiv == True the corresponing changes are set in underlyings.py.
                else:
                    asset = Underlying(ticker=ticker)
                asset.set_data(
                    data[ticker]
                )  # The data from the DB is assigned to the asset.
                if fillna:
                    asset.fillna()  # Linear interpolation when values are missing (usually "Volatility" and/or "Dividend Rate")
                dic[ticker] = asset
            return dic
