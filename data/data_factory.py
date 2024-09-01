import shelve
import sys
import os

# import beautifulData as bd  We can not include this here, otherwise the unit test failed since beautifulData can not be imported.
import pandas as pd

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

databases_path = os.path.expanduser("~/ArfimaTools/Databases/")


# def readDropboxPath():
#     dropbox_json_path = os.path.join(os.getenv("LOCALAPPDATA"), "Dropbox", "info.json")
#     dropbox_json_info = json.load(open(dropbox_json_path))
#     try:
#         dropbox_path = dropbox_json_info["business"]["path"]
#     except:
#         try:
#             dropbox_path = dropbox_json_info["personal"]["path"]
#         except:
#             raise ValueError("Dropbox json info not found")
#
#     return dropbox_path
#
#
# db_tools_path = readDropboxPath() + "/Inputs/"
# sys.path.insert(1, db_tools_path)
#
# from db_tools import BeautifulDataAFSStyle

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
    def __init__(self, db_object):
        self.db_object = db_object

    def import_discount_curves(
        self, *tickers, start_date, end_date, interpolate_missing_dates=True
    ):
        """
        Instantiate objects of child classes of DiscountCurve using data of the (Dropbox) DB.

        Parameters
        ----------
        tickers :  list
            Tickers to be imported.
        start_date : pandas.DatetimeIndex
            First date.
        end_date : pandas.DatetimeIndex
            Last date.
        interpolate_missing_dates :  bool
            If True a linear interpolation is used for having values every day of the year.

        Returns
        -------
        dict
            Dictionary with the discount curves (objects of child class of DiscountCurve).

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
                curves[ticker].fit(temp_data)
            setattr(curves[ticker], "ticker", ticker)
            setattr(curves[ticker], "type", temp_data_dic["Curve Specs"].loc["Type"])
        return curves

    def import_underlying(
        self,
        *tickers,
        start_date="19000101",
        end_date="21000101",
        fillna=True,
        asset_kind=LognormalAsset
    ):
        """

        Parameters
        ----------
        tickers
        start_date
        end_date
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
                asset.fillna()  # Linear interpolation when values are missing (usually "Volatility" and/or "Dividend Rate")
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
