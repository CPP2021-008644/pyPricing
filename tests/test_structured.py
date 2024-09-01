import unittest
import sys
import os
import pandas as pd
import numpy as np
import pyfeng as pf
from datetime import datetime

pypricing_directory = os.path.expanduser("~/ArfimaTools/pypricing")
sys.path.insert(1, pypricing_directory)

import afslibrary as afs  # Note that data_factory_bd is not imported (otherwise the Job will fail since beautifulData can not be imported.

dbtools_directory = os.path.expanduser("~/ArfimaTools/afsdb")
sys.path.insert(1, dbtools_directory)
import db_tools

db = db_tools.BeautifulDataAFSStyleXL()
factory = afs.DataFactory(db)
assets = factory.import_underlying("SX5E", "RTY")
data = np.array(
    [
        (0.1259434783, 0, 3745.15, "2020-01-01", "00:00:00"),
        (0.2433090909, 0, 3337.77, "2020-09-02", "00:00:00"),
        (0.2431136364, 0, 3304.22, "2020-09-03", "00:00:00"),
        (0.2429181818, 0, 3260.59, "2020-09-04", "00:00:00"),
        (0.2427227273, 0, 3304.22, "2020-09-07", "00:00:00"),
        (0.2425272727, 0, 3267.37, "2020-09-08", "00:00:00"),
        (0.2423318182, 0, 3324.83, "2020-09-09", "00:00:00"),
        (0.1577714286, 0, 4223.97, "2021-10-26", "00:00:00"),
        (0.156247619, 0, 4220.88, "2021-10-27", "00:00:00"),
        (0.1547238095, 0, 4233.87, "2021-10-28", "00:00:00"),
        (0.2147681818, 0, 4109.51, "2021-11-29", "00:00:00"),
    ]
)
df = pd.DataFrame(
    data, columns=["Volatility", "Dividend Rate", "Price", "Date", "Time"]
)
format = "%Y-%m-%d %H:%M:%S"
df["Datetime"] = pd.to_datetime(
    df["Date"] + " " + df["Time"].astype("string"), format=format
)
df["Volatility"] = df["Volatility"].astype(float)
df["Price"] = df["Price"].astype(float)
df["Dividend Rate"] = df["Dividend Rate"].astype(float)
df.set_index("Datetime", inplace=True)
df = df.drop(["Date", "Time"], axis=1)
fake_normal_asset = afs.NormalAsset("fake_SX5E")
fake_normal_asset.set_data(data=df)
fake_normal_assets = {"fake_SX5E": fake_normal_asset}
normal_assets = factory.import_underlying("SX5E", asset_kind=afs.NormalAsset)
calendars = factory.import_calendar("Act360", "Act365", "ActSecs")
discount_curves = factory.import_discount_curves(
    "EURIBOR", "USD LIBOR", start_date="20200131", end_date="20220930"
)


data_div = np.array(
    [
        (0.1259434783, 0.034995, 3745.15, "2020-01-01", "00:00:00"),
        (0.2433090909, 0.034304, 3337.77, "2020-09-02", "00:00:00"),
        (0.2431136364, 0.0341, 3304.22, "2020-09-03", "00:00:00"),
        (0.2429181818, 0.03406, 3260.59, "2020-09-04", "00:00:00"),
        (0.2427227273, 0.03401, 3304.22, "2020-09-07", "00:00:00"),
        (0.2425272727, 0.03396, 3267.37, "2020-09-08", "00:00:00"),
        (0.1577714286, 0.029334, 4223.97, "2021-10-26", "00:00:00"),
        (0.156247619, 0.029304, 4220.88, "2021-10-27", "00:00:00"),
        (0.1547238095, 0.029264, 4233.87, "2021-10-28", "00:00:00"),
        (0.2147681818, 0.029234, 4109.51, "2021-11-29", "00:00:00"),
    ]
)
df_div = pd.DataFrame(
    data_div, columns=["Volatility", "Dividend Rate", "Price", "Date", "Time"]
)
format = "%Y-%m-%d %H:%M:%S"
df_div["Datetime"] = pd.to_datetime(
    df_div["Date"] + " " + df_div["Time"].astype("string"), format=format
)
df_div["Volatility"] = df_div["Volatility"].astype(float)
df_div["Price"] = df_div["Price"].astype(float)
df_div["Dividend Rate"] = df_div["Dividend Rate"].astype(float)
df_div.set_index("Datetime", inplace=True)
df_div = df_div.drop(["Date", "Time"], axis=1)
fake_lognormal_asset = afs.LognormalAsset("fake_SX5E_div")
fake_lognormal_asset.set_data(data=df_div)
fake_lognormal_assets = {"fake_SX5E_div": fake_lognormal_asset}
fake_lognormal_asset.yieldsdiv = True


class TestPrices(unittest.TestCase):
    # Class attributes common to the different tests
    K = 3505.22
    K_put = 3304
    K_call = 3280
    maturity = maturity1 = "20211027"
    maturity2 = "20220223"
    fake_ticker = "fake_SX5E"
    ticker = "SX5E"
    discount_curve = "EURIBOR"
    dates = ["20200903", "20200907"]

    def test_price_bachelier_without_dividends(self):
        """
        Test the price for vanilla Bachelier options without dividends using MC and the analytical formula.
        """
        c = afs.Call(
            underlying=fake_normal_assets[TestPrices.fake_ticker],
            strike=TestPrices.K_call,
            maturity=TestPrices.maturity,
            calendar=calendars["Act360"],
        )
        p = afs.Put(
            underlying=fake_normal_assets[TestPrices.fake_ticker],
            strike=TestPrices.K_put,
            maturity=TestPrices.maturity,
            calendar=calendars["Act360"],
        )
        tau = calendars["Act360"].interval(TestPrices.dates, TestPrices.maturity)
        tau = pd.Series(tau, index=pd.DatetimeIndex(TestPrices.dates))
        # To call the price from MCProduct
        price_mc_p = afs.MCProduct.get_px(
            p,
            TestPrices.dates,
            discount_curves[TestPrices.discount_curve],
            no_calcs=1,
        )
        price_mc_c = afs.MCProduct.get_px(
            c,
            TestPrices.dates,
            discount_curves[TestPrices.discount_curve],
            no_calcs=1,
        )
        # Analytical derivative price
        price_p = p.get_px(TestPrices.dates, discount_curves[TestPrices.discount_curve])
        price_c = c.get_px(TestPrices.dates, discount_curves[TestPrices.discount_curve])
        rel_error_p = (price_mc_p - price_p) / price_p
        rel_error_c = (price_mc_c - price_c) / price_c
        # Error analytical-vs-MC
        self.assertAlmostEqual(np.max(np.abs(rel_error_p.values)), 0, delta=0.01)
        self.assertAlmostEqual(np.max(np.abs(rel_error_c.values)), 0, delta=0.01)

    def test2_price_bachelier_without_dividends(self):
        """
        Test the price of the sum of two vanilla Bachelier calls without dividends using
        MC and the analytical formula. It checks that the discounts are being calculated
        properly for a product with multiple payments.
        """
        c1 = afs.Call(
            underlying=fake_normal_assets[TestPrices.fake_ticker],
            strike=TestPrices.K_call,
            maturity=TestPrices.maturity1,
            calendar=calendars["Act360"],
        )
        c2 = afs.Call(
            underlying=fake_normal_assets[TestPrices.fake_ticker],
            strike=TestPrices.K_call,
            maturity=TestPrices.maturity2,
            calendar=calendars["Act360"],
        )
        c = c1 + c2

        # To call the price from MCProduct
        price_mc_c = afs.MCProduct.get_px(
            c,
            TestPrices.dates,
            discount_curves[TestPrices.discount_curve],
            no_calcs=1,
        )

        # Analytical derivative price
        price_c1 = c1.get_px(
            TestPrices.dates, discount_curves[TestPrices.discount_curve]
        )
        price_c2 = c2.get_px(
            TestPrices.dates, discount_curves[TestPrices.discount_curve]
        )
        price_c = price_c1 + price_c2
        rel_error_c = (price_mc_c - price_c) / price_c
        # Error analytical-vs-MC
        self.assertAlmostEqual(np.max(np.abs(rel_error_c.values)), 0, delta=0.01)

    def test_price_blackscholes(self):
        """
        Test the price for vanilla Black-Scholes options using MC and the analytical formula.
        """
        c = afs.Call(
            assets[TestPrices.ticker],
            TestPrices.K,
            TestPrices.maturity,
            calendars["Act360"],
        )
        p = afs.Put(
            assets[TestPrices.ticker],
            TestPrices.K,
            TestPrices.maturity,
            calendars["Act360"],
        )
        tau = calendars["Act360"].interval(TestPrices.dates, TestPrices.maturity)
        tau = pd.Series(tau, index=pd.DatetimeIndex(TestPrices.dates))
        # To call the price from MCProduct
        price_mc_p = afs.MCProduct.get_px(
            p, TestPrices.dates, discount_curves[TestPrices.discount_curve]
        )
        price_mc_c = afs.MCProduct.get_px(
            c, TestPrices.dates, discount_curves[TestPrices.discount_curve]
        )
        # Analytical derivative price
        price_p = p.get_px(TestPrices.dates, discount_curves[TestPrices.discount_curve])
        price_c = c.get_px(TestPrices.dates, discount_curves[TestPrices.discount_curve])
        rel_error_p = (price_mc_p - price_p) / price_p
        rel_error_c = (price_mc_c - price_c) / price_c
        # Error analytical-vs-MC
        self.assertAlmostEqual(np.max(np.abs(rel_error_p.values)), 0, delta=0.01)
        self.assertAlmostEqual(np.max(np.abs(rel_error_c.values)), 0, delta=0.01)

    def test2_price_blackscholes(self):
        """
        Test the price of the sum of two vanilla Black-Scholes calls using MC and the
        analytical formula. It checks that the discounts are being calculated
        properly for a product with multiple payments.
        """
        c1 = afs.Call(
            assets[TestPrices.ticker],
            TestPrices.K,
            TestPrices.maturity1,
            calendars["Act360"],
        )
        c2 = afs.Put(
            assets[TestPrices.ticker],
            TestPrices.K,
            TestPrices.maturity2,
            calendars["Act360"],
        )
        c = c1 + c2

        # To call the price from MCProduct
        price_mc_c = afs.MCProduct.get_px(
            c, TestPrices.dates, discount_curves[TestPrices.discount_curve]
        )
        # Analytical derivative price
        price_c1 = c1.get_px(
            TestPrices.dates, discount_curves[TestPrices.discount_curve]
        )
        price_c2 = c2.get_px(
            TestPrices.dates, discount_curves[TestPrices.discount_curve]
        )
        price_c = price_c1 + price_c2

        # Error analytical-vs-MC
        rel_error_c = (price_mc_c - price_c) / price_c

        self.assertAlmostEqual(np.max(np.abs(rel_error_c.values)), 0, delta=0.01)


class TestGreeks(unittest.TestCase):
    # Class attributes common to the different tests
    K = 1590.482
    K_gamma = 3304
    K_put = 4502.45
    K_call = 2507.11
    K_gamma_bach = K_rho_bach = 3303.968
    K_vega_bach = 3272.99139
    maturity = "20211027"
    maturity_gamma = "20221027"
    maturity_gamma_bach = "20200908"
    fake_ticker_div = "fake_SX5E_div"
    fake_ticker = "fake_SX5E"
    ticker = "SX5E"
    discount_curve = "EURIBOR"
    dates = ["20200903", "20200908"]
    dates_bach = ["20200903", "20200907"]

    def test_delta_blackscholes(self):
        """
        Test the delta for vanilla Black-Scholes options using MC and the analytical formula.
        """

        p = afs.Put(
            assets[TestGreeks.ticker],
            TestGreeks.K,
            TestGreeks.maturity,
            calendars["Act360"],
            nominal=1,
        )
        c = afs.Call(
            assets[TestGreeks.ticker],
            TestGreeks.K,
            TestGreeks.maturity,
            calendars["Act360"],
            nominal=1,
        )
        q = assets[TestGreeks.ticker].get_divrate(TestGreeks.dates)
        tau = calendars["Act360"].interval(TestGreeks.dates, TestGreeks.maturity)
        tau = pd.Series(tau, index=pd.DatetimeIndex(TestGreeks.dates))
        # To call the delta from MCProduct
        delta_mc_p = afs.MCProduct.get_delta(
            p,
            TestGreeks.dates,
            discount_curves[TestGreeks.discount_curve],
            monte_carlo=True,
        )
        delta_mc_c = afs.MCProduct.get_delta(
            c,
            TestGreeks.dates,
            discount_curves[TestGreeks.discount_curve],
            monte_carlo=True,
        )
        # Analytical difference
        delta_diff_p = afs.MCProduct.get_delta(
            p, TestGreeks.dates, discount_curves[TestGreeks.discount_curve]
        )
        delta_diff_c = afs.MCProduct.get_delta(
            c, TestGreeks.dates, discount_curves[TestGreeks.discount_curve]
        )
        # Analytical derivative
        delta_p = p.get_delta(
            TestGreeks.dates, discount_curves[TestGreeks.discount_curve]
        )
        delta_c = c.get_delta(
            TestGreeks.dates, discount_curves[TestGreeks.discount_curve]
        )
        rel_error_p = (delta_mc_p - delta_p) / delta_p
        rel_error_diff_p = (delta_diff_p - delta_p) / delta_p
        rel_error_c = (delta_mc_c - delta_c) / delta_c
        rel_error_diff_c = (delta_diff_c - delta_c) / delta_c
        # By definition of the payoff and nominal set to 1
        cp_par_an = delta_c - delta_p - np.exp(-q * tau)
        cp_par_mc = delta_mc_c - delta_mc_p - np.exp(-q * tau)
        # Error analytical-vs-MC
        self.assertAlmostEqual(np.max(np.abs(rel_error_p.values)), 0, delta=0.01)
        self.assertAlmostEqual(np.max(np.abs(rel_error_c.values)), 0, delta=0.01)
        # Error analytical-vs-analytical diff
        self.assertAlmostEqual(np.max(np.abs(rel_error_diff_p.values)), 0, delta=0.001)
        self.assertAlmostEqual(np.max(np.abs(rel_error_diff_c.values)), 0, delta=0.001)
        # Call-Put parity, this should be zero with no error
        self.assertAlmostEqual(np.max(np.abs(cp_par_an.values)), 0, delta=0.001)
        self.assertAlmostEqual(np.max(np.abs(cp_par_mc.values)), 0, delta=0.001)

    def test_delta_bachelier_without_dividends(self):
        """
        Test the delta for vanilla Bachelier options without dividends using MC and the analytical formula.
        """
        import pdb

        p = afs.Put(
            fake_normal_assets[TestGreeks.fake_ticker],
            3303.968 * 0.995,
            TestGreeks.maturity,
            calendars["Act360"],
            nominal=1,
        )
        c = afs.Call(
            fake_normal_assets[TestGreeks.fake_ticker],
            3303.968 * 0.995,
            TestGreeks.maturity,
            calendars["Act360"],
            nominal=1,
        )
        q = fake_normal_assets[TestGreeks.fake_ticker].get_divrate(
            TestGreeks.dates_bach
        )
        tau = calendars["Act360"].interval(TestGreeks.dates_bach, TestGreeks.maturity)
        tau = pd.Series(tau, index=pd.DatetimeIndex(TestGreeks.dates_bach))

        # To call the delta from MCProduct
        delta_mc_p = afs.MCProduct.get_delta(
            p,
            TestGreeks.dates_bach,
            discount_curves[TestGreeks.discount_curve],
            monte_carlo=True,
        )
        delta_mc_c = afs.MCProduct.get_delta(
            c,
            TestGreeks.dates_bach,
            discount_curves[TestGreeks.discount_curve],
            monte_carlo=True,
        )
        # Analytical difference
        delta_diff_p = afs.MCProduct.get_delta(
            p, TestGreeks.dates_bach, discount_curves[TestGreeks.discount_curve]
        )
        delta_diff_c = afs.MCProduct.get_delta(
            c, TestGreeks.dates_bach, discount_curves[TestGreeks.discount_curve]
        )
        # Analytical derivative
        delta_p = p.get_delta(
            TestGreeks.dates_bach, discount_curves[TestGreeks.discount_curve]
        )
        delta_c = c.get_delta(
            TestGreeks.dates_bach, discount_curves[TestGreeks.discount_curve]
        )

        rel_error_diff_mc_p = (delta_diff_p - delta_mc_p) / delta_diff_p
        rel_error_diff_mc_c = (delta_diff_c - delta_mc_c) / delta_diff_c

        # By definition of the payoff and default nominal being 100
        cp_par_an = delta_c - delta_p - np.exp(-q * tau)
        cp_par_mc = delta_mc_c - delta_mc_p - np.exp(-q * tau)

        # Error MonteCarlo-vs-analytical diff
        self.assertAlmostEqual(
            np.max(np.abs(rel_error_diff_mc_p.values)), 0, delta=0.001
        )
        self.assertAlmostEqual(
            np.max(np.abs(rel_error_diff_mc_c.values)), 0, delta=0.001
        )

        # Call-Put parity, this should be zero with no error
        self.assertAlmostEqual(np.max(np.abs(cp_par_an.values)), 0, delta=0.001)
        self.assertAlmostEqual(np.max(np.abs(cp_par_mc.values)), 0, delta=0.001)

    def test_gamma_blackscholes(self):
        """
        Test the gamma for vanilla Black-Scholes options using MC and the analytical formula.
        """

        p = afs.Put(
            assets[TestGreeks.ticker],
            TestGreeks.K_gamma,
            TestGreeks.maturity_gamma,
            calendars["Act360"],
        )
        c = afs.Call(
            assets[TestGreeks.ticker],
            TestGreeks.K_gamma,
            TestGreeks.maturity_gamma,
            calendars["Act360"],
        )

        # To call the gamma from MCProduct
        gamma_mc_p = afs.MCProduct.get_gamma(
            p,
            TestGreeks.dates,
            discount_curves[TestGreeks.discount_curve],
            monte_carlo=True,
        )
        gamma_mc_c = afs.MCProduct.get_gamma(
            c,
            TestGreeks.dates,
            discount_curves[TestGreeks.discount_curve],
            monte_carlo=True,
        )

        # Analytical difference
        gamma_diff_p = afs.MCProduct.get_gamma(
            p, TestGreeks.dates, discount_curves[TestGreeks.discount_curve]
        )
        gamma_diff_c = afs.MCProduct.get_gamma(
            c, TestGreeks.dates, discount_curves[TestGreeks.discount_curve]
        )

        # Analytical derivative
        gamma_p = p.get_gamma(
            TestGreeks.dates, discount_curves[TestGreeks.discount_curve]
        )
        gamma_c = c.get_gamma(
            TestGreeks.dates, discount_curves[TestGreeks.discount_curve]
        )

        rel_error_p = (gamma_mc_p - gamma_p) / gamma_p
        rel_error_diff_p = (gamma_diff_p - gamma_p) / gamma_p
        rel_error_c = (gamma_mc_c - gamma_c) / gamma_c
        rel_error_diff_c = (gamma_diff_c - gamma_c) / gamma_c

        # Error analytical-vs-MC
        self.assertAlmostEqual(np.max(np.abs(rel_error_p.values)), 0, delta=0.01)
        self.assertAlmostEqual(np.max(np.abs(rel_error_c.values)), 0, delta=0.01)

        # Error analytical-vs-analytical diff
        self.assertAlmostEqual(np.max(np.abs(rel_error_diff_p.values)), 0, delta=0.01)
        self.assertAlmostEqual(np.max(np.abs(rel_error_diff_c.values)), 0, delta=0.01)

    def test_gamma_bachelier_without_dividends(self):
        """
        Test the gamma for vanilla Bachelier options without dividends using MC and the analytical formula.
        """

        p = afs.Put(
            fake_normal_assets[TestGreeks.fake_ticker],
            TestGreeks.K_gamma_bach,
            TestGreeks.maturity_gamma_bach,
            calendars["Act360"],
        )
        c = afs.Call(
            fake_normal_assets[TestGreeks.fake_ticker],
            TestGreeks.K_gamma_bach,
            TestGreeks.maturity_gamma_bach,
            calendars["Act360"],
        )

        # To call the gamma from MCProduct
        gamma_mc_p = afs.MCProduct.get_gamma(
            p,
            TestGreeks.dates_bach,
            discount_curves[TestGreeks.discount_curve],
            monte_carlo=True,
            no_calcs=1,
        )
        gamma_mc_c = afs.MCProduct.get_gamma(
            c,
            TestGreeks.dates_bach,
            discount_curves[TestGreeks.discount_curve],
            monte_carlo=True,
            no_calcs=1,
        )

        # Analytical difference
        gamma_diff_p = afs.MCProduct.get_gamma(
            p, TestGreeks.dates_bach, discount_curves[TestGreeks.discount_curve]
        )
        gamma_diff_c = afs.MCProduct.get_gamma(
            c, TestGreeks.dates_bach, discount_curves[TestGreeks.discount_curve]
        )

        rel_error_mc_diff_p = (gamma_mc_p - gamma_diff_p) / gamma_diff_p
        rel_error_mc_diff_c = (gamma_mc_c - gamma_diff_c) / gamma_diff_c

        # Error analytical diff-vs-MC
        self.assertAlmostEqual(
            np.max(np.abs(rel_error_mc_diff_p.values)), 0, delta=0.01
        )
        self.assertAlmostEqual(
            np.max(np.abs(rel_error_mc_diff_c.values)), 0, delta=0.01
        )

    def test_vega_blackscholes(self):
        """
        Test the vega for vanilla Black-Scholes options using MC and the analytical formula.
        """

        p = afs.Put(
            assets[TestGreeks.ticker],
            TestGreeks.K_put,
            TestGreeks.maturity,
            calendars["Act360"],
        )
        c = afs.Call(
            assets[TestGreeks.ticker],
            TestGreeks.K_call,
            TestGreeks.maturity,
            calendars["Act360"],
        )

        # To call the vega from MCProduct
        vega_mc_p = afs.MCProduct.get_vega(
            p,
            TestGreeks.dates,
            discount_curves[TestGreeks.discount_curve],
            monte_carlo=True,
            no_calcs=1,
        )
        vega_mc_c = afs.MCProduct.get_vega(
            c,
            TestGreeks.dates,
            discount_curves[TestGreeks.discount_curve],
            monte_carlo=True,
            no_calcs=1,
        )

        # Analytical difference
        vega_diff_p = afs.MCProduct.get_vega(
            p, TestGreeks.dates, discount_curves[TestGreeks.discount_curve]
        )
        vega_diff_c = afs.MCProduct.get_vega(
            c, TestGreeks.dates, discount_curves[TestGreeks.discount_curve]
        )

        # Analytical derivative
        vega_p = p.get_vega(
            TestGreeks.dates, discount_curves[TestGreeks.discount_curve]
        )
        vega_c = c.get_vega(
            TestGreeks.dates, discount_curves[TestGreeks.discount_curve]
        )

        rel_error_p = (vega_mc_p - vega_p) / vega_p
        rel_error_diff_p = (vega_diff_p - vega_p) / vega_p
        rel_error_c = (vega_mc_c - vega_c) / vega_c
        rel_error_diff_c = (vega_diff_c - vega_c) / vega_c

        # Error analytical-vs-MC
        self.assertAlmostEqual(np.max(np.abs(rel_error_p.values)), 0, delta=0.01)
        self.assertAlmostEqual(np.max(np.abs(rel_error_c.values)), 0, delta=0.01)

        # Error analytical-vs-analytical diff
        self.assertAlmostEqual(np.max(np.abs(rel_error_diff_p.values)), 0, delta=0.01)
        self.assertAlmostEqual(np.max(np.abs(rel_error_diff_c.values)), 0, delta=0.01)

    def test_vega_bachelier_without_dividends(self):
        """
        Test the vega for vanilla Bachelier options without dividends using MC and the analytical formula.
        """

        p = afs.Put(
            fake_normal_assets[TestGreeks.fake_ticker],
            TestGreeks.K_vega_bach,
            TestGreeks.maturity_gamma,
            calendars["Act360"],
        )
        c = afs.Call(
            fake_normal_assets[TestGreeks.fake_ticker],
            TestGreeks.K_vega_bach,
            TestGreeks.maturity_gamma,
            calendars["Act360"],
        )

        # To call the vega from MCProduct
        vega_mc_p = afs.MCProduct.get_vega(
            p,
            TestGreeks.dates_bach,
            discount_curves[TestGreeks.discount_curve],
            monte_carlo=True,
            no_calcs=1,
        )
        vega_mc_c = afs.MCProduct.get_vega(
            c,
            TestGreeks.dates_bach,
            discount_curves[TestGreeks.discount_curve],
            monte_carlo=True,
            no_calcs=1,
        )
        # Analytical difference
        vega_diff_p = afs.MCProduct.get_vega(
            p, TestGreeks.dates_bach, discount_curves[TestGreeks.discount_curve]
        )
        vega_diff_c = afs.MCProduct.get_vega(
            c, TestGreeks.dates_bach, discount_curves[TestGreeks.discount_curve]
        )
        # Analytical derivative
        vega_p = p.get_vega(
            TestGreeks.dates_bach, discount_curves[TestGreeks.discount_curve]
        )
        vega_c = c.get_vega(
            TestGreeks.dates_bach, discount_curves[TestGreeks.discount_curve]
        )
        rel_error_p = (vega_mc_p - vega_p) / vega_p
        rel_error_diff_p = (vega_diff_p - vega_p) / vega_p
        rel_error_c = (vega_mc_c - vega_c) / vega_c
        rel_error_diff_c = (vega_diff_c - vega_c) / vega_c

        # Error analytical-vs-MC
        self.assertAlmostEqual(np.max(np.abs(rel_error_p.values)), 0, delta=0.01)
        self.assertAlmostEqual(np.max(np.abs(rel_error_c.values)), 0, delta=0.01)

        # Error analytical-vs-analytical diff
        self.assertAlmostEqual(np.max(np.abs(rel_error_diff_p.values)), 0, delta=0.01)
        self.assertAlmostEqual(np.max(np.abs(rel_error_diff_c.values)), 0, delta=0.01)

    def test_rho_blackscholes(self):
        """
        Test the rho for vanilla Black-Scholes options using MC and the analytical formula.
        """

        p = afs.Put(
            assets[TestGreeks.ticker],
            TestGreeks.K_put,
            TestGreeks.maturity,
            calendars["Act360"],
        )
        c = afs.Call(
            assets[TestGreeks.ticker],
            TestGreeks.K_call,
            TestGreeks.maturity,
            calendars["Act360"],
        )

        # To call the rho from MCProduct
        rho_mc_p = afs.MCProduct.get_rho(
            p,
            TestGreeks.dates,
            discount_curves[TestGreeks.discount_curve],
            monte_carlo=True,
            no_calcs=1,
        )
        rho_mc_c = afs.MCProduct.get_rho(
            c,
            TestGreeks.dates,
            discount_curves[TestGreeks.discount_curve],
            monte_carlo=True,
            no_calcs=1,
        )

        # Analytical difference
        rho_diff_p = afs.MCProduct.get_rho(
            p, TestGreeks.dates, discount_curves[TestGreeks.discount_curve]
        )
        rho_diff_c = afs.MCProduct.get_rho(
            c, TestGreeks.dates, discount_curves[TestGreeks.discount_curve]
        )

        # Analytical derivative
        rho_p = p.get_rho(TestGreeks.dates, discount_curves[TestGreeks.discount_curve])
        rho_c = c.get_rho(TestGreeks.dates, discount_curves[TestGreeks.discount_curve])

        rel_error_p = (rho_mc_p - rho_p) / rho_p
        rel_error_diff_p = (rho_diff_p - rho_p) / rho_p
        rel_error_c = (rho_mc_c - rho_c) / rho_c
        rel_error_diff_c = (rho_diff_c - rho_c) / rho_c

        # Error analytical-vs-MC
        self.assertAlmostEqual(np.max(np.abs(rel_error_p.values)), 0, delta=0.01)
        self.assertAlmostEqual(np.max(np.abs(rel_error_c.values)), 0, delta=0.01)

        # Error analytical-vs-analytical diff
        self.assertAlmostEqual(np.max(np.abs(rel_error_diff_p.values)), 0, delta=0.01)
        self.assertAlmostEqual(np.max(np.abs(rel_error_diff_c.values)), 0, delta=0.01)

    def test_rho_bachelier_without_dividends(self):
        """
        Test the rho for vanilla Bachelier options without dividends using MC and the analytical formula.
        """

        p = afs.Put(
            fake_normal_assets[TestGreeks.fake_ticker],
            TestGreeks.K_put,
            TestGreeks.maturity_gamma,
            calendars["Act360"],
        )
        c = afs.Call(
            fake_normal_assets[TestGreeks.fake_ticker],
            TestGreeks.K_call,
            TestGreeks.maturity_gamma,
            calendars["Act360"],
        )

        # To call the rho from MCProduct
        rho_mc_p = afs.MCProduct.get_rho(
            p,
            TestGreeks.dates_bach,
            discount_curves[TestGreeks.discount_curve],
            monte_carlo=True,
            no_calcs=1,
        )
        rho_mc_c = afs.MCProduct.get_rho(
            c,
            TestGreeks.dates_bach,
            discount_curves[TestGreeks.discount_curve],
            monte_carlo=True,
            no_calcs=1,
        )

        # Analytical difference
        rho_diff_p = afs.MCProduct.get_rho(
            p, TestGreeks.dates_bach, discount_curves[TestGreeks.discount_curve]
        )
        rho_diff_c = afs.MCProduct.get_rho(
            c, TestGreeks.dates_bach, discount_curves[TestGreeks.discount_curve]
        )

        # Analytical derivative
        rho_p = p.get_rho(
            TestGreeks.dates_bach, discount_curves[TestGreeks.discount_curve]
        )
        rho_c = c.get_rho(
            TestGreeks.dates_bach, discount_curves[TestGreeks.discount_curve]
        )

        rel_error_p = (rho_mc_p - rho_p) / rho_p
        rel_error_diff_p = (rho_diff_p - rho_p) / rho_p
        rel_error_c = (rho_mc_c - rho_c) / rho_c
        rel_error_diff_c = (rho_diff_c - rho_c) / rho_c

        # Error analytical-vs-MC
        self.assertAlmostEqual(np.max(np.abs(rel_error_p.values)), 0, delta=0.01)
        self.assertAlmostEqual(np.max(np.abs(rel_error_c.values)), 0, delta=0.01)

        # Error analytical-vs-analytical diff
        self.assertAlmostEqual(np.max(np.abs(rel_error_diff_p.values)), 0, delta=0.01)
        self.assertAlmostEqual(np.max(np.abs(rel_error_diff_c.values)), 0, delta=0.01)

    def test1_theta_blackscholes(self):
        """
        Test#1: test the theta for vanilla Black-Scholes options using multiple MC methods
        and the analytical formula.
        In this test, a discount curve with constant interest rate (CRDC) is used.
        Therefore, all methods give approximately the same result for theta's value.
        N.B.: We test the constant_curve=False Monte Carlo method here too.
        """
        K_put = 3540.61
        K_call = 3088.61
        discount_curve = afs.CRDC(r=0.002, calendar=calendars["Act360"])

        p = afs.Put(
            assets[TestGreeks.ticker],
            K_put,
            TestGreeks.maturity,
            calendars["Act360"],
        )
        c = afs.Call(
            assets[TestGreeks.ticker], K_call, TestGreeks.maturity, calendars["Act360"]
        )

        # Calls the theta from MCProduct, but uses analytical finite differences
        theta_mc_put = afs.MCProduct.get_theta(
            p,
            TestGreeks.dates,
            discount_curve,
            constant_curve=True,
            no_calcs=2,
        )

        # Calls the theta from MCProduct, maturity method
        theta_mc_put_mat = afs.MCProduct.get_theta(
            p,
            TestGreeks.dates,
            discount_curve,
            maturity_method=True,
            constant_curve=True,
            no_calcs=2,
        )

        # Calls the theta from MCProduct, Monte Carlo method
        theta_mc_put_iry = afs.MCProduct.get_theta(
            p,
            TestGreeks.dates,
            discount_curve,
            monte_carlo=True,
            constant_curve=True,
            no_sims=10**8,
            no_calcs=2,
        )

        theta_mc_put_irn = afs.MCProduct.get_theta(
            p,
            TestGreeks.dates,
            discount_curve,
            constant_curve=False,
            monte_carlo=True,
            no_sims=10**8,
            no_calcs=2,
        )

        # Calls the theta from Put, uses analytical formula
        theta_put = p.get_theta(
            TestGreeks.dates, discount_curve, is_interest_rate_constant=True
        )

        rel_error_put = (theta_mc_put - theta_put) / theta_put
        rel_error_put_mat = (theta_mc_put_mat - theta_put) / theta_put
        rel_error_put_iry = (theta_mc_put_iry - theta_put) / theta_put
        rel_error_put_irn = (theta_mc_put_irn - theta_put) / theta_put
        self.assertAlmostEqual(np.max(np.abs(rel_error_put.values)), 0, delta=0.01)
        self.assertAlmostEqual(np.max(np.abs(rel_error_put_mat.values)), 0, delta=0.01)
        self.assertAlmostEqual(np.max(np.abs(rel_error_put_iry.values)), 0, delta=0.01)
        self.assertAlmostEqual(np.max(np.abs(rel_error_put_irn.values)), 0, delta=0.01)

        # Calls the theta from MCProduct, but uses analytical finite differences
        theta_mc_call = afs.MCProduct.get_theta(
            c,
            TestGreeks.dates,
            discount_curve,
            constant_curve=True,
            no_calcs=2,
        )
        # Calls the theta from MCProduct, maturity method
        theta_mc_call_mat = afs.MCProduct.get_theta(
            c,
            TestGreeks.dates,
            discount_curve,
            maturity_method=True,
            constant_curve=True,
            no_calcs=2,
        )

        # Calls the theta from MCProduct, constant_curve=True and Monte Carlo method
        theta_mc_call_iry = afs.MCProduct.get_theta(
            c,
            TestGreeks.dates,
            discount_curve,
            constant_curve=True,
            monte_carlo=True,
            no_sims=10**8,
            no_calcs=2,
        )

        theta_mc_call_irn = afs.MCProduct.get_theta(
            c,
            TestGreeks.dates,
            discount_curve,
            constant_curve=False,
            monte_carlo=True,
            no_sims=10**8,
            no_calcs=2,
        )

        # Calls the theta from Call, uses analytical formula
        theta_call = c.get_theta(
            TestGreeks.dates, discount_curve, is_interest_rate_constant=True
        )

        rel_error_call = (theta_mc_call - theta_call) / theta_call
        rel_error_call_mat = (theta_mc_call_mat - theta_call) / theta_call
        rel_error_call_iry = (theta_mc_call_iry - theta_call) / theta_call
        rel_error_call_irn = (theta_mc_call_irn - theta_call) / theta_call
        self.assertAlmostEqual(np.max(np.abs(rel_error_call.values)), 0, delta=0.01)
        self.assertAlmostEqual(np.max(np.abs(rel_error_call_mat.values)), 0, delta=0.01)
        self.assertAlmostEqual(np.max(np.abs(rel_error_call_iry.values)), 0, delta=0.01)
        self.assertAlmostEqual(np.max(np.abs(rel_error_call_irn.values)), 0, delta=0.01)

    def test2_theta_blackscholes(self):
        """
        Test#2: test the theta for vanilla Black-Scholes options using multiple MC methods
        and the analytical formula (through the finite difference approximation).
        In this test, a discount curve with non-constant interest rate is used.
        Therefore, the maturity method and the constant_curve=True MC method
        cannot be exploited here.
        Furthermore, we do not compare the MC prices with the analytical one because we cannot take
        a time interval small enough (the minimum is one day) to properly approximate the limit
        in the definition of the derivative.
        """
        K_put = 3540.61
        K_call = 3088.61

        p = afs.Put(
            assets[TestGreeks.ticker],
            K_put,
            TestGreeks.maturity,
            calendars["Act360"],
        )
        c = afs.Call(
            assets[TestGreeks.ticker], K_call, TestGreeks.maturity, calendars["Act360"]
        )

        # Calls the theta from MCProduct, but uses analytical finite differences
        theta_mc_put = afs.MCProduct.get_theta(
            p,
            TestGreeks.dates,
            discount_curves[TestGreeks.discount_curve],
            no_calcs=2,
        )

        # Calls the theta from MCProduct, Monte Carlo method
        theta_mc_put_mc = afs.MCProduct.get_theta(
            p,
            TestGreeks.dates,
            discount_curves[TestGreeks.discount_curve],
            monte_carlo=True,
            no_calcs=2,
        )

        # Calls the theta from MCProduct, constant_curve=False and Monte Carlo method
        theta_mc_put_irn = afs.MCProduct.get_theta(
            p,
            TestGreeks.dates,
            discount_curves[TestGreeks.discount_curve],
            constant_curve=False,
            monte_carlo=True,
            no_sims=10**8,
            no_calcs=2,
        )

        rel_error_put_mc = (theta_mc_put_mc - theta_mc_put) / theta_mc_put
        rel_error_put_irn = (theta_mc_put_irn - theta_mc_put) / theta_mc_put
        self.assertAlmostEqual(np.max(np.abs(rel_error_put_mc.values)), 0, delta=0.01)
        self.assertAlmostEqual(np.max(np.abs(rel_error_put_irn.values)), 0, delta=0.01)

        # Calls the theta from MCProduct, but uses analytical finite differences
        theta_mc_call = afs.MCProduct.get_theta(
            c,
            TestGreeks.dates,
            discount_curves[TestGreeks.discount_curve],
            no_calcs=2,
        )

        # Calls the theta from MCProduct, Monte Carlo method
        theta_mc_call_mc = afs.MCProduct.get_theta(
            c,
            TestGreeks.dates,
            discount_curves[TestGreeks.discount_curve],
            monte_carlo=True,
            no_calcs=2,
        )

        # Calls the theta from MCProduct, constant_curve=False and Monte Carlo method
        theta_mc_call_irn = afs.MCProduct.get_theta(
            c,
            TestGreeks.dates,
            discount_curves[TestGreeks.discount_curve],
            constant_curve=False,
            monte_carlo=True,
            no_calcs=2,
        )

        rel_error_call_mc = (theta_mc_call_mc - theta_mc_call) / theta_mc_call
        rel_error_call_irn = (theta_mc_call_irn - theta_mc_call) / theta_mc_call
        self.assertAlmostEqual(np.max(np.abs(rel_error_call_mc.values)), 0, delta=0.01)
        self.assertAlmostEqual(np.max(np.abs(rel_error_call_irn.values)), 0, delta=0.01)

    def test1_theta_bachelier_without_dividends(self):
        """
        Test#1: test the theta for vanilla Bachelier options without dividends using multiple MC methods
        and the analytical formula.
        In this test, a discount curve with constant interest rate (CRDC) is used.
        Therefore, all methods give approximately the same result for theta's value.
        N.B.: We test the constant_curve=False Monte Carlo method here too.
        """

        K_bachelier_call = 3088.61
        K_bachelier_put = 3540.61
        discount_curve = afs.CRDC(r=0.002, calendar=calendars["Act360"])

        c = afs.Call(
            fake_normal_assets[TestGreeks.fake_ticker],
            K_bachelier_call,
            TestGreeks.maturity,
            calendars["Act360"],
        )
        p = afs.Put(
            fake_normal_assets[TestGreeks.fake_ticker],
            K_bachelier_put,
            TestGreeks.maturity,
            calendars["Act360"],
        )

        # Calls the theta from MCProduct, but uses analytical finite differences
        theta_mc_put = afs.MCProduct.get_theta(
            p,
            TestGreeks.dates,
            discount_curve,
            constant_curve=True,
            no_calcs=2,
        )

        # Calls the theta from MCProduct, maturity method
        theta_mc_put_mat = afs.MCProduct.get_theta(
            p,
            TestGreeks.dates,
            discount_curve,
            maturity_method=True,
            constant_curve=True,
            no_calcs=2,
        )

        # Calls the theta from MCProduct, Monte Carlo method
        theta_mc_put_iry = afs.MCProduct.get_theta(
            p,
            TestGreeks.dates,
            discount_curve,
            monte_carlo=True,
            constant_curve=True,
            no_calcs=2,
        )

        # Calls the theta from MCProduct, constant_curve=False and Monte Carlo method
        theta_mc_put_irn = afs.MCProduct.get_theta(
            p,
            TestGreeks.dates,
            discount_curve,
            constant_curve=False,
            monte_carlo=True,
            no_calcs=2,
        )

        # Calls the theta from Put, uses analytical formula
        theta_put = p.get_theta(
            TestGreeks.dates, discount_curve, is_interest_rate_constant=True
        )

        rel_error_put = (theta_mc_put - theta_put) / theta_put
        rel_error_put_mat = (theta_mc_put_mat - theta_put) / theta_put
        rel_error_put_iry = (theta_mc_put_iry - theta_put) / theta_put
        rel_error_put_irn = (theta_mc_put_irn - theta_put) / theta_put
        self.assertAlmostEqual(np.max(np.abs(rel_error_put.values)), 0, delta=0.01)
        self.assertAlmostEqual(np.max(np.abs(rel_error_put_mat.values)), 0, delta=0.01)
        self.assertAlmostEqual(np.max(np.abs(rel_error_put_iry.values)), 0, delta=0.01)
        self.assertAlmostEqual(np.max(np.abs(rel_error_put_irn.values)), 0, delta=0.01)

        # Calls the theta from MCProduct, but uses analytical finite differences
        theta_mc_call = afs.MCProduct.get_theta(
            c, TestGreeks.dates, discount_curve, constant_curve=True, no_calcs=2
        )
        # Calls the theta from MCProduct, maturity method
        theta_mc_call_mat = afs.MCProduct.get_theta(
            c,
            TestGreeks.dates,
            discount_curve,
            maturity_method=True,
            constant_curve=True,
            no_calcs=2,
        )

        # Calls the theta from MCProduct, constant_curve=True and Monte Carlo method
        theta_mc_call_iry = afs.MCProduct.get_theta(
            c,
            TestGreeks.dates,
            discount_curve,
            constant_curve=True,
            monte_carlo=True,
            no_calcs=2,
        )

        # Calls the theta from MCProduct, constant_curve=False and Monte Carlo method
        theta_mc_call_irn = afs.MCProduct.get_theta(
            c,
            TestGreeks.dates,
            discount_curve,
            constant_curve=False,
            monte_carlo=True,
            no_calcs=2,
        )

        # Calls the theta from Call, uses analytical formula
        theta_call = c.get_theta(
            TestGreeks.dates, discount_curve, is_interest_rate_constant=True
        )

        rel_error_call = (theta_mc_call - theta_call) / theta_call
        rel_error_call_mat = (theta_mc_call_mat - theta_call) / theta_call
        rel_error_call_iry = (theta_mc_call_iry - theta_call) / theta_call
        rel_error_call_irn = (theta_mc_call_irn - theta_call) / theta_call
        self.assertAlmostEqual(np.max(np.abs(rel_error_call.values)), 0, delta=0.01)
        self.assertAlmostEqual(np.max(np.abs(rel_error_call_mat.values)), 0, delta=0.01)
        self.assertAlmostEqual(np.max(np.abs(rel_error_call_iry.values)), 0, delta=0.01)
        self.assertAlmostEqual(np.max(np.abs(rel_error_call_irn.values)), 0, delta=0.01)

    def test2_theta_bachelier_without_dividends(self):
        """
        Test#2: test the theta for vanilla Bachelier options without dividends using multiple MC methods
        and the analytical formula (through the finite difference approximation).
        In this test, a discount curve with non-constant interest rate is used.
        Therefore, the maturity method and the constant_curve=True MC method
        cannot be exploited here.
        Furthermore, we do not compare the MC prices with the analytical one because we cannot take
        a time interval small enough (the minimum is one day) to properly approximate the limit
        in the definition of the derivative.
        """
        K_bachelier_call = 3088.61
        K_bachelier_put = 3540.61

        c = afs.Call(
            fake_normal_assets[TestGreeks.fake_ticker],
            K_bachelier_call,
            TestGreeks.maturity,
            calendars["Act360"],
        )
        p = afs.Put(
            fake_normal_assets[TestGreeks.fake_ticker],
            K_bachelier_put,
            TestGreeks.maturity,
            calendars["Act360"],
        )

        # Calls the theta from MCProduct, but uses analytical finite differences
        theta_mc_put = afs.MCProduct.get_theta(
            p, TestGreeks.dates, discount_curves[TestPrices.discount_curve], no_calcs=2
        )

        # Calls the theta from MCProduct, Monte Carlo method
        theta_mc_put_mc = afs.MCProduct.get_theta(
            p,
            TestGreeks.dates,
            discount_curves[TestPrices.discount_curve],
            monte_carlo=True,
            no_calcs=2,
        )

        # Calls the theta from MCProduct, constant_curve=False and Monte Carlo method
        theta_mc_put_irn = afs.MCProduct.get_theta(
            p,
            TestGreeks.dates,
            discount_curves[TestPrices.discount_curve],
            constant_curve=False,
            monte_carlo=True,
            no_calcs=2,
        )

        rel_error_put_mc = (theta_mc_put_mc - theta_mc_put) / theta_mc_put
        rel_error_put_irn = (theta_mc_put_irn - theta_mc_put) / theta_mc_put
        self.assertAlmostEqual(np.max(np.abs(rel_error_put_mc.values)), 0, delta=0.01)
        self.assertAlmostEqual(np.max(np.abs(rel_error_put_irn.values)), 0, delta=0.01)

        # Calls the theta from MCProduct, but uses analytical finite differences
        theta_mc_call = afs.MCProduct.get_theta(
            c, TestGreeks.dates, discount_curves[TestPrices.discount_curve], no_calcs=2
        )

        # Calls the theta from MCProduct, Monte Carlo method
        theta_mc_call_mc = afs.MCProduct.get_theta(
            c,
            TestGreeks.dates,
            discount_curves[TestPrices.discount_curve],
            monte_carlo=True,
            no_calcs=2,
        )

        # Calls the theta from MCProduct, constant_curve=False and Monte Carlo method
        theta_mc_call_irn = afs.MCProduct.get_theta(
            c,
            TestGreeks.dates,
            discount_curves[TestPrices.discount_curve],
            constant_curve=False,
            monte_carlo=True,
            no_calcs=2,
        )

        rel_error_call_mc = (theta_mc_call_mc - theta_mc_call) / theta_mc_call
        rel_error_call_irn = (theta_mc_call_irn - theta_mc_call) / theta_mc_call
        self.assertAlmostEqual(np.max(np.abs(rel_error_call_mc.values)), 0, delta=0.01)
        self.assertAlmostEqual(np.max(np.abs(rel_error_call_irn.values)), 0, delta=0.01)


class TestLookback(unittest.TestCase):
    # Class attributes common to the different tests
    ticker = "SX5E"
    discount_curve = "EURIBOR"

    def test1_lookback_option(self):
        """
        Test lookback call options using MC and the analytical formula.
        """
        dates = ["20200831"]
        obsdates = pd.date_range(
            start=pd.to_datetime("20200831"),
            end=pd.to_datetime("20200831 6:00:00"),
            periods=20000,
        )
        calendar = calendars["ActSecs"]
        assets[TestLookback.ticker].yieldsdiv = False
        discount_curve = afs.CRDC(1, calendar=calendar)

        lc = afs.Lookback(
            underlying=assets[TestLookback.ticker],
            obsdates=obsdates,
            strike=None,
            kind="call",
            calendar=calendar,
        )

        price_mc_lc = afs.MCProduct.get_px(
            lc,
            dates=dates,
            discount_curve=discount_curve,
            no_sims=2 * 10**4,
            no_calcs=1,
        )

        price_lc = lc.get_px(
            dates=dates,
            discount_curve=discount_curve,
        )

        rel_error_lc = (price_mc_lc - price_lc) / price_lc
        self.assertAlmostEqual(np.max(np.abs(rel_error_lc.values)), 0, delta=0.015)

    def test2_lookback_option(self):
        """
        Test broadcasting with more than one valuation date in lookback call options using MC and the analytical formula.
        """
        dates = ["20200831", "20200901"]
        obsdates = pd.date_range(start="20200831", end="20210901")
        calendar = calendars["Act365"]
        n_sims = 10**5
        r = 1
        assets[TestLookback.ticker].yieldsdiv = False
        discount_curve = afs.CRDC(r, calendar=calendar)

        lc = afs.Lookback(
            underlying=assets[TestLookback.ticker],
            obsdates=obsdates,
            strike=None,
            kind="call",
            calendar=calendar,
        )

        price_mc_lc = afs.MCProduct.get_px(
            lc,
            dates=dates,
            discount_curve=discount_curve,
            no_sims=n_sims,
            no_calcs=1,
        )

        price_lc = lc.get_px(
            dates=dates,
            discount_curve=discount_curve,
        )

        rel_error_lc = (price_mc_lc - price_lc) / price_lc
        self.assertAlmostEqual(np.max(np.abs(rel_error_lc.values)), 0, delta=0.03)


class TestScenarios(unittest.TestCase):
    def test_get_risk_matrix(self):
        K = 1590.482
        maturity = "20211027"
        ticker = "RTY"
        discount_curve = "USD LIBOR"
        p = afs.Put(
            assets[ticker],
            strike=0.9 * K,
            maturity=maturity,
            calendar=calendars["Act365"],
            nominal=1,
            implied_volatility=0.2,
        )
        scenarios = {
            "spot": np.array([1600, 1800]),
            "future_date": ["20200910", "20200910"],
            "implied_vol": np.array([0.21, 0.25]),
        }
        df_risk_pricing, dict_greeks_scenarios = p.get_risk_matrix(
            ["20200903"], discount_curves[discount_curve], scenarios
        )
        assert np.all(
            df_risk_pricing["Discrete PnL"].values
            == [-13.248958852116315, -41.433257327274134]
        )

    def test_greeks_fictitious_option_call(self):
        """
        Test higher order greeks of Black-Scholes call options.
        """
        dc = afs.CRDC(0.05)
        date = "20200903"
        c = afs.Call(
            fake_lognormal_assets[TestGreeks.fake_ticker_div],
            3304,
            "20211027",
            calendars["Act365"],
            nominal=1,
        )
        price_c = c.get_px(date, dc)
        delta_c = c.get_delta(date, dc)
        theta_c = c.get_theta(date, dc) / 365
        gamma_c = c.get_gamma(date, dc)
        rho_c = c.get_rho(date, dc) / 100
        vega_c = c.get_vega(date, dc) / 100
        vomma_c = c.get_vomma(date, dc) / 10000
        vanna_c = c.get_vanna(date, dc) / 100
        charm_c = c.get_charm(date, dc) / 365
        color_c = c.get_color(date, dc) / 365
        speed_c = c.get_speed(date, dc)

        rel_error_price_c = (price_c.values - 355.898975) / 355.898975
        rel_error_delta_c = (delta_c.values - 0.557236) / 0.557236
        rel_error_gamma_c = (gamma_c.values - 0.000436853) / 0.000436853
        rel_error_theta_c = (theta_c.values - (-0.417615)) / (-0.417615)
        rel_error_vega_c = (vega_c.values - 13.310794) / 13.310794
        rel_error_rho_c = (rho_c.values - 17.050785) / 17.050785
        rel_error_vomma_c = (vomma_c.values - (-0.006579)) / (-0.006579)
        rel_error_vanna_c = (vanna_c.values - 0.000926547) / 0.000926547
        rel_error_charm_c = (charm_c.values - (-0.0000377)) / (-0.0000377)
        rel_error_color_c = (color_c.values - 0.000000583) / 0.000000583
        rel_error_speed_c = (speed_c.values - (-0.0000002340126)) / (-0.0000002340126)

        self.assertAlmostEqual(np.abs(rel_error_price_c), 0, delta=0.001)
        self.assertAlmostEqual(np.abs(rel_error_delta_c), 0, delta=0.001)
        self.assertAlmostEqual(np.abs(rel_error_gamma_c), 0, delta=0.001)
        self.assertAlmostEqual(np.abs(rel_error_theta_c), 0, delta=0.001)
        self.assertAlmostEqual(np.abs(rel_error_vega_c), 0, delta=0.001)
        self.assertAlmostEqual(np.abs(rel_error_rho_c), 0, delta=0.001)
        self.assertAlmostEqual(np.abs(rel_error_vomma_c), 0, delta=0.001)
        self.assertAlmostEqual(np.abs(rel_error_vanna_c), 0, delta=0.001)
        self.assertAlmostEqual(np.abs(rel_error_charm_c), 0, delta=0.001)
        self.assertAlmostEqual(np.abs(rel_error_color_c), 0, delta=0.001)
        self.assertAlmostEqual(np.abs(rel_error_speed_c), 0, delta=0.001)

    def test_greeks_fictitious_option_put(self):
        """
        Test higher order greeks of Black-Scholes put options.
        """
        dc = afs.CRDC(0.05)
        date = "20200903"
        p = afs.Put(
            fake_lognormal_assets[TestGreeks.fake_ticker_div],
            3304,
            "20211027",
            calendars["Act365"],
            nominal=1,
        )
        price_p = p.get_px(date, dc)
        delta_p = p.get_delta(date, dc)
        theta_p = p.get_theta(date, dc) / 365
        gamma_p = p.get_gamma(date, dc)
        rho_p = p.get_rho(date, dc) / 100
        vega_p = p.get_vega(date, dc) / 100
        vomma_p = p.get_vomma(date, dc) / 10000
        vanna_p = p.get_vanna(date, dc) / 100
        charm_p = p.get_charm(date, dc) / 365
        color_p = p.get_color(date, dc) / 365
        speed_p = p.get_speed(date, dc)

        rel_error_price_p = (price_p.values - 298.222808) / 298.222808
        rel_error_delta_p = (delta_p.values - (-0.404375)) / (-0.404375)
        rel_error_gamma_p = (gamma_p.values - 0.000436853) / 0.000436853
        rel_error_theta_p = (theta_p.values - (-0.287105)) / (-0.287105)
        rel_error_vega_p = (vega_p.values - 13.310794) / 13.310794
        rel_error_rho_p = (rho_p.values - (-18.761653)) / (-18.761653)
        rel_error_vomma_p = (vomma_p.values - (-0.006579)) / (-0.006579)
        rel_error_vanna_p = (vanna_p.values - 0.000926547) / 0.000926547
        rel_error_charm_p = (charm_p.values - (-0.0001275383117)) / (-0.0001275383117)
        rel_error_color_p = (color_p.values - 0.00000058303458) / 0.00000058303458
        rel_error_speed_p = (speed_p.values - (-0.0000002340126)) / (-0.0000002340126)

        self.assertAlmostEqual(np.abs(rel_error_price_p), 0, delta=0.001)
        self.assertAlmostEqual(np.abs(rel_error_delta_p), 0, delta=0.001)
        self.assertAlmostEqual(np.abs(rel_error_gamma_p), 0, delta=0.001)
        self.assertAlmostEqual(np.abs(rel_error_theta_p), 0, delta=0.001)
        self.assertAlmostEqual(np.abs(rel_error_vega_p), 0, delta=0.001)
        self.assertAlmostEqual(np.abs(rel_error_rho_p), 0, delta=0.001)
        self.assertAlmostEqual(np.abs(rel_error_vomma_p), 0, delta=0.001)
        self.assertAlmostEqual(np.abs(rel_error_vanna_p), 0, delta=0.001)
        self.assertAlmostEqual(np.abs(rel_error_charm_p), 0, delta=0.001)
        self.assertAlmostEqual(np.abs(rel_error_color_p), 0, delta=0.001)
        self.assertAlmostEqual(np.abs(rel_error_speed_p), 0, delta=0.001)


class TestImpliedVol(unittest.TestCase):
    K = 3287.44816
    maturity = "20211027"
    ticker = "SX5E"
    discount_curve = "EURIBOR"
    dates = ["20200903", "20200907"]

    def test_implied_volatility(self):
        """
        Test the proper functioning of attribute implied_volatility.
        """
        c = afs.Call(
            assets[TestImpliedVol.ticker],
            TestImpliedVol.K,
            TestImpliedVol.maturity,
            calendars["Act360"],
            implied_volatility=None,
        )
        price_c = c.get_px(
            TestImpliedVol.dates, discount_curves[TestImpliedVol.discount_curve]
        )
        delta_c = c.get_delta(
            TestImpliedVol.dates, discount_curves[TestImpliedVol.discount_curve]
        )
        gamma_c = c.get_gamma(
            TestImpliedVol.dates, discount_curves[TestImpliedVol.discount_curve]
        )
        vega_c = c.get_vega(
            TestImpliedVol.dates, discount_curves[TestImpliedVol.discount_curve]
        )
        theta_c = c.get_theta(
            TestImpliedVol.dates, discount_curves[TestImpliedVol.discount_curve]
        )
        rho_c = c.get_rho(
            TestImpliedVol.dates, discount_curves[TestImpliedVol.discount_curve]
        )
        vol = assets[TestImpliedVol.ticker].get_vol(dates=TestImpliedVol.dates)
        c_vol = afs.Call(
            assets[TestImpliedVol.ticker],
            TestImpliedVol.K,
            TestImpliedVol.maturity,
            calendars["Act360"],
            implied_volatility=vol,
        )
        price_c_vol = c_vol.get_px(
            TestImpliedVol.dates, discount_curves[TestImpliedVol.discount_curve]
        )
        delta_c_vol = c_vol.get_delta(
            TestImpliedVol.dates, discount_curves[TestImpliedVol.discount_curve]
        )
        gamma_c_vol = c_vol.get_gamma(
            TestImpliedVol.dates, discount_curves[TestImpliedVol.discount_curve]
        )
        vega_c_vol = c_vol.get_vega(
            TestImpliedVol.dates, discount_curves[TestImpliedVol.discount_curve]
        )
        theta_c_vol = c_vol.get_theta(
            TestImpliedVol.dates, discount_curves[TestImpliedVol.discount_curve]
        )
        rho_c_vol = c_vol.get_rho(
            TestImpliedVol.dates, discount_curves[TestImpliedVol.discount_curve]
        )

        diff_price = price_c - price_c_vol
        diff_delta = delta_c - delta_c_vol
        diff_gamma = gamma_c - gamma_c_vol
        diff_vega = vega_c - vega_c_vol
        diff_theta = theta_c - theta_c_vol
        diff_rho = rho_c - rho_c_vol

        self.assertAlmostEqual(np.max(np.abs(diff_price.values)), 0, delta=0.000001)
        self.assertAlmostEqual(np.max(np.abs(diff_delta.values)), 0, delta=0.000001)
        self.assertAlmostEqual(np.max(np.abs(diff_gamma.values)), 0, delta=0.000001)
        self.assertAlmostEqual(np.max(np.abs(diff_vega.values)), 0, delta=0.000001)
        self.assertAlmostEqual(np.max(np.abs(diff_theta.values)), 0, delta=0.000001)
        self.assertAlmostEqual(np.max(np.abs(diff_rho.values)), 0, delta=0.000001)


class TestVaR(unittest.TestCase):
    K = 3524.22
    maturity = "20211027"
    ticker = "SX5E"
    discount_curve = "EURIBOR"
    dates = ["20200903", "20200907"]

    def test_var(self):
        """
        Test the Value at Risk (VaR) computation using MC method and the analytical formula.
        """
        c = afs.Call(
            assets[TestVaR.ticker],
            TestVaR.K,
            TestVaR.maturity,
            calendars["Act360"],
            implied_volatility=None,
        )

        an_var_c = c.get_var(
            dates=TestVaR.dates,
            discount_curve=discount_curves[TestVaR.discount_curve],
            time_horizon=1,
            conf_level=0.05,
        )

        c.monotonicity_price_function = None
        mc_var_c = c.get_var(
            dates=TestVaR.dates,
            discount_curve=discount_curves[TestVaR.discount_curve],
            time_horizon=1,
            conf_level=0.05,
            no_sims=10**4,
            no_sims_price=10**6,
            no_calcs_price=1,
        )

        rel_error_c = (mc_var_c - an_var_c) / an_var_c

        self.assertAlmostEqual(np.max(np.abs(rel_error_c.values)), 0, delta=0.05)
        # We set such a high value of delta=0.05 for the relative error to avoid
        # increasing the number of simulations, no_sims, too much and
        # burdening the test in terms of required execution time.
        # We conducted a test by increasing the number of simulations, no_sims,
        # to 5*10**5. In this case, the relative error is found to be 0.00608053


class TestZCBond(unittest.TestCase):
    dates = ["20200505", "20200512"]
    calendar = calendars["Act360"]
    maturity = "20210505"

    def test_zcbond_theta(self):
        "Test get_theta method for ZCBond."
        zcb = afs.ZCBond(maturity=TestZCBond.maturity, calendar=TestZCBond.calendar)
        discount_curve = discount_curves["USD LIBOR"]
        theta_an = zcb.get_theta(TestZCBond.dates, discount_curve, constant_curve=False)
        dates_p = pd.to_datetime(TestZCBond.dates) + pd.tseries.offsets.BDay()
        dates_m = pd.to_datetime(TestZCBond.dates) - pd.tseries.offsets.BDay()
        theta_an_diff = (
            zcb.nominal
            * (
                discount_curve.get_value(dates_p, zcb.maturity)
                - discount_curve.get_value(dates_m, zcb.maturity)
            )
            / (zcb.calendar.interval(dates_m, dates_p))
        )

        rel_error = (theta_an_diff - theta_an) / theta_an
        self.assertAlmostEqual(np.max(np.abs(rel_error.values)), 0, delta=0.001)

        discount_curve_crdc = afs.CRDC(r=0.1, calendar=calendars["Act360"])
        theta_an_crdc = zcb.get_theta(
            TestZCBond.dates, discount_curve_crdc, constant_curve=False
        )
        theta_an_cc_crdc = zcb.get_theta(
            TestZCBond.dates, discount_curve_crdc, constant_curve=True
        )
        theta_an_diff_crdc = (
            zcb.nominal
            * (
                discount_curve_crdc.get_value(dates_p, zcb.maturity)
                - discount_curve_crdc.get_value(dates_m, zcb.maturity)
            )
            / (zcb.calendar.interval(dates_m, dates_p))
        )

        rel_error_crdc = (theta_an_diff_crdc - theta_an_crdc) / theta_an_crdc
        rel_error_cc_crdc = (theta_an_cc_crdc - theta_an_crdc) / theta_an_crdc
        self.assertAlmostEqual(np.max(np.abs(rel_error_crdc.values)), 0, delta=0.001)
        self.assertAlmostEqual(np.max(np.abs(rel_error_cc_crdc.values)), 0, delta=0.001)


if __name__ == "__main__":
    unittest.main()
