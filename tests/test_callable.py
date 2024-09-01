import unittest
import sys
import os
import pandas as pd
import numpy as np
import random


pypricing_directory = os.path.expanduser("~/ArfimaTools/pypricing")
sys.path.insert(1, pypricing_directory)

import afslibrary as afs  # Note that data_factory_bd is not imported (otherwise the Job will fail since beautifulData can not be imported.

dbtools_directory = os.path.expanduser("~/ArfimaTools/afsdb")
sys.path.insert(1, dbtools_directory)
import db_tools

db = db_tools.BeautifulDataAFSStyleXL()
factory = afs.DataFactory(db)
underlyings = factory.import_underlying("SX5E", "RTY")
calendars = factory.import_calendar("Act360", "Act365")
# discount_curves = factory.import_discount_curves("EURIBOR", "USD LIBOR", start_date="20200831", end_date="20200930")


def select_one_test(test_class):
    """Return a suite containing only a randomly chosen test method."""
    test_names = unittest.TestLoader().getTestCaseNames(test_class)
    selected_test = random.choice(test_names)
    while selected_test == "test_DLIB":
        selected_test = random.choice(test_names)
    return unittest.defaultTestLoader.loadTestsFromName(selected_test, test_class)


class TestAmericanOptions(unittest.TestCase):
    def test_DLIB(self):
        """Example p.9 of American Monte Carlo DLIB."""
        calendar = "Act360"
        paths = np.array(
            [
                [8.0, 11.0, 15.0, 18.0],
                [8.0, 10.0, 11.0, 6.0],
                [8.0, 9.0, 6.0, 6.0],
                [8.0, 6.0, 5.0, 8.0],
                [8.0, 4.0, 3.0, 2.0],
            ]
        )
        paths = paths[:, :, np.newaxis, np.newaxis]
        curve = afs.CRDC(0.0, calendars[calendar])
        strike = 9
        pay_dates = afs.dates_formatting(
            ["20200202", "20240201", "20250201", "20260201"]
        )
        dates_dic = {}
        for i in range(pay_dates.size):
            dates_dic[pay_dates[i]] = i
        simulation_data = [[afs.dates_formatting(pay_dates[0]), paths, 0, dates_dic]]
        american_put = afs.AmericanVanillaOption(
            underlying=underlyings["SX5E"],
            strike=strike,
            kind="put",
            calendar=calendars[calendar],
            pay_dates=pay_dates,
            nominal=1,
        )
        american_call = afs.AmericanVanillaOption(
            underlying=underlyings["SX5E"],
            strike=strike,
            kind="call",
            calendar=calendars[calendar],
            pay_dates=pay_dates,
            nominal=1,
        )
        european_call = afs.Call(
            underlying=underlyings["SX5E"],
            strike=strike,
            calendar=calendars[calendar],
            maturity=pay_dates[-1],
            nominal=1,
        )
        european_put = afs.Put(
            underlying=underlyings["SX5E"],
            strike=strike,
            calendar=calendars[calendar],
            maturity=pay_dates[-1],
            nominal=1,
        )
        engine = afs.RegressionMC()
        # Get the actual values from the functions
        american_put_LS = engine.compute_price_from_simulations(
            american_put, curve, simulation_data, method=("LS", 1)
        )
        american_put_TvR = engine.compute_price_from_simulations(
            american_put, curve, simulation_data, method=("TvR", 3)
        )
        european_put_MC = (
            afs.DeterministicVolDiffusionMC().compute_price_from_simulations(
                european_put, curve, simulation_data
            )
        )
        american_call_LS = engine.compute_price_from_simulations(
            american_call, curve, simulation_data, method=("LS", 3)
        )
        american_call_TvR = engine.compute_price_from_simulations(
            american_call, curve, simulation_data, method=("TvR", 3)
        )
        european_call_MC = (
            afs.DeterministicVolDiffusionMC().compute_price_from_simulations(
                european_call, curve, simulation_data
            )
        )
        # Expected values
        expected_values = [3.2, 3.4, 2.8, 2.2, 2.2, 1.8]
        # Actual values
        actual_values = [
            american_put_LS,
            american_put_TvR,
            european_put_MC,
            american_call_LS,
            american_call_TvR,
            european_call_MC,
        ]
        # Compare actual values to expected values
        for expected, actual in zip(expected_values, actual_values):
            self.assertAlmostEqual(expected, actual.values[0])

    def test_Glasserman(self):
        """Example of Glasserman, Chapter 8, p.463"""
        # Data
        calendar = "Act360"
        curve = afs.CRDC(0.05, calendars[calendar])
        strike = 100
        dates = ["20200203"]
        pay_dates = pd.DatetimeIndex(
            [
                pd.to_datetime(date) + pd.DateOffset(days=i * 360 / 3)
                for date in dates
                for i in range(1, 10)
            ]
        )
        data = pd.DataFrame(
            index=underlyings["SX5E"].get_dates().union(underlyings["RTY"].get_dates()),
            columns=["Price", "Volatility", "Dividend Rate"],
        )
        data["Volatility"] = 0.2
        data["Dividend Rate"] = 0.1
        underlyings["SX5E"].set_data(data)
        underlyings["RTY"].set_data(data)
        corr = {}
        for date in dates:
            corr[pd.to_datetime(date)] = np.array([[1, 0], [0, 1]])
        underlyings_multi = afs.MultiAsset(underlyings["SX5E"], underlyings["RTY"])
        underlyings_multi.corr = corr
        american_call = afs.AmericanVanillaOption(
            underlying=underlyings_multi,
            strike=strike,
            kind="call",
            calendar=calendars[calendar],
            pay_dates=pay_dates,
            nominal=1,
            phi="max",
        )
        engine_am = afs.RegressionMC()
        no_paths = 10**6
        # prices = np.array([90, 100, 110])
        prices = np.array([90, 100, 110])
        methods = [("LS", 2), ("TvR", 2)]
        actual = {}
        for price in prices:
            for method in methods:
                data["Price"] = price
                underlyings["SX5E"].set_data(data)
                underlyings["RTY"].set_data(data)
                actual[(price, method)] = engine_am.price(
                    american_call, dates, curve, no_paths, method=method
                )
        expected = {
            (90, ("TvR", 2)): 8.27,
            (90, ("LS", 2)): 7.99,
            (100, ("TvR", 2)): 14.08,
            (100, ("LS", 2)): 13.78,
            (110, ("TvR", 2)): 21.38,
            (110, ("LS", 2)): 21.16,
        }
        for key, actual_value in actual.items():
            with self.subTest(
                key=key
            ):  # To make it clearer which combination of price and method failed the test, if any do.
                self.assertAlmostEqual(
                    actual_value.values[0], expected[key], delta=expected[key] * 0.02
                )  # 2 percent error, just in case. In general less. Specially if we
                # allow recalculations, which will slow the test.

    def test_Longstaff_Schwartz(self):
        calendar = "Act365"
        no_paths = 10**5
        curve = afs.CRDC(0.06, calendars[calendar])
        strike = 40
        date = "20200203"
        date = pd.to_datetime(date)
        data = pd.DataFrame(
            index=underlyings["SX5E"].get_dates(),
            columns=["Price", "Volatility", "Dividend Rate"],
        )
        data["Dividend Rate"] = 0
        underlyings["SX5E"].set_data(data)
        underlyings_single = underlyings["SX5E"]

        method = ("LS", 6)
        engine_am = afs.RegressionMC()
        prices = np.array([36, 44])
        Ts = [1, 2]
        vols = [0.2, 0.4]
        actual = {}
        actual_eur = {}
        for price, vol, T in zip(prices, vols, Ts):
            data["Price"] = price
            data["Volatility"] = vol
            underlyings["SX5E"].set_data(data)
            underlyings["RTY"].set_data(data)
            pay_dates = pd.date_range(
                start=date, end=date + pd.DateOffset(years=T), periods=50 * T
            )
            american_put = afs.AmericanVanillaOption(
                underlying=underlyings_single,
                strike=strike,
                kind="put",
                calendar=calendars[calendar],
                pay_dates=pay_dates,
                nominal=1,
            )
            european_put = afs.Put(
                underlying=underlyings_single,
                maturity=pay_dates[-1],
                strike=strike,
                calendar=calendars[calendar],
                nominal=1,
            )
            actual[(price, vol, T)] = engine_am.price(
                american_put, date, curve, no_paths, method=method, no_calcs=10
            )
            actual_eur[(price, vol, T)] = european_put.get_px(date, curve)
        expected = {(36, 0.2, 1): 4.472, (44, 0.4, 2): 5.622}
        expected_eur = {(36, 0.2, 1): 3.844, (44, 0.4, 2): 5.202}
        for key, actual_value in actual.items():
            with self.subTest(
                key=key
            ):  # To make it clearer which combination of price and method failed the test, if any do.
                self.assertAlmostEqual(
                    actual_value.values[0], expected[key], delta=expected[key] * 0.02
                )  # 2 percent error, just in case. In general less. Specially if we
                # allow recalculations, which will slow the test.
        for key, actual_value in actual_eur.items():
            with self.subTest(
                key=key
            ):  # To make it clearer which combination of price and method failed the test, if any do.
                self.assertAlmostEqual(
                    actual_value.values[0],
                    expected_eur[key],
                    delta=expected_eur[key] * 0.001,
                )

    def test_bermudan_limit_call_premium(self):
        """Using the same example now we check that the call price has no premium. Also check that Bermudan option with one pay dates collapses to the European case."""
        T = 2
        calendar = "Act365"
        no_paths = 10**5
        curve = afs.CRDC(0.06, calendars[calendar])
        strike = 40
        date = "20200203"
        date = pd.to_datetime(date)
        data = pd.DataFrame(
            index=underlyings["SX5E"].get_dates(),
            columns=["Price", "Volatility", "Dividend Rate"],
        )
        data["Dividend Rate"] = 0
        data["Price"] = 44
        data["Volatility"] = 0.2
        underlyings["SX5E"].set_data(data)
        underlyings_single = underlyings["SX5E"]
        pay_dates = pd.date_range(
            start=date, end=date + pd.DateOffset(years=T), periods=50 * T
        )
        european_call = afs.Call(
            underlying=underlyings_single,
            maturity=pay_dates[-1],
            strike=strike,
            calendar=calendars[calendar],
            nominal=1,
        )
        american_call = afs.AmericanVanillaOption(
            underlying=underlyings_single,
            strike=strike,
            kind="call",
            calendar=calendars[calendar],
            pay_dates=pay_dates,
            nominal=1,
        )
        one_date_berm_call = afs.AmericanVanillaOption(
            underlying=underlyings_single,
            strike=strike,
            kind="call",
            calendar=calendars[calendar],
            pay_dates=pd.DatetimeIndex([pay_dates[-1]]),
            nominal=1,
        )
        engine_am = afs.RegressionMC()
        price_LS = engine_am.price(
            american_call, date, curve, no_paths, method=("LS", 6), no_calcs=10
        ).values[0]
        price_TvR = engine_am.price(
            american_call, date, curve, no_paths, method=("TvR", 6), no_calcs=10
        ).values[0]
        price_limit = engine_am.price(
            one_date_berm_call, date, curve, no_paths, method=("TvR", 6), no_calcs=10
        ).values[0]
        price_eur = european_call.get_px(date, curve).values[0]
        delta = 0.01 * price_eur
        with self.subTest(msg="LS vs TvR"):
            self.assertAlmostEqual(price_LS, price_TvR, delta=delta)

        with self.subTest(msg="LS vs Limit"):
            self.assertAlmostEqual(price_LS, price_limit, delta=delta)

        with self.subTest(msg="LS vs Eur"):
            self.assertAlmostEqual(price_LS, price_eur, delta=delta)

        with self.subTest(msg="TvR vs Limit"):
            self.assertAlmostEqual(price_TvR, price_limit, delta=delta)

        with self.subTest(msg="TvR vs Eur"):
            self.assertAlmostEqual(price_TvR, price_eur, delta=delta)

        with self.subTest(msg="Limit vs Eur"):
            self.assertAlmostEqual(price_limit, price_eur, delta=delta)

    def test_american_from_european(self):
        """Using the same example now we check that the call price has no premium. Also check that Bermudan option with one pay dates collapses to the European case."""
        T = 2
        calendar = "Act365"
        no_paths = (
            10**3
        )  # As we reuse draws, we can do it with few paths, it must be the same
        curve = afs.CRDC(0.06, calendars[calendar])
        strike = 40
        date = "20200203"
        date = pd.to_datetime(date)
        data = pd.DataFrame(
            index=underlyings["SX5E"].get_dates(),
            columns=["Price", "Volatility", "Dividend Rate"],
        )
        data["Dividend Rate"] = 0
        data["Price"] = 44
        data["Volatility"] = 0.2
        underlyings["SX5E"].set_data(data)
        underlyings_single = underlyings["SX5E"]
        pay_dates = pd.date_range(
            start=date, end=date + pd.DateOffset(years=T), periods=50 * T
        )
        european_call = afs.Call(
            underlying=underlyings_single,
            maturity=pay_dates[-1],
            strike=strike,
            calendar=calendars[calendar],
            nominal=1,
        )
        american_call = afs.AmericanVanillaOption(
            underlying=underlyings_single,
            strike=strike,
            kind="call",
            calendar=calendars[calendar],
            pay_dates=pay_dates,
            nominal=1,
        )
        american_eur_call = afs.AmericanFromEuropean(
            underlyings_single, pay_dates, calendars[calendar], european_call
        )
        engine_am = afs.RegressionMC()
        price_am = engine_am.price(
            american_call, date, curve, no_paths, method=("LS", 2)
        ).values[0]
        price_am_eur = engine_am.price(
            american_eur_call, date, curve, no_paths, method=("LS", 2), reuse_draws=True
        ).values[0]
        self.assertAlmostEqual(price_am, price_am_eur)


# if __name__ == '__main__':  # To test one method randomly, to reduce computation time
#     suite = select_one_test(TestAmericanOptions)
#     unittest.TextTestRunner().run(unittest.defaultTestLoader.loadTestsFromName("test_DLIB", TestAmericanOptions))  # This is fast, always tested.
#     unittest.TextTestRunner().run(suite)

if __name__ == "__main__":
    unittest.main()
