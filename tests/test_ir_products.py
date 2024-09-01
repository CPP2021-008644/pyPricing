import unittest
import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime

pypricing_directory = os.path.expanduser("~/ArfimaTools/pypricing")
sys.path.insert(1, pypricing_directory)
import \
    afslibrary as afs  # Note that data_factory_bd is not imported (otherwise the Job will fail since beautifulData can not be imported.

dbtools_directory = os.path.expanduser("~/ArfimaTools/afsdb")
sys.path.insert(1, dbtools_directory)
import db_tools

dbtools_directory = os.path.expanduser("~/ArfimaTools/afsdb")
sys.path.insert(1, dbtools_directory)
db = db_tools.BeautifulDataAFSStyleXL()
factory = afs.DataFactory(db)
calendars = factory.import_calendar("Act360")
discount_curves = factory.import_discount_curves(
    "EURIBOR", "USD LIBOR", start_date="20180101", end_date="20241231"
)


def check_prices_approx_equal(self, *prices, delta=0.01):
    """Check that all combinations of prices are approximately equal."""
    n = len(prices)
    for i in range(n):
        for j in range(i + 1, n):
            self.assertAlmostEqual(prices[i], prices[j], delta=delta)


class TestIRProducts(unittest.TestCase):
    # Interest rate model
    date = pd.to_datetime("20200131")
    date_index = pd.DatetimeIndex([date])
    g2pp = afs.G2PlusPlusShortRate(discount_curves["EURIBOR"], calendars["Act360"])
    g2pp.parameters = g2pp.parameters.reindex(date_index)
    a = 0.00116
    b = 0.083
    sigma = 0.00108
    eta = 0.00102
    rho = -0.5
    g2pp.parameters["vol_x"] = sigma
    g2pp.parameters["reversion_x"] = a
    g2pp.parameters["vol_y"] = eta
    g2pp.parameters["reversion_y"] = b
    g2pp.parameters["correlation"] = rho
    calendar = "Act360"
    discount_curve = "EURIBOR"
    mc = afs.SRDeterministicVolDiffusionMC(g2pp)

    # noinspection PyUnresolvedReferences
    def test_caplets(self):
        """
        Test the price of several caplets using the analytical formula and MC.
        """

        # Caps date, only for caps
        dates_caps = ["20211029"]
        dates = dates_caps
        dates_index = pd.to_datetime(dates)

        # Interest rate model: G2++
        g2pp = afs.G2PlusPlusShortRate(discount_curves["EURIBOR"], calendars["Act360"])
        g2pp.parameters = g2pp.parameters.reindex(dates_index)
        g2pp.parameters["vol_x"] = 0.005352
        g2pp.parameters["reversion_x"] = 0.192024
        g2pp.parameters["vol_y"] = 0.005352
        g2pp.parameters["reversion_y"] = 0.192024
        g2pp.parameters["correlation"] = -0.5

        # Interest rate model: HW
        hw = afs.OneFactorHullWhiteShortRate(discount_curves["EURIBOR"],
                                             calendars["Act360"])
        hw.parameters = hw.parameters.reindex(dates_index)
        hw.parameters = g2pp.parameters.reindex(dates_index)
        hw.parameters["vol"] = 0.005352
        hw.parameters["reversion"] = 0.192024

        # Interest rate model: Vasicek
        vasicek = afs.VasicekShortRate(discount_curves["EURIBOR"], calendars["Act360"])
        vasicek.parameters = vasicek.parameters.reindex(dates_index)
        vasicek.parameters["vol"] = 0.005352
        vasicek.parameters["reversion"] = 0.192024
        vasicek.parameters["mean"] = 0.01

        calendar = "Act360"
        date = pd.to_datetime("20211029")
        no_caplets = 10  # Even number
        tenor = np.array(no_caplets / 2, dtype=int)
        for model in [vasicek, hw, g2pp]:
            # Data frames
            df_comp = pd.DataFrame(columns=["MC", "Analytic"])
            caps_df = pd.DataFrame(
                index=np.arange(1),
                columns=["Maturity", "Tenor", "Pay Frequency", "Quote"]
            )
            caps_df.Maturity = 0
            caps_df.Tenor = tenor
            caps_df.Quote = 0  # Irrelevant now
            caps_df["Pay Frequency"] = 6
            caps_df, all_forwards, all_T, all_deltas, all_bonds = model.compute_cap_data(
                date,
                caps_df,
                discount_curves["EURIBOR"],
                effective_date_interval=0,
                vol_type="normal",
            )

            params = model.parameters.values[0]
            strikes = np.array([-0.1])
            pay_freq = np.array([6])
            maturities = np.array([tenor])

            analytic = model.cap_price_function(
                params,
                strikes,
                maturities,
                pay_freq,
                all_T,
                all_deltas,
                all_bonds,
                return_caplets=True,
            )
            df_comp["Analytic"] = analytic[strikes[0]]
            t1 = datetime.now()
            prices_mc = np.full(no_caplets, np.nan)
            if len(pay_freq) == 1:
                for no_c in range(no_caplets):
                    obsdates = pd.date_range(
                        date, periods=1, freq=pd.DateOffset(months=pay_freq[0])
                    )
                    obsdates.freq = None  # Otherwise it is problematic for the union with pay_dates, tenor_dates

                    date = date + pd.DateOffset(
                        months=pay_freq[0]
                    )  # We increase the date from the previous iteration
                    date_plus = date + pd.DateOffset(months=pay_freq[0])
                    forward_rate = afs.ForwardRate(
                        discount_curves["EURIBOR"],
                        date,
                        date_plus,
                        calendars[calendar],
                        pay_freq[0],
                    )

                    # obsdates = pd.date_range(date, periods=1, freq=pd.DateOffset(months=pay_freq[0]))
                    pay_dates = pd.date_range(
                        date, periods=1, freq=pd.DateOffset(months=pay_freq[0])
                    )
                    pay_dates.freq = None

                    caps = afs.CapFloor(
                        rate=forward_rate,
                        obsdates=obsdates,
                        strike=strikes,
                        calendar=calendars[calendar],
                        kind="cap",
                        nominal=1,
                        pay_dates=pay_dates,
                    )
                    price_mc = caps.get_px(
                        dates, discount_curves["EURIBOR"], model, no_sim=10 ** 6
                    )
                    prices_mc[no_c] = price_mc.values[0]
            else:
                for no_c in range(no_caplets):
                    obsdates = pd.date_range(
                        date, periods=1, freq=pd.DateOffset(months=pay_freq)
                    )
                    obsdates.freq = None  # Otherwise it is problematic for the union with pay_dates, tenor_dates

                    date = date + pd.DateOffset(
                        months=pay_freq
                    )  # We increase the date from the previous iteration
                    date_plus = date + pd.DateOffset(months=pay_freq)
                    forward_rate = afs.ForwardRate(
                        discount_curves["EURIBOR"],
                        date,
                        date_plus,
                        calendars[calendar],
                        pay_freq,
                    )

                    # obsdates = pd.date_range(date, periods=1, freq=pd.DateOffset(months=pay_freq))
                    pay_dates = pd.date_range(
                        date, periods=1, freq=pd.DateOffset(months=pay_freq)
                    )
                    pay_dates.freq = None

                    caps = afs.CapFloor(
                        rate=forward_rate,
                        obsdates=obsdates,
                        strike=strikes,
                        calendar=calendars[calendar],
                        kind="cap",
                        nominal=1,
                        pay_dates=pay_dates,
                    )
                    price_mc = caps.get_px(
                        dates, discount_curves["EURIBOR"], model, no_sim=10 ** 6
                    )
                    prices_mc[no_c] = price_mc.values[0]

            t2 = datetime.now()
            dt = (t2 - t1).total_seconds()
            df_comp["MC"] = prices_mc
            df_quot = df_comp["MC"] / df_comp["Analytic"]
            df_comp["Difference (bp)"] = (df_comp["MC"] - df_comp["Analytic"]) * 10000

            self.assertAlmostEqual(df_quot.values.all(), 1, delta=0.0001)
            self.assertAlmostEqual(df_comp["Difference (bp)"].values.all(), 1,
                                   delta=0.2)
            self.assertLess(dt, 10)  # Should be less than 10 second.

    # noinspection PyUnresolvedReferences
    def test_knock_out_swap(self):
        offset = 100
        tenor = 10
        date_alpha = pd.to_datetime("20200131") + pd.DateOffset(days=offset)
        date_beta = date_alpha + pd.DateOffset(years=tenor)
        swap_rate = afs.LognormalSwapRate(discount_curves["EURIBOR"], date_alpha,
                                          date_beta,
                                          6, 6, legs_calendars=calendars["Act360"],
                                          tenor_length=tenor)
        # Now, the barrier
        spread = 0.01 * np.arange(10)
        gearing = 2
        path_dep_rate = afs.PathDependentRate(swap_rate, spread, gearing)
        tl_barrier = afs.KnockOutRate(path_dep_rate, barrier=0.1, kind="up-and-out")

        rate = afs.ProductRate(swap_rate, tl_barrier)
        rate_magic = swap_rate * tl_barrier

        obsdates = pd.date_range(date_alpha, date_beta - pd.DateOffset(years=1),
                                 freq=pd.DateOffset(years=1))
        ko_swap = afs.GeneralSwap(rate, obsdates, effective_date=date_alpha,
                                  end_date=date_beta, floating_freq=12, fixed_freq=12,
                                  legs_calendars=calendars[TestIRProducts.calendar],
                                  nominal=1)
        price = TestIRProducts.mc.price(ko_swap, TestIRProducts.date,
                                        discount_curves["EURIBOR"], 10000).values[0]
        ko_swap_magic = afs.GeneralSwap(rate_magic, obsdates, effective_date=date_alpha,
                                        end_date=date_beta, floating_freq=12,
                                        fixed_freq=12,
                                        legs_calendars=calendars[
                                            TestIRProducts.calendar],
                                        nominal=1)
        price_magic = TestIRProducts.mc.price(ko_swap_magic, TestIRProducts.date,
                                              discount_curves["EURIBOR"], 10000).values[
            0]
        self.assertAlmostEqual(price, price_magic, delta=0.01)

    # noinspection PyUnresolvedReferences
    def test_tarn_global_floor_swap(self):
        offset = 100
        tenor = 10
        calendar = TestIRProducts.calendar
        date_alpha = pd.to_datetime("20200131") + pd.DateOffset(days=offset)
        date_beta = date_alpha + pd.DateOffset(years=tenor)
        swap_rate = afs.LognormalSwapRate(discount_curves["EURIBOR"], date_alpha,
                                          date_beta,
                                          6, 6, legs_calendars=calendars["Act360"],
                                          tenor_length=tenor)
        dates = pd.date_range(date_alpha, date_beta, freq=pd.DateOffset(years=1))
        cumulative_rate = afs.CumulativeRate(swap_rate, dates=dates,
                                             calendar=calendars["Act360"])
        floor = afs.GlobalFloorRate(swap_rate, floor=1, dates=dates,
                                    calendar=calendars["Act360"])

        tarn = afs.KnockOutRate(cumulative_rate, barrier=0.1, kind="up-and-out")
        rate = afs.ProductRate(tarn, swap_rate)
        floor_rate = afs.AdditionRate(floor, swap_rate)
        floor_rate_magic = floor + swap_rate  # Using the magic method

        obsdates = pd.date_range(date_alpha, date_beta - pd.DateOffset(years=1),
                                 freq=pd.DateOffset(years=1))
        tarn_swap = afs.GeneralSwap(rate, obsdates, effective_date=date_alpha,
                                    end_date=date_beta, floating_freq=12, fixed_freq=12,
                                    legs_calendars=calendars[calendar], nominal=1)
        global_floor_swap = afs.GeneralSwap(floor_rate, obsdates,
                                            effective_date=date_alpha,
                                            end_date=date_beta, floating_freq=12,
                                            fixed_freq=12,
                                            legs_calendars=calendars[calendar],
                                            nominal=1)
        global_floor_swap_magic = afs.GeneralSwap(floor_rate_magic, obsdates,
                                                  effective_date=date_alpha,
                                                  end_date=date_beta, floating_freq=12,
                                                  fixed_freq=12,
                                                  legs_calendars=calendars[calendar],
                                                  nominal=1)

        tarn_rate_compact = afs.TARNRate(rate=swap_rate, dates=dates,
                                         calendar=calendars[calendar], barrier=0.1)
        tarn_swap_comp = afs.GeneralSwap(tarn_rate_compact, obsdates,
                                         effective_date=date_alpha, end_date=date_beta,
                                         floating_freq=12, fixed_freq=12,
                                         legs_calendars=calendars[calendar], nominal=1)

        tarn_swap_comp2 = afs.TARN(swap_rate, dates, calendars[calendar], 0.1, obsdates,
                                   date_alpha, date_beta, 12, 12, calendars[calendar],
                                   1)

        floor_rate_comp = afs.GFlooredRate(rate=swap_rate, floor=1, dates=dates,
                                           calendar=calendars[calendar])
        global_floor_swap_comp = afs.GeneralSwap(floor_rate, obsdates,
                                                 effective_date=date_alpha,
                                                 end_date=date_beta, floating_freq=12,
                                                 fixed_freq=12,
                                                 legs_calendars=calendars[calendar],
                                                 nominal=1)
        global_floor_swap_comp_sq = afs.GeneralSwap(floor_rate_comp, obsdates,
                                                    effective_date=date_alpha,
                                                    end_date=date_beta,
                                                    floating_freq=12,
                                                    fixed_freq=12,
                                                    legs_calendars=calendars[calendar],
                                                    nominal=1)
        price_1 = TestIRProducts.mc.price(tarn_swap, TestIRProducts.date,
                                          discount_curves["EURIBOR"], 10000).values[0]
        price_2 = TestIRProducts.mc.price(tarn_swap_comp, TestIRProducts.date,
                                          discount_curves["EURIBOR"], 10000).values[0]
        price_3 = TestIRProducts.mc.price(tarn_swap_comp2, TestIRProducts.date,
                                          discount_curves["EURIBOR"], 10000).values[0]
        check_prices_approx_equal(self, price_1, price_2, price_3)

        price_1 = TestIRProducts.mc.price(global_floor_swap, TestIRProducts.date,
                                          discount_curves["EURIBOR"], 10000).values[0]
        price_2 = TestIRProducts.mc.price(global_floor_swap_magic, TestIRProducts.date,
                                          discount_curves["EURIBOR"], 10000).values[0]
        price_3 = TestIRProducts.mc.price(global_floor_swap_comp, TestIRProducts.date,
                                          discount_curves["EURIBOR"], 10000).values[0]
        price_4 = \
            TestIRProducts.mc.price(global_floor_swap_comp_sq, TestIRProducts.date,
                                    discount_curves["EURIBOR"], 10000).values[0]
        check_prices_approx_equal(self, price_1, price_2, price_3, price_4)

    # noinspection PyUnresolvedReferences
    def test_spread_coupon(self):
        discount_curve = TestIRProducts.discount_curve
        calendar = TestIRProducts.calendar
        cms_rates = afs.CMSDiffRate(discount_curves[discount_curve], [2, 10],
                                    "20220115", 6,
                                    12, legs_calendars=calendars[calendar])
        spread = 0.01
        gearing = 2
        cap = 0.1
        floor = 0

        end_date_0 = pd.to_datetime("20220115") + pd.DateOffset(years=2)
        end_date_1 = pd.to_datetime("20220115") + pd.DateOffset(years=10)
        cms_rate_0 = afs.LognormalSwapRate(discount_curves[discount_curve], "20220115",
                                           end_date_0, 6, 12,
                                           legs_calendars=calendars[calendar],
                                           tenor_length=2)
        cms_rate_1 = afs.LognormalSwapRate(discount_curves[discount_curve], "20220115",
                                           end_date_1, 6, 12,
                                           legs_calendars=calendars[calendar],
                                           tenor_length=10)

        cms_diff_rates = afs.DifferenceRate(cms_rate_0, cms_rate_1)

        cms_spread = afs.CMSSpreadCoupon(cms_rates, "20220115", spread, gearing, cap,
                                         floor,
                                         calendars[calendar])
        cms_spread_d = afs.CMSSpreadCoupon(cms_diff_rates, "20220115", spread, gearing,
                                           cap,
                                           floor, calendars[calendar])

        rate = afs.GSCFRate(cms_diff_rates, gearing=gearing, spread=spread, cap=cap,
                            floor=floor)
        cms_spread_prod = afs.Forward(underlying=rate, maturity="20220115", strike=0,
                                      calendar=calendars[calendar], nominal=100)

        rates = [cms_rate_0, cms_rate_1]
        cms_spread_compact = afs.CMSSpreadForward(rates, gearing, spread, cap, floor,
                                                  maturity="20220115", strike=0,
                                                  calendar=calendars[calendar],
                                                  nominal=100)
        price_1 = TestIRProducts.mc.price(cms_spread, TestIRProducts.date,
                                          discount_curves["EURIBOR"], 10000).values[0]
        price_2 = TestIRProducts.mc.price(cms_spread_d, TestIRProducts.date,
                                          discount_curves["EURIBOR"], 10000).values[0]
        price_3 = TestIRProducts.mc.price(cms_spread_prod, TestIRProducts.date,
                                          discount_curves["EURIBOR"], 10000).values[0]
        price_4 = TestIRProducts.mc.price(cms_spread_compact, TestIRProducts.date,
                                          discount_curves["EURIBOR"], 10000).values[0]
        check_prices_approx_equal(self, price_1, price_2, price_3, price_4)


if __name__ == "__main__":
    unittest.main()
