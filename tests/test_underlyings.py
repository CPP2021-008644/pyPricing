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
assets = factory.import_underlying("SD3E", "RTY")
assets_heston = factory.import_underlying("SD3E", asset_kind=afs.Heston)
calendars = factory.import_calendar("Act365")
discount_curves = factory.import_discount_curves(
    "USD LIBOR", start_date="20190101", end_date="20241231"
)


class TestUnderlyings(unittest.TestCase):
    def test_BS(self):
        """
        Test the price for a Put option assuming BS for the underlying.
        """
        K = 1500
        maturity = "20211027"
        ticker = "RTY"
        calendar = "Act365"
        discount_curve = "USD LIBOR"
        valuation_date = "20200903"
        product = afs.Put(assets[ticker], 0.9 * K, maturity, calendars[calendar])
        price_an = product.get_px("20200903", discount_curves[discount_curve]).values[0]
        t1 = datetime.now()
        price_mc = (
            afs.DiffusionMC()
            .price(
                product,
                valuation_date,
                discount_curve=discount_curves[discount_curve],
                no_sims=10**5,
                no_calcs=10,
            )
            .values[0]
        )
        t2 = datetime.now()
        dt1 = (t2 - t1).total_seconds()
        self.assertAlmostEqual((price_an- price_mc)/price_an,0, delta=0.1)
        self.assertLess(dt1, 1)  # Should be less than 1 second.

    def test_Heston(self):
        """
        Test the price for a call assuming Heston for the underlying.
        """
        valuation_date_str = "20230228"
        valuation_date = pd.to_datetime(valuation_date_str)
        tenor = 10  # unit years
        end_date_str = valuation_date + pd.DateOffset(years=tenor)
        end_date = pd.to_datetime(end_date_str)
        start_date = pd.to_datetime(valuation_date) + pd.DateOffset(days=1)
        time_step = 30  # unit days
        obs_dates = pd.date_range(
            start=start_date, end=end_date, freq=pd.DateOffset(days=time_step)
        )
        heston = assets_heston[
            "SD3E"
        ]  # We check that the index is imported without errors here.
        x0 = 100  # Although we fix a particular value for S_0.
        epsilon = 1
        kappa = 0.5
        rho = -0.9
        theta = 0.04
        dates_index = pd.to_datetime(
            [valuation_date]
        )  # Since we only have one date we write [valuation_date] in order to have a DatetimeIndex.
        heston.parameters = heston.parameters.reindex(dates_index)
        heston.parameters["kappa"] = kappa
        heston.parameters["epsilon"] = epsilon
        heston.parameters["theta"] = theta
        heston.parameters["rho"] = rho
        heston.parameters["v0"] = theta
        heston.parameters["x0"] = x0
        strike = 100
        underlying = heston

        def fpayoff_call(s):
            return np.where(s - strike > 0, s - strike, 0)

        zero_credit_risk = afs.CRDC(0)  # We assume zero-credit risk (as in PyFENG)
        call_heston = afs.ProductFromFunction(
            underlying=underlying,
            obsdates=obs_dates,
            pastmatters=False,
            func=fpayoff_call,
            calendar=calendars["Act365"],
        )
        n_trials = 1  # WE then set a sufficiently high tolerance to prevent the test from taking too long.
        t1 = datetime.now()
        price_mc = (
            afs.DiffusionMC(time_step=time_step)
            .price(
                call_heston,
                valuation_date,
                discount_curve=zero_credit_risk,
                no_sims=10**5,
                no_calcs=n_trials,
            )
            .values[0]
        )
        t2 = datetime.now()
        dt1 = (t2 - t1).total_seconds()
        sigma, vov, mr, rho, texp, spot = theta, epsilon, kappa, rho, tenor, x0
        m = pf.HestonFft(sigma, vov=vov, mr=mr, rho=rho)
        price_pf = m.price_fft(strike, spot, texp)
        self.assertAlmostEqual(
            price_mc, price_pf, delta=0.3
        )  # A sufficiently high tolerance to prevent the test from taking too long.
        self.assertLess(dt1, 20)  # Should be less than 20 second.


if __name__ == "__main__":
    unittest.main()
