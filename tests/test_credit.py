import unittest
import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime

pypricing_directory = os.path.expanduser("~/ArfimaTools/pypricing")
sys.path.insert(1, pypricing_directory)

import afslibrary as afs  # Note that data_factory_bd is not imported (otherwise the Job will fail since beautifulData can not be imported).

dbtools_directory = os.path.expanduser("~/ArfimaTools/afsdb")
sys.path.insert(1, dbtools_directory)
import db_tools

db = db_tools.BeautifulDataAFSStyleXL()
factory = afs.DataFactory(db)
calendars = factory.import_calendar("Act360")


class TestCredit(unittest.TestCase):
    """
    In the initialization we also construct the discount curve.

    References
    ----------
        - O’Kane, D. (2008). Modelling Single-name and Multi-name Credit Derivatives (first edition).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.payment_dates_0 = None
        self.payment_dates = None
        self.discount_curve = None
        self.discounts = None
        values = list(
            [
                1.000000,
                0.996449,
                0.985213,
                0.973973,
                0.961504,
                0.947857,
                0.936601,
                0.924726,
                0.913373,
                0.902181,
                0.891177,
                0.880308,
                0.869570,
                0.858799,
                0.848282,
                0.837779,
                0.827293,
                0.816938,
                0.806934,
                0.796834,
                0.786860,
            ]
        )
        self.discounts = values
        # # # Discount curve
        # For constructing the discount curve we use only the payment dates after the effective date
        payment_dates_0 = list(
            [
                "2008-01-18",
                "2008-02-15",
                "2008-05-15",
                "2008-08-15",
                "2008-11-17",
                "2009-02-16",
                "2009-05-15",
                "2009-08-17",
                "2009-11-16",
                "2010-02-15",
                "2010-05-17",
                "2010-08-16",
                "2010-11-15",
                "2011-02-15",
                "2011-05-16",
                "2011-08-15",
                "2011-11-15",
                "2012-02-15",
                "2012-05-15",
                "2012-08-15",
                "2012-11-15",
            ]
        )
        payment_dates_0 = pd.to_datetime(payment_dates_0)
        self.payment_dates_0 = payment_dates_0
        payment_dates = list(
            [
                "2006-11-15",
                "2007-02-15",
                "2007-05-15",
                "2007-08-15",
                "2007-11-15",
                "2008-02-15",
                "2008-05-15",
                "2008-08-15",
                "2008-11-17",
                "2009-02-16",
                "2009-05-15",
                "2009-08-17",
                "2009-11-16",
                "2010-02-15",
                "2010-05-17",
                "2010-08-16",
                "2010-11-15",
                "2011-02-15",
                "2011-05-16",
                "2011-08-15",
                "2011-11-15",
                "2012-02-15",
                "2012-05-15",
                "2012-08-15",
                "2012-11-15",
            ]
        )
        payment_dates = pd.to_datetime(payment_dates)
        self.payment_dates = payment_dates
        # Constructing the discount curve
        dates = self.payment_dates_0
        years_diff = (dates - dates[0]) / pd.Timedelta(days=360)
        values_2d = np.array(self.discounts).reshape(1, -1)
        df_new = pd.DataFrame(values_2d, index=[dates[0]], columns=years_diff.values)
        df_new_values = df_new.apply(lambda p: (p - 1) / float(p.name), axis=0)
        discount_curve_LIBOR = afs.DepositCurve(
            calendar=calendars["Act360"], interest_type="simple", semiannual=False
        )
        discount_curve_LIBOR.fit(df_new_values)
        self.discount_curve = discount_curve_LIBOR

    def test_discount_curve(self):
        """
        Testing the discount curve.
        """
        a = self.discount_curve.get_value(self.payment_dates_0[0], self.payment_dates_0)
        b = self.discounts
        self.assertLess(
            np.mean(100 * np.abs(a - b) / b), 10**-10
        )  # Error less than 10**-10 %.

    def test_price_CDS(self):
        """
        Test CDS price. Example 7.8 EXAMPLE CDS VALUATION in [O'Kane]. For now, only RPV01 is tested.
        Note that the accrued coupon is tested in this example.
        """

        # Probability Structure
        tenors = [0.67, 1.17, 2.17, 3.17, 4.17, 5.17, 7.17, 10.17]  # In years
        hazard_rates = [2.732, 2.436, 3.035, 3.584, 4.176, 6.436, 5.704, 6.296]
        hazard_rates = [
            rate * 10**-2 for rate in hazard_rates
        ]  # We want hazard rates per unit
        cds_curve = afs.CDSCurve(
            "CDS Survival",
            True,
            calendar=calendars["Act360"],
            discount_curve=self.discount_curve,
        )
        arr = np.array(hazard_rates)
        arr_2d = arr.reshape(1, -1)
        cols = tenors
        df_hazard_rates = pd.DataFrame(
            arr_2d, index=[self.payment_dates_0[0]], columns=cols
        )

        # Equivalent to the result of CDSCurve.fit
        cds_curve.hazard_rates = df_hazard_rates
        cds_curve.fitting_dates = df_hazard_rates.index
        cds_curve.interpolation_data = -df_hazard_rates
        cds_curve.params = {
            -1: -df_hazard_rates,
            0: -df_hazard_rates,
            1: -df_hazard_rates,
        }  # For this example we do not consider the perturbations.
        # CDS
        recovery_rate = 0.4
        nominal = -10 * 10**6  # Short position
        premium_rate = 180 * 10**-4

        evaluation_date = pd.to_datetime("2008-01-18")

        cds_pypricing = afs.CDS(
            cds_curve,
            effective_date=evaluation_date,
            maturity=self.payment_dates[-1],
            premium_rate=premium_rate,
            premium_dates=self.payment_dates,
            nominal=nominal,
        )
        print(self.discount_curve)
        t1 = datetime.now()
        rpvbp = (
            cds_pypricing.get_rpvp_par_spread(
                dates=evaluation_date, discount_curve=self.discount_curve
            )
            .iloc[:, 0]
            .values[0]
        )
        t2 = datetime.now()
        rpvbp_okane = 4.2082  # See
        relative_error = np.abs(rpvbp - rpvbp_okane) * 100 / rpvbp_okane  # Error in %
        time_computation = (t2 - t1).total_seconds()
        self.assertLess(relative_error, 0.1)  # Error less than 0.1 %.
        self.assertLess(time_computation, 0.1)  # Should be less than 0.1 second.


if __name__ == "__main__":
    unittest.main()
