import unittest
import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime

try:
    from . import functions as afsfun
except (ImportError, ModuleNotFoundError, ValueError):
    import pricing.functions as afsfun

pypricing_directory = os.path.expanduser("~/ArfimaTools/pypricing")
sys.path.insert(1, pypricing_directory)

import afslibrary as afs  # Note that data_factory_bd is not imported (otherwise the Job will fail since beautifulData can not be imported).

dbtools_directory = os.path.expanduser("~/ArfimaTools/afsdb")
sys.path.insert(1, dbtools_directory)
import db_tools

db = db_tools.BeautifulDataAFSStyleXL()
factory = afs.DataFactory(db)
calendars = factory.import_calendar("Act365", "Act360")
discount_curves = factory.import_discount_curves(
    "USD LIBOR", "EURIBOR", start_date="20200101", end_date="20220930"
)


class TestDiscountCurves(unittest.TestCase):
    def test_bootstrapping_CDS(self):
        """
        Test to obtain the hazard rates from CDS spreads.
        """
        risk_free_r = 0.02
        cds_spreads = np.array([0.00150, 0.01, 0.0350, 0.0750, 0.0950, 0.11])
        discount_curve = afs.CRDC(r=risk_free_r, calendar=calendars["Act365"])
        arr_2d = cds_spreads.reshape(1, -1)
        date = pd.to_datetime("2007-05-15")
        cols = [1, 2, 3, 4, 8, 12]  # tenors in years
        cds_spreads = pd.DataFrame(arr_2d, index=[date], columns=cols)
        cds_curve = afs.CDSCurve(
            "Test CDS curve",
            True,
            calendar=calendars["Act365"],
            discount_curve=discount_curve,
        )
        t1 = datetime.now()
        cds_curve.fit(cds_spreads)
        hazards = cds_curve.hazard_rates
        t2 = datetime.now()
        # Results using QuantLib, see TestsCRS/Tst_discount_curves/2023_07_13_Quantlib.py
        hazards_expected = np.array(
            [0.00248776, 0.0327625, 0.15869722, 0.43339152, 0.25562948, 0.7868441]
        )
        mean_errors = (
            np.mean(np.abs(hazards_expected - hazards.values) / hazards_expected) * 100
        )  # Errors in %
        time_computation = (t2 - t1).total_seconds()
        self.assertLess(mean_errors, 0.25)  # Error less than 0.25%.
        self.assertLess(time_computation, 0.3)  # Should be less than 0.3 second.


async def get_discount_table_df():
    from uglyData.ingestor.tslib import TSLib
    from uglyData.ingestor.timescale import ON_CONFLICT

    dbservice = "pi"

    tlib = TSLib()
    await tlib.connect(service=dbservice)

    ctdquery = """SELECT * FROM pricing.discount_curve_ppoly
    ORDER BY discount_curve ASC, dtime ASC"""

    return await tlib.conn.fetch(ctdquery, output="dataframe")


class TestReconstructionDiscountCurve(unittest.TestCase):
    async def test_reconstruction_discount_curve(self):
        discount_table_df = await get_discount_table_df()
        discount_curve = discount_curves["EURIBOR"]
        dates = ["20200505", "20200605"]
        ticker_discount_curve = "EURIBOR"
        calendar = calendars["Act360"]

        reconstructed = await afsfun.disc_curve_reconstruction(
            dates=dates,
            discount_table_df=discount_table_df,
            ticker_discount_curve=ticker_discount_curve,
            calendar=calendar,
            method="bond_spline",
        )
        df_original1 = discount_curve.get_value(dates, "20210810")
        df_reconstructed1 = reconstructed.get_value(dates, "20210810")
        rel_error1 = (df_reconstructed1 - df_original1) / df_original1

        df_original2 = discount_curve.get_value(dates, "20250810")
        df_reconstructed2 = reconstructed.get_value(dates, "20250810")
        rel_error2 = (df_reconstructed2 - df_original2) / df_original2

        df_original3 = discount_curve.get_value(dates, "20300810")
        df_reconstructed3 = reconstructed.get_value(dates, "20300810")
        rel_error3 = (df_reconstructed3 - df_original3) / df_original3

        df_original4 = discount_curve.get_value(dates, "20350810")
        df_reconstructed4 = reconstructed.get_value(dates, "20350810")
        rel_error4 = (df_reconstructed4 - df_original4) / df_original4

        df_original5 = discount_curve.get_value(dates, "20400810")
        df_reconstructed5 = reconstructed.get_value(dates, "20400810")
        rel_error5 = (df_reconstructed5 - df_original5) / df_original5

        self.assertAlmostEqual(np.max(np.abs(rel_error1)), 0, delta=0.00001)
        self.assertAlmostEqual(np.max(np.abs(rel_error2)), 0, delta=0.0005)
        self.assertAlmostEqual(np.max(np.abs(rel_error3)), 0, delta=0.0005)
        self.assertAlmostEqual(np.max(np.abs(rel_error4)), 0, delta=0.002)
        self.assertAlmostEqual(np.max(np.abs(rel_error5)), 0, delta=0.002)


if __name__ == "__main__":
    unittest.main()
