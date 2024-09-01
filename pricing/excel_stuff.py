import numpy as np
import pandas as pd

try:
    from . import (
        functions as afsfun,
    )  # (Relative) import needed for the workspace. In this case __package__ is 'pypricing.pricing'
except (ImportError, ModuleNotFoundError, ValueError):
    import pricing.functions as afsfun  # (Absolute) local import


class ToleranceTableObject:
    def __add__(self, other):
        z = ToleranceTableObject()
        setattr(
            z,
            "compute_discount_tolerances",
            lambda *x: self.compute_discount_tolerances(*x)
            + other.compute_discount_tolerances(*x),
        )
        setattr(
            z,
            "compute_equity_tolerances",
            lambda *x: self.compute_equity_tolerances(*x)
            + other.compute_equity_tolerances(*x),
        )
        return z

    def __sub__(self, other):
        z = ToleranceTableObject()
        setattr(
            z,
            "compute_discount_tolerances",
            lambda *x: self.compute_discount_tolerances(*x)
            - other.compute_discount_tolerances(*x),
        )
        setattr(
            z,
            "compute_equity_tolerances",
            lambda *x: self.compute_equity_tolerances(*x)
            - other.compute_equity_tolerances(*x),
        )
        return z

    def __rmul__(self, other):
        z = ToleranceTableObject()
        setattr(
            z,
            "compute_discount_tolerances",
            lambda *x: other * self.compute_discount_tolerances(*x),
        )
        setattr(
            z,
            "compute_equity_tolerances",
            lambda *x: other * self.compute_equity_tolerances(*x),
        )
        return z

    def __neg__(self):
        z = ToleranceTableObject()
        setattr(
            z,
            "compute_discount_tolerances",
            lambda *x: -self.compute_discount_tolerances(*x),
        )
        setattr(
            z,
            "compute_equity_tolerances",
            lambda *x: -self.compute_equity_tolerances(*x),
        )
        return z

    def compute_discount_tolerances(self, dates, discount_curve):
        """
        Construct a table filled with the computed tolerances values of the product at the given dates.
        The tolerances are computed as the difference between the price of the product with a slight modification (i.e., a bump)
        of the value of interest rate (or credit spread) and the original price.

        Parameters
        ----------
        dates : pandas.DatetimeIndex, list of strings, pandas.Timestamp, or string
            Dates at which to compute the tolerances. It can be a single date (as a pandas.Timestamp or its string representation) or an array-like object
            (as a pandas.DatetimeIndex or a list of its string representation) containing dates.
        discount_curve : pricing.discount_curves.DiscountCurve
            Gives the discounts for the computations.

        Returns
        -------
        pandas.Dataframe
            The table with tolerances. The index of this data frame is ``dates``, while the different kind of
            positive and negative bumps are the columns.

        Notes
        ----------
        Let us make an example for the interest rate tolerance in order to explain how it is computed mathematically:

        .. math::
            \\Delta V = V(t, D + \\Delta D, \\ldots) - V(t, D, \\ldots)\,,

        where:

        - :math:`\\Delta V` is the tolerance for interest rates,
        - :math:`V(t, D, \\ldots)` is the price of the product at time `t` without the interest rate bump.
          Its price depends on the discount curve used, :math:`D`, and, possibly, other parameters.
        - :math:`V(t, D + \\Delta D, \\ldots)` is the price of the product at time `t`
          with the interest rate bump, :math:`\\Delta D`, represented as a change in the discount curve.

        See Also
        --------
        discount_curves.YieldCurve.fit
            For more details on how the bumps are calculated.

        """

        dates = afsfun.dates_formatting(dates)
        int_header = pd.MultiIndex.from_product(
            [["Tipos de Interés"], ["+5bp", "-5bp"]]
        )
        cred_header = pd.MultiIndex.from_product(
            [["Spread Crédito"], ["+20bp", "-20bp"]]
        )
        headers = pd.MultiIndex.union(int_header, cred_header, sort=False)
        table = pd.DataFrame(index=headers, columns=dates).transpose()
        prices = self.get_px(dates, discount_curve)
        discount_curve.p = -1
        if hasattr(self, "coupon_curve"):
            if self.coupon_curve.spot_curve.is_ir_curve:
                self.coupon_curve.spot_curve.p = -1
        if hasattr(self, "curve"):
            if self.curve.is_ir_curve:
                self.curve.p = -1
        table["Tipos de Interés", "-5bp"] = self.get_px(dates, discount_curve) - prices
        discount_curve.p = 1
        if hasattr(self, "coupon_curve"):
            if self.coupon_curve.spot_curve.is_ir_curve:
                self.coupon_curve.spot_curve.p = 1
        if hasattr(self, "curve"):
            if self.curve.is_ir_curve:
                self.curve.p = 1
        table["Tipos de Interés", "+5bp"] = self.get_px(dates, discount_curve) - prices
        discount_curve.p = 0
        if hasattr(self, "coupon_curve"):
            if self.coupon_curve.spot_curve.is_ir_curve:
                self.coupon_curve.spot_curve.p = 0
        if hasattr(self, "curve"):
            if self.curve.is_ir_curve:
                self.curve.p = 0

        if hasattr(self, "credit_curve"):
            self.credit_curve.p = -1
        if hasattr(self, "curve"):
            if self.curve.is_credit_curve:
                self.curve.p = -1
        table["Spread Crédito", "-20bp"] = self.get_px(dates, discount_curve) - prices
        if hasattr(self, "credit_curve"):
            self.credit_curve.p = 1
        if hasattr(self, "curve"):
            if self.curve.is_credit_curve:
                self.curve.p = 1
        table["Spread Crédito", "+20bp"] = self.get_px(dates, discount_curve) - prices
        if hasattr(self, "credit_curve"):
            self.credit_curve.p = 0
        if hasattr(self, "curve"):
            if self.curve.is_credit_curve:
                self.curve.p = 0
        return table

    def generate_tolerances_table(self, dates, discount_curve, extended_table=False):
        dates = afsfun.dates_formatting(dates)
        bond_headers = pd.MultiIndex.from_product(
            [["Bond sensitivities"], ["TIR", "Dur."]]
        )
        int_header = pd.MultiIndex.from_product(
            [["Tipos de Interés"], ["-5bp", "+5bp"]]
        )
        headers = pd.MultiIndex.union(bond_headers, int_header, sort=False)
        cred_header = pd.MultiIndex.from_product(
            [["Spread Crédito"], ["-20bp", "+20bp"]]
        )
        headers = pd.MultiIndex.union(headers, cred_header, sort=False)
        vol_header = pd.MultiIndex.from_product(
            [["EQ Volatilidad"], ["-200bp", "+200bp"]]
        )
        headers = pd.MultiIndex.union(headers, vol_header, sort=False)
        div_header = pd.MultiIndex.from_product(
            [["EQ Dividendos"], ["-100bp", "+100bp"]]
        )
        headers = pd.MultiIndex.union(headers, div_header, sort=False)
        final_tol_header = pd.MultiIndex.from_product(
            [["Precio ExCupón"], ["-", "=", "+"]]
        )
        headers = pd.MultiIndex.union(headers, final_tol_header, sort=False)

        table = pd.DataFrame(index=headers, columns=dates).transpose()
        table["Precio ExCupón", "="] = self.get_px(
            dates, discount_curve, no_sims=10**6, no_calcs=10
        )

        disc_header = pd.MultiIndex.union(int_header, cred_header, sort=False)
        table[disc_header] = self.compute_discount_tolerances(dates, discount_curve)

        eq_headers = pd.MultiIndex.union(vol_header, div_header, sort=False)
        if hasattr(self, "compute_equity_tolerances"):
            table[eq_headers] = self.compute_equity_tolerances(dates, discount_curve)
        else:
            table[eq_headers] = 0

        temp_table = table[
            [
                ("Tipos de Interés", "+5bp"),
                ("Tipos de Interés", "-5bp"),
                ("Spread Crédito", "+20bp"),
                ("Spread Crédito", "-20bp"),
                ("EQ Volatilidad", "+200bp"),
                ("EQ Volatilidad", "-200bp"),
                ("EQ Dividendos", "+100bp"),
                ("EQ Dividendos", "-100bp"),
            ]
        ]
        temp_table = np.sum(temp_table * (temp_table >= 0), axis=1)
        table["Precio ExCupón", "+"] = (
            table["Precio ExCupón", "="].values + temp_table.values
        )
        temp_table = table[
            [
                ("Tipos de Interés", "+5bp"),
                ("Tipos de Interés", "-5bp"),
                ("Spread Crédito", "+20bp"),
                ("Spread Crédito", "-20bp"),
                ("EQ Volatilidad", "+200bp"),
                ("EQ Volatilidad", "-200bp"),
                ("EQ Dividendos", "+100bp"),
                ("EQ Dividendos", "-100bp"),
            ]
        ]
        temp_table = np.sum(temp_table * (temp_table <= 0), axis=1)
        table["Precio ExCupón", "-"] = (
            table["Precio ExCupón", "="].values + temp_table.values
        )
        if not extended_table:
            table = table.dropna(axis=1, how="all")
        return table
