import numpy as np
import pandas as pd

try:
    from .excel_stuff import (
        ToleranceTableObject,
    )  # (Relative) import needed for the workspace. In this case __package__ is 'pypricing.pricing'
except (ImportError, ModuleNotFoundError, ValueError):
    from pricing.excel_stuff import ToleranceTableObject  # (Absolute) local import
try:
    from . import functions as afsfun
except (ImportError, ModuleNotFoundError, ValueError):
    import pricing.functions as afsfun


class CDS(ToleranceTableObject):
    """
    CDS object as described in [O'Kane].

    Parameters
    ----------
    cds_survival :  pricing.discount_curves.CDSCurve
        Survival probability curve constructed from spreads. We will use the calendar defined here, i.e., the same day count convention for bootstrapping and pricing.
    effective_date : str
        The effective date (as defined in Section 5.2 of [O’Kane, 2008]) in 'YYYY-MM-DD' format.
    maturity : str
        Maturity date (T in Section 5.2 of [O’Kane, 2008]) in 'YYYY-MM-DD' format.
    premium_rate : float
        Premium coupon per unit (100 bp = 0.01 per unit)
    premium_dates : pandas.DatetimeIndex
        Premium payment dates of the CDS contract. By default, IMM dates (20 March, 20 June, 20 September and 20 December of each year) are used. Note that the effective date
        is not included here although maturity is (the first ‘IMM’ date T years after the effective date).
    nominal : float, default = 100.
        Notional amount. For a short position (short protection), negative nominal.
    discretization : str, default = 'monthly'
        Time discretization for the integral in the Protection Leg (see page 106 [O’Kane, 2008]). The available methods are 'monthly' and 'daily'.
        In both methods we also include the payment dates. The day of the month considered in the monthly discretization is the effective_date day of the month.


    References
    ----------
        - [O’Kane, 2008] O’Kane, D. (2008). Modelling Single-name and Multi-name Credit Derivatives (first edition).

    Notes
    ------
    - Note that maturity is calculated as the first ‘IMM’ date T years after the effective date [O'Kane]. In consequence, if the maturity introduced is not a IMM date,
      the next IMM date is considered.
    """

    def __init__(
        self,
        cds_survival,
        effective_date,
        maturity,
        premium_rate,
        premium_dates=None,
        nominal=100,
        discretization="monthly",
    ):
        self.cds_curve = cds_survival
        self.calendar = (
            self.cds_curve.calendar
        )  # We use the same day count convention for bootstrapping and pricing.
        self.premium_rate = premium_rate
        self.discretization = discretization
        if premium_dates is None:
            self.premium_dates = cds_survival.get_imm_dates(effective_date, maturity)
            self.imm_bool = True  # IMM dates are considered for this contract
        else:
            self.premium_dates = premium_dates
            self.imm_bool = False  # IMM dates are considered for this contract
        self.nominal = nominal

    def get_rpvp_par_spread(self, dates, discount_curve):
        """
        Return RPV01(t, T), as described in Eq. (6.4) of [O'Kane], and the break-even spread, as described in Eq. (6.5) of [O'Kane].

        Parameters
        ----------
        dates : pandas.DatetimeIndex
            Valuation dates, :math:`t` in [O'Kane].
        discount_curve : pricing.discount_curves.DiscountCurve
            Gives the discounts, :math:`Z(t, t')` in [O'Kane].

        Returns
        -------
        pandas.Dataframe
            The index is a pandas.DatetimeIndex with the dates :math:`t^0_k`. The first column is RPV01 :math:`(t^0_k, T)`. The second column is :math:`S(t^0_k, T)`.
        Notes
        ------
            - We assume that t are payment dates so there is no accrued premium.
        """
        dates = afsfun.dates_formatting(dates)
        rpvbp = np.full(dates.size, np.nan)
        integral = np.full(dates.size, np.nan)
        rpvbp = np.empty(dates.size)
        for i in range(dates.size):
            # We first consider the additional terms arising when t is not a payment date (Sec. 6.5.1 of [O'Kane])
            if not np.isin(dates[i], self.premium_dates):
                if self.imm_bool:
                    imm_months = np.array([3, 6, 9, 12])
                    imm_month_prev = imm_months[
                        imm_months <= np.array([dates[i].month])
                    ][-1]
                    start_date_temp = dates[i].replace(day=15, month=imm_month_prev)
                    end_date_temp = dates[i].replace(day=25, month=imm_month_prev)
                    tnam1 = self.cds_curve.get_imm_dates(
                        start_date=start_date_temp, end_date=end_date_temp
                    )[0]  # t_{n^*-1}
                    tna = self.premium_dates[tnam1 < self.premium_dates][0]  # t_{n^*}
                else:
                    tnam1 = self.premium_dates[self.premium_dates < dates[i]][
                        -1
                    ]  # t_{n^*-1}
                    tna = self.premium_dates[tnam1 < self.premium_dates][0]  # t_{n^*}
                za = discount_curve.get_value(dates[i], tna)  # Z(t, t_{n^*})
                qa = self.cds_curve.get_value(
                    dates[i], tna, self.calendar
                )  # Q(t, t_{n^*})
                delta_1 = self.calendar.interval(tnam1, dates[i])  # Delta(t_{n^*-1}, t)
                delta_2 = self.calendar.interval(dates[i], tna)  # Delta(t, t_{n^*})
                delta_3 = self.calendar.interval(
                    tnam1, tna
                )  # Delta(t_{n^*-1}, t_{n^*})
                term_1 = delta_1 * za * (1 - qa)
                term_2 = 1 / 2 * delta_2 * za * (1 - qa)
                term_3 = delta_3 * za * qa
                accrued = term_1 + term_2 + term_3
                rpvbp[i] = accrued[0]
                fut_pay_dates = self.premium_dates[tna < self.premium_dates]
                fut_pay_dates_0 = self.premium_dates[tna <= self.premium_dates]
            else:  # In this case we simply apply Eq. (6.4)
                # accrued = 0
                fut_pay_dates = self.premium_dates[dates[i] < self.premium_dates]
                fut_pay_dates_0 = np.unique(np.insert(dates[i], 0, fut_pay_dates))
                rpvbp[i] = 0

            intervals_payment = np.array(
                [
                    self.calendar.interval(
                        fut_pay_dates_0[j - 1], fut_pay_dates_0[j]
                    ).item()
                    for j in range(1, len(fut_pay_dates_0))
                ]
            )
            if self.discretization == "monthly":
                # Monthly
                n_months = len(
                    pd.date_range(start=dates[i], end=fut_pay_dates[-1], freq="MS")
                )
                integral_dates = np.array(
                    [dates[i]]
                    + [
                        dates[i] + pd.DateOffset(months=n)
                        for n in range(1, n_months + 1)
                    ]
                )  # Effective and payment dates included
            elif self.discretization == "daily":
                # Daily
                n_days = len(
                    pd.date_range(start=dates[i], end=fut_pay_dates[-1], freq="D")
                )
                integral_dates = np.array(
                    [dates[i]]
                    + [dates[i] + pd.DateOffset(days=n) for n in range(1, n_days + 1)]
                )  # Effective and payment dates included
            else:
                raise NameError(f"The method {self.discretization} is not implemented.")

            integral_dates = pd.concat(
                [pd.Series(integral_dates), pd.Series(fut_pay_dates_0)]
            ).sort_values()
            # integral_dates = pd.concat([pd.Series(integral_dates)]).sort_values()  # Without the payment dates
            integral_dates = np.array(
                integral_dates.to_list()
            )  # np.array of pd.TimeStamps
            integral_dates = np.unique(
                integral_dates
            )  # We remove (possible) duplicated dates

            q_integral = self.cds_curve.get_value(
                dates[i], integral_dates, self.calendar
            )  # The first value should be 1.
            z_integral = discount_curve.get_value(
                dates[i], integral_dates, self.calendar
            )  # The first value should be 1.
            q_payment = self.cds_curve.get_value(
                dates[i], fut_pay_dates_0, self.calendar
            )
            z_payment = discount_curve.get_value(dates[i], fut_pay_dates, self.calendar)

            # RPVBP
            rpvbp[i] += (1 / 2) * np.sum(
                intervals_payment * z_payment * (q_payment[:-1] + q_payment[1:])
            )  # Eq. (6.4) of [O`Kane].
            # Integral
            integral[i] = (1 / 2) * np.sum(
                (z_integral[:-1] + z_integral[1:]) * (q_integral[:-1] - q_integral[1:])
            )
            # print(f'Integral 1: {integral[i]}')
            # integral[i] = np.sum(z_integral[1:] * (q_integral[:-1] - q_integral[1:]))  # Sometimes this approximation is used
            # print(f'Integral 2: {integral[i]}')

        rpvbp = pd.Series(rpvbp, index=dates)
        # print(f'Accrued: {accrued*self.nominal}')
        print(f"Default leg: {(1 - self.cds_curve.recovery) * integral * self.nominal}")
        par_spreads = (
            (1 - self.cds_curve.recovery) * integral / rpvbp
        )  # Eq. (6.5) together with the first equation in Section 6.6.1 of [O'Kane].
        df = pd.DataFrame({"RPV01": rpvbp, "S(t,T)": par_spreads})
        return df

    def get_px(self, dates, discount_curve):
        """
        CDS price (buyer of protection point of view).

        Parameters
        ----------
        dates : pandas.DatetimeIndex, list of strings, pandas.Timestamp, or string
            Valuation dates. It can be a single date (as a pandas.Timestamp or its string representation) or an array-like object
            (as a pandas.DatetimeIndex or a list of its string representation) containing dates.
        discount_curve: pricing.discount_curves.DiscountCurve
            Discount curve.

        Returns
        -------
        pandas.Series
            Prices for each valuation date.

        """
        dates = afsfun.dates_formatting(dates)
        df = self.get_rpvp_par_spread(dates, discount_curve)
        rpvbp = df["RPV01"]
        parspread = df["S(t,T)"]

        # print(f"Premium leg: {-self.premium_rate*rpvbp*self.nominal}")
        # print(f"RPV01: {rpvbp}")
        # print(f"Break even spread (bp): {parspread*10**4}")
        px = (
            parspread * rpvbp - self.premium_rate * rpvbp
        )  # First equation in Section 6.6.1 of [O'Kane].

        return self.nominal * px

    def get_px_artur(self, dates, discount_curve, clean=False):
        """
        Alternative pricing method including accrued coupon.

        Protection.Buyer point of view

        Parameters
        ----------
        dates
        discount_curve
        clean

        Returns
        -------

        """
        dates = afsfun.dates_formatting(dates)
        df = self.get_rpvp_par_spread(dates, discount_curve)
        rpvbp = df["RPV01"]
        parspread = df["S(t,T)"]

        # print(f"Coupon/Premium Leg: {- self.premium_rate * rpvbp}")
        # print(f"Protection/Default Leg: {parspread*rpvbp}")
        px = (
            parspread - self.premium_rate
        ) * rpvbp  # First equation in page 99 of [O'Kane].
        # print(f'px before accrued: {px}')
        if not clean:
            # TODO: Check if this is an approximation.
            z = np.full(dates.size, np.nan)
            deltas = np.full(dates.size, np.nan)
            for i in range(dates.size):
                previous_coupon_dates = self.premium_dates[
                    self.premium_dates <= dates[i]
                ]
                if previous_coupon_dates.size == 0:
                    deltas[i] = 0
                    z[i] = 0
                else:
                    deltas[i] = self.calendar.interval(
                        self.premium_dates[self.premium_dates <= dates[i]][-1], dates[i]
                    )
                    z[i] = discount_curve.get_value(
                        dates[i], self.premium_dates[self.premium_dates > dates[i]][0]
                    )
            accrued_coupon = (
                self.premium_rate * deltas * z
            )  # TODO: Should be the one in page 107 of [O'Kane] (WHY Q(t, t_n) DOES NOT APPEAR?)
            px -= accrued_coupon
        # print(f'px after accrued: {px}')
        return self.nominal * px


class DefaultableBond:
    pass


class CDO:
    pass
