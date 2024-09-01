import numpy as np
import pandas as pd

try:
    from .excel_stuff import (
        ToleranceTableObject,
    )  # (Relative) import needed for the workspace. In this case __package__ is 'pypricing.pricing'
except (ImportError, ModuleNotFoundError, ValueError):
    from pricing.excel_stuff import ToleranceTableObject  # (Absolute) local import
try:
    from .discount_curves import CRDC
except (ImportError, ModuleNotFoundError, ValueError):
    from pricing.discount_curves import CRDC
from scipy import optimize
from math import ceil
from itertools import accumulate

try:
    from . import functions as afsfun
except (ImportError, ModuleNotFoundError, ValueError):
    import pricing.functions as afsfun

zero_credit_risk = CRDC(0)


class Bond(ToleranceTableObject):
    def __init__(self, effective_date, maturity, market_spread=0, rating=None):
        self.effective_date = effective_date
        self.maturity = maturity
        self.rating = rating
        self.market_spread = market_spread

        self.px_fun = lambda *x: self.get_px(*x)
        self.disc_tols_fun = lambda *x: self.compute_discount_tolerances(*x)

    def __add__(self, other):
        if (
            self.effective_date == other.effective_date
            and self.maturity == other.maturity
        ):
            z = Bond(self.effective_date, self.maturity)
            z.px_fun = lambda *x: self.get_px(*x) + other.get_px(*x)
            z.disc_tols_fun = lambda *x: self.disc_tols_fun(*x) + other.disc_tols_fun(
                *x
            )
            return z
        else:
            print("Effective and/or maturity do not match")
            return None

    def __sub__(self, other):
        if (
            self.effective_date == other.effective_date
            and self.maturity == other.maturity
        ):
            z = Bond(self.effective_date, self.maturity)
            z.px_fun = lambda *x: self.get_px(*x) - other.get_px(*x)
            z.disc_tols_fun = lambda *x: self.disc_tols_fun(*x) - other.disc_tols_fun(
                *x
            )
            return z
        else:
            print("Effective and/or maturity do not match")
            return None

    def get_px(self, date, discount_curve):
        """
        Get price method.

        Warnings
        --------
        - Old, undocumented and unchecked module, review and correct before using. 

        """
        return self.px_fun(date, discount_curve)

    def compute_discount_tolerances(self, dates, discount_curve):
        int_header = pd.MultiIndex.from_product(
            [["Tipos de Interés"], ["-5bp", "+5bp"]]
        )
        cred_header = pd.MultiIndex.from_product(
            [["Spread Crédito"], ["-20bp", "+20bp"]]
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

    def fit_market_spread(self, date, price, discount_curve):
        def difference(spread):
            self.market_spread = spread
            diff = price - self.get_px(date, discount_curve).values[0]
            return diff

        spread = optimize.newton(difference, 0.01)
        self.market_spread = spread


class Sinkable(Bond):
    def __init__(
        self,
        effective_date,
        maturity,
        freq,
        principal_amounts,
        interest_amounts,
        calendar,
        credit_curve=zero_credit_risk,
        rating=None,
        market_spread=0,
    ):
        self.effective_date = effective_date
        self.maturity = maturity
        self.freq = freq
        self.pay_dates = pd.date_range(
            self.effective_date, self.maturity, freq=pd.DateOffset(months=self.freq)
        )
        Bond.__init__(
            self,
            self.pay_dates[0],
            self.pay_dates[-1],
            rating=rating,
            market_spread=market_spread,
        )
        self.principal_amounts = np.asarray(principal_amounts)
        self.interest_amounts = np.asarray(interest_amounts)
        self.calendar = calendar
        self.credit_curve = credit_curve

    def get_outstanding(self, dates):
        dates = afsfun.dates_formatting(dates)
        outstanding = pd.Series(index=dates, name="Outstanding amount")
        for date in dates:
            outstanding.loc[date] = np.sum(
                self.principal_amounts[self.pay_dates[1:] > date]
            )
        return outstanding

    def get_px(self, dates, discount_curve, percentage=True, clean=True):
        dates = afsfun.dates_formatting(dates)
        px = pd.Series(index=dates, name="Price")
        for date in dates:
            boolean = self.pay_dates[1:] > date
            principal = self.principal_amounts[boolean]
            interest = self.interest_amounts[boolean]
            future_pay_dates = self.pay_dates[self.pay_dates > date]
            # accrued coupon:
            previous_date = self.pay_dates[self.pay_dates <= date][-1]
            if clean:
                d = self.calendar.interval(
                    previous_date, date
                ) / self.calendar.interval(previous_date, future_pay_dates[0])
                interest[0] = interest[0] * (1 - d)
            rf_discount = discount_curve.get_value(
                date, future_pay_dates, self.calendar
            )
            credit_discount = self.credit_curve.get_value(
                date, future_pay_dates, self.calendar
            )
            spread_discount = np.exp(
                -self.market_spread * self.calendar.interval(date, future_pay_dates)
            )

            # zero recovery
            zero_recovery = np.sum(
                spread_discount * credit_discount * rf_discount * (principal + interest)
            )

            # Recovery
            if hasattr(self.credit_curve, "recovery"):
                R = self.credit_curve.recovery
                Q = np.ones(future_pay_dates.size + 1)
                Q[1:] = self.credit_curve.get_value(
                    date, future_pay_dates, self.calendar
                )
                differences = Q[:-1] - Q[1:]
                rec_principals = np.zeros(principal.size)
                for i in range(principal.size):
                    rec_principals[i] = np.sum(principal[i:])
                interest_recovery = R * np.sum(
                    (rec_principals + interest / 2) * rf_discount * differences
                )
                recovery = interest_recovery
            else:
                recovery = 0

            # price
            price = zero_recovery + recovery
            if percentage:
                price = price / np.sum(principal) * 100
            px.loc[date] = price
        return px


class FixedCouponSinkable(Sinkable):
    def __init__(
        self,
        effective_date,
        maturity,
        freq,
        coupon_rate,
        principal_amounts,
        calendar,
        credit_curve=zero_credit_risk,
        rating=None,
        market_spread=0,
        repays_nominal=True,
    ):
        self.coupon_rate = coupon_rate
        self.nominal = np.sum(np.asarray(principal_amounts))
        Sinkable.__init__(
            self,
            effective_date,
            maturity,
            freq,
            principal_amounts,
            0,
            calendar,
            credit_curve=credit_curve,
            rating=rating,
            market_spread=market_spread,
        )
        if not repays_nominal:
            principal_amounts = np.zeros(self.pay_dates.size - 1)
        if np.asarray(principal_amounts).shape == ():
            principal_amounts = np.zeros(self.pay_dates.size - 1)
            principal_amounts[-1] = self.principal_amounts
        self.principal_amounts = principal_amounts
        interest_amounts = np.zeros(self.pay_dates.size - 1)
        deltas = self.calendar.interval(self.pay_dates[:-1], self.pay_dates[1:])
        for i in range(self.pay_dates.size - 1):
            interest_amounts[i] = (
                deltas[i] * self.coupon_rate * np.sum(self.principal_amounts[i:])
            )
        self.interest_amounts = interest_amounts


class FloatingCouponSinkable(Sinkable):
    def __init__(
        self,
        effective_date,
        maturity,
        freq,
        coupon_curve,
        spread,
        principal_amounts,
        calendar,
        floor=None,
        cap=None,
        credit_curve=zero_credit_risk,
        rating=None,
        market_spread=0,
        repays_nominal=True,
    ):
        self.coupon_curve = coupon_curve
        self.spread = spread
        self.floor = floor
        self.cap = cap
        self.nominal = np.sum(np.asarray(principal_amounts))
        Sinkable.__init__(
            self,
            effective_date,
            maturity,
            freq,
            principal_amounts,
            0,
            calendar,
            credit_curve=credit_curve,
            rating=rating,
            market_spread=market_spread,
        )
        if not repays_nominal:
            principal_amounts = np.zeros(self.pay_dates.size - 1)
        if np.asarray(principal_amounts).shape == ():
            principal_amounts = np.zeros(self.pay_dates.size - 1)
            principal_amounts[-1] = self.principal_amounts
        self.principal_amounts = principal_amounts

    def get_px(self):
        # compute coupons
        pass


class CouponBond(Bond):
    def __init__(
        self,
        effective_date,
        maturity,
        freq,
        calendar,
        credit_curve=zero_credit_risk,
        nominal=100,
        rating=None,
        repays_nominal=True,
        market_spread=0,
    ):
        self.nominal = nominal
        self.effective_date = effective_date
        self.maturity = maturity
        self.freq = freq
        self.coupon_dates = pd.date_range(
            self.effective_date, self.maturity, freq=pd.DateOffset(months=self.freq)
        )
        self.credit_curve = credit_curve
        self.calendar = calendar
        self.repays_nominal = repays_nominal
        Bond.__init__(
            self, effective_date, maturity, rating=rating, market_spread=market_spread
        )

    def get_px(
        self, dates, discount_curve, coupon_rates, bounds=False, with_recovery=True
    ):
        dates = afsfun.dates_formatting(dates)

        deltas_coupons = self.calendar.interval(
            self.coupon_dates[:-1], self.coupon_dates[1:]
        )
        deltas_coupons = deltas_coupons.reshape(deltas_coupons.size, 1)
        if np.asarray(coupon_rates).size != 1:
            coupon_rates = coupon_rates.reshape(deltas_coupons.size, 1)

        Z = np.full((self.coupon_dates.size, dates.size), np.nan)
        Q = np.full((self.coupon_dates.size, dates.size), np.nan)
        deltas_dates = np.zeros((self.coupon_dates.size, dates.size))
        for i in range(dates.size):
            Z[:, i] = discount_curve.get_value(
                dates[i], self.coupon_dates, self.calendar
            )
            Q[:, i] = self.credit_curve.get_value(
                dates[i], self.coupon_dates, self.calendar
            )
            deltas_dates[dates[i] <= self.coupon_dates, i] = self.calendar.interval(
                dates[i], self.coupon_dates
            )[dates[i] <= self.coupon_dates]

        # the following change in deltas takes care of the accrued coupon
        deltas = deltas_dates[1:] - deltas_dates[:-1]
        zero_recovery = (
            np.sum(coupon_rates * deltas * Z[1:] * Q[1:], axis=0)
            + Z[-1] * Q[-1] * self.repays_nominal
        )

        if hasattr(self.credit_curve, "recovery"):
            # multiplying by boolean removes the difference between first premium date, and previous premium date
            # (recall curve gives 1 for past dates)
            differences = (Q[:-1] - Q[1:]) * (Q[:-1] < 1)
            R = self.credit_curve.recovery
            recovery = R * np.sum(
                (self.repays_nominal + coupon_rates * deltas / 2) * Z[1:] * differences,
                axis=0,
            )
        else:
            recovery = 0

        prices = pd.Series(zero_recovery + recovery * with_recovery, index=dates)
        if bounds:
            prices = pd.DataFrame(prices, columns=["Price"])
            prices["Zero Recovery"] = zero_recovery
            prices["No credit risk"] = (
                np.sum(coupon_rates * deltas * Z[1:], axis=0)
                + Z[-1] * self.repays_nominal
            )
            prices = prices[["Zero Recovery", "Price", "No credit risk"]]
        return self.nominal * prices


class FixedCoupon(CouponBond):
    def __init__(
        self,
        effective_date,
        maturity,
        freq,
        coupon_rate,
        calendar,
        credit_curve=zero_credit_risk,
        nominal=100,
        rating=None,
        market_spread=0,
        repays_nominal=True,
    ):
        """
        freq (coupon frequency) should be number of months
        """
        CouponBond.__init__(
            self,
            effective_date,
            maturity,
            freq,
            calendar,
            credit_curve=credit_curve,
            nominal=nominal,
            rating=rating,
            market_spread=market_spread,
            repays_nominal=repays_nominal,
        )
        self.coupon_rate = coupon_rate

    def get_px(self, dates, discount_curve, bounds=False):
        prices = CouponBond.get_px(
            self, dates, discount_curve, self.coupon_rate + self.market_spread, bounds
        )
        return prices


class FloatingCoupon(CouponBond):
    def __init__(
        self,
        effective_date,
        maturity,
        freq,
        coupon_curve,
        calendar,
        spread=0,
        floor=None,
        cap=None,
        rating=None,
        market_spread=0,
        credit_curve=zero_credit_risk,
        nominal=100,
        repays_nominal=True,
    ):
        CouponBond.__init__(
            self,
            effective_date,
            maturity,
            freq,
            calendar,
            credit_curve=credit_curve,
            nominal=nominal,
            rating=rating,
            repays_nominal=repays_nominal,
            market_spread=market_spread,
        )
        self.coupon_curve = coupon_curve
        self.spread = spread
        self.floor = floor
        self.cap = cap

    def get_px(self, dates, discount_curve, bounds=False):
        dates = afsfun.dates_formatting(dates)

        # frates gives the rates on all coupon dates, but CouponBond kills them where appropriate
        frates = np.zeros((dates.size, self.coupon_dates.size - 1))
        for i in range(dates.size):
            if np.min(dates[i] >= self.coupon_dates) == 0:
                n = np.argmin(dates[i] >= self.coupon_dates)
                frates[i, n - 1 :] = self.coupon_curve.get_fvalue(
                    dates[i], self.coupon_dates[n - 1 : -1]
                )
        coupon_rates = frates + self.spread
        if self.floor is not None:
            coupon_rates = coupon_rates * (coupon_rates >= self.floor) + self.floor * (
                coupon_rates < self.floor
            )
        if self.cap is not None:
            coupon_rates = coupon_rates * (coupon_rates <= self.cap) + self.cap * (
                coupon_rates > self.cap
            )
        prices = CouponBond.get_px(self, dates, discount_curve, coupon_rates, bounds)
        return prices


class MBS(Bond):
    def __init__(
        self,
        effective_date,
        maturity,
        freq,
        calendar,
        market_spread=0,
        fee_rate=0,
        rating=None,
    ):
        Bond.__init__(
            self, effective_date, maturity, market_spread=market_spread, rating=rating
        )
        self.freq = freq
        self.coupon_dates = pd.date_range(
            self.effective_date, self.maturity, freq=pd.DateOffset(months=self.freq)
        )
        self.calendar = calendar
        self.fee_rate = fee_rate
        self.data = pd.DataFrame(
            columns=[
                "AMT_OUTSTANDING",
                "MTG_WAM",
                "MTG_WACPN",
                "MTG_EQV_CPR",
                "DELINQUENCIES_OVER_30_DAYS_%",
            ]
        )
        # set default value 1 when querying key not in dictionary
        self.ratings_dic = {
            "Aaa": 0,
            "AAA": 0,
            "Aa1": 0.1,
            "AA+": 0.1,
            "Aa2": 0.2,
            "AA": 0.2,
            "Aa3": 0.3,
            "AA-": 0.3,
            "A1": 0.4,
            "A+": 0.4,
            "A2": 0.5,
            "A": 0.5,
            "A3": 0.6,
            "A-": 0.6,
            "Baa1": 0.7,
            "BBB+": 0.7,
            "Baa2": 0.8,
            "BBB": 0.8,
            "Baa3": 0.9,
            "BBB-": 0.9,
        }
        self.default_severity = 0.5

    def import_data(self, data_frame):
        if "RATING" in data_frame.columns:
            data_frame = data_frame[
                [
                    "AMT_OUTSTANDING",
                    "MTG_WAM",
                    "MTG_WACPN",
                    "MTG_EQV_CPR",
                    "DELINQUENCIES_OVER_30_DAYS_%",
                    "RATING",
                ]
            ]
        else:
            data_frame = data_frame[
                [
                    "AMT_OUTSTANDING",
                    "MTG_WAM",
                    "MTG_WACPN",
                    "MTG_EQV_CPR",
                    "DELINQUENCIES_OVER_30_DAYS_%",
                ]
            ]
        data_frame[["MTG_WACPN", "MTG_EQV_CPR", "DELINQUENCIES_OVER_30_DAYS_%"]] = (
            data_frame[["MTG_WACPN", "MTG_EQV_CPR", "DELINQUENCIES_OVER_30_DAYS_%"]]
            / 100
        )
        for date in data_frame.index:
            self.data.loc[date] = data_frame.loc[date]

    def get_px(self, dates, discount_curve):
        
        dates = afsfun.dates_formatting(dates)
        prices = pd.Series(index=dates, dtype="float64")
        for date in dates:
            near_dates_with_data = self.data.index[
                self.data.index >= date - pd.DateOffset(months=1)
            ]
            near_dates_with_data = near_dates_with_data[
                near_dates_with_data <= date + pd.DateOffset(months=1)
            ]
            if near_dates_with_data.size == 0:
                print("No data near enough to date {}".format(date))
                pass
            elif near_dates_with_data.size == 1:
                date_data = near_dates_with_data[0]
            else:
                date_data = near_dates_with_data[near_dates_with_data >= date][0]
            amt_outstanding = self.data["AMT_OUTSTANDING"].loc[date_data]
            if amt_outstanding == 0:
                prices.loc[date] = 0
            else:
                wam = self.data["MTG_WAM"].loc[date_data]
                wac_period = self.data["MTG_WACPN"].loc[date_data] * 12 / self.freq
                delinquency = self.data["DELINQUENCIES_OVER_30_DAYS_%"].loc[date_data]
                CPR = self.data["MTG_EQV_CPR"].loc[date_data]
                spm = 1 - (1 - CPR) ** (self.freq / 12)
                if "RATING" in self.data.columns:
                    severity = self.ratings_dic.get(
                        self.data["RATING"].loc[date_data], 1
                    )
                elif self.rating is not None:
                    severity = self.ratings_dic.get(self.rating, 1)
                else:
                    print("Using default severity")
                    severity = self.default_severity
                # summing 1 gives ceiling and not floor
                wam_periods = ceil(wam / self.freq)
                future_coupon_dates = self.coupon_dates[self.coupon_dates > date]
                no_periods = np.min([wam_periods, future_coupon_dates.size])
                periods_vector = no_periods - np.arange(no_periods)
                future_coupon_dates = future_coupon_dates[:no_periods]
                # compute principal payments
                balance = np.ones(no_periods + 1)
                balance[0] = amt_outstanding
                scheduled_amortization = (
                    amt_outstanding
                    * wac_period
                    * (1 + wac_period) ** (-periods_vector)
                    / (1 - (1 + wac_period) ** (-no_periods))
                )

                loss = delinquency * scheduled_amortization
                recovery = np.zeros(no_periods)
                if delinquency != 0:
                    for i in range(no_periods - 1):
                        recovery_amount = (1 - severity) * loss[i]
                        no_rp = no_periods - i - 1
                        rp_vector = no_rp - np.arange(no_rp)
                        recovery_amounts = (
                            recovery_amount
                            * wac_period
                            * (1 + wac_period) ** (-rp_vector)
                            / (1 - (1 + wac_period) ** (-no_rp))
                        )
                        recovery[i + 1 :] = recovery[i + 1 :] + np.flip(
                            recovery_amounts
                        )

                prepayments = np.zeros(no_periods)
                if spm != 0:
                    for i in range(no_periods):
                        prepayments[i] = (balance[i] - scheduled_amortization[i]) * spm
                        no_rp = no_periods - i - 1
                        rp_vector = no_rp - np.arange(no_rp)
                        lost_from_prepay = (
                            prepayments[i]
                            * wac_period
                            * (1 + wac_period) ** (-rp_vector)
                            / (1 - (1 + wac_period) ** (-no_rp))
                        )
                        scheduled_amortization[i + 1 :] = (
                            scheduled_amortization[i + 1 :] - lost_from_prepay
                        )
                        balance[i + 1] = (
                            balance[i] - scheduled_amortization[i] - prepayments[i]
                        )
                else:
                    balance[1:] = amt_outstanding - np.array(
                        list(accumulate(scheduled_amortization))
                    )

                coupon_calc_dates = self.coupon_dates[
                    self.coupon_dates > date - pd.DateOffset(months=self.freq)
                ][:-1]
                coupon_calc_dates = coupon_calc_dates[: periods_vector.size]
                coupon_rates = self.get_coupon_rates(date, coupon_calc_dates)
                # length of future_coupon_dates was already adjusted to no_periods above
                deltas = self.calendar.interval(date, future_coupon_dates)
                if deltas.size > 1:
                    deltas[1:] = deltas[1:] - deltas[:-1]
                coupons = deltas * coupon_rates
                interest = coupons * balance[:-1]

                # compute fees
                fees = self.fee_rate * 12 / self.freq * scheduled_amortization

                # compute discounted cashflows and price
                rf_discount = discount_curve.get_value(
                    date, future_coupon_dates, self.calendar
                )

                # print("Principal", 100 * np.sum(rf_discount * scheduled_amortization) / amt_outstanding)
                # print("Prepayments", 100 * np.sum(rf_discount * prepayments) / amt_outstanding)
                # print("Losses", 100 * np.sum(rf_discount * loss) / amt_outstanding)
                # print("Recovery", 100 * np.sum(rf_discount * recovery) / amt_outstanding)
                # print("Interest", 100 * np.sum(rf_discount * interest) / amt_outstanding)

                spread_discount = np.exp(
                    -self.market_spread
                    * self.calendar.interval(date, future_coupon_dates)
                )
                cash_flows = (
                    interest
                    + scheduled_amortization
                    + prepayments
                    - loss
                    + recovery
                    - fees
                )
                cash_flows = spread_discount * rf_discount * cash_flows
                prices.loc[date] = 100 * np.sum(cash_flows) / amt_outstanding
        return prices


class FloatingCouponMBS(MBS):
    def __init__(
        self,
        effective_date,
        maturity,
        freq,
        coupon_curve,
        spread,
        calendar,
        market_spread=0,
        floor=0,
        cap=None,
        fee_rate=0,
        rating=None,
    ):
        MBS.__init__(
            self,
            effective_date,
            maturity,
            freq,
            calendar,
            market_spread=market_spread,
            fee_rate=fee_rate,
            rating=rating,
        )
        self.coupon_curve = coupon_curve
        self.spread = spread
        self.floor = floor
        self.cap = cap

    def get_coupon_rates(self, date, coupon_calc_dates):
        # frates gives rates for all coupon dates (past dates are actual rates, not forward)
        frates = self.coupon_curve.get_fvalue(date, coupon_calc_dates, self.calendar)
        frates = frates.reshape(
            frates.size,
        )
        coupon_rates = frates + self.spread
        if self.floor is not None:
            coupon_rates = coupon_rates * (coupon_rates >= self.floor) + self.floor * (
                coupon_rates < self.floor
            )
        if self.cap is not None:
            coupon_rates = coupon_rates * (coupon_rates <= self.cap) + self.cap * (
                coupon_rates > self.cap
            )
        return coupon_rates


class FixedCouponMBS(MBS):
    def __init__(
        self,
        effective_date,
        maturity,
        freq,
        coupon_rate,
        calendar,
        market_spread=0,
        fee_rate=0,
        rating=None,
    ):
        MBS.__init__(
            self,
            effective_date,
            maturity,
            freq,
            calendar,
            market_spread=market_spread,
            fee_rate=fee_rate,
            rating=rating,
        )
        self.coupon_rate = coupon_rate

    def get_coupon_rates(self, date, coupon_calc_dates):
        """
        Get coupon rates method.

        Warnings
        --------
        - Old, undocumented and unchecked module, review and correct before using. 

        """
        return self.coupon_rate


# ----------------------------------------------------------------------------------------------------------------------
# MARF
# ----------------------------------------------------------------------------------------------------------------------


class MARF(Bond):
    """
    MARF class.

    Warnings
    --------
    - This class is empty.
    """

    def __init__(
        self,
        effective_date,
        maturity,
        freq,
        coupon_rate,
        calendar,
        issued_amount,
        rating,
        market_spread=0,
    ):
        pass
