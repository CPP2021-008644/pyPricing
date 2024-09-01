import numpy as np
import pandas as pd

try:
    from .structured import (
        Derivative,
        MCProduct,
        QuantoMultiAsset,
        Vanilla,
    )  # (Relative) import needed for the workspace. In this case __package__ is 'pypricing.pricing'
except (ImportError, ModuleNotFoundError, ValueError):
    from pricing.structured import (
        Derivative,
        MCProduct,
        QuantoMultiAsset,
        Vanilla,
    )  # (Absolute) local import
try:
    from .discount_curves import CRDC
except (ImportError, ModuleNotFoundError, ValueError):
    from pricing.discount_curves import CRDC
try:
    from . import functions as afsfun
except (ImportError, ModuleNotFoundError, ValueError):
    import pricing.functions as afsfun

zero_credit_risk = CRDC(0)


# class GeneralCallable(Derivative, MCProduct):
#     def __init__(
#         self,
#         underlyings,
#         barriers,
#         strikes,
#         fixed_rate,
#         fixed_pay_dates,
#         rate_curve,
#         spread,
#         floating_pay_dates,
#         call_amounts,
#         call_dates,
#         calendar,
#         ty,
#         nominal=100,
#         repays_nominal=True,
#         credit_curve=zero_credit_risk,
#     ):
#         # We always assume the product pays the natural floating rate
#         # (as in 3mo rate for quarterly payments)
#         if np.asarray(barriers).ndim == 0:
#             barriers = [barriers]
#         self.barrier = np.array(barriers)
#         if np.asarray(strikes).ndim == 0:
#             strikes = [strikes]
#         self.strike = np.array(strikes)
#         if np.asarray(fixed_pay_dates).ndim == 0:
#             fixed_pay_dates = [fixed_pay_dates]
#         self.fixed_dates = pd.to_datetime(fixed_pay_dates).sort_values()
#         self.effective_date = self.fixed_dates[0]
#         self.fixed_taus = calendar.interval(self.fixed_dates[:-1], self.fixed_dates[1:])
#         floating_pay_dates = afsfun.dates_formatting(floating_pay_dates)
#         self.floating_taus = calendar.interval(
#             self.floating_dates[:-1], self.floating_dates[1:]
#         )
#         self.pay_dates = pd.DatetimeIndex.union(
#             self.floating_dates, self.fixed_dates
#         ).sort_values()
#         self.fixed_rate = np.asarray(fixed_rate) * np.ones(self.fixed_dates.size)
#         self.spread = np.asarray(spread) * np.ones(self.floating_dates.size)
#         call_dates = afsfun.dates_formatting(call_dates)
#         self.call_amounts = np.asarray(call_amounts) * np.ones(self.call_dates.size)
#
#         obsdates = pd.DatetimeIndex.union(self.call_dates, [self.pay_dates[-1]])
#         self.nominal = nominal
#         self.repays_nominal = repays_nominal
#
#         def payoff(price, dates_dic, *others):
#             index = dates_dic[self.pay_dates[-1]]
#             price = price[:, index]
#             # CAREFUL: in Callable we go long a call, but in Reverse, we go SHORT a put (hence a + both times)
#             # print(price)
#             payoffs = (
#                 float(self.repays_nominal)
#                 + (price / self.strike - 1) * (price >= self.barrier) * (ty == "call")
#                 + (price / self.strike - 1) * (price <= self.barrier) * (ty == "put")
#             )
#             # print(payoffs)
#             payoffs = np.min(payoffs, axis=2)
#             # print(payoffs)
#             return payoffs
#
#         if type(underlyings) != list:
#             underlyings = [underlyings]
#         Derivative.__init__(
#             self, underlyings, obsdates, calendar, payoff, credit_curve=credit_curve
#         )
#
#         def early_exercise(continuation_values, n, discounts):
#             boolean = (self.fixed_dates > self.obsdates[n]) & (
#                 self.fixed_dates <= self.obsdates[n + 1]
#             )
#             fixed_dates = self.fixed_dates[boolean]
#             fixed_taus = self.fixed_taus[boolean[1:]]
#             fixed_amounts = self.fixed_rate[boolean] * fixed_taus
#             pay_dates = self.pay_dates[
#                 (self.pay_dates > self.obsdates[n])
#                 & (self.pay_dates <= self.obsdates[n + 1])
#             ]
#             boolean_fixed = pay_dates.isin(fixed_dates)
#             boolean = (self.floating_dates > self.obsdates[n]) & (
#                 self.floating_dates <= self.obsdates[n + 1]
#             )
#             floating_dates = self.floating_dates[boolean]
#             boolean_floating = pay_dates.isin(floating_dates)
#             if discounts.ndim == 2:
#                 fixed_discounts = discounts[boolean_fixed]
#                 floating_discounts = discounts[boolean_floating]
#             else:
#                 fixed_discounts = discounts[:, boolean_fixed]
#                 floating_discounts = discounts[:, boolean_floating]
#             fixed_amounts = fixed_discounts * fixed_amounts.reshape(
#                 (fixed_amounts.size, 1)
#             )
#             if floating_discounts.size > 0:
#                 shape = (1,) + discounts.shape[1:]
#                 floating_amounts = (
#                     np.concatenate((np.ones(shape), floating_discounts[:-1]))
#                     - floating_discounts
#                 )
#                 # if floating_amounts.ndim == 3:
#                 #     to_plot = floating_amounts.reshape((floating_amounts.shape[0], floating_amounts.shape[-1], floating_amounts.shape[-2]))
#                 #     for array in to_plot:
#                 #         plt.plot(array[0])
#                 boolean = (self.floating_dates > self.obsdates[n]) & (
#                     self.floating_dates <= self.obsdates[n + 1]
#                 )
#                 floating_taus = self.floating_taus[boolean[1:]]
#                 spread_amounts = (self.spread[boolean] * floating_taus).reshape(
#                     (floating_taus.size, 1)
#                 ) * floating_discounts
#             else:
#                 floating_amounts = np.zeros(fixed_amounts.shape)
#                 spread_amounts = np.zeros(fixed_amounts.shape)
#             if discounts.ndim == 2:
#                 intermediate_cash_flows = np.sum(
#                     spread_amounts + floating_amounts, axis=0
#                 ) + np.sum(fixed_amounts, axis=0)
#             else:
#                 intermediate_cash_flows = np.sum(
#                     spread_amounts + floating_amounts, axis=1
#                 ) + np.sum(fixed_amounts, axis=1)
#             # print(intermediate_cash_flows)
#             continuation = continuation_values + intermediate_cash_flows
#             if ty == "put":
#                 early_exercise_amount = np.minimum(continuation, self.call_amounts[n])
#             else:
#                 early_exercise_amount = np.maximum(continuation, self.call_amounts[n])
#             # print(early_exercise_amount)
#             return self.nominal * early_exercise_amount
#
#         self.early_exercise = early_exercise
#
#         def remaining_coupon(dates, discounts):
#             next_call = self.obsdates[self.obsdates > dates[-1]][0]
#             previous_call = self.call_dates[self.call_dates <= dates[0]]
#             if previous_call.size != 0:
#                 previous_date = previous_call[-1]
#             else:
#                 previous_date = self.effective_date
#             coupon_amounts = np.full(dates.size, np.nan)
#             for i in range(dates.size):
#                 boolean = (self.fixed_dates > dates[i]) & (
#                     self.fixed_dates <= next_call
#                 )
#                 fixed_dates = self.fixed_dates[boolean]
#                 fixed_taus = self.fixed_taus[boolean[1:]]
#                 fixed_taus[0] = self.calendar.interval(dates[i], fixed_dates[0])
#                 fixed_amounts = self.fixed_rate[boolean] * fixed_taus
#                 pay_dates = self.pay_dates[
#                     (self.pay_dates > previous_date) & (self.pay_dates <= next_call)
#                 ]
#                 fixed_discounts = discounts[:, i][pay_dates.isin(fixed_dates)]
#                 fixed_amounts = fixed_discounts * fixed_amounts
#                 boolean = (self.floating_dates > dates[i]) & (
#                     self.floating_dates <= next_call
#                 )
#                 floating_dates = self.floating_dates[boolean]
#                 floating_discounts = discounts[:, i][pay_dates.isin(floating_dates)]
#                 if floating_discounts.size > 0:
#                     floating_amounts = (
#                         np.concatenate(([1], floating_discounts[:-1]))
#                         - floating_discounts
#                     )
#                     floating_taus = self.floating_taus[boolean[1:]]
#                     # print(floating_amounts/(floating_taus*floating_discounts))
#                     unaccrued_factor = (
#                         self.calendar.interval(dates[i], floating_dates[0])
#                         / floating_taus[0]
#                     )
#                     floating_amounts[0] = unaccrued_factor * floating_amounts[0]
#                     floating_taus[0] = unaccrued_factor * floating_taus[0]
#                     spread_amounts = (
#                         self.spread[boolean] * floating_taus * floating_discounts
#                     )
#                 else:
#                     floating_amounts = 0
#                     spread_amounts = 0
#                 coupon_amounts[i] = np.sum(spread_amounts + floating_amounts) + np.sum(
#                     fixed_amounts
#                 )
#             return coupon_amounts
#
#         self.remaining_coupons = remaining_coupon
#
#
# class CallableFRConvertible(GeneralCallable):
#     def __init__(
#         self,
#         underlyings,
#         barriers,
#         strikes,
#         fixed_rate,
#         pay_dates,
#         call_amounts,
#         call_dates,
#         calendar,
#         nominal=100,
#         credit_curve=zero_credit_risk,
#     ):
#         GeneralCallable.__init__(
#             self,
#             underlyings,
#             barriers,
#             strikes,
#             fixed_rate,
#             pay_dates,
#             None,
#             0,
#             [],
#             call_amounts,
#             call_dates,
#             calendar,
#             "call",
#             nominal=nominal,
#             repays_nominal=True,
#             credit_curve=credit_curve,
#         )
#
#
# class CallableFRReverseConvertible(GeneralCallable):
#     def __init__(
#         self,
#         underlyings,
#         barriers,
#         strikes,
#         fixed_rate,
#         pay_dates,
#         call_amounts,
#         call_dates,
#         calendar,
#         nominal=100,
#         credit_curve=zero_credit_risk,
#     ):
#         GeneralCallable.__init__(
#             self,
#             underlyings,
#             barriers,
#             strikes,
#             fixed_rate,
#             pay_dates,
#             None,
#             0,
#             [],
#             call_amounts,
#             call_dates,
#             calendar,
#             "put",
#             nominal=nominal,
#             repays_nominal=True,
#             credit_curve=credit_curve,
#         )
#
#
# class QuantoCallableConvertible(CallableFRConvertible, QuantoMultiAsset):
#     def __int__(
#         self,
#         underlyings,
#         exchange_rate,
#         barriers,
#         strikes,
#         fixed_rate,
#         pay_dates,
#         call_amounts,
#         call_dates,
#         calendar,
#         nominal=100,
#         credit_curve=zero_credit_risk,
#     ):
#         CallableFRConvertible.__init__(
#             self,
#             underlyings,
#             barriers,
#             strikes,
#             fixed_rate,
#             pay_dates,
#             call_amounts,
#             call_dates,
#             calendar,
#             nominal=nominal,
#             credit_curve=credit_curve,
#         )
#         QuantoMultiAsset.__init__(self, underlyings, exchange_rate)
#
#
# class QuantoCallableReverseConvertible(CallableFRReverseConvertible, QuantoMultiAsset):
#     def __init__(
#         self,
#         underlyings,
#         xchange_rate,
#         barriers,
#         strikes,
#         fixed_rate,
#         pay_dates,
#         call_amounts,
#         call_dates,
#         calendar,
#         nominal=100,
#         credit_curve=zero_credit_risk,
#     ):
#         CallableFRReverseConvertible.__init__(
#             self,
#             underlyings,
#             barriers,
#             strikes,
#             fixed_rate,
#             pay_dates,
#             call_amounts,
#             call_dates,
#             calendar,
#             nominal=nominal,
#             credit_curve=credit_curve,
#         )
#         QuantoMultiAsset.__init__(self, underlyings, xchange_rate)
#
#
# class CancelableSwap(GeneralCallable):
#     def __init__(
#         self,
#         fixed_rate,
#         effective_date,
#         end_date,
#         fixed_freq,
#         floating_freq,
#         spread,
#         cancel_dates,
#         calendar,
#         nominal=100,
#         credit_curve=zero_credit_risk,
#     ):
#         fixed_dates = pd.date_range(
#             start=effective_date, end=end_date, freq=pd.DateOffset(months=fixed_freq)
#         )
#         floating_dates = pd.date_range(
#             start=effective_date, end=end_date, freq=pd.DateOffset(months=floating_freq)
#         )
#         GeneralCallable.__init__(
#             self,
#             [sx5e, spx],
#             np.inf,
#             1,
#             -fixed_rate,
#             fixed_dates,
#             None,
#             spread,
#             floating_dates,
#             0,
#             cancel_dates,
#             calendar,
#             "call",
#             nominal=nominal,
#             repays_nominal=False,
#             credit_curve=credit_curve,
#         )


# New version, old was wrong, see "Planning American-Bermudan style products"
class Callable:
    def __init__(self, underlying, pay_dates, calendar):
        self.underlying = underlying
        self.pay_dates = afsfun.dates_formatting(pay_dates)
        self.calendar = calendar


class AmericanFromEuropean(Callable):
    def __init__(self, underlying, pay_dates, calendar, eur_prod):
        Callable.__init__(
            self, underlying=underlying, pay_dates=pay_dates, calendar=calendar
        )
        self.maturity = pay_dates[-1]
        self.obsdates = pay_dates
        self.eur_prod = eur_prod
        self.pastmatters = self.eur_prod.pastmatters


class AmericanVanillaOption(Callable):
    def __init__(
        self,
        underlying,
        strike,
        pay_dates,
        kind,
        calendar,
        nominal=100.0,
        credit_curve=zero_credit_risk,
        maturity=None,
        phi="min",
    ):
        # We want the exercise value for all dates and not a payment series that, after discounting, are add up. It is better to rewrite the payoff method.
        if maturity is None:
            self.maturity = pay_dates[-1]
        self.obsdates = pd.to_datetime(pay_dates).sort_values()
        self.maturity = self.obsdates[-1]
        self.pastmatters = False  # For path dependent options, self.pastmatters = True, TODO: check this
        self.credit_curve = credit_curve
        self.strike = strike
        self.nominal = nominal
        self.kind = kind
        if phi == "min":
            self.phi = np.min
        elif phi == "max":
            self.phi = np.max
        else:
            raise ValueError("Value of self.phi must be 'min' or 'max'")
        Callable.__init__(
            self, underlying=underlying, pay_dates=pay_dates, calendar=calendar
        )

    def payoff(self, prices, dates_dic, n):
        if (
            prices.ndim == 4
        ):  # I.e., when prices.ndim == 4 (the older possibility was prices.ndim == 3)
            prices = self.phi(
                prices, axis=-1, keepdims=False
            )  # When the price depends on several underlyings it is defined as the minimum over them.
        payoffs = (prices - self.strike) * (prices >= self.strike) - (
            prices - self.strike
        ) * (self.kind == "put")
        payoffs_matrix = payoffs
        return self.nominal * payoffs_matrix
