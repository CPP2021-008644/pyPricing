import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import copy

try:
    from .structured import VanillaStrategy
except (ImportError, ModuleNotFoundError, ValueError):
    from pricing.structured import VanillaStrategy

try:
    from .functions import dates_formatting
except (ImportError, ModuleNotFoundError, ValueError):
    from pricing.functions import dates_formatting


class Strategy:
    def __init__(
        self, discount_curve, initial_underlying, initial_vanillas, initial_date
    ):
        self.discount_curve = discount_curve
        self.initial_underlying = initial_underlying
        self.initial_vanillas = initial_vanillas
        self.initial_date = dates_formatting(initial_date)[0]

    def simulate_pnl(
        self,
        time_series,
        rebalancing_rule,
        decompose_pnl=False,
        underlying_fees=0,
        intradata_time_series=None,
    ):
        time_series = time_series[time_series.index >= self.initial_date]
        options = self.initial_vanillas
        options = copy.deepcopy(options)
        underlying_shares = self.initial_underlying
        underlying_shares_series = pd.Series()
        delta_series = pd.Series()
        underlying_pnl_series = pd.Series()
        option_pnl_series = pd.Series()
        fees_pnl_series = pd.Series()
        underlying_pnl = 0
        option_pnl = 0
        acc_fees_pnl = 0
        previous_underlying = 0
        previous_underlying_last_time_data_option = 0
        previous_implied_volatilities = 0
        previous_option_premium = 0
        last_evaluation_strategy = copy.deepcopy(VanillaStrategy(options))
        last_evaluation_date = None
        acc_pnl_table = None
        daily_pnl = None
        pnl_underlying_intradata = 0
        fees_extra_intradata = 0
        for i, date in enumerate(time_series.index):
            if date > min([x.maturity for x in options]):
                break
            underlying = time_series.loc[date]["underlying"].tolist()
            implied_volatilities = (
                time_series.filter(like="implied volatility").loc[date].tolist()
            )
            dividends = time_series.filter(like="dividend").loc[date].tolist()
            strategy = VanillaStrategy(options)
            strategy.update_options(
                date, underlying, implied_volatilities, dividend_yield=dividends
            )
            option_premium = strategy.map_method(
                "get_px", date, self.discount_curve
            ).loc[date]["Total Strategy"]
            if i != 0:  # PnL update
                underlying_pnl += (
                    underlying_shares * (underlying - previous_underlying)
                    + pnl_underlying_intradata
                )
                underlying_pnl_series.loc[date] = underlying_pnl

                option_pnl += option_premium - previous_option_premium
                option_pnl_series.loc[date] = option_pnl

            (
                delta_pre_rebalance,
                rebalance_flag,
                new_underlying_shares,
                new_options,
            ) = rebalancing_rule.rebalance(
                date, underlying_shares, strategy, self.discount_curve
            )

            # We also take into account the fees to initialise the strategy
            fees_pnl = (
                -np.abs(new_underlying_shares - underlying_shares) * underlying_fees
            ) + fees_extra_intradata
            acc_fees_pnl += fees_pnl
            fees_pnl_series.loc[date] = acc_fees_pnl

            if decompose_pnl and i != 0:
                scenarios = {
                    "spot": np.array([underlying]),
                    "future_date": [date],
                    "implied_vol": np.array([implied_volatilities]),
                    "future_dividends": np.array([dividends]),
                }
                df_risk_pricing = last_evaluation_strategy.get_risk_matrix(
                    [last_evaluation_date],
                    self.discount_curve,
                    scenarios,
                    return_scenarios_greeks=False,
                )
                df_risk_pricing = df_risk_pricing.loc["Total Strategy"].reset_index(
                    drop=True
                )
                df_risk_pricing["Transaction Fees PnL"] = fees_pnl
                daily = copy.deepcopy(df_risk_pricing)
                daily.index = [date]
                daily["underlying PnL"] = (
                    underlying_shares * (underlying - previous_underlying)
                    + pnl_underlying_intradata
                )
                daily["vega"] = (
                    last_evaluation_strategy.map_method(
                        "get_vega", [last_evaluation_date], self.discount_curve
                    )["Total Strategy"].iloc[0]
                ) / 100
                daily["vega/delta"] = (
                    last_evaluation_strategy.map_method(
                        "get_vega", [last_evaluation_date], self.discount_curve
                    ).iloc[0]
                    / 100
                    / last_evaluation_strategy.map_method(
                        "get_delta", [last_evaluation_date], self.discount_curve
                    ).iloc[0]
                ).iloc[-1]
                daily["gamma"] = last_evaluation_strategy.map_method(
                    "get_gamma", [last_evaluation_date], self.discount_curve
                )["Total Strategy"][0]
                daily["spot_change"] = (
                    underlying - previous_underlying_last_time_data_option
                )
                daily["moneyness_change"] = (
                    options[0].strike / underlying
                    - options[0].strike / previous_underlying_last_time_data_option
                )
                daily["vol_change"] = list(
                    np.array(implied_volatilities)
                    - np.array(previous_implied_volatilities)
                )
                if daily_pnl is None:
                    daily_pnl = daily
                else:
                    daily_pnl = pd.concat([daily_pnl, daily], axis=0)

                if acc_pnl_table is None:
                    acc_pnl_table = df_risk_pricing
                else:
                    acc_pnl_table = acc_pnl_table.add(df_risk_pricing)

            if rebalance_flag:
                options = new_options
                underlying_shares = new_underlying_shares
            else:
                options = strategy.vanillas

            if decompose_pnl:
                last_evaluation_strategy = copy.deepcopy(VanillaStrategy(options))
                last_evaluation_date = date

            underlying_shares_series.loc[date] = underlying_shares
            delta_series.loc[date] = delta_pre_rebalance
            previous_underlying = underlying
            previous_underlying_last_time_data_option = underlying
            previous_implied_volatilities = implied_volatilities
            previous_option_premium = (
                VanillaStrategy(options)
                .map_method("get_px", date, self.discount_curve)
                .loc[date]["Total Strategy"]
            )

            # This is to provide underlying time series intra option data
            if intradata_time_series is not None:
                # Parameters for first loop
                date_to_start = date
                delta_start_options = copy.deepcopy(
                    VanillaStrategy(options)
                ).map_method("get_delta", [date], self.discount_curve)[
                    "Total Strategy"
                ][
                    0
                ]
                gamma_start = copy.deepcopy(VanillaStrategy(options)).map_method(
                    "get_gamma", [date], self.discount_curve
                )["Total Strategy"][0]
                underlying_shares_to_start = underlying_shares
                delta_start = delta_start_options + underlying_shares_to_start
                spot_to_start = underlying
                delta_threshold = rebalancing_rule.delta_threshold
                pnl_underlying_intradata = 0
                fees_extra_intradata = 0
                while True:
                    if i + 1 < len(time_series.index):
                        future_date = time_series.index[i + 1]
                        intradata_filter = intradata_time_series[
                            (intradata_time_series.index >= date_to_start)
                            & (intradata_time_series.index <= future_date)
                        ]
                    else:
                        intradata_filter = intradata_time_series[
                            (intradata_time_series.index >= date_to_start)
                        ]

                    spot_delta_0 = spot_to_start - delta_start / gamma_start
                    threshold_relative = np.abs(delta_threshold / gamma_start)
                    indices = np.where(
                        np.abs(np.array(intradata_filter) - spot_delta_0)
                        > threshold_relative
                    )[0]
                    if len(indices) > 0:
                        first_index = indices[0]
                        first_value = intradata_filter.iloc[first_index].iloc[0]
                    else:
                        break

                    pnl_underlying_intradata += underlying_shares_to_start * (
                        first_value - spot_to_start
                    )
                    delta_value = (
                        (first_value - spot_delta_0)
                        / threshold_relative
                        * np.sign(gamma_start)
                    )
                    N_shares_change = -np.round(delta_value)
                    underlying_shares_to_start += N_shares_change

                    if np.abs((first_value - spot_delta_0) / threshold_relative) < 1:
                        raise ValueError(
                            "There is an error, check why we did not reach threshold despite being here."
                        )
                    fees_extra_intradata += -np.abs(N_shares_change) * underlying_fees
                    date_to_start = intradata_filter.index[first_index]
                    spot_to_start = first_value
                    delta_start = delta_value + N_shares_change

                    if np.abs(delta_start) > 1:
                        raise ValueError(
                            "There is an error, check why our initial_delta is so big."
                        )

                previous_underlying = spot_to_start
                underlying_shares = underlying_shares_to_start

        if decompose_pnl:
            # It takes the sum of all changes in the option, and add the final pnl of the underlying
            # this underlying PnL already takes into account the intradata case
            acc_pnl_table["Underlying PnL"] = underlying_pnl
            acc_pnl_table["Option PnL"] = acc_pnl_table["Total PnL"]
            acc_pnl_table["Total PnL"] = (
                acc_pnl_table["Option PnL"]
                + acc_pnl_table["Underlying PnL"]
                + acc_pnl_table["Transaction Fees PnL"]
            )
            acc_pnl_table["Discrete PnL Error (%)"] = (
                acc_pnl_table["Discrete PnL Error"] / acc_pnl_table["Option PnL"] * 100
            )
            return (
                underlying_shares_series,
                fees_pnl_series,
                underlying_pnl_series,
                option_pnl_series,
                delta_series,
                acc_pnl_table,
                daily_pnl,
            )

        return (
            underlying_shares_series,
            fees_pnl_series,
            underlying_pnl_series,
            option_pnl_series,
            delta_series,
        )


class DeltaHedging:
    def __init__(self, delta_threshold):
        self.delta_threshold = delta_threshold

    def rebalance(self, date, underlying_shares, strategy, discount_curve):
        delta_options = strategy.map_method("get_delta", date, discount_curve).loc[
            date
        ]["Total Strategy"]
        delta = delta_options + underlying_shares

        if abs(delta) > self.delta_threshold:
            rebalance_flag = True
            new_underlying_shares = -np.round(delta_options)
            return delta, rebalance_flag, new_underlying_shares, strategy.vanillas
        else:
            return delta, False, underlying_shares, strategy.vanillas
