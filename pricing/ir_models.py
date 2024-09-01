import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy import optimize
from scipy.integrate import quadrature, quad
import plotly.express as px

try:
    from . import functions as afsfun
except (ImportError, ModuleNotFoundError, ValueError):
    import pricing.functions as afsfun

# -----------------------------------------------------------------------
# short rate
# -----------------------------------------------------------------------


class ShortRateModel:
    """
    An abstract base class for short rate models used in financial simulations.

    This class provides a common interface and basic functionality for different types of short rate models. It should be subclassed to implement specific short rate model behaviors.

    Parameters
    ----------
    spot_curve : pricing.discount_curves.DiscountCurve
        Discount curve observed from market data, i.e., :math:`T\\to P^M(0, T) \\text{ for }T>0.`

    calendar : data.calendars.DayCountCalendar
            Specifies the calendar used for computation. In particular, it gives :math:`\\text{DC}(t, T)` or :math:`\\tau(t,T)`,

    Attributes
    ----------
    spot_curve : pricing.discount_curves.DiscountCurve
        Discount curve observed from market data, i.e., :math:`T\\to P^M(0, T) \\text{ for }T>0.`

    calendar : data.calendars.DayCountCalendar
            Specifies the calendar used for computation. In particular, it gives :math:`\\text{DC}(t, T)` or :math:`\\tau(t,T)`,

    parameters : np.ndarray
        Parameters of the short rate model.

    no_factors : int
        The number of factors in the short rate model.

    calibration_data : pandas.Dataframe
        Data used for calibrating the model.

    recover_spot : bool
        Indicates whether the spot rate can be recovered from the model.

    N : np.vectorize
        A vectorized version of the cumulative distribution function of the standard normal distribution.

    n : np.vectorize
        A vectorized version of the probability density function of the standard normal distribution.

    Notes
    -----
    - The `ShortRateModel` class is designed to be a parent class for specific short rate model implementations.
    - The actual model-specific behavior (like path generation, calibration, etc.) must be implemented in subclasses.
    - The `N` and `n` attributes are vectorized versions of functions from the `scipy.stats.norm` module.
    """

    def __init__(self, spot_curve, calendar):
        self.spot_curve = spot_curve
        self.calendar = calendar
        self.parameters = None
        self.no_factors = None
        self.calibration_data = None
        self.recover_spot = None

        self.N = np.vectorize(norm.cdf)
        self.n = np.vectorize(norm.pdf)

    def cap_price_function(
        self, params, strikes, maturities, pay_freq, all_T, all_deltas, all_bonds
    ):
        """
        Return cap prices according to the model.

        Parameters
        ----------
        params: np.ndarray
            Array of model parameters.
        strikes : np. ndarray
            Array of cap strikes.
        maturities : list
            A list of cap maturities.
        pay_freq : list
            A list of cap frequencies.
        all_T : dict
            Dictionary of times T for different caps for each frequency (keys).
        all_deltas: dict
            Idem but for time intervals.
        all_bonds: dict
            Idem but for bond prices.

        Returns
        -------
        numpy.ndarray
            Array of cap prices.
        """
        pass

    def caplet_price_function(
        self, params, strikes, maturities, pay_freq, all_T, all_deltas, all_bonds
    ):
        """
        Return caplet prices according to the model

        Parameters
        ----------
        params: np.ndarray
            Array of model parameters.
        strikes : np. ndarray
            Array of cap strikes.
        maturities : list
            A list of cap maturities.
        pay_freq : list
            A list of cap frequencies.
        all_T : dict
            Dictionary of times T for different caplets for each frequency (keys).
        all_deltas: dict
            Idem but for time intervals.
        all_bonds: dict
            Idem but for bond prices.

        Returns
        -------
        numpy.ndarray
            Array of cap prices.
        """
        pass

    def compute_all_bonds(
        self, date, caps_df, effective_date_interval=2, bond_type="get"
    ):
        """
        Compute the dictionary of bonds needed for a cap.

        Parameters
        ----------
        date : pandas.Timestamp
            Valuation date, the first argument (:math:`t`) of :math:`\\textnormal{FP}(t; T, S)`.
        caps_df : pandas.DataFrame
            The DataFrame containing cap (or caplet) information. The columns, for caps, are: "Maturity", "Tenor", "Pay Frequency", "Quote". It could also contain "Strike".
        effective_date_interval : int
            Effective day interval to compute end dates
        bond_type : str
            This argument specifies the way the bond is calculated. "get" uses ``get_value`` from spot_curve and "compute" uses compute_spot_bond.

        Returns
        -------
        dict
            Dictionary of all_bonds for the cap.
        """
        tenors = caps_df.Tenor.values
        pay_freq = caps_df["Pay Frequency"].values
        all_freqs = [
            i for i in [3, 6] if (pay_freq == i).max() == 1
        ]  # Returns the subset of [3, 6] that appears in pay_freq
        all_offsets = {
            n: n * np.arange(1 + int(12 / n) * (tenors[pay_freq == n]).max())
            for n in all_freqs
        }  # Dictionary for list of 1 + 12/month_freq * max_tenor_n elements
        all_end_dates = {
            n: [
                date
                + pd.Timedelta(days=effective_date_interval)
                + pd.DateOffset(months=offset)
                for offset in all_offsets[n]
            ]
            for n in all_freqs
        }

        if bond_type == "compute":
            all_bonds = {}
            for n in all_freqs:
                dates = [date] * len(all_end_dates[n])
                beg_dates = pd.DatetimeIndex(dates)
                all_bonds[n] = self.compute_spot_bond(
                    beg_dates, all_end_dates[n], calendar=self.calendar
                )  # Bonds P(t=date, T=end date)
        elif bond_type == "get":
            all_bonds = {
                n: self.spot_curve.get_value(
                    date, all_end_dates[n], calendar=self.calendar
                )
                for n in all_freqs
            }  # Bonds P(t=date, T=end date)
        else:
            raise ValueError("bond_type must be either 'compute' or 'get'. ")
        return all_bonds

    def compute_cap_data(
        self,
        date,
        caps_df,
        discount_curve,
        effective_date_interval=2,
        vol_type="normal",
    ):
        """
        Compute cap parameters strike, forwards, T, deltas, bonds...

        Parameters
        ----------
        date : pandas.Timestamp
            Valuation date, the first argument (:math:`t`) of :math:`\\textnormal{FP}(t; T, S)`.
        caps_df : pandas.DataFrame
            The DataFrame containing cap (or caplet) information. The column, for caps, are: "Maturity", "Tenor", "Pay Frequency", "Quote". It could also contain "Strike".
        discount_curve : pricing.discount_curves.DiscountCurve , default = None
            Gives the discounts for the computation of the strikes if these are not given.
        effective_date_interval : int
            Effective day interval to compute end dates
        vol_type : str, default = "Normal"
            Black formula kind used for cap prices and vegas.

        Returns
        -------
        pandas.DataFrame, dict, ditc, dict, dict
            caps_df, all_forwards, all_T, all_deltas, all_bonds

        """
        caps_df = caps_df[caps_df.Maturity == 0]
        tenors = caps_df.Tenor.values
        pay_freq = caps_df["Pay Frequency"].values

        all_freqs = [
            i for i in [3, 6] if (pay_freq == i).max() == 1
        ]  # Returns the subset of [3, 6] that appears in pay_freq
        all_offsets = {
            n: n * np.arange(1 + int(12 / n) * (tenors[pay_freq == n]).max())
            for n in all_freqs
        }  # Dictionary for list of 1 + 12/month_freq * max_tenor_n elements
        # all_end_dates = {n: [date+pd.Timedelta(days=effective_date_interval)+pd.DateOffset(months=offset) for offset in all_offsets[n]] for n in all_freqs}
        all_end_dates = {
            n: [
                date
                + pd.DateOffset(days=effective_date_interval)
                + pd.DateOffset(months=offset)
                for offset in all_offsets[n]
            ]
            for n in all_freqs
        }  # Dictionary with
        # list of dates = date + 2 days + offset moths
        all_T = {
            n: self.calendar.interval(date, all_end_dates[n]) for n in all_freqs
        }  # Dictionary with calendar interval between dates and end dates
        all_deltas = {
            n: all_T[n][1:] - all_T[n][:-1] for n in all_freqs
        }  # Dictionary for T[i+1] - T[i]
        all_bonds = self.compute_all_bonds(
            date, caps_df, effective_date_interval=effective_date_interval
        )
        all_forwards = {
            n: (all_bonds[n][:-1] / all_bonds[n][1:] - 1) / all_deltas[n]
            for n in all_freqs
        }  # F(t=date; T=all_end_dates[i], S=all_end_dates[i+1]), (1.20) of BM
        if "Strike" not in caps_df.columns:
            caps_df["Strike"] = np.full(len(caps_df), np.nan)
            if discount_curve is not None:
                all_discounts = {
                    n: discount_curve.get_value(
                        date, all_end_dates[n], calendar=self.calendar
                    )
                    for n in all_freqs
                }
            else:
                all_discounts = {
                    n: self.spot_curve.get_value(
                        date, all_end_dates[n], calendar=self.calendar
                    )
                    for n in all_freqs
                }

            for i in caps_df.index:
                n = caps_df.loc[i, "Pay Frequency"]
                tenor = caps_df.loc[i, "Tenor"]
                deltas = all_deltas[n][: tenor * int(12 / n)]
                discounts = all_discounts[n][1 : tenor * int(12 / n) + 1]
                forwards = all_forwards[n][: tenor * int(12 / n)]
                caps_df.loc[i, "Strike"] = np.sum(
                    deltas * forwards * discounts
                ) / np.sum(deltas * discounts)

        for i in caps_df.index:
            sigma = caps_df.loc[i, "Quote"]
            tenor = caps_df.loc[i, "Tenor"]
            strike = caps_df.loc[i, "Strike"]
            n = caps_df.loc[i, "Pay Frequency"]
            T = all_T[n][: int(tenor * int(12 / n))]
            deltas = all_deltas[n][: int(tenor * int(12 / n))]
            bonds = all_bonds[n][1 : int(tenor * int(12 / n)) + 1]
            forwards = all_forwards[n][: int(tenor * int(12 / n))]

            if vol_type == "lognormal":
                mask = sigma == 0 or np.sqrt(T) == 0
                d = np.empty_like(T)
                d[mask] = (np.sign(np.log(forwards / strike)) * np.inf)[mask]
                d[~mask] = (
                    np.log(forwards[~mask] / strike) + (sigma**2 * T[~mask]) / 2
                ) / (sigma * np.sqrt(T[~mask]))
                caps_df.loc[i, "Price"] = np.sum(
                    deltas
                    * bonds
                    * (forwards * self.N(d) - strike * self.N(d - sigma * np.sqrt(T)))
                )
                # caps_df.loc[i, "Vega"] = np.sum(deltas * bonds * strike * np.sqrt(T) * self.n(d - sigma * np.sqrt(T)))
                caps_df.loc[i, "Vega"] = np.sum(
                    deltas * bonds * forwards * np.sqrt(T) * self.n(d)
                )
            elif vol_type == "normal":
                mask = sigma == 0 or np.sqrt(T) == 0
                d = np.empty_like(T)
                d[mask] = (np.sign(forwards - strike) * np.inf)[mask]
                d[~mask] = ((forwards - strike) / (sigma * np.sqrt(T[~mask])))[~mask]
                caps_df.loc[i, "Price"] = np.sum(
                    deltas
                    * bonds
                    * ((forwards - strike) * self.N(d) + sigma * np.sqrt(T) * self.n(d))
                )
                caps_df.loc[i, "Vega"] = np.sum(deltas * bonds * np.sqrt(T) * self.n(d))

        return caps_df, all_forwards, all_T, all_deltas, all_bonds

    def compute_forward_bond(self, dates, beg_dates, tenor_dates, calendar, srsim):
        """
        Compute the forward bond price, :math:`\\text{FP}_{t^0}(t^1,T_{l-1}, T_l)`, for the given dates and stochastic factors. This function computes the forward bond price
        as:

        .. math::
            \\text{FP}_{t^0}\\left(t^1,T_{l-1}, T_l\\right) := \\frac{P_{t^0}\\left(t^1, T_l\\right)}{P_{t^0}\\left(t_1,T_{l-1}\\right)}\,.

        For the dates :math:`\\{t^0_k\\}_{k=0}^K` ,  :math:`\\{t^1_j\\}_{j=0}^J` , :math:`\\{T_l\\}_{l=0}^{L}`, this function computes
        the forward bond price according to the specified model, defining :math:`T_{-1}:=t^1_j` for :math:`l=0`. It follows the definition in [Brigo and Mercurio, 2006]
        (see page XXXVI) and makes explicit that the bond price depends on the stochastic factors of the model.

        Parameters
        ----------
        dates : pandas.DatetimeIndex, list of strings, pandas.Timestamp, or string
            Represents the dates :math:`\\{t^0_k\\}_{k=0}^K`, serving as the origin for calibration.
        beg_dates : pandas.DatetimeIndex, list of strings, pandas.Timestamp, or string
            Represents :math:`\\{t^1_j\\}_{j=0}^J`, the first set of dates.
        tenor_dates : pandas.DatetimeIndex, list of strings, pandas.Timestamp, or string
            Represents :math:`\\{T_l\\}_{l=0}^{L}`, the tenor structure.
        calendar : data.calendars.DayCountCalendar
            Specifies the calendar used for computation. In particular, it gives :math:`\\text{DC}(t, T)` or :math:`\\tau(t,T)`.
        srsim : numpy.ndarray
            Simulations of the stochastic factors, denoted as :math:`\\underline{x}_i(t^0_k, t^1_j)`,
            same notation as in :py:meth:`compute_future_bond <pricing.ir_models.ShortRateModel.compute_future_bond>`.
            That is, ``srsim[i,j,k,l]`` = :math:`x^l_i(t^0_k, t^1_j)`.

        Returns
        -------
        numpy.ndarray
            Returns an array named ``forwards`` such that:

            - ``forwards[i, j, k, 0]`` = :math:`\\text{FP}_{t^0_k}(t^1_j,t^1_j, T_l, \\underline{x}_i(t^0_k, t^1_j))=P_{t^0_k}(t^1_j, T_0, \\underline{x}_i(t^0_k, t^1_j))` if :math:`l=0`.
            - ``forwards[i, j, k, l]`` = :math:`\\text{FP}_{t^0_k}(t^1_j,T_{l-1}, T_l, \\underline{x}_i(t^0_k, t^1_j))` otherwise.

        Notes
        -----
        - Constraint :math:`t^0_k \\le t^1_j` must hold for all :math:`j, k`.
        - The subscript of :math:`P` in :math:`\\text{FP}_{t^0}` signifies the time considered as origin for calibration, see :py:meth:`pricing.ir_models.ShortRateModel.compute_future_bond`.
        - The result can be used to calculate a forward according to:
          :math:`F(t; T, S) = \\frac{1}{S-T}(\\text{FP}^{-1}(t; T, S)-1)` for :math:`t\\le T\\le S`.

        Warnings
        --------
        - To calculate a forward we need :math:`\\text{FP}^{-1}`, not :math:`\\text{FP}` itself.

        References
        -------
        - [Brigo and Mercurio, 2006] Brigo & Mercurio (2006). Interest Rate Models–Theory and Practice.
        """

        # Dates formatting
        dates, beg_dates, tenor_dates = afsfun.dates_formatting(
            dates, beg_dates, tenor_dates
        )

        if beg_dates.min() < dates.max():
            raise ValueError("beg_dates must be later than date.")

        # Forwards calculation
        fb = np.full(
            (srsim.shape[0], beg_dates.size, dates.size, tenor_dates.size), np.nan
        )
        for j in range(beg_dates.size):
            bonds = self.compute_future_bond(
                dates=dates,
                beg_dates=beg_dates[j],
                end_dates=tenor_dates,
                calendar=calendar,
                srsim=srsim[:, [j], :, :],
            )
            fb[:, j, :, :] = np.transpose(bonds[:, :, :, 0], (0, 2, 1))

        den = np.full(fb.shape, np.nan)
        den[:, :, :, 0] = 1
        den[:, :, :, 1:] = fb[:, :, :, :-1]

        forwards = fb / den
        return forwards

    def compute_forward_bond_old(self, date, step_dates, final_dates, calendar, srsim):
        """
        Return the forward bonds according to a specific model.

        Parameters
        ----------
        date : pandas.Timestamp or str
            Valuation date, the first argument (:math:`t`) of :math:`\\textnormal{FP}(t; T, S)`.
        step_dates : pandas.DatetimeIndex or list
            Step dates for forward bonds, the second argument (:math:`T`) of :math:`\\textnormal{FP}(t; T, S)`.
        final_dates : pandas.DatetimeIndex or list
            Final dates for forward bonds, the third argument (:math:`S`) of :math:`\\textnormal{FP}(t; T, S)`.
        calendar : data.calendars.DayCountCalendar
            Calendar for computation of time intervals.
        srsim : numpy.ndarray
            Array of simulations of the stochastic factors at date.
        Returns
        -------
        numpy.ndarray
            Array of :math:`\\textnormal{FP}(t; T, S)`.
        """

        # Dates formatting
        date = pd.to_datetime(date)
        step_dates, final_dates = afsfun.dates_formatting(step_dates, final_dates)
        if step_dates.min() < date:
            raise ValueError("beg_dates must be later than date.")

        # Forwards calculation
        numerator = np.full((srsim.shape[0], final_dates.size), np.nan)
        for i in range(final_dates.size):
            numerator[:, i] = self.compute_future_bond(
                dates=date,
                beg_dates=step_dates[0],
                end_dates=final_dates[i],
                calendar=calendar,
                srsim=srsim,
            )[:, 0, 0, 0]

        denominator = np.full((srsim.shape[0], step_dates.size - 1), np.nan)
        for i in range(step_dates.size - 1):
            denominator[:, i] = self.compute_future_bond(
                dates=date,
                beg_dates=step_dates[0],
                end_dates=step_dates[i + 1],
                calendar=calendar,
                srsim=srsim,
            )[:, 0, 0, 0]

        forward_bonds = numerator / denominator
        return forward_bonds

    def compute_future_bond(self, dates, beg_dates, end_dates, calendar, srsim):
        """
        Compute the bond price, :math:`P(t, T)`, according to a specified model.

        This function computes the bond price based on specified model parameters and given
        simulation data. It relies on several input dates and stochastic simulations to
        compute and return the bond price.

        Parameters
        ----------
        dates : pandas.DatetimeIndex, list of strings, pandas.Timestamp, or string
            Calibration time represented by :math:`\\{t^0_k\\}_{k=0}^K`. It serves as the origin for valuation.
        beg_dates : pandas.DatetimeIndex, list of strings, pandas.Timestamp, or string
            Represents :math:`\\{t^1_j\\}_{j=0}^J`, the first set of dates.
        end_dates : pandas.DatetimeIndex, list of strings, pandas.Timestamp, or string
            Represents :math:`\\{t^2_j\\}_{j=0}^{J'}`, the second set of dates.
        calendar : data.calendars.DayCountCalendar
            Specifies the calendar used for computation. In particular, it gives :math:`\\text{DC}(t, T)` or :math:`\\tau(t,T)`,
        srsim : numpy.ndarray
            Simulations of the stochastic factors, denoted as :math:`\\underline{x}_i(t^0_k, t^1_j)`, where ``srsim[i,j,k,l]`` = :math:`x^l_i(t^0_k, t^1_j)`
            signifies the `i`-th simulation at time :math:`t^1_j` with origin :math:`t^0_k` for the `l`-th stochastic component.

        Returns
        -------
        numpy.ndarray
            Returns an array representing :math:`P_{t^0_k}(t^1_j, t^2_j, \\underline{x}_i(t^0_k, t^1_j))`. That is, if the return variable is ``future_bond``, then

            ``future_bond[i, j, k, 0]`` = :math:`P_{t^0_k}(t^1_j, t^2_j, \\underline{x}_i(t^0_k, t^1_j))`.

        Notes
        -----
        1. The subscript of :math:`P`, :math:`t^0`, signifies the time considered as the origin for "calibration", i.e., :math:`P_{t^0}(t^0,T)=P^M(t^0,T)` corresponds to
           market data. As we saw in ``generate_paths``, this will only be the case for some models. For those models in which the spot curve is not
           recovered, once the parameters are fixed, $t_0$ is irrelevant.

        2. The bond price relies on the stochastic factors of the model :math:`\\underline{x}`, such as :math:`\\underline{x}=(x,y)` for G2++.

        3. The constraint :math:`t^0_k \\le t^1_j \\le t^2_j` must hold for all `j`, `k`.

        4. **Broadcasting rules**: Either :math:`J=J'` or the minimum of :math:`J+1, J'+1` must be 1. In the returned array, `j` must be in the range of
           :math:`\\{0, ... \\max(J, J')\\}`.

        5. If :math:`\\{t^0_k\\}_{k=0}^K = \\{t^1_j\\}_{j=0}^J` in increasing sequences of time, then Constraint 3 doesn't need to hold. However, only elements below the
           diagonal (:math:`j\\ge k` for a given `i`) of the array make sense.
        """

        pass

    def compute_implied_cap_vol(
        self,
        model_prices,
        strikes,
        maturities,
        pay_freq,
        all_forwards,
        all_T,
        all_deltas,
        all_bonds,
        vol_type="normal",
    ):
        """
        For a given set of cap specifications, return the implied volatility according to the specified Black or Bachelier's formula.

        Parameters
        ----------
        model_prices : numpy.ndarray
            Array of cap prices
        maturities : list
            A list of cap maturities.
        pay_freq : list
            A list of cap frequencies.
        strikes : numpy.ndarray
            Array of strikes
        maturities : numpy.ndarray
            Array of maturities
        pay_freq : numpy.ndarray
            Array of frequencies
        all_T : dict
            Dictionary of times T for different caps for each frequency (keys).
        all_forwards : dict
            Idem but for forwards
        all_deltas: dict
            Idem but for time intervals.
        all_bonds: dict
            Idem but for bond prices.
        vol_type : str, default = "normal"
            Black formula kind used for cap prices.

        Returns
        -------
        numpy.ndarray
            Implied volatilities.
        """
        implied_vols = np.full(model_prices.size, np.nan)

        for i in range(model_prices.size):
            n = int(pay_freq[i])
            T = all_T[n][
                : int(maturities[i] * int(12 / n))
            ]  # int is needed so slicing does not raise errors
            deltas = all_deltas[n][: int(maturities[i] * int(12 / n))]
            bonds = all_bonds[n][1 : int(maturities[i] * int(12 / n)) + 1]
            # discounts = all_discounts[n][1:maturities[i] * int(12 / n) + 1]
            forwards = all_forwards[n][: int(maturities[i] * int(12 / n))]

            def difference(sigma):
                if vol_type == "lognormal":
                    d = (np.log(forwards / strikes[i]) + (sigma**2 * T) / 2) / (
                        sigma * np.sqrt(T)
                    )
                    price = np.sum(
                        deltas
                        * bonds
                        * (
                            forwards * self.N(d)
                            - strikes[i] * self.N(d - sigma * np.sqrt(T))
                        )
                    )
                    # Check following formula!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                    # vega = np.sum(deltas * bonds * strikes * np.sqrt(T) * self.n(d - sigma * np.sqrt(T)))
                    vega = np.sum(deltas * bonds * forwards * np.sqrt(T) * self.n(d))
                elif vol_type == "normal":
                    d = (forwards - strikes[i]) / (sigma * np.sqrt(T))
                    price = np.sum(
                        deltas
                        * bonds
                        * (
                            (forwards - strikes[i]) * self.N(d)
                            + sigma * np.sqrt(T) * self.n(d)
                        )
                    )
                    vega = np.sum(deltas * bonds * np.sqrt(T) * self.n(d))
                # noinspection PyUnboundLocalVariable
                diff = (model_prices[i] - price) ** 2
                # noinspection PyUnboundLocalVariable
                return diff, -2 * (model_prices[i] - price) * vega

            def difference_plain(sigma):
                if vol_type == "lognormal":
                    d = (np.log(forwards / strikes[i]) + (sigma**2 * T) / 2) / (
                        sigma * np.sqrt(T)
                    )
                    price = np.sum(
                        deltas
                        * bonds
                        * (
                            forwards * self.N(d)
                            - strikes[i] * self.N(d - sigma * np.sqrt(T))
                        )
                    )
                    # Check following formula!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                    # vega = np.sum(deltas * bonds * strikes * np.sqrt(T) * self.n(d - sigma * np.sqrt(T)))
                    vega = np.sum(deltas * bonds * forwards * np.sqrt(T) * self.n(d))
                elif vol_type == "normal":
                    d = (forwards - strikes[i]) / (sigma * np.sqrt(T))
                    price = np.sum(
                        deltas
                        * bonds
                        * (
                            (forwards - strikes[i]) * self.N(d)
                            + sigma * np.sqrt(T) * self.n(d)
                        )
                    )
                    vega = np.sum(deltas * bonds * np.sqrt(T) * self.n(d))
                # noinspection PyUnboundLocalVariable
                diff = model_prices[i] - price
                # noinspection PyUnboundLocalVariable
                return diff, -vega

            if vol_type == "lognormal":
                delta = (forwards - strikes[i]) / 2
                x0 = np.mean(
                    np.sqrt(2 * np.pi / T)
                    * (model_prices[i] - delta)
                    / (forwards - delta)
                )  # Using the Bharadia-Christofides-Salkin model
                xtol = None
            elif vol_type == "normal":
                x0 = 0.001
                xtol = None
            else:
                x0 = np.nan
                xtol = np.nan

            # sigma0 = model_prices[i]/np.sum(np.sqrt(T/(2*np.pi)))
            # sigma = optimize.root_scalar(difference_plain, x0=x0, method="newton", xtol=xtol, fprime=True, options={"disp": True}).root
            sigma_min = optimize.minimize(
                (lambda x: difference(x)[0]),
                x0=x0,
                bounds=((0, 1.5),),
                method="powell",
                options={"xatol": 1e-12, "fatol": 1e-12, "disp": True},
            ).x[0]
            sigma = sigma_min
            # print("sigma 0", sigma)
            if (
                np.abs(sigma) > 1.5
            ):  # To chech if something is going wrong with the inversion
                print(f"strikes, maturities:{strikes, maturities}")
            #     print("Sigma plain: ", sigma)
            #     sigma2 = optimize.root_scalar(difference, x0=x0, method="newton", xtol=xtol, fprime=True, options={"disp": True}).root
            #     print("Sigma square: ", sigma2)
            #     sigma_min = optimize.minimize((lambda x: difference(x)[0]), x0=x0, bounds=((0, 1.5),), method="powell",
            #                                   options={"xatol": 1e-12, "fatol": 1e-12, "disp": True}).x[0]
            #     print("Sigma min: ", sigma_min)
            #
            #     t = np.arange(0, 2, 0.005)
            #     df = pd.DataFrame(data=np.array([np.vectorize(difference)(t)[0], np.vectorize(difference_plain)(t)[0]]).transpose(), index=t,
            #                       columns=["Difference^2", "Difference"])
            #     # noinspection PyCompatibility
            #     title = f'sigma = {sigma}'
            #     fig = px.line(df, title=title)
            #     # fig.show()
            implied_vols[i] = sigma
        return implied_vols

    def compute_spot_bond(self, beg_dates, end_dates, calendar):
        """
        Return the spot bond price according to a specific model.

        Parameters
        ----------
        beg_dates : pandas.DatetimeIndex or list
            Beginning dates, the first argument (:math:`t`), :math:`P_{0}(t,T)`. Mathematically represented as :math:`\\{t^1_j\\}_{j=0}^J`.
        end_dates : pandas.DatetimeIndex or list
            Final dates, the second argument (:math:`T`), :math:`P_{0}(t,T)`. Mathematically represented as :math:`\\{t^2_j\\}_{j=0}^{J'}`.
        calendar : data.calendars.DayCountCalendar
            Calendar used for computations, in particular, it provides :math:`\\text{DC}(t, T)` or :math:`\\tau(t,T)`.

        Returns
        -------
        numpy.ndarray
            The spot bond price array :math:`P_{0}(t,T)`.

        Notes
        -----
        For the dates :math:`\\{t^1_j\\}_{j=0}^J` and :math:`\\{t^2_j\\}_{j=0}^{J'}`, this method computes :math:`P_{t^1}(t^1, t^2)` according to the specified model. More precisely, it returns the array:

        .. math::
            \\left(P_{t^1_j}(t^1_j, t^2_j)\\right)_{j}\\,,

        Clearly, this computation is independent of the simulation as it only depends on the initial curve. This is the spot curve if we recover it using :math:`\\varphi`, as in G2++ and Hull-White models. \
        Another way to see it is using the previous method `compute_future_bond` and making explicit that the bond price depends on the stochastic factors of the model, :math:`\\underline{x}`. Here, the simulations are :math:`\\underline{x}(t^1, t^1)`, always the initial condition, hence independent of the simulation.

        The following broadcasting rule must be satisfied:

        .. math::
            \\text{either }\\quad J=J' \\quad\\text{ or }\\quad \\min\\{J+1,J'+1\\}=1\\,,

        In the returned array, :math:`j \\in \\{1,\\ldots, \\max(J, J')\\}`. Recall the use of zero-based indexing in Python.
        """

        # initial_values = self.get_risk_factors(beg_dates)
        # short_rate = np.full((1, 1, 1, self.no_factors), np.nan)
        # short_rate[:, 0] = initial_values
        #
        # # bonds = np.full(beg_dates.size, np.nan)
        # # for i in range(beg_dates.size):
        # #     bonds[i] = self.compute_future_bond(dates=beg_dates[i], beg_dates=beg_dates[i], end_dates=end_dates[i], calendar=calendar, srsim=short_rate[:, [0]])[0, 0, 0]
        # bonds = self.compute_future_bond(dates=beg_dates, beg_dates=beg_dates, end_dates=end_dates, calendar=calendar, srsim=short_rate[:, [0], :, :])
        # bonds = bonds[0, :, :].diagonal()

        # Dates formatting
        beg_dates, end_dates = afsfun.dates_formatting(beg_dates, end_dates)

        initial_values = self.get_risk_factors(beg_dates)
        nf = self.no_factors
        size = beg_dates.size
        if not type(initial_values) == int:  # Reshaping for broadcasting
            # noinspection PyUnresolvedReferences
            initial_values = initial_values.reshape(size, nf)
        short_rate = np.full((1, size, size, nf), np.nan)
        short_rate[0] = initial_values

        # WAY 1: The problem of this method is that as beg_dates are equal to dates, the conditions of dates=<beg_dates, as vector, is violated. On the other hand,
        # broadcasting follows naturally
        bonds = self.compute_future_bond(
            dates=beg_dates,
            beg_dates=beg_dates,
            end_dates=end_dates,
            calendar=calendar,
            srsim=short_rate,
        )
        bonds = bonds[
            0, :, :, 0
        ].diagonal()  # We want P_{t^1_j}(t^1_j, t^2_j), not P_{t^1_k}(t^1_j, t^2_j)

        # # WAY 2: It is better to avoid using loops
        # bonds = np.full(size, np.nan)
        # for i in range(beg_dates.size):  # We have to do it for each component to avoid the problems mentioned above
        #     sr = short_rate[:, [i], :, :]
        #     sr = sr[:, :, [i], :]  # For some reason, sr[:, [i], [i], :] eliminates one dimension
        #     bonds[i] = self.compute_future_bond(dates=beg_dates[i], beg_dates=beg_dates[i], end_dates=end_dates[i], calendar=calendar, srsim=sr)

        # # WAY 3: vectorize
        # @np.vectorize  # Otherwise, np.fromfunction uses [i] instead of i
        # def bonds_f(i):
        #     sr = short_rate[:, i:i+1, i:i+1, :]  # In this way the dimensions are preserved
        #     bonds_i = self.compute_future_bond(dates=beg_dates[i], beg_dates=beg_dates[i], end_dates=end_dates, calendar=calendar, srsim=sr)[0, 0, 0]
        #     return bonds_i
        # bonds = np.fromfunction(bonds_f, shape=(size,), dtype=int)  # Arguments must be of int type for slicing

        return bonds

    def compute_vols(
        self,
        date,
        caps_df,
        discount_curve=None,
        effective_date_interval=2,
        vol_type="normal",
    ):
        """
        Compute the volatility values for a given set of caps according to the model.

        Parameters
        ----------
        date : pandas.Timestamp
            Valuation date, the first argument (:math:`t`) of :math:`\\textnormal{FP}(t; T, S)`.
        caps_df : pandas.DataFrame
            The DataFrame containing cap (or caplet) information. The column, for caps, are: "Maturity", "Tenor", "Pay Frequency", "Quote". It could also contain "Strike".
        discount_curve : pricing.discount_curves.DiscountCurve , default = None
            Gives the discounts for the computation of the strikes if these are not given.
        effective_date_interval : int
            Effective day interval to compute end dates
        vol_type : str, default = "Normal"
            Black formula kind used for cap prices and vegas.

        Returns
        -------
        pandas.DataFrame
            A DataFrame with index the maturities and columns the volatility values according to the model and to the actual market quotes.
        """
        tenors = caps_df.Tenor.values
        pay_freq = caps_df["Pay Frequency"].values
        caps_df, all_forwards, all_T, all_deltas, all_bonds = self.compute_cap_data(
            date, caps_df, discount_curve, effective_date_interval
        )

        model_prices = self.cap_price_function(
            self.parameters.loc[date].values,
            caps_df["Strike"].values,
            tenors,
            pay_freq,
            all_T,
            all_deltas,
            all_bonds,
        )
        model_imp_vol = self.compute_implied_cap_vol(
            model_prices,
            caps_df["Strike"].values,
            tenors,
            pay_freq,
            all_forwards,
            all_T,
            all_deltas,
            all_bonds,
            vol_type,
        )

        market_vol = caps_df.Quote.values
        vol_df = pd.DataFrame(
            data=np.array([model_imp_vol, market_vol]).transpose(),
            columns=["Model volatilities", "Market volatilities"],
            index=tenors,
        )
        return vol_df

    @staticmethod
    def dates_formatting_cfb(dates, beg_dates, end_dates):
        """Dates formatting for compute_future_bonds method"""
        dates, beg_dates, end_dates = afsfun.dates_formatting(
            dates, beg_dates, end_dates
        )

        # if np.array_equal(beg_dates, dates) and dates.size > 1:
        #     print("Some beg_dates are later than dates as beg_dates equal dates, only below-diagonal terms makes sense.")
        if beg_dates.min() < dates.max() and not np.array_equal(beg_dates, dates):
            raise ValueError("beg_dates must be later than dates.")
        if (
            beg_dates.size != end_dates.size
            and min([beg_dates.size, end_dates.size]) != 1
        ):  # Broadcasting rules
            raise ValueError("beg_dates and end_dates must follow broadcasting rules.")

        return dates, beg_dates, end_dates

    def fit_to_caps(
        self,
        date,
        caps_df,
        discount_curve=None,
        effective_date_interval=2,
        vol_type="normal",
        initial_guess=None,
        recalc=True,
    ):
        """
        Adjust parameters to caps or caplets.

        Parameters
        ----------
        date : pandas.Timestamp
            Valuation date.
        caps_df : pandas.DataFrame
            The DataFrame containing cap (or caplet) information. The column, for caps, are: "Maturity", "Tenor", "Pay Frequency", "Quote". It could also contain "Strike".
        discount_curve : pricing.discount_curves.DiscountCurve , default = None
            Gives the discounts for the computation of the strikes if these are not given.
        effective_date_interval: int, default = 2
            Date interval between date and the offsets.
        vol_type : str, default = "Normal"
            Black formula kind used for cap prices and vegas.
        initial_guess : numpy.ndarray, default = None
            Initial guess for numerical minimization. If None, get_numerical_parameters is used.
        recalc : bool, default = True
            If ``True``, caplets are used instead of caps.

        Returns
        -------
        None
            It assigns the parameters that minimize the difference of prices divided by the vega.
        """
        self.calibration_data = caps_df
        tenors = caps_df.Tenor.values
        pay_freq = caps_df["Pay Frequency"].values
        caps_df, all_forwards, all_T, all_deltas, all_bonds = self.compute_cap_data(
            date, caps_df, discount_curve, effective_date_interval, vol_type
        )

        def difference(params):
            """Returns difference of model prices and market prices divided by the vega, which tries to approximate the difference of volatilities, up to first term."""
            if (
                not self.recover_spot
            ):  # Model bonds will not coincide with the ones of the spot curve
                self.parameters.loc[date] = params
                all_bonds_comp = self.compute_all_bonds(
                    date,
                    caps_df,
                    effective_date_interval=effective_date_interval,
                    bond_type="compute",
                )
                model_prices = self.cap_price_function(
                    params,
                    caps_df["Strike"].values,
                    tenors,
                    pay_freq,
                    all_T,
                    all_deltas,
                    all_bonds_comp,
                )
            else:
                model_prices = self.cap_price_function(
                    params,
                    caps_df["Strike"].values,
                    tenors,
                    pay_freq,
                    all_T,
                    all_deltas,
                    all_bonds,
                )
            differences = np.sum(
                ((caps_df["Price"].values - model_prices) / caps_df["Vega"].values) ** 2
            )
            return differences

        initial_guess0, bounds = self.get_numerical_parameters()
        # if initial_guess is None:
        #     initial_guess = initial_guess0
        # minimum = optimize.minimize(difference, initial_guess, bounds=bounds, method="powell", options={"xatol": 1e-12, "fatol": 1e-12, "disp": True}).x
        minimum = optimize.dual_annealing(func=difference, bounds=bounds).x
        if recalc:
            minimum = optimize.minimize(
                difference,
                minimum,
                bounds=bounds,
                method="powell",
                options={"xatol": 1e-12, "fatol": 1e-12, "disp": True},
            ).x

        self.parameters.loc[date] = minimum

    def generate_draws(
        self, no_obsdates, no_valuation_dates, no_sims, start_dates=None
    ):
        """
        Normal random variables needed for generating the paths of volatilities and assets prices.
        This method is needed for the new class DiffusionMC.

        Parameters
        ----------
        no_obsdates : int
            Number of observation dates.
        no_valuation_dates : int
            Number of valuation dates.
        no_sims : int
            Number of simulations.
        start_dates: pandas.Timestamp
            Starting date. Default ``None``.

        Returns
        -------
        np.array
            draws[i,j,k,l] is a four dimensional numpy array with all the normals.
        """
        no_factors = self.no_factors
        draws = np.random.randn(no_sims, no_obsdates, no_valuation_dates, no_factors)
        return draws

    def generate_paths(
        self, start_dates, step_dates, short_rate_draws, intervals, forward_measure=True
    ):
        """
        Return the stochastic variable needed for the computation of the future bond.

        Parameters
        ----------
        start_dates : pandas.DatetimeIndex
            Starting dates for the simulation. Mathematically represented as :math:`\\{t^0_k\\}_{k=0}^K`.

        step_dates : pandas.DatetimeIndex
            Step dates for the simulation. Mathematically represented as :math:`\\{t^1_j\\}_{j=0}^J`.

        short_rate_draws : np.ndarray
            A 4-dimensional array with the random draws generated for the simulation. In mathematical terms,
            this corresponds to the standard conventions for the indexes :math:`(i,j,k,l)`.

        intervals : np.ndarray
            Time intervals for the simulation. Represented as :math:`\\textrm{intervals}[j,k]=\\tau(t^1_{j-1}, t^1_{j})`
            for :math:`j>0` and :math:`\\textrm{intervals}[0,k]=\\tau(t^0_{k}, t^1_{0})`. The function :math:`\\tau`
            represents the day count convention and will depend on the calendar used. For simplicity, it can be
            thought of as :math:`\\tau(t,t')=t'-t` as a year fraction.

        forward_measure : bool, default = True
            If True, it uses the forward measure for the simulation.

        Returns
        -------
        np.ndarray
            Simulation paths.

        Notes
        -----
        It returns the stochastic variable needed for the computation of the future bond, see :py:meth:`compute_future_bond <pricing.ir_models.ShortRateModel.compute_future_bond>`. More precisely, in general,

        .. math::
            r(t)=\\varphi(t)+\\sum_{l=0}^{L} x^l(t)\\,,

        for bond pricing we will be interested in :math:`\\underline{x}=(x^l)_{l=0}^L`. The term :math:`\\varphi(t)` is introduced to recover the initial spot curve :math:`T\\to P^M(0, T) \\text{ for }T>0`, so if the model cannot do this, automatically :math:`\\varphi\\equiv 0`. In particular,

        - **Vasicek**: :math:`r(t)`, see (3.9) of BM,
        - **Hull-White**: :math:`x(t)`, see page 75 of BM,
        - **G2++**: :math:`(x(t), y(t))`, see Lemma 4.2.2 of BM.

        Let us call this :math:`\\underline{x}=(x^l)_{l=0}^L`. More specifically, it returns a four-dimensional array of the form:

        .. math::
            \\left(x_{i}^l(t^0_k,t^1_{j-1})\\right)_{i,j\\in\\{0, \\ldots, J+1\\},k,l}\\,,

        for the :math:`i`-th simulation of the :math:`k`-th start date, :math:`t^0_k` and :math:`l`-th component evaluated at the time corresponding to the :math:`j`-th step date, :math:`t^1_j`, \
        (:math:`j=-1` corresponds to the starting date, see intervals above). Note that for a given valuation date (fixed :math:`k`) we have :math:`t^1_{j=-1, k} := t^0_k`, which is consistent with \
        the definition of `intervals` given above.

        **Remark**: following Python conventions, in the whole document we will use the so-called zero-based indexing, i.e., :math:`\\underline{x}=(x^l)_{l=0}^L` instead of :math:`\\underline{x}=(x^l)_{l=1}^{L+1}`.

        """

        pass

    def generate_short_rate(
        self, start_dates, step_dates, short_rate_draws, intervals, forward_measure=True
    ):
        """
        Return the short rate simulations according to a specified model.

        Parameters
        ----------
        start_dates : pandas.DatetimeIndex
            Starting dates for the simulation. Mathematically represented as :math:`\\{t^0_k\\}_{k=0}^K`.

        step_dates : pandas.DatetimeIndex
            Step dates for the simulation. Mathematically represented as :math:`\\{t^1_j\\}_{j=0}^J`.

        short_rate_draws : np.ndarray
            A 4-dimensional array with the random draws generated for the simulation. Corresponds to the standard conventions for the indexes :math:`i,j,k,l`.

        intervals : np.ndarray
            Time intervals for the simulation. Represented as:

            .. math::
                \\textrm{intervals}[j,k]=
                \\begin{cases}
                \\textrm{intervals}[j,k]=\\tau(t^1_{j-1}, t^1_{j})\\text{ for }j>0 \\text{ (independent of } k\\text{)},\\\\
                \\textrm{intervals}[0,k]=\\tau(t^0_{k}, t^1_{0})\\text{ for }j=0.
                \\end{cases}

            The function :math:`\\tau` represents the day count convention and will depend on the calendar used. For simplicity, it can be thought of as :math:`\\tau(t,t')=t'-t` as a year fraction. \
            See Chapter 1 of [Brigo and Mercurio, 2006].

        forward_measure : bool, default = True
            If True, it uses the forward measure for the simulation.

        Returns
        -------
        np.ndarray
            Short rate paths.

        Notes
        -----
        Similar as above, it returns :math:`r(t)` according to the specified model. More specifically, it returns a four-dimensional array of the form:

        .. math::
            \\left(r_{i}(t^0_k,t^1_j)\\right)_{i,j,k,l=0}\\,,

        for the :math:`i`-th simulation of the :math:`k`-th start date at the time corresponding to the :math:`j`-th step date (:math:`j=0` correspond to the starting date).
        """
        pass

    def get_numerical_parameters(self, *k):
        """Return a guess of parameters of the model and their allowed regions."""
        return np.nan, np.nan

    def get_risk_factors(self, dates):
        """Return the value of the risk factors for a given dates"""
        pass

    def get_value(self, dates):
        """
        Calculate the short rate values for given dates based on the spot curve.

        This method computes the short rate values for specified dates. The calculation involves determining
        bond prices for a one-day period starting from each date and then using these prices to derive the
        short rates.

        Parameters
        ----------
        dates : pandas.DatetimeIndex, list of strings, pandas.Timestamp, or string
            The dates for which the short rate values are to be calculated. It can be a
            single date (as a pandas.Timestamp or its string representation) or an array-like object
            (as a pandas.DatetimeIndex or a list of its string representation) containing dates.

        Returns
        -------
        short_rate_values : ndarray
            An array of short rate values corresponding to the provided dates.

        Notes
        -----
        The calculation of short rate values, :math:`r`, from the spot curve, :math:`P^M`, is based on the following formula:

        .. math::
            r(\\bar{t}) = -\\frac{\\log(P^M(t, t+\\Delta))}{\\Delta}

        where :math:`P^M(t, t+\\Delta)` is the bond price for a one-day period starting from the given date and :math:`\\bar{t}\\in [t,t+\\Delta]`.
        The `days_in_year` attribute from the calendar is used to compute :math:`\\Delta`. Obviously, the mean value approximation is being used.
        """
        dates = afsfun.dates_formatting(dates)
        dates_plus_one = dates + pd.Timedelta("1D")
        bonds = self.spot_curve.get_value(dates, dates_plus_one, self.calendar)
        short_rate_values = -np.log(bonds) * self.calendar.days_in_year
        assert self.calendar.days_in_year == 1 / self.calendar.interval(
            dates, dates_plus_one
        )
        return short_rate_values

    def strip_caplet_vols(self):
        """Compute caplet implied vols from cap prices. See "Eight ways to strip your caplets: An introduction to caplet stripping" """
        pass


class BootstrappedCaplets(ShortRateModel):
    """
    A class representing a bootstrapped model for caplets, inheriting from ShortRateModel.

    This class is designed to model caplets, which are options on short-term interest rates,
    using a bootstrapped approach. It extends the ShortRateModel by including volatility
    information specific to caplets.

    Parameters
    ----------
    spot_curve : pricing.discount_curves.DiscountCurve
        The initial spot curve used in the short rate model.

    calendar : data.calendars.DayCountCalendar
        The calendar object used for day count conventions and date calculations in the model.

    vol_type : str
        The type of volatility model used for caplet pricing. This could be, for example,
        'lognormal' or 'normal', depending on the specifics of the model implementation.

    Attributes
    ----------
    vol_type : str
        Stores the type of volatility model specified at initialization.

    See Also
    --------
    ShortRateModel : The parent class from which this class inherits.
    """

    def __init__(self, spot_curve, calendar, vol_type):
        ShortRateModel.__init__(self, spot_curve, calendar)
        self.vol_type = vol_type

    def strip(self, date, caps_df, discount_curve=None):
        """
        Return the caplet volatility surface obtained using the bootstrapping method.

        Parameters
        ----------
        date : pandas.DatetimeIndex, list of strings, pandas.Timestamp, or string
            Valuation date. It can be a single date (as a pandas.Timestamp or its string representation) or an array-like object (as a pandas.DatetimeIndex or a list of its string
            representation) containing dates.
        discount_curve : pricing.discount_curves.DiscountCurve , default = None
            Gives the discounts for the computation of the strikes if these are not given.
        caps_df : pandas.DataFrame
            Data frame where the index is the set of maturities and the columns the different strikes, fixed. Not ATM strikes.

        Returns
        -------
        numpy.ndarray
            For each frequency, a dictionary such that for each strike a data frame is returned with the caplet implied volatilities.
        """
        pay_freqs = np.unique(caps_df["Pay Frequency"])
        implied_vol = {}
        for freq in pay_freqs:
            caps_df_freq = caps_df.loc[caps_df["Pay Frequency"] == freq]
            strikes = np.unique(caps_df_freq["Strike"])
            implied_vol[freq] = {}
            for strike in strikes:
                caps_df_strike = caps_df_freq.loc[caps_df_freq["Strike"] == strike]
                caps_df_strike = caps_df_strike.dropna(
                    axis=0, how="any"
                )  # Remove rows with any nan
                implied_vol[freq][strike] = pd.DataFrame(
                    [], index=caps_df_strike["Tenor"], columns=[strike]
                )
                (
                    caps_df_strike,
                    all_forwards,
                    all_T,
                    all_deltas,
                    all_bonds,
                ) = self.compute_cap_data(
                    date, caps_df_strike, discount_curve, vol_type=self.vol_type
                )
                prices = caps_df_strike["Price"].values
                cap_diff = np.full(prices.size, np.nan)
                cap_diff[0] = prices[0]
                cap_diff[1:] = np.diff(prices)
                maturities = caps_df_strike["Tenor"].values
                for j in range(cap_diff.size):
                    ind_max = int(maturities[j] * (12 / freq))
                    if j == 0:
                        ind_min = 0
                    else:
                        ind_min = int(maturities[j - 1] * (12 / freq))

                    all_T_diff = {freq: all_T[freq][ind_min:ind_max]}
                    all_forwards_diff = {freq: all_forwards[freq][ind_min:ind_max]}
                    all_bonds_diff = {freq: all_bonds[freq][ind_min : ind_max + 1]}
                    all_deltas_diff = {freq: all_deltas[freq][ind_min:ind_max]}

                    implied_vol[freq][strike].iloc[j] = float(
                        self.compute_implied_cap_vol(
                            model_prices=cap_diff[j].reshape(
                                1,
                            ),
                            strikes=strike.reshape(
                                1,
                            ),
                            maturities=maturities[j].reshape(
                                1,
                            ),
                            pay_freq=freq.reshape(
                                1,
                            ),
                            all_forwards=all_forwards_diff,
                            all_T=all_T_diff,
                            all_deltas=all_deltas_diff,
                            all_bonds=all_bonds_diff,
                            vol_type=self.vol_type,
                        )
                    )
        return implied_vol


class VasicekShortRate(ShortRateModel):
    """
    A class representing the Vasicek short rate model, inheriting from ``ShortRateModel``.

    The Vasicek model is a mathematical model describing the evolution of interest rates.
    It is a type of one-factor short rate model as it describes interest rate movements
    driven by only one source of market risk. The model is characterized by its mean reversion
    feature, where interest rates tend to revert to a long-term mean level.

    Parameters
    ----------
    discount_curve : pricing.discount_curves.DiscountCurve
        The initial discount curve used in the Vasicek model.

    calendar : data.calendars.DayCountCalendar
        The calendar object used for day count conventions and date calculations in the model.

    Attributes
    ----------
    parameters : pandas.DataFrame
        A DataFrame to hold the parameters of the Vasicek model including volatility ('vol'),
        speed of mean reversion ('reversion'), and long-term mean ('mean').

    no_factors : int
        The number of factors in the model, which is 1 for the Vasicek model.

    recover_spot : bool
        Indicates whether the spot rate can be recovered from the model, which is False for the Vasicek model.

    Notes
    -----
    - The Vasicek model assumes that, in the risk-neutral measure, the short rate follows a stochastic process defined as:

      .. math::
        dr = k(\\theta - r)dt + \\sigma\\cdot dW\,,

      where :math:`k` is the speed of mean reversion, :math:`\\theta` is the long-term mean,
      :math:`\\sigma` is the volatility, and :math:`W` is a Wiener process.

    See Also
    --------
    ShortRateModel : The parent class from which this class inherits.

    References
    ----------
    - [Brigo and Mercurio, 2006] Brigo & Mercurio (2006). Interest Rate Models–Theory and Practice. Chapter 3.

    """

    def __init__(self, discount_curve, calendar):
        ShortRateModel.__init__(self, discount_curve, calendar)
        self.parameters = pd.DataFrame(columns=["vol", "reversion", "mean"])
        self.no_factors = 1
        self.recover_spot = False

    def cap_price_function(
        self,
        params,
        strikes,
        maturities,
        pay_freq,
        all_T,
        all_deltas,
        all_bonds,
        return_caplets=False,
    ):
        sigma = params[0]
        k = params[1]
        # theta = params[2]  # This parameters only affects bond prices
        b = {n: (1 - np.exp(-k * all_deltas[n])) / k for n in all_T.keys()}

        sigma_p = {
            n: sigma * b[n] * np.sqrt((1 - np.exp(-2 * k * all_T[n][:-1])) / (2 * k))
            for n in all_T.keys()
        }
        if return_caplets:
            model_prices = {}
        else:
            model_prices = np.full(strikes.size, np.nan)
        for i in range(strikes.size):
            n = pay_freq[i]
            temp_bonds = all_bonds[n][: int(12 / n) * maturities[i] + 1]
            temp_deltas = all_deltas[n][: int(12 / n) * maturities[i]]
            temp_sigma = sigma_p[n][: int(12 / n) * maturities[i]]
            h = (
                np.log(
                    temp_bonds[1:] * (1 + strikes[i] * temp_deltas) / temp_bonds[:-1]
                )
                / temp_sigma
                + temp_sigma / 2
            )
            caplets = temp_bonds[:-1] * self.N(-h + temp_sigma) - (
                1 + strikes[i] * temp_deltas
            ) * temp_bonds[1:] * self.N(-h)
            if return_caplets:
                model_prices[strikes[i]] = caplets
            else:
                model_prices[i] = np.sum(caplets)
        return model_prices

    def compute_future_bond(self, dates, beg_dates, end_dates, calendar, srsim):
        if srsim.shape[3] > 1:
            print(
                "short_rate_draws has more than one stochastic factor (fourth dimension bigger than one). Selecting only first component."
            )
        srsim = srsim[:, :, :, 0]

        # Dates formatting
        dates, beg_dates, end_dates = ShortRateModel.dates_formatting_cfb(
            dates, beg_dates, end_dates
        )

        # Parameters of the model
        sigma = self.parameters.loc[dates, "vol"].values
        k = self.parameters.loc[dates, "reversion"].values
        theta = self.parameters.loc[dates, "mean"].values

        # Intermediate steps
        taus = calendar.interval(beg_dates, end_dates)
        taus = taus.reshape((taus.size, 1))
        B = (1 - np.exp(-k * taus)) / k
        A = np.exp(
            (theta - 0.5 * (sigma / k) ** 2) * (B - taus) - (sigma * B) ** 2 / (4 * k)
        )

        # Discounts
        disc = A * np.exp(-B * srsim)
        disc = disc[:, :, :, np.newaxis]  # For convention considerations

        return disc

    def generate_paths(
        self, start_dates, step_dates, short_rate_draws, intervals, forward_measure=True
    ):
        if short_rate_draws.shape[3] > 1:
            print(
                "short_rate_draws has more than one stochastic factor (fourth dimension bigger than one)."
            )
        short_rate_draws = short_rate_draws[:, :, :, 0]  # We eliminate the last index
        short_rate_start = self.get_risk_factors(start_dates)

        # Parameters of the model
        sigma = self.parameters.loc[start_dates, "vol"].values
        k = self.parameters.loc[start_dates, "reversion"].values
        theta = self.parameters.loc[start_dates, "mean"].values

        # Intermediate computations
        exp_inter = np.exp(-k * intervals)
        cond_var = sigma**2 / (2 * k) * (1 - exp_inter**2)
        t = np.add.accumulate(intervals, axis=0)
        T = np.sum(intervals, axis=0)  # noinspection PyPep8Naming
        MT = (theta - sigma**2 / (k**2)) * (1 - exp_inter) + sigma**2 / (
            2 * k**2
        ) * np.exp(-k * (T - t)) * (1 - exp_inter**2)  # noinspection PyPep8Naming
        shape = (
            short_rate_draws.shape[0],
            short_rate_draws.shape[1] + 1,
            short_rate_draws.shape[2],
        )  # One more element for the second index, no draws for the first obs date

        # Short rate calculations
        short_rate = np.full(shape, np.nan)
        short_rate[:, 0] = short_rate_start
        # Recall that for a numpy array z, z[:, 0, :] == z[:, 0] is True
        for i in range(1, short_rate.shape[1]):
            cond_mean = short_rate[:, i - 1] * exp_inter[i - 1] + MT[i - 1]
            short_rate[:, i] = (
                cond_mean + np.sqrt(cond_var[i - 1]) * short_rate_draws[:, i - 1]
            )

        return short_rate.reshape(short_rate.shape + (1,))

    def generate_short_rate(
        self, start_dates, step_dates, short_rate_draws, intervals, forward_measure=True
    ):
        return self.generate_paths(
            start_dates, step_dates, short_rate_draws, intervals, forward_measure
        )

    def get_numerical_parameters(self, *k):
        initial_guess = np.array([0.005352, 0.192024, 0.01])
        bounds = [(0, 1), (-1, 1), (0, 1)]
        return initial_guess, bounds

    def get_risk_factors(self, dates):
        return self.get_value(dates)


class OneFactorHullWhiteShortRate(ShortRateModel):
    """
    A class representing the One-Factor Hull-White short rate model, inheriting from ``ShortRateModel``.

    The One-Factor Hull-White model is an extension of the Vasicek model that allows for a time-dependent
    mean reversion level. It is also a one-factor model that captures the evolution of interest rates with
    mean reversion, but with more flexibility to fit the current term structure of interest rates.

    Parameters
    ----------
    discount_curve : pricing.discount_curves.DiscountCurve
        The initial discount curve used in the Hull-White model.

    calendar : data.calendars.DayCountCalendar
        The calendar object used for day count conventions and date calculations in the model.

    Attributes
    ----------
    parameters : pandas.DataFrame
        A DataFrame to hold the parameters of the Hull-White model, including volatility ('vol')
        and speed of mean reversion ('reversion').

    no_factors : int
        The number of factors in the model, which is 1 for the Hull-White model.

    recover_spot : bool
        Indicates whether the spot rate can be recovered from the model, which is True for the Hull-White model.

    Notes
    -----
    - In the Hull-White model, the short rate follows a stochastic process in the risk-neutral measure defined as:

      .. math::
        dr = (\\theta(t) - a \\cdot r)dt + \\sigma\\cdot dW\,,

      where :math:`a` is the speed of mean reversion, :math:`\\theta(t)` is a time-dependent function ensuring
      the model fits the initial term structure of interest rates, :math:`\\sigma` is the volatility,
      and :math:`W` is a Wiener process.

    - The Hull-White model's ability to fit the current term structure makes innecessary to specify the parameter \
    of asymptotic mean reversion.

    See Also
    --------
    ShortRateModel : The parent class from which this class inherits.

    References
    ----------
    - [Brigo and Mercurio, 2006] Brigo & Mercurio (2006). Interest Rate Models–Theory and Practice. Chapter 3.

    """

    def __init__(self, discount_curve, calendar):
        ShortRateModel.__init__(self, discount_curve, calendar)
        self.parameters = pd.DataFrame(columns=["vol", "reversion"])
        self.no_factors = 1
        self.recover_spot = True

    def cap_price_function(
        self,
        params,
        strikes,
        maturities,
        pay_freq,
        all_T,
        all_deltas,
        all_bonds,
        return_caplets=False,
    ):
        sigma = params[0]
        a = params[1]
        b = {n: (1 - np.exp(-a * all_deltas[n])) / a for n in all_T.keys()}

        sigma_p = {
            n: sigma * b[n] * np.sqrt((1 - np.exp(-2 * a * all_T[n][:-1])) / (2 * a))
            for n in all_T.keys()
        }
        if return_caplets:
            model_prices = {}
        else:
            model_prices = np.full(strikes.size, np.nan)
        for i in range(strikes.size):
            n = pay_freq[i]
            temp_bonds = all_bonds[n][: int(12 / n) * maturities[i] + 1]
            temp_deltas = all_deltas[n][: int(12 / n) * maturities[i]]
            temp_sigma = sigma_p[n][: int(12 / n) * maturities[i]]
            h = (
                np.log(
                    temp_bonds[1:] * (1 + strikes[i] * temp_deltas) / temp_bonds[:-1]
                )
                / temp_sigma
                + temp_sigma / 2
            )
            caplets = temp_bonds[:-1] * self.N(-h + temp_sigma) - (
                1 + strikes[i] * temp_deltas
            ) * temp_bonds[1:] * self.N(-h)
            if return_caplets:
                model_prices[strikes[i]] = caplets
            else:
                model_prices[i] = np.sum(caplets)
        return model_prices

    def compute_future_bond(self, dates, beg_dates, end_dates, calendar, srsim):
        # Dates formatting
        dates, beg_dates, end_dates = ShortRateModel.dates_formatting_cfb(
            dates, beg_dates, end_dates
        )

        # Parameters
        sigma = self.parameters.loc[dates, "vol"].values
        a = self.parameters.loc[dates, "reversion"].values

        # Time variables
        t = np.full((beg_dates.size, dates.size), np.nan)
        for i in range(t.shape[0]):
            t[i] = calendar.interval(dates, beg_dates[i])
        T = np.full((end_dates.size, dates.size), np.nan)
        for i in range(T.shape[0]):
            T[i] = calendar.interval(dates, end_dates[i])

        # Computation of V_exp:=1/2[V(t,T) - V(0,T) + V(0,t)]
        aux0 = (np.exp((3.0 * (a * t)))) + (
            (-4.0 * (np.exp((a * ((2.0 * t) + T)))))
            + (3.0 * (np.exp((a * (t + (2.0 * T))))))
        )
        aux1 = (-1.0 + (np.exp((a * t)))) * (
            (((np.exp((2.0 * (a * t)))) + aux0) - (np.exp((2.0 * (a * T)))))
            * (sigma**2)
        )
        V_exp = -0.25 * ((a**-3.0) * ((np.exp((-2.0 * (a * (t + T))))) * aux1))

        # Bond calculations
        market_forward_bonds = self.spot_curve.compute_fbond(
            dates, beg_dates, end_dates, calendar
        )
        disc = market_forward_bonds * np.exp(
            V_exp + (np.exp(-a * (T - t))) / a * srsim[:, :, :, 0]
        )
        disc = disc[:, :, :, np.newaxis]  # For convention considerations

        return disc

    def compute_future_bond_old(self, dates, beg_dates, end_dates, calendar, srsim):
        if srsim.shape[3] > 1:
            print(
                "short_rate_draws has more than one stochastic factor (fourth dimension bigger than one). Selecting only first component."
            )
        srsim = srsim[:, :, :, 0]

        dates, beg_dates, end_dates = afsfun.dates_formatting(
            dates, beg_dates, end_dates
        )

        if beg_dates.min() < dates.max():
            print("Somes dates are later than other beg_dates.")
            # raise ValueError("beg_dates must be later than dates.")

        sigma = self.parameters.loc[dates, "vol"].values
        a = self.parameters.loc[dates, "reversion"].values
        t = np.full((beg_dates.size, dates.size), np.nan)
        for i in range(t.shape[0]):
            t[i] = calendar.interval(dates, beg_dates[i])
        taus = calendar.interval(beg_dates, end_dates)
        taus = taus.reshape((taus.size, 1))
        B = (1 - np.exp(-a * taus)) / a
        market_fbonds = self.spot_curve.compute_fbond(
            dates, beg_dates, end_dates, calendar
        )
        beg_plus_one = beg_dates + pd.Timedelta("1D")
        inf_fbonds = self.spot_curve.compute_fbond(
            dates, beg_dates, beg_plus_one, calendar
        )
        forwards = -np.log(inf_fbonds) * calendar.days_in_year
        A = market_fbonds * np.exp(
            B * forwards - sigma**2 / (4 * a) * (1 - np.exp(-2 * a * t)) * B**2
        )
        disc = np.full((srsim.shape[0],) + A.shape, np.nan)
        for i in range(srsim.shape[0]):
            disc[i] = A * np.exp(-B * srsim[i])
        return disc

    def fit_to_atm_caps(
        self,
        date,
        caps_df,
        discount_curve,
        vol_type="normal",
        effective_date_interval=2,
        initial_guess=np.array([0.001, 0.1]),
    ):
        date = pd.to_datetime(date)
        caps_df, all_T, all_deltas, all_bonds = ShortRateModel.get_market_data(
            self,
            date,
            caps_df,
            discount_curve,
            vol_type=vol_type,
            effective_date_interval=effective_date_interval,
        )

        def difference(x):
            sigma = x[0]
            a = x[1]
            b = {n: (1 - np.exp(-a * all_deltas[n] / 2)) / a for n in all_T.keys()}
            sigma_p = {
                n: sigma
                * b[n]
                * np.sqrt((1 - np.exp(-2 * a * all_T[n][:-1])) / (2 * a))
                for n in all_T.keys()
            }
            model_prices = np.full(len(caps_df), np.nan)
            for i in caps_df.index:
                freq = caps_df.loc[i, "Pay Frequency"]
                maturity = caps_df.loc[i, "Tenor"]
                strike = caps_df.loc[i, "Strike"]
                temp_bonds = all_bonds[freq][: int(12 / freq) * maturity + 1]
                temp_deltas = all_deltas[freq][: int(12 / freq) * maturity]
                temp_sigma = sigma_p[freq][: int(12 / freq) * maturity]
                h = (
                    np.log(
                        temp_bonds[1:] * (1 + strike * temp_deltas) / temp_bonds[:-1]
                    )
                    / temp_sigma
                    + temp_sigma / 2
                )
                caplets = temp_bonds[:-1] * self.N(-h + temp_sigma) - (
                    1 + strike * temp_deltas
                ) * temp_bonds[1:] * self.N(-h)
                model_prices[i] = np.sum(caplets)
            differences = np.sum(
                ((caps_df.Price.values - model_prices) / caps_df.Vega) ** 2
            )
            return differences

        minimum = optimize.minimize(
            difference,
            initial_guess,
            method="powell",
            # bounds=[(0, 1), (0, 1)],
            options={"xatol": 1e-12, "fatol": 1e-12, "disp": True},
        ).x
        self.parameters.loc[date] = minimum

    def generate_paths(
        self, start_dates, step_dates, short_rate_draws, intervals, forward_measure=True
    ):
        if short_rate_draws.shape[3] > 1:
            print(
                "short_rate_draws has more than one stochastic factor (fourth dimension bigger than one)."
            )
        short_rate_draws = short_rate_draws[:, :, :, 0]

        # Parameters
        sigma = self.parameters.loc[start_dates, "vol"].values
        a = self.parameters.loc[start_dates, "reversion"].values

        # Intermediary steps
        e_int = np.exp(-a * intervals)
        variance = sigma**2 / (2 * a) * (1 - e_int**2)
        t = np.add.accumulate(intervals, axis=0)
        # alpha = forwards + (sigma * (1 - np.exp(-a * t)) / a) ** 2 / 2
        T = np.sum(intervals, axis=0)
        MT = (sigma / a) ** 2 * (
            (1 - e_int) - (np.exp(-a * (T - t)) * (1 - e_int**2)) / 2
        )

        # Short rate components dynamics
        shape = (
            short_rate_draws.shape[0],
            short_rate_draws.shape[1] + 1,
            short_rate_draws.shape[2],
        )
        short_rate = np.full(shape, np.nan)
        short_rate[:, 0] = self.get_risk_factors(start_dates)
        # import matplotlib.pyplot as plt
        # plt.plot(alpha)
        if short_rate_draws.shape[1] > 1:
            short_rate[:, 1] = -MT[0] + np.sqrt(variance[0]) * short_rate_draws[:, 0]
        for i in range(short_rate.shape[1] - 2):
            mean = short_rate[:, i + 1] * e_int[i + 1] - MT[i + 1]
            short_rate[:, i + 2] = (
                mean + np.sqrt(variance[i + 1]) * short_rate_draws[:, i + 1]
            )

        return short_rate.reshape(short_rate.shape + (1,))

    def generate_short_rate(
        self, start_dates, step_dates, short_rate_draws, intervals, forward_measure=True
    ):
        # Data
        short_rate_values = self.get_value(start_dates)
        step_plus_one = step_dates + pd.Timedelta("1D")
        fbonds = self.spot_curve.compute_fbond(
            start_dates, step_dates, step_plus_one, self.calendar
        )
        # IF ANYTHING FAILS; CHECK THIS FIRST (DISCOUNT CURVE MIGHT BE WORKING BADLY)
        forwards = -np.log(fbonds) * self.calendar.days_in_year
        pd.Series(
            forwards.reshape((forwards.size,)), index=step_dates, name="forwards"
        ).plot()

        # Parameters
        sigma = self.parameters.loc[start_dates, "vol"].values
        a = self.parameters.loc[start_dates, "reversion"].values

        # Short rate components dynamics
        t = np.add.accumulate(intervals, axis=0)
        alpha = forwards + (sigma * (1 - np.exp(-a * t)) / a) ** 2 / 2
        short_rate_components = self.generate_paths(
            start_dates=start_dates,
            step_dates=step_dates,
            short_rate_draws=short_rate_draws,
            intervals=intervals,
            forward_measure=forward_measure,
        )
        shape = (
            short_rate_draws.shape[0],
            short_rate_draws.shape[1] + 1,
            short_rate_draws.shape[2],
            1,
        )
        short_rate = np.full(shape, np.nan)
        short_rate[:, 0] = short_rate_values
        short_rate[:, 1:, :, 0] = short_rate_components[:, 1:, :, 0] + alpha

        return short_rate

    def get_numerical_parameters(self, *k):
        initial_guess = np.array([0.001, 0.1])
        bounds = [(0, 1), (-1, 1)]
        return initial_guess, bounds

    def get_risk_factors(self, dates):
        return 0


class G2PlusPlusShortRate(ShortRateModel):
    """
    A class representing the G2++ short rate model, inheriting from ``ShortRateModel``.

    The G2++ model is a two-factor model that enhances the Gaussian short rate models by incorporating two sources of risk. It is particularly known for its ability to capture a wider variety of term structures and interest rate dynamics compared to single-factor models.

    Parameters
    ----------
    discount_curve : pricing.discount_curves.DiscountCurve
        The initial discount curve used in the G2++ model.

    calendar : data.calendars.DayCountCalendar
        The calendar object used for day count conventions and date calculations in the model.

    Attributes
    ----------
    parameters : pandas.DataFrame
        A DataFrame to hold the parameters of the G2++ model, including volatilities ('vol_x', 'vol_y'),
        speed of mean reversion ('reversion_x', 'reversion_y'), and the correlation between the two factors.

    no_factors : int
        The number of factors in the model, which is 2 for the G2++ model.

    recover_spot : bool
        Indicates whether the spot rate can be recovered from the model, which is True for the G2++ model.

    Notes
    -----
    - In the G2++ model, the short rate is assumed to follow a stochastic process defined as:

      .. math::
        r(t) = x(t) + y(t) + \\varphi(t)\,,

      where :math:`x(t)` and :math:`y(t)` are two independent Gaussian processes with their own volatilities and mean reversion rates,
      and :math:`\\varphi(t)` is a deterministic shift function ensuring the fit to the initial term structure.

    - The model is capable of fitting a wide variety of term structures due to its two-factor nature, and it allows for the correlation between these factors.

    See Also
    --------
    ShortRateModel : The parent class from which this class inherits.

    References
    ----------
    - [Brigo and Mercurio, 2006] Brigo & Mercurio (2006). Interest Rate Models–Theory and Practice. Chapter 4.
    """

    def __init__(self, discount_curve, calendar):
        ShortRateModel.__init__(self, discount_curve, calendar)
        self.parameters = pd.DataFrame(
            columns=["vol_x", "reversion_x", "vol_y", "reversion_y", "correlation"]
        )
        self.no_factors = 2
        self.recover_spot = True

    def cap_price_function(
        self,
        params,
        strikes,
        maturities,
        pay_freq,
        all_T,
        all_deltas,
        all_bonds,
        return_caplets=False,
    ):
        """Return_caplet only implemented for this model."""
        sigma = params[0]
        a = params[1]
        eta = params[2]
        b = params[3]
        rho = params[4]

        sigma_p = {
            n: np.sqrt(
                sigma**2
                / (2 * a**3)
                * (1 - np.exp(-a * all_deltas[n])) ** 2
                * (1 - np.exp(-2 * a * all_T[n][:-1]))
                + eta**2
                / (2 * b**3)
                * (1 - np.exp(-b * all_deltas[n])) ** 2
                * (1 - np.exp(-2 * b * all_T[n][:-1]))
                + 2
                * rho
                * (sigma * eta / (a * b * (a + b)))
                * (1 - np.exp(-a * all_deltas[n]))
                * (1 - np.exp(-b * all_deltas[n]))
                * (1 - np.exp(-(a + b) * all_T[n][:-1]))
            )
            for n in all_T.keys()
        }
        if return_caplets:
            model_prices = {}
        else:
            model_prices = np.full(strikes.size, np.nan)
        for i in range(strikes.size):
            n = pay_freq[i]
            temp_bonds = all_bonds[n][: int(12 / n) * maturities[i] + 1]
            temp_deltas = all_deltas[n][: int(12 / n) * maturities[i]]
            temp_sigma = sigma_p[n][: int(12 / n) * maturities[i]]
            mask = temp_sigma == 0
            caplets = np.empty_like(temp_sigma)
            caplets[mask] = (
                -(1 + strikes[i] * temp_deltas)
                * temp_bonds[1:]
                * self.N(
                    np.sign(
                        np.log(
                            temp_bonds[:-1]
                            / ((1 + strikes[i] * temp_deltas) * temp_bonds[1:])
                        )
                    )
                    * np.inf
                )
                + temp_bonds[:-1]
                * self.N(
                    np.sign(
                        np.log(
                            temp_bonds[:-1]
                            / ((1 + strikes[i] * temp_deltas) * temp_bonds[1:])
                        )
                    )
                    * np.inf
                )
            )[mask]
            caplets[~mask] = -(1 + strikes[i] * temp_deltas[~mask]) * temp_bonds[1:][
                ~mask
            ] * self.N(
                np.log(
                    temp_bonds[:-1][~mask]
                    / ((1 + strikes[i] * temp_deltas[~mask]) * temp_bonds[1:][~mask])
                )
                / temp_sigma[~mask]
                - temp_sigma[~mask] / 2
            ) + temp_bonds[:-1][~mask] * self.N(
                np.log(
                    temp_bonds[:-1][~mask]
                    / ((1 + strikes[i] * temp_deltas[~mask]) * temp_bonds[1:][~mask])
                )
                / temp_sigma[~mask]
                + temp_sigma[~mask] / 2
            )
            if return_caplets:
                model_prices[strikes[i]] = caplets
            else:
                model_prices[i] = np.sum(caplets)
        return model_prices

    def compute_future_bond_pieces(
        self,
        dates,
        beg_dates,
        end_dates,
        calendar,
        sigma,
        a,
        eta,
        b,
        rho,
        return_times=False,
    ):
        """A, B's of the future bond, see [Brigo and Mercurio, 2006]. Only done for this model."""
        # Time variables
        t = np.full((beg_dates.size, dates.size), np.nan)
        for i in range(t.shape[0]):
            t[i] = calendar.interval(dates, beg_dates[i])
        T = np.full((end_dates.size, dates.size), np.nan)
        for i in range(T.shape[0]):
            T[i] = calendar.interval(dates, end_dates[i])
        # taus = calendar.interval(beg_dates, end_dates)
        # taus = taus.reshape((taus.size, 1))

        # Calculation of V_exp:=1/2[V(t,T) - V(0,T) + V(0,t)]
        aux0 = (np.exp((3.0 * (b * t)))) + (
            (-4.0 * (np.exp((b * ((2.0 * t) + T)))))
            + (3.0 * (np.exp((b * (t + (2.0 * T))))))
        )
        aux1 = (-1.0 + (np.exp((b * t)))) * (
            (((np.exp((2.0 * (b * t)))) + aux0) - (np.exp((2.0 * (b * T))))) * (eta**2)
        )
        aux2 = (-1.0 + (np.exp((b * t)))) * (
            ((np.exp((b * t))) - (np.exp((b * T)))) * (eta * (rho * sigma))
        )
        aux3 = (np.exp((3.0 * (a * t)))) + (
            (-4.0 * (np.exp((a * ((2.0 * t) + T)))))
            + (3.0 * (np.exp((a * (t + (2.0 * T))))))
        )
        aux4 = (-1.0 + (np.exp((a * t)))) * (
            (((np.exp((2.0 * (a * t)))) + aux3) - (np.exp((2.0 * (a * T)))))
            * (sigma**2)
        )
        aux5 = (np.divide(((b**-2.0) * ((np.exp((-b * (t + T)))) * aux2)), a)) + (
            -0.25 * ((a**-3.0) * ((np.exp((-2.0 * (a * (t + T))))) * aux4))
        )
        aux6 = (np.exp((b * (t + T)))) * (
            ((np.exp((2.0 * (a * t)))) + (np.exp((a * T)))) - (np.exp((a * t)))
        )
        aux7 = (((np.exp((2.0 * ((a + b) * t)))) + (np.exp(((a + b) * T)))) - aux6) - (
            np.exp(((a + b) * t))
        )
        aux8 = (np.exp((b * (t + T)))) * (
            (-1.0 + (np.exp((a * t)))) * ((np.exp((a * t))) - (np.exp((a * T))))
        )
        aux9 = (a**-2.0) * (
            (np.exp((-(a + b) * (t + T))))
            * (((a * aux7) - (b * aux8)) * (eta * (rho * sigma)))
        )
        V_exp = (
            (-0.25 * ((b**-3.0) * ((np.exp((-2.0 * (b * (t + T))))) * aux1))) + aux5
        ) - (np.divide((np.divide(aux9, (a + b))), b))

        # Bond calculations
        market_forward_bonds = self.spot_curve.compute_fbond(
            dates, beg_dates, end_dates, calendar
        )

        if return_times:
            return (
                market_forward_bonds * np.exp(V_exp),
                (1 - np.exp(-a * (T - t))) / a,
                (1 - np.exp(-b * (T - t))) / b,
                t,
                T,
            )
        else:
            return (
                market_forward_bonds * np.exp(V_exp),
                (1 - np.exp(-a * (T - t))) / a,
                (1 - np.exp(-b * (T - t))) / b,
            )

    def compute_future_bond(
        self,
        dates,
        beg_dates,
        end_dates,
        calendar,
        srsim,
        params=None,
        return_pieces=False,
    ):
        # Dates formatting
        dates, beg_dates, end_dates = ShortRateModel.dates_formatting_cfb(
            dates, beg_dates, end_dates
        )

        # Parameters
        if params is None:
            sigma = self.parameters.loc[dates, "vol_x"].values
            eta = self.parameters.loc[dates, "vol_y"].values
            a = self.parameters.loc[dates, "reversion_x"].values
            b = self.parameters.loc[dates, "reversion_y"].values
            rho = self.parameters.loc[dates, "correlation"].values
        else:
            sigma = params[0]
            a = params[1]
            eta = params[2]
            b = params[3]
            rho = params[4]

        A, B_a, B_b = self.compute_future_bond_pieces(
            dates, beg_dates, end_dates, calendar, sigma, a, eta, b, rho
        )

        disc = A * np.exp(-B_a * srsim[:, :, :, 0] - B_b * srsim[:, :, :, 1])
        disc = disc[:, :, :, np.newaxis]  # For convention considerations
        return disc

    def fit_to_atm_caps(
        self,
        date,
        caps_df,
        discount_curve,
        vol_type="normal",
        initial_guess=np.array([0.0081429, 0.0234, 0.0020949, 0.0015]),
        rho=-0.2536,
    ):
        date = pd.to_datetime(date)
        caps_df = caps_df[caps_df.Maturity == 0]

        caps_df, all_T, all_deltas, all_bonds = ShortRateModel.get_market_data(
            self, date, caps_df, discount_curve, vol_type=vol_type
        )

        def difference(x):
            sigma = x[0]
            a = x[1]
            eta = x[2]
            b = x[3]

            sigma_p = {
                n: np.sqrt(
                    sigma**2
                    / (2 * a**3)
                    * (1 - np.exp(-a * all_deltas[n])) ** 2
                    * (1 - np.exp(-2 * a * all_T[n][:-1]))
                    + eta**2
                    / (2 * b**3)
                    * (1 - np.exp(-b * all_deltas[n])) ** 2
                    * (1 - np.exp(-2 * b * all_T[n][:-1]))
                    + 2
                    * rho
                    * (sigma * eta / (a * b * (a + b)))
                    * (1 - np.exp(-a * all_deltas[n]))
                    * (1 - np.exp(-b * all_deltas[n]))
                    * (1 - np.exp(-(a + b) * all_T[n][:-1]))
                )
                for n in all_T.keys()
            }

            model_prices = np.full(len(caps_df), np.nan)
            for i in caps_df.index:
                freq = caps_df.loc[i, "Pay Frequency"]
                maturity = caps_df.loc[i, "Tenor"]
                strike = caps_df.loc[i, "Strike"]
                temp_bonds = all_bonds[freq][: int(12 / freq) * maturity + 1]
                temp_deltas = all_deltas[freq][: int(12 / freq) * maturity]
                temp_sigma = sigma_p[freq][: int(12 / freq) * maturity]
                caplets = -(1 + strike * temp_deltas) * temp_bonds[1:] * self.N(
                    np.log(
                        temp_bonds[:-1] / ((1 + strike * temp_deltas) * temp_bonds[1:])
                    )
                    / temp_sigma
                    - temp_sigma / 2
                ) + temp_bonds[:-1] * self.N(
                    np.log(
                        temp_bonds[:-1] / ((1 + strike * temp_deltas) * temp_bonds[1:])
                    )
                    / temp_sigma
                    + temp_sigma / 2
                )
                model_prices[i] = np.sum(caplets)

            differences = np.sum((caps_df.Price.values - model_prices) ** 2)
            return differences

        minimum = optimize.minimize(
            difference,
            initial_guess,
            method="powell",
            bounds=[(0, 0.5), (0, 1), (0, 0.5), (0, 1)],
            options={"xatol": 1e-8, "fatol": 1e-8, "disp": True},
        ).x

        self.parameters.loc[date] = [
            minimum[0],
            minimum[1],
            minimum[2],
            minimum[3],
            rho,
        ]

    def generate_paths(
        self, start_dates, step_dates, short_rate_draws, intervals, forward_measure=True
    ):
        # Data
        # step_plus_one = step_dates + pd.Timedelta("1D")
        # fbonds = self.spot_curve.compute_fbond(start_dates, step_dates, step_plus_one, self.calendar)
        # # IF ANYTHING FAILS; CHECK THIS FIRST (DISCOUNT CURVE MIGHT BE WORKING BADLY)
        # forwards = -np.log(fbonds) * self.calendar.days_in_year
        # if short_rate_draws.shape[2] == 1:  # Plot if there is only one valuation date
        #     print("'Infinitesimal' forwards plot")
        #     series = pd.Series(forwards.reshape((forwards.size,)), index=step_dates, name="forwards")
        #     df = pd.DataFrame(series)
        #     fig = px.line(df, y='forwards')
        #     fig.show()

        # Parameters
        sigma = self.parameters.loc[start_dates, "vol_x"].values
        eta = self.parameters.loc[start_dates, "vol_y"].values
        a = self.parameters.loc[start_dates, "reversion_x"].values
        b = self.parameters.loc[start_dates, "reversion_y"].values
        rho = self.parameters.loc[start_dates, "correlation"].values

        # Intermediate steps
        exp_inter_a = np.exp(-a * intervals)
        exp_inter_b = np.exp(-b * intervals)
        exp_inter_apb = np.exp(-(a + b) * intervals)
        t = np.add.accumulate(intervals, axis=0)
        T = np.sum(intervals, axis=0)

        variance_a = sigma**2 / (2 * a) * (1 - exp_inter_a**2)
        variance_b = eta**2 / (2 * b) * (1 - exp_inter_b**2)
        MT_x = (
            ((sigma / a) ** 2 + rho * (sigma * eta) / (a * b)) * (1 - exp_inter_a)
            - (sigma / a) ** 2 * 1 / 2 * (np.exp(-a * (T - t)) * (1 - exp_inter_a**2))
            - (rho * sigma * eta)
            / (b * (b + a))
            * np.exp(-b * (T - t))
            * (1 - exp_inter_apb)
        )
        MT_y = (
            ((eta / b) ** 2 + rho * (sigma * eta) / (a * b)) * (1 - exp_inter_b)
            - (eta / b) ** 2 * (np.exp(-b * (T - t)) * (1 - exp_inter_b**2)) / 2
            - (rho * sigma * eta)
            / (a * (b + a))
            * np.exp(-a * (T - t))
            * (1 - exp_inter_apb)
        )

        # Short rate components dynamics
        shape = (
            short_rate_draws.shape[0],
            short_rate_draws.shape[1] + 1,
            short_rate_draws.shape[2],
            short_rate_draws.shape[3],
        )
        short_rate = np.full(shape, np.nan)
        short_rate[:, 0, :, :] = 0  # Initial values for the components
        if short_rate_draws.shape[1] >= 1:
            short_rate[:, 1, :, 0] = (
                -MT_x[0] + np.sqrt(variance_a[0]) * short_rate_draws[:, 0, :, 0]
            )
            short_rate[:, 1, :, 1] = -MT_y[0] + (
                rho * np.sqrt(variance_b[0]) * short_rate_draws[:, 0, :, 0]
                + np.sqrt(1 - rho**2)
                * np.sqrt(variance_b[0])
                * short_rate_draws[:, 0, :, 1]
            )
        for j in range(
            2, short_rate.shape[1]
        ):  # Broadcasting cannot work for recursive definitions
            short_rate[:, j, :, 0] = (
                short_rate[:, j - 1, :, 0] * exp_inter_a[j - 1]
                - MT_x[j - 1]
                + np.sqrt(variance_a[j - 1]) * short_rate_draws[:, j - 1, :, 0]
            )
            short_rate[:, j, :, 1] = (
                short_rate[:, j - 1, :, 1] * exp_inter_b[j - 1]
                - MT_y[j - 1]
                + (
                    rho * np.sqrt(variance_b[j - 1]) * short_rate_draws[:, j - 1, :, 0]
                    + np.sqrt(1 - rho**2)
                    * np.sqrt(variance_b[j - 1])
                    * short_rate_draws[:, j - 1, :, 1]
                )
            )
        return short_rate

    def generate_short_rate(
        self, start_dates, step_dates, short_rate_draws, intervals, forward_measure=True
    ):
        # Parameters
        sigma = self.parameters["vol_x"].iloc[0]
        eta = self.parameters["vol_y"].iloc[0]
        a = self.parameters["reversion_x"].iloc[0]
        b = self.parameters["reversion_y"].iloc[0]
        rho = self.parameters["correlation"].iloc[0]

        # Data
        short_rate_values = self.get_value(start_dates)
        step_plus_one = step_dates + pd.Timedelta("1D")
        fbonds = self.spot_curve.compute_fbond(
            start_dates, step_dates, step_plus_one, self.calendar
        )
        # IF ANYTHING FAILS; CHECK THIS FIRST (DISCOUNT CURVE MIGHT BE WORKING BADLY)
        forwards = -np.log(fbonds) * self.calendar.days_in_year
        print("'Infinitesimal' forwards plot")
        if short_rate_draws.shape[2] == 1:  # Plot if there is only one valuation date
            print("'Infinitesimal' forwards plot")
            series = pd.Series(
                forwards.reshape((forwards.size,)), index=step_dates, name="forwards"
            )
            df = pd.DataFrame(series)
            fig = px.line(df, y="forwards")
            fig.show()

        # Short rate dynamics
        t = np.add.accumulate(intervals, axis=0)
        var_phi = (
            forwards
            + (sigma * (1 - np.exp(-a * t)) / a) ** 2 / 2
            + (sigma * (1 - np.exp(-b * t)) / b) ** 2 / 2
            + rho
            * (sigma * eta)
            / (a * b)
            * (1 - np.exp(-a * t))
            * (1 - np.exp(-b * t))
        )
        short_rate_components = self.generate_paths(
            start_dates=start_dates,
            step_dates=step_dates,
            short_rate_draws=short_rate_draws,
            intervals=intervals,
            forward_measure=forward_measure,
        )
        shape = (
            short_rate_draws.shape[0],
            short_rate_draws.shape[1] + 1,
            short_rate_draws.shape[2],
            1,
        )
        short_rate = np.full(shape, np.nan)
        short_rate[:, 0] = short_rate_values
        short_rate[:, 1:, :, 0] = (
            short_rate_components[:, 1:, :, 0]
            + short_rate_components[:, 1:, :, 1]
            + var_phi
        )  # var_phi has no axis for simulations but numpy assumes
        # it is the same for the first index, i.e.,   short_rate_components[:, 1:, :, 1] + var_phi == short_rate_components[:, 1:, :, 1] + var_phi_rep where
        # var_phi_rep = np.repeat(var_phi[np.newaxis, :], short_rate_draws.shape[0], axis=0) (we add a new axis, the first, and fill it with repeated values)

        return short_rate

    def get_factors_correlation(self):
        return np.array([[1, self.parameters["correlation"].iloc[-1]], [0, 1]])

    def get_numerical_parameters(self, *k):
        initial_guess = np.array([0.0081429, 0.0234, 0.0020949, 0.0015, -0.9])
        bounds = [(0, 0.5), (0, 1), (0, 0.5), (0, 1), (-1, 1)]
        return initial_guess, bounds

    def get_risk_factors(self, dates):
        return 0

    def swaption_price_function_SP(
        self, params, swaptions_df, all_T, all_deltas, all_bonds
    ):
        """Approximation analytic formula Schrager & Pelsser. Not the same signature as in caps_price_function."""
        sigma = params[0]
        a = params[1]
        eta = params[2]
        b = params[3]
        rho = params[4]

        strikes = swaptions_df["Strike"].values
        pay_freq = swaptions_df["Pay Frequency"].values
        maturities = swaptions_df["Tenor"].values

        prices = np.full(strikes.size, np.nan)
        for i in range(strikes.size):
            freq = pay_freq[i]
            bonds = all_bonds[freq][: int(12 / freq) * maturities[i] + 1]
            taus = all_deltas[freq][: int(12 / freq) * maturities[i]]
            T = all_T[freq][: int(12 / freq) * maturities[i] + 1]
            n = int(12 / freq) * maturities[i]

            BPV_0 = sum(taus * bonds[1:])  # Only evaluated at t=0
            PVBP = np.full(bonds.size, np.nan)  # Evaluated at t=0
            for j, bond in enumerate(bonds):
                PVBP[j] = bond / BPV_0
            forwards = (bonds[:-1] / bonds[1:] - 1) / taus
            strike = np.sum(taus * forwards * bonds[1:]) / np.sum(
                taus * bonds[1:]
            )  # ATM swaption
            if not strike == strikes[i]:
                raise AttributeError(
                    "The current implementation of the formula only works for ATM swaptions."
                )

            def exp_T(x):
                exp = np.full(T.size, np.nan)
                for i, time in enumerate(T):
                    exp[i] = np.exp(-x * time)
                return exp

            def C_alpha_beta(x):
                exp_T_x = exp_T(x)
                return (
                    1
                    / x
                    * (
                        np.exp(-x * T[0]) * PVBP[0]
                        - np.exp(-x * T[n]) * PVBP[n]
                        - strike * sum(taus * PVBP[1:] * exp_T_x[1:])
                    )
                )

            vol = np.sqrt(
                sigma**2 * C_alpha_beta(a) ** 2 * (np.exp(2 * a * T[0]) - 1) / (2 * a)
                + eta**2 * C_alpha_beta(b) ** 2 * (np.exp(2 * b * T[0]) - 1) / (2 * b)
                + 2
                * rho
                * sigma
                * eta
                * C_alpha_beta(a)
                * C_alpha_beta(b)
                * (np.exp((a + b) * T[0]) - 1)
                / (a + b)
            )
            prices[i] = BPV_0 * vol / np.sqrt(2 * np.pi)

        return prices  # Unit nominal

    def swaption_price_function(
        self,
        date,
        maturity_date,
        payment_dates,
        calendar,
        strike,
        params,
        omega=1,
        nominal=100,
    ):
        """Semi-closed formula of Brigo and Mercurio. The signature is completely different to swaption_price_function_SP because we need dates, not just intervals"""
        sigma = params[0]
        a = params[1]
        eta = params[2]
        b = params[3]
        rho = params[4]

        c = np.full((payment_dates.size, 1), np.nan)
        date, maturity_date, payment_dates = ShortRateModel.dates_formatting_cfb(
            date, maturity_date, payment_dates
        )
        dates = payment_dates.union(maturity_date)
        for i in range(1, dates.size):
            c[i - 1] = strike * calendar.interval(dates[i - 1], dates[i])
        c[-1] += 1

        A, B_a, B_b, t, T_old = self.compute_future_bond_pieces(
            date,
            maturity_date,
            payment_dates,
            calendar,
            sigma,
            a,
            eta,
            b,
            rho,
            return_times=True,
        )
        T = t.reshape(
            1
        )  # Following the new notation of this section, see "Notas ir_models". Also, reshape

        def jamshidian_y(x):
            def jamshidian_f(y):
                j_f = np.sum(c * A * np.exp(-B_a * x - B_b * y), axis=0) - 1
                j_f = np.float(j_f)
                return j_f

            # data = [np.float(jamshidian_f(i)) for i in np.arange(-10, 10, 0.01)]
            # index = [i for i in np.arange(-10, 10, 0.01)]
            # df = pd.DataFrame(data, index=index, columns=["Jamshidian Function"])
            # fig = px.line(df)
            # fig.show()

            inf_b = -10
            while jamshidian_f(inf_b) < 0:
                sup_b = inf_b
                inf_b -= 10

            try:
                sup_b
            except NameError:
                sup_b = 10
            while jamshidian_f(sup_b) > 0:
                inf_b = sup_b
                sup_b += 10

            sol = optimize.bisect(jamshidian_f, inf_b, sup_b)
            if np.abs(jamshidian_f(sol)) > 10 ** (-5):
                print("Malfunctioning of Jamshidian solution.\n.")
                print("x", x)
                print("Solution", sol)
                print("Minimum, J_x(sol)=", jamshidian_f(sol))
                print("A", A)
                print("B", B_a, B_b)
                print("c", c)
                # data = [np.float(jamshidian_f(i)) for i in np.arange(-10, 10, 0.01)]
                # index = [i for i in np.arange(-10, 10, 0.01)]
                # df = pd.DataFrame(data, index=index, columns=["Jamshidian Function"])
                # fig = px.line(df)
                # fig.show()
                # raise ValueError("Check the solution.")

            return sol

        # Similar to generate_paths, but some changes needed. To unify it in a common method seems more complex than doing this
        exp_a = np.exp(-a * T)
        exp_b = np.exp(-b * T)
        exp_apb = np.exp(-(a + b) * T)
        variance_a = sigma**2 / (2 * a) * (1 - exp_a**2)
        variance_b = eta**2 / (2 * b) * (1 - exp_b**2)
        MT_x = (
            ((sigma / a) ** 2 + rho * (sigma * eta) / (a * b)) * (1 - exp_a)
            - (sigma / a) ** 2 * 1 / 2 * (1 - exp_a**2)
            - (rho * sigma * eta) / (b * (b + a)) * (1 - exp_apb)
        )
        MT_y = (
            ((eta / b) ** 2 + rho * (sigma * eta) / (a * b)) * (1 - exp_b)
            - (eta / b) ** 2 * (1 - exp_b**2) / 2
            - (rho * sigma * eta) / (a * (b + a)) * (1 - exp_apb)
        )

        # Parameters
        mu_x = -MT_x
        mu_y = -MT_y
        sigma_x = np.sqrt(variance_a)
        sigma_y = np.sqrt(variance_b)
        rho_xy = (rho * sigma * eta) / ((a + b) * sigma_x * sigma_y) * (1 - exp_apb)

        # Functions
        def h1(x):
            return (jamshidian_y(x) - mu_y) / (sigma_y * np.sqrt(1 - rho_xy**2)) - (
                rho_xy * (x - mu_x)
            ) / (sigma_x * np.sqrt(1 - rho_xy**2))

        def h2(x):
            return h1(x) + B_b * sigma_y * np.sqrt(1 - rho_xy**2)

        def lambda_f(x):
            return c * A * np.exp(-B_a * x)

        def kappa(x):
            return -B_b * (
                mu_y
                - 1 / 2 * (1 - rho_xy**2) * sigma_y**2 * B_b
                + rho_xy * sigma_y * (x - mu_x) / sigma_x
            )

        Phi = np.vectorize(norm.cdf)
        phi = np.vectorize(norm(loc=mu_x, scale=sigma_x).pdf)

        @np.vectorize
        def integrand(x):
            return np.float(
                phi(x)
                * (
                    Phi(-omega * h1(x))
                    - np.sum(
                        lambda_f(x) * np.exp(kappa(x)) * Phi(-omega * h2(x)), axis=0
                    )
                )
            )

        # inf = -10
        # while integrand(inf) <= 0.0000001:
        #     inf += 0.01
        # sup = np.float(inf + 2*(mu_x-inf))

        inf = mu_x - sigma_x
        n = 1
        while integrand(inf) > 10 ** (-8):
            n += 1
            inf = mu_x - n * sigma_x

        inf = mu_x - n * sigma_x

        n = 1
        sup = mu_x + n * sigma_x
        while integrand(sup) > 10 ** (-8):
            n += 1
            sup = mu_x + n * sigma_x

        sup = mu_x + n * sigma_x

        delta = (sup - inf) / 1000
        data = [np.float(integrand(i)) for i in np.arange(inf, sup, delta)]
        index = [i for i in np.arange(inf, sup, delta)]
        df = pd.DataFrame(data, index=index, columns=["Integrand"])
        fig = px.line(df)
        fig.show()

        print(inf, sup)

        integral = quadrature(integrand, inf, sup)[0]
        if inf < -10 or sup > 10:
            print("Check integral_p")
        integral_p = quad(integrand, -10, 10)[0]
        integral = np.array([integral, integral_p])
        return (
            nominal
            * omega
            * self.spot_curve.get_value(date, maturity_date, self.calendar)
            * integral
        )
