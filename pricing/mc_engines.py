import numpy as np
import pandas as pd
import plotly.graph_objs as go

from scipy.stats import norm, multivariate_normal
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

try:
    from . import functions as afsfun
except (ImportError, ModuleNotFoundError, ValueError):
    import pricing.functions as afsfun


class StatisticsGatherer:
    def __init__(self):
        self.payoffs = 0
        self.no_sims = 0

    def load(self, payoffs):
        # we add every time for the case when it gathers from MC run of two products
        self.payoffs = self.payoffs + payoffs.reshape(
            payoffs.size,
        )
        self.no_sims = self.payoffs.size
        return None

    def reset_gatherer(self):
        self.payoffs = 0
        self.no_sims = 0
        return None

    def histogram(self, return_data=False):
        if self.payoffs is None:
            print("No loaded data")
            return None
        hist, bin_edges = np.histogram(self.payoffs, bins=100)
        trace = go.Histogram(
            x=self.payoffs, histnorm=""
        )  # We can change histnorm in order to normalize the histogram.
        layout = go.Layout(
            title="Histogram of the payoff",
            xaxis=dict(title="P(S_T)"),
            yaxis=dict(title=None),
        )
        fig = go.Figure(data=[trace], layout=layout)
        fig.show()
        if return_data:
            return hist
        else:
            return None

    def running_mean(self, return_data=False, plot=True):
        if self.payoffs is None:
            print("No loaded data")
            return None
        means = np.add.accumulate(self.payoffs)
        means = means / (np.arange(self.payoffs.size) + 1)
        if plot:
            import matplotlib.pyplot as plt

            plt.plot(means)
        if return_data:
            return means
        else:
            return None

    def compute_std(self, return_data=False, plot=True):
        if self.payoffs is None:
            print("No loaded data")
            return None
        # std = self.payoffs.std()
        means = self.running_mean(return_data=True, plot=False)
        std = np.zeros(self.payoffs.size - 1)
        for i in range(std.size):
            std[i] = means[i:].std()
        if plot:
            import matplotlib.pyplot as plt

            plt.plot(std)
        if return_data:
            return std
        else:
            return None


class DiffusionMC:
    """
    Implement Monte Carlo simulation techniques.

    The price of a financial product is determined using Stochastic Volatility models for the underlyings.
    Normal and Lognormal dynamics are considered as particular cases.

    Attributes
    ----------
        time_step :float, default = numpy.inf.
            Time step for the discretization in days. By default, ``time_step = numpy.inf`` so ``obs_dates = sim_dates``

        discretization_method : string, default = 'unbiased'.
            Discretization method for the simulation. By default, an unbiased generate_paths is used but generate_paths_euler
            ('euler') is also implemented in some cases.

    """

    def __init__(self, time_step=np.inf, discretization_method="unbiased"):
        self.time_step = time_step
        self.discretization_method = discretization_method
        self.draws = None  # To store draws

    def compute_intervals(self, start_dates, step_dates, calendar):
        """
        Time intervals between simulation dates and valuation dates.

        Parameters
        ----------
        start_dates : pandas.DatetimeIndex, list of strings, pandas.Timestamp, or string
            Valuation dates. It can be a single date (as a pandas.Timestamp or its string representation) or an array-like object (as a pandas.DatetimeIndex or a list of its string
            representation) containing dates.
        step_dates : pandas.DatetimeIndex, list of strings, pandas.Timestamp, or string
            Simulation dates. It can be a single date (as a pandas.Timestamp or its string representation) or an array-like object (as a pandas.DatetimeIndex or a list of its
            string representation) containing dates.
        calendar : data.calendars.DayCountCalendar
            Calendar used for the day count convention.
        Returns
        -------
        numpy.ndarray
            Array of intervals.
        References
        -----
        See Section "compute_intervals" in "MC engines Documentation.ipynb" for details.

        """
        intervals = np.zeros((start_dates.size, step_dates.size))
        for i in range(start_dates.size):
            taus = calendar.interval(start_dates[i], step_dates)
            if taus.size > 1:
                taus[1:] = taus[1:] - taus[:-1]
            intervals[i] = taus
        intervals = intervals.transpose((1, 0))
        return intervals

    def compute_correlation_matrix(self, assets, dates):
        if hasattr(assets, "components"):
            corr = assets.get_correlation_matrix(dates)
        else:
            corr = np.array([1])
        return corr

    def simulate_asset(
        self,
        assets,
        discount_curve,
        start_dates,
        simulation_dates,
        no_sims,
        calendar,
        quanto_correction=0,
        forward_measure=False,
        for_pricing=False,
        reuse_draws=False,
    ):
        """
        Simulate asset paths given start dates, simulation dates, and the number of simulations.

        Assumes independence between assets and the short rate, always works in the forward measure.
        It doesn't work in the risk-neutral measure as it doesn't keep track of the money-market account.

        Parameters
        ----------
        assets : object
            The assets object which may or may not have specific attributes such as 'components' or 'generate_rates'.

        start_dates : pandas.DatetimeIndex, list of strings, pandas.Timestamp, or string
            Starting dates for the simulations.

        simulation_dates : pandas.DatetimeIndex, list of strings, pandas.Timestamp, or string
            Dates for the simulation intervals.

        no_sims : int
            Number of simulations.

        calendar : data.calendars.DayCountCalendar
            Calendar used for the day count convention.

        quanto_correction : float, default=0
            Quanto correction factor.

        discount_curve : pricing.discount_curves.DiscountCurve
            Gives the discounts for the computation of the T-forward measure numeraires.

        forward_measure : bool, default=True
            Indicator for whether to use forward measure. Note: the function always uses forward measure regardless of this input.

        for_pricing : bool, default=False
            Argument added for consistency; not used in the function.

        reuse_draws : bool, default=False
            If True, the method uses previously computed draws. Otherwise, new draws are generated.
            This option can be useful for computing greeks in Monte Carlo.

        Returns
        -------
        numpy.ndarray
            Array of simulated asset paths.

        """
        # matrix intervals
        start_dates, simulation_dates = afsfun.dates_formatting(
            start_dates, simulation_dates
        )
        intervals = self.compute_intervals(
            start_dates=start_dates, step_dates=simulation_dates, calendar=calendar
        )
        if not reuse_draws:
            draws = assets.generate_draws(
                no_obsdates=intervals.shape[0],
                no_valuation_dates=intervals.shape[1],
                no_sims=no_sims,
                start_dates=start_dates,
            )
            self.draws = draws
        else:
            draws = self.draws
        if self.discretization_method == "unbiased":
            paths = assets.generate_paths(
                start_dates=start_dates,
                step_dates=simulation_dates,
                draws=draws,
                intervals=intervals,
                discount_curve=discount_curve,
                forward_measure=forward_measure,
            )
        else:
            method_name = "generate_paths_" + self.discretization_method
            if hasattr(assets, method_name):
                paths = getattr(assets, method_name)(
                    start_dates=start_dates,
                    step_dates=simulation_dates,
                    draws=draws,
                    intervals=intervals,
                    forward_measure=forward_measure,
                )
            else:
                methods_names = [
                    str(method)
                    for method in dir(assets)
                    if method.startswith("generate_paths")
                ]
                raise AttributeError(
                    f"{type(assets)} does not have the method {method_name}. The available methods are: {methods_names}."
                )

        if forward_measure:
            to_steps = np.full((simulation_dates.size + 1, start_dates.size), np.nan)
            to_steps[0] = discount_curve.get_value(
                dates=start_dates,
                future_dates=simulation_dates[-1],
                calendar=calendar,
            )
            for i in range(simulation_dates.size):
                to_steps[i + 1] = discount_curve.compute_fbond(
                    dates=start_dates,
                    beg_dates=simulation_dates[i],
                    end_dates=simulation_dates[-1],
                    calendar=calendar,
                )
            to_steps = to_steps.reshape(to_steps.shape + (1,))
            # Since assets.generate_paths gives paths = S_t/N_t,
            # the forward price is backed out of paths by multiplying by N_t. Assuming deterministic rates,
            # N_t = forward bond = to_steps, so it turns out to be:
            paths = paths * to_steps
        # adjust for currency effects
        if quanto_correction != 0:
            I = np.ones(draws.shape)
            quanto_correction = (quanto_correction * intervals).reshape(
                (intervals.shape + (1,))
            ) * I
            quanto_correction_big = np.ones((draws.shape[0],) + intervals.shape)
            quanto_correction_big[:, 1:, :] = np.exp(
                np.add.accumulate(quanto_correction, 1)
            )
            paths = quanto_correction_big * paths

        return paths

    def generate_paths_for_pricing(
        self, product, dates, discount_curve, no_sims, reuse_draws=False
    ):
        """
        Generate paths for pricing a financial product through simulations.

        We override this method to introduce more simulation dates (needed for discretization).

        Parameters
        ----------
        product : object
            The financial product to be valued.
        dates : pandas.DatetimeIndex, list of strings, pandas.Timestamp, or string
            Valuation dates for which paths will be generated. It can be a single date (as a pandas.Timestamp or its string representation) or an array-like object (as a
            pandas.DatetimeIndex or a list of its string representation) containing dates.
        discount_curve : pricing.discount_curves.DiscountCurve
            The discount curve used in valuation.
        no_sims : int
            The number of simulations to perform.
        reuse_draws: bool, default=False
            Reuse the draws. Default is ``False".

        Returns
        -------
        list of tuples
            A list of tuples containing simulation information. Each tuple contains an array of dates which Simulate dates for this iteration (d), an array which Simulates paths
            for the financial product (paths) and an int which indicates the number of past observations considered in the simulation (n).
        dict
            Dictionary mapping observation dates to observation indices.

        Notes
        -----
        This function generates simulated paths for valuing a financial product using a Monte Carlo simulation approach.
        It is adjusted to include more simulation dates and consider past observations if necessary.
        """
        dates = afsfun.dates_formatting(dates)
        underlying = product.underlying
        # if np.asarray(underlyings).shape == (): underlyings = [underlyings]
        calendar = product.calendar
        if dates.size == 0:
            print("No data available for any of the given dates; aborting simulation.")
            return pd.Series(name="Price ({} sims)".format(no_sims))
        # checking if there are data for past observation dates
        obs_dates = product.obsdates
        if product.pastmatters:
            tempobspast = obs_dates[obs_dates < dates[-1]]
            if tempobspast.size != 0:
                # n = min([pd.DatetimeIndex.intersection(tempobspast, underlying[i].get_dates()).size for i in range(underlying.no_components)])
                n = pd.DatetimeIndex.intersection(
                    tempobspast, underlying.get_dates()
                ).size
                if n != tempobspast.size:
                    print(
                        "Data for past observation dates insufficient for simulation."
                    )
                    pass
        # sorting cases of valuation between observation dates
        # adding [] just simplifies the filtering expression
        # We define the simulation dates and change obs_dates -> sim_dates (indeed, the code in DeterministicVolDiffusionMC is a particular case of
        # this one, in which simulation_dates = obs_dates).
        dt = self.time_step
        if dt == np.inf:  # Normal or Lognormal
            inter_dates = obs_dates
        else:
            inter_dates = pd.date_range(
                dates[0],
                obs_dates[-1],
                freq=pd.DateOffset(days=dt),
                inclusive="neither",
            )
            # We generate intermediate dates
            # between the first valuation date and the last observation date.
        sim_dates = inter_dates.union(obs_dates)
        # We add all the obs_dates (removing possible duplicated dates).
        L = [pd.to_datetime([])] + [dates[dates <= simdate] for simdate in sim_dates]
        L = [(L[i + 1][~L[i + 1].isin(L[i])], sim_dates[i]) for i in range(len(L) - 1)]
        # The filtered sub-list only contains elements that are not in
        #  the previous sub-list.
        L = [l for l in L if l[0].size != 0]
        # evaluating between each two observation dates
        simulations_data = []
        for d, simdate in L:
            # simplifying, since interest rates effect is very small
            step_dates = sim_dates[sim_dates >= simdate]
            if hasattr(product, "is_quanto"):
                quanto_correction = product.compute_correction(dates)
            else:
                quanto_correction = 0
            # TODO: The following method might not work properly when
            # reuse_draws=True and there are multiple dates in dates.
            # In fact, this for loop overwrites self.draws for each date in dates.
            # Consequently, it's possible that the number of draws for one date
            # is not the same as for another date, leading to a broadcasting error.
            # Even if the number of draws is the same for two different dates,
            # reuse_draws=True doesn't actually use the same draws for the same date,
            # since self.draws has been overwritten by draws from other dates.
            # Find a way to solve this problem.
            paths = self.simulate_asset(
                assets=underlying,
                discount_curve=discount_curve,
                start_dates=d,
                simulation_dates=step_dates,
                no_sims=no_sims,
                calendar=calendar,
                quanto_correction=quanto_correction,
                forward_measure=True,
                for_pricing=True,
                reuse_draws=reuse_draws,
            )
            paths = paths[:, 1:]  # We remove the first date (valuation date).
            mask = step_dates.isin(
                obs_dates
            )  # Boolean indexing to get a boolean array representing observation dates.
            indices = np.where(mask)[
                0
            ]  # Positions (integer indices) of observation dates.
            paths = paths[:, indices]  # We only need the prices for observation dates.
            # computing payoffs
            if product.pastmatters:
                # dealing with past
                # adding in prices of past observation dates if they matter
                dates_dic = dict(zip(obs_dates, np.arange(obs_dates.size)))
                tempobspast = obs_dates[
                    obs_dates < simdate
                ]  # This should not change, we are only obtaining the value of previous dates (we don't need to simulate).
                n = tempobspast.size
                if n != 0:
                    paths_new = np.zeros(
                        (
                            paths.shape[0],
                            paths.shape[1] + tempobspast.size,
                            paths.shape[2],
                            paths.shape[3],
                        )
                    )
                    obs_data = underlying.get_value(tempobspast).values
                    if obs_data.ndim == 1:
                        obs_data = obs_data.reshape((obs_data.size, 1))
                    obs_data = obs_data * np.ones(
                        (d.size, obs_data.shape[0], obs_data.shape[1])
                    )
                    obs_data = obs_data.transpose((1, 0, 2))
                    paths_new[:, : tempobspast.size] = obs_data
                    paths_new[:, tempobspast.size :] = paths
                    paths = paths_new
            else:
                n = 0
                obs_dates_2 = obs_dates[
                    obs_dates >= simdate
                ]  # This is the definition of step_dates in generate_paths_for_pricing of DeterministicVolDiffusionMC.
                dates_dic = dict(
                    zip(obs_dates_2, np.arange(obs_dates_2.size))
                )  # We also need to change step_dates here and use obs_dates_2.
            simulations_data.append((d, paths, n, dates_dic))
        return simulations_data

    def compute_disc_payoff_from_simulations(
        self, product, discount_curve, simulation_data
    ):
        """
        This method computes discounted payoffs from simulation data.

        Parameters
        ----------
        product : object
            The financial product to be valued.

        discount_curve : pricing.discount_curves.DiscountCurve
            An object that represents a discount curve, which can be used to compute discount factors for different dates.

        simulation_data : list
            A list containing simulation details. It should be a tuple containing:
                - valuation_dates: A pandas DatetimeIndex indicating the dates of each valuation.
                - paths: A numpy array of shape (num_simulations, num_obs_dates, num_val_dates, num_assets) representing the simulated asset prices.
                - n: An integer (not used in this function).
                - dates_dic: A dictionary mapping pandas Timestamps to integer indices, typically indicating the position of each date in the `paths` array.

        Returns
        -------
        numpy.ndarray
            Array of discounted payoffs from simulation data.
        pandas.DatetimeIndex
            Valuation dates.

        Notes
        -----
        This method does not call for path generation, so that we can repeatedly compute discounted payoffs from the same simulation matrix.
        """
        payoffs_lists = []
        dates = pd.to_datetime([])
        for valuation_dates, paths, n, dates_dic in simulation_data:
            dates = pd.DatetimeIndex.union(dates, valuation_dates)
            # the next might not coincide with product.obsdates because we might simulate for multiple assets at once
            step_dates = pd.to_datetime(list(dates_dic.keys())).sort_values()

            payoffs = product.payoff(paths, dates_dic, n)

            # Original discount calculation
            disc_old = np.full((step_dates.size - n, valuation_dates.size), np.nan)
            for i in range(step_dates.size - n):
                disc_old[i] = discount_curve.get_value(
                    dates=valuation_dates,
                    future_dates=step_dates[n + i],
                    calendar=product.calendar,
                )

            # [ARS] New calculation. See Section 2.7, 2.8 of Brigo and Mercurio for details.
            pay_dates = (
                product.pay_dates
            )  # TODO: this only works if step_dates equals obsdates, simulation_data should be modified

            term_disc = self.compute_simulated_bonds(
                valuation_dates,
                step_dates[n:],
                step_dates[-1],
                discount_curve,
                product.calendar,
            )
            spot_disc = discount_curve.get_value(
                dates=valuation_dates,
                future_dates=step_dates[-1],
                calendar=product.calendar,
            )
            defer_disc = self.compute_simulated_bonds(
                valuation_dates,
                step_dates[n:],
                pay_dates[n:],
                discount_curve,
                product.calendar,
            )
            disc = (spot_disc * defer_disc) / term_disc
            payoffs_lists.append(np.sum(disc * payoffs, axis=1))

        # joining everything together (don't forget payoffs_lists is a list of numpys)
        all_payoffs = np.concatenate(payoffs_lists, axis=1)

        return all_payoffs, dates

    def compute_price_from_simulations(
        self, product, discount_curve, simulation_data, statistics_gatherer=None
    ):
        """
        This method computes prices from simulation data.

        Parameters
        ----------
        product : object
            The financial product to be valued.

        discount_curve : pricing.discount_curves.DiscountCurve
            An object that represents a discount curve, which can be used to compute discount factors for different dates.

        simulation_data : list
            A list containing simulation details. It should be a tuple containing:
                - valuation_dates: A pandas DatetimeIndex indicating the dates of each valuation.
                - paths: A numpy array of shape (num_simulations, num_obs_dates, num_val_dates, num_assets) representing the simulated asset prices.
                - n: An integer (not used in this function).
                - dates_dic: A dictionary mapping pandas Timestamps to integer indices, typically indicating the position of each date in the `paths` array.
        statistics_gatherer : callable, default=None
            A function or method to gather and process statistics based on the simulation results.

        Returns
        -------
        numpy.ndarray
            Array of discounted payoffs from simulation data.
        pandas.DatetimeIndex
            Valuation dates.

        Notes
        -----
        This method does not call for path generation, so that we can repeatedly compute prices from the same simulation matrix.
        """
        # payoffs_lists = []  # TODO : delete after new method is checked
        # dates = pd.to_datetime([])
        # for valuation_dates, paths, n, dates_dic in simulation_data:
        #     dates = pd.DatetimeIndex.union(dates, valuation_dates)
        #     # the next might not coincide with product.obsdates because we might simulate for multiple assets at once
        #     step_dates = pd.to_datetime(list(dates_dic.keys())).sort_values()
        #
        #     payoffs = product.payoff(paths, dates_dic, n)
        #
        #     # Original discount calculation
        #     disc_old = np.full((step_dates.size-n, valuation_dates.size), np.nan)
        #     for i in range(step_dates.size-n):
        #         disc_old[i] = discount_curve.get_value(dates=valuation_dates, future_dates=step_dates[n+i], calendar=product.calendar)
        #
        #     # [ARS] New calculation. See Section 2.7, 2.8 of Brigo and Mercurio for details.
        #     pay_dates = product.pay_dates  # TODO: this only works if step_dates equals obsdates, simulation_data should be modified
        #
        #     term_disc = self.compute_simulated_bonds(valuation_dates, step_dates[n:], step_dates[-1], discount_curve, product.calendar)
        #     spot_disc = discount_curve.get_value(dates=valuation_dates, future_dates=step_dates[-1], calendar=product.calendar)
        #     defer_disc = self.compute_simulated_bonds(valuation_dates, step_dates[n:], pay_dates[n:], discount_curve, product.calendar)
        #     disc = (spot_disc*defer_disc)/term_disc
        #     payoffs_lists.append(np.sum(disc * payoffs, axis=1))
        #
        # # joining everything together (don't forget payoffs_lists is a list of numpys)
        # all_payoffs = np.concatenate(payoffs_lists, axis=1)
        all_payoffs, dates = self.compute_disc_payoff_from_simulations(
            product, discount_curve, simulation_data
        )
        # loading the statistics gatherer, if any
        if hasattr(statistics_gatherer, "load"):
            statistics_gatherer.load(all_payoffs)
        # computing price estimates
        prices = np.mean(all_payoffs, axis=0)
        no_sims = all_payoffs.shape[0]
        price_series = pd.Series(
            prices, index=dates, dtype="float64", name="Price ({} sims)".format(no_sims)
        )
        return price_series

    def price(
        self,
        product,
        dates,
        discount_curve,
        no_sims,
        statistics_gatherer=None,
        no_calcs=1,
        reuse_draws=False,
        method=None,
    ):
        """
        Compute the price of a financial product through simulations.

        Parameters
        ----------
        product : object
            The financial product to be priced.
        dates : pandas.DatetimeIndex, list of strings, pandas.Timestamp, or string
            Valuation dates for which prices will be computed. It can be a single date (as a pandas.Timestamp or its string representation) or an array-like object (as a
            pandas.DatetimeIndex or a list of its string representation) containing dates.
        discount_curve : pricing.discount_curves.DiscountCurve
            The discount curve used in valuation.
        no_sims : int
            The number of simulations to perform.
        statistics_gatherer : object, optional
            An optional statistics gatherer object for collecting simulation results.
        no_calcs : int, default = 1
            The price is computed no_calcs times and then the mean value is returned. This can help prevent memory errors when increasing the number of simulations.
        reuse_draws : bool, default=False
            If True, the method uses previously computed draws. Otherwise, new draws are generated.
            This option can be useful for computing greeks in Monte Carlo.
        Returns
        -------
        pandas.Series
            A pandas Series containing the computed prices for each valuation date.

        Notes
        -----
        This function computes the price of a financial product using Monte Carlo simulations. It generates paths for pricing and collects simulation results based on the provided
        parameters. The computed prices are returned as a pandas Series indexed by valuation dates.
        """
        # separated path generation from pricing so we can use the same paths for sensitivity computations (e.g., tols)
        dates = afsfun.dates_formatting(dates)
        valid_dates = dates[dates <= product.maturity]
        if valid_dates.size != 0:
            prices_df = pd.DataFrame()
            for _ in range(no_calcs):
                simulations_data = self.generate_paths_for_pricing(
                    product, valid_dates, discount_curve, no_sims, reuse_draws
                )
                try:
                    prices = self.compute_price_from_simulations(
                        product,
                        discount_curve,
                        simulations_data,
                        statistics_gatherer,
                        method,
                    )
                except TypeError:  # If the class has no method attribute
                    prices = self.compute_price_from_simulations(
                        product, discount_curve, simulations_data, statistics_gatherer
                    )
                prices_df = pd.concat([prices_df, prices], axis=1)
            prices = prices_df.mean(axis=1)  # This is again a Series.
        else:
            prices = pd.Series()
        prices = prices.reindex(
            dates, fill_value=0
        )  # If valuation_dates > maturity, the prices are zero.
        return prices

    def compute_simulated_bonds(
        self, dates, beg_dates, end_dates, discount_curve, calendar
    ):
        end_dates = afsfun.dates_formatting(end_dates)
        term_disc_num = np.full(
            (end_dates.size, dates.size), np.nan
        )  # get_value cannot handle multiple beg_dates
        for j, end_date in enumerate(end_dates):
            term_disc_num[j] = discount_curve.get_value(
                dates=dates, future_dates=end_date, calendar=calendar
            )

        term_disc_den = np.full(
            (beg_dates.size, dates.size), np.nan
        )  # get_value cannot handle multiple beg_dates
        for j, beg_date in enumerate(beg_dates):
            term_disc_den[j] = discount_curve.get_value(
                dates=dates, future_dates=beg_date, calendar=calendar
            )
        term_disc = term_disc_num / term_disc_den
        term_disc = term_disc[
            np.newaxis, ...
        ]  # For convention considerations, first index reserved to simulations

        return term_disc


class DeterministicVolDiffusionMC:  # TODO: implement this class as a special case of DiffusionMC
    def __init__(self, randno_gen=multivariate_normal.rvs):
        self.randno_gen = randno_gen

    def generate_draws(self, corr, no_obsdates, no_valuation_dates, no_sims):
        """
        Generate correlated random draws based on a provided correlation matrix.

        Parameters
        ----------
        corr : numpy.ndarray
            Correlation matrix. This can be a 2D or 3D array.
        no_obsdates : int
            Number of observation dates.
        no_valuation_dates : int
            Number of valuation dates.
        no_sims : int
            Number of simulations.

        Returns
        -------
        numpy.ndarray
            A 4D array containing correlated random draws. The dimensions are
            (number of simulations, number of observation dates, number of valuation dates, number of assets).

        Notes
        -----
        This method generates random draws :math:`d` from a standard normal distribution with
        shape (``no_sims``, ``no_obsdates``) and covariance equal to the identity matrix. The method then
        reshapes the ``draws_small`` array to include the number of assets in the last dimension.

        The new ``draws`` array is created by replicating ``draws_small`` along a new axis for the
        number of valuation dates. The ``draws`` array is then transposed to put the valuation dates
        as the third dimension.

        The correlation structure is applied by multiplying the Cholesky decomposition of the correlation
        matrix :math:`B` with the corresponding slice of ``draws``. In mathematical terms, if the
        correlation matrix is :math:`C`, the Cholesky decomposition is given by :math:`B = Chol(C)`.
        The correlated draws :math:`d_c` are then computed by :math:`d_c = Bd`.

        If the ``corr`` size is 1, the correlation structure does not need to be applied, so ``b_draw``
        is set to be equal to ``draws``.

        References
        -----
        See Section "generate_draws" in "MC engines Documentation.ipynb" for details.
        """
        no_assets = corr.shape[-1]
        draws_small = self.randno_gen(
            cov=np.identity(no_assets), size=(no_sims, no_obsdates)
        )
        draws_small = draws_small.reshape((no_sims, no_obsdates) + (no_assets,))
        draws = np.full((no_valuation_dates,) + draws_small.shape, 1) * draws_small
        draws = draws.transpose((1, 2, 0, 3))
        if corr.size > 1:
            if corr.ndim == 2:
                b = np.linalg.cholesky(corr) * np.ones(
                    (no_valuation_dates,) + corr.shape
                )
            elif corr.ndim == 3 and corr.shape[2] == 1:
                b = np.linalg.cholesky(corr[0]) * np.ones(
                    (no_valuation_dates,) + corr[0].shape
                )
            else:
                b = np.full(corr.shape, np.nan)
                for i in range(corr.shape[0]):
                    b[i] = np.linalg.cholesky(corr[i])
            b_draw = np.full(draws.shape, np.nan)
            for i in range(draws.shape[0]):
                for j in range(draws.shape[1]):
                    for k in range(draws.shape[2]):
                        b_draw[i, j, k] = np.matmul(b[k], draws[i, j, k])
        else:
            b_draw = draws
        return b_draw

    def compute_intervals(self, start_dates, step_dates, calendar):
        """
        Time intervals between simulation dates and valuation dates.

        Parameters
        ----------
        start_dates : pandas.DatetimeIndex, list of strings, pandas.Timestamp, or string
            Valuation dates for which paths will be generated. It can be a single date (as a pandas.Timestamp or its string representation) or an array-like object (as a
            pandas.DatetimeIndex or a list of its string representation) containing dates.
        step_dates : pandas.DatetimeIndex, list of strings, pandas.Timestamp, or string
            Simulation dates for which paths will be generated. It can be a single date (as a pandas.Timestamp or its string representation) or an array-like object (as a
            pandas.DatetimeIndex or a list of its string representation) containing dates.
        calendar : data.calendars.DayCountCalendar
            Calendar used for the day count convention.
        Returns
        -------
        numpy.ndarray
            Array of intervals.
        References
        -----
        See Section "compute_intervals" in "MC engines Documentation.ipynb" for details.

        """
        intervals = np.zeros((start_dates.size, step_dates.size))
        for i in range(start_dates.size):
            taus = calendar.interval(start_dates[i], step_dates)
            if taus.size > 1:
                taus[1:] = taus[1:] - taus[:-1]
            intervals[i] = taus
        intervals = intervals.transpose((1, 0))
        return intervals

    def compute_correlation_matrix(self, assets, dates):
        if hasattr(assets, "components"):
            corr = assets.get_correlation_matrix(dates)
        else:
            corr = np.array([1])
        return corr

    def simulate_asset(
        self,
        assets,
        discount_curve,
        start_dates,
        simulation_dates,
        no_sims,
        calendar,
        quanto_correction=0,
        forward_measure=False,
        for_pricing=False,
    ):
        """
        This method automatically compiles from assets the parameters necessary for the MC hardcore
        for_pricing if True, disables the adjustment by numeraire, delaying it so the routine for IR sensitivity works
        """
        # matrix intervals
        start_dates, simulation_dates = afsfun.dates_formatting(
            start_dates, simulation_dates
        )
        intervals = self.compute_intervals(
            start_dates=start_dates, step_dates=simulation_dates, calendar=calendar
        )
        corr = self.compute_correlation_matrix(assets, start_dates)
        draws = self.generate_draws(
            corr=corr,
            no_obsdates=intervals.shape[0],
            no_valuation_dates=intervals.shape[1],
            no_sims=no_sims,
        )
        paths = assets.generate_paths(
            start_dates=start_dates,
            step_dates=simulation_dates,
            draws=draws,
            intervals=intervals,
            discount_curve=discount_curve,
            forward_measure=forward_measure,
        )

        if forward_measure:
            to_steps = np.full((simulation_dates.size + 1, start_dates.size), np.nan)
            to_steps[0] = discount_curve.get_value(
                dates=start_dates,
                future_dates=simulation_dates[-1],
                calendar=calendar,
            )
            for i in range(simulation_dates.size):
                to_steps[i + 1] = discount_curve.compute_fbond(
                    dates=start_dates,
                    beg_dates=simulation_dates[i],
                    end_dates=simulation_dates[-1],
                    calendar=calendar,
                )
            to_steps = to_steps.reshape(to_steps.shape + (1,))
            paths = paths * to_steps

        # adjust for currency effects
        if quanto_correction != 0:
            I = np.ones(draws.shape)
            quanto_correction = (quanto_correction * intervals).reshape(
                (intervals.shape + (1,))
            ) * I
            quanto_correction_big = np.ones((draws.shape[0],) + intervals.shape)
            quanto_correction_big[:, 1:, :] = np.exp(
                np.add.accumulate(quanto_correction, 1)
            )
            paths = quanto_correction_big * paths

        return paths

    def generate_paths_for_pricing(self, product, dates, discount_curve, no_sims):
        """
        date format: %Y%m%d
        This methods essentially just sorts all dates, calls for simulate_asset, then concatenates the data from past obsdates
        """
        dates = afsfun.dates_formatting(dates)
        underlying = product.underlying
        # if np.asarray(underlyings).shape == (): underlyings = [underlyings]
        calendar = product.calendar

        if dates.size == 0:
            print("No data available for any of the given dates; aborting simulation.")
            return pd.Series(name="Price ({} sims)".format(no_sims))
        # checking if there are data for past observation dates
        obs_dates = product.obsdates
        if product.pastmatters:
            tempobspast = obs_dates[obs_dates < dates[-1]]
            if tempobspast.size != 0:
                n = pd.DatetimeIndex.intersection(
                    tempobspast, underlying.get_dates()
                ).size
                if n != tempobspast.size:
                    print(
                        "Data for past observation dates insufficient for simulation."
                    )
                    pass
        L = [pd.to_datetime([])] + [dates[dates <= obsdate] for obsdate in obs_dates]
        L = [
            (L[i + 1][~L[i + 1].isin(L[i])], obs_dates[i]) for i in range(len(L) - 1)
        ]  # The filtered sub-list only contains elements that are not in the previous sub-list.
        L = [l for l in L if l[0].size != 0]
        simulations_data = []
        for d, obsdate in L:
            # simplifying, since interest rates effect is very small
            step_dates = obs_dates[obs_dates >= obsdate]
            if hasattr(product, "is_quanto"):
                quanto_correction = product.compute_correction(dates)
            else:
                quanto_correction = 0
            paths = self.simulate_asset(
                assets=underlying,
                discount_curve=discount_curve,
                start_dates=d,
                simulation_dates=step_dates,
                no_sims=no_sims,
                calendar=calendar,
                quanto_correction=quanto_correction,
                forward_measure=True,
                for_pricing=True,
            )
            paths = paths[:, 1:]
            # computing payoffs
            if product.pastmatters:
                # dealing with past and adding in prices of past observation dates if they matter
                dates_dic = dict(zip(obs_dates, np.arange(obs_dates.size)))
                tempobspast = obs_dates[obs_dates < obsdate]
                n = tempobspast.size
                if n != 0:
                    paths_new = np.zeros(
                        (
                            paths.shape[0],
                            paths.shape[1] + tempobspast.size,
                            paths.shape[2],
                            paths.shape[3],
                        )
                    )
                    obs_data = underlying.get_value(tempobspast).values
                    if obs_data.ndim == 1:
                        obs_data = obs_data.reshape((obs_data.size, 1))
                    obs_data = obs_data * np.ones(
                        (d.size, obs_data.shape[0], obs_data.shape[1])
                    )
                    obs_data = obs_data.transpose((1, 0, 2))
                    paths_new[:, : tempobspast.size] = obs_data
                    paths_new[:, tempobspast.size :] = paths
                    paths = paths_new
            else:
                n = 0
                dates_dic = dict(zip(step_dates, np.arange(step_dates.size)))
            simulations_data.append((d, paths, n, dates_dic))
        return simulations_data

    def compute_price_from_simulations(
        self, product, discount_curve, simulation_data, statistics_gatherer=None
    ):
        """
        Compute prices from simulation data.

        Note:
             It does not call for path generation, so that we can repeatedly compute prices from the same simulation matrix.

        """
        payoffs_lists = []
        dates = pd.to_datetime([])
        for valuation_dates, paths, n, dates_dic in simulation_data:
            dates = pd.DatetimeIndex.union(dates, valuation_dates)
            # the next might not coincide with product.obsdates because we might simulate for multiple assets at once
            step_dates = pd.to_datetime(list(dates_dic.keys())).sort_values()

            payoffs = product.payoff(paths, dates_dic, n)

            # Original discount calculation
            disc_old = np.full((step_dates.size - n, valuation_dates.size), np.nan)
            for i in range(step_dates.size - n):
                disc_old[i] = discount_curve.get_value(
                    dates=valuation_dates,
                    future_dates=step_dates[n + i],
                    calendar=product.calendar,
                )

            # [ARS] New calculation. See Section 2.7, 2.8 of Brigo and Mercurio for details.
            pay_dates = (
                product.pay_dates
            )  # TODO: this only works if step_dates equals obsdates, simulation_data should be modified

            term_disc = self.compute_simulated_bonds(
                valuation_dates,
                step_dates[n:],
                step_dates[-1],
                discount_curve,
                product.calendar,
            )
            spot_disc = discount_curve.get_value(
                dates=valuation_dates,
                future_dates=step_dates[-1],
                calendar=product.calendar,
            )
            defer_disc = self.compute_simulated_bonds(
                valuation_dates,
                step_dates[n:],
                pay_dates[n:],
                discount_curve,
                product.calendar,
            )
            disc = (spot_disc * defer_disc) / term_disc
            payoffs_lists.append(np.sum(disc * payoffs, axis=1))

        # joining everything together (don't forget payoffs_lists is a list of numpys)
        all_payoffs = np.concatenate(payoffs_lists, axis=1)
        # loading the statistics gatherer, if any
        if hasattr(statistics_gatherer, "load"):
            statistics_gatherer.load(all_payoffs)
        # computing price estimates
        prices = np.mean(all_payoffs, axis=0)
        no_sims = all_payoffs.shape[0]
        price_series = pd.Series(
            prices, index=dates, dtype="float64", name="Price ({} sims)".format(no_sims)
        )
        return price_series

    def price(
        self,
        product,
        dates,
        discount_curve,
        no_sims,
        statistics_gatherer=None,
        no_calcs=1,
    ):
        """
        Calculate the price of a financial product through Monte Carlo simulations.

        Parameters
        ----------
        product : object
            The financial product for which the price will be calculated.
        dates : date or list of dates
            Valuation dates for which the prices will be computed.
        discount_curve : pricing.discount_curves.DiscountCurve
            The discount curve used in the valuation.
        no_sims : int
            The number of simulations to perform.
        statistics_gatherer : object, optional
            An optional statistics gatherer object for collecting simulation results.
        no_calcs : int, default = 1
            The price is computed no_calcs times, and then the mean value is returned.

        Returns
        -------
        pandas.Series
            A pandas Series containing the computed prices for each valuation date.

        Notes
        -----
        This function calculates the price of a financial product using Monte Carlo simulations. It generates paths for pricing and collects simulation results based on the
        provided parameters. The computed prices are returned as a pandas Series indexed by valuation dates.
        """

        # separated path generation from pricing so we can use the same paths for sensitivity computations (e.g., tols)
        dates = afsfun.dates_formatting(dates)
        valid_dates = dates[dates <= product.maturity]
        if valid_dates.size != 0:
            prices_df = pd.DataFrame()
            for _ in range(no_calcs):
                simulations_data = self.generate_paths_for_pricing(
                    product, valid_dates, discount_curve, no_sims
                )
                prices = self.compute_price_from_simulations(
                    product, discount_curve, simulations_data, statistics_gatherer
                )
                prices_df = pd.concat([prices_df, prices], axis=1)
            prices = prices_df.mean(axis=1)  # This is again a Series.
        else:
            prices = pd.Series()
        prices = prices.reindex(dates, fill_value=0)
        return prices

    def compute_simulated_bonds(
        self, dates, beg_dates, end_dates, discount_curve, calendar
    ):
        end_dates = afsfun.dates_formatting(end_dates)
        term_disc_num = np.full(
            (end_dates.size, dates.size), np.nan
        )  # get_value cannot handle multiple beg_dates
        for j, end_date in enumerate(end_dates):
            term_disc_num[j] = discount_curve.get_value(
                dates=dates, future_dates=end_date, calendar=calendar
            )

        term_disc_den = np.full(
            (beg_dates.size, dates.size), np.nan
        )  # get_value cannot handle multiple beg_dates
        for j, beg_date in enumerate(beg_dates):
            term_disc_den[j] = discount_curve.get_value(
                dates=dates, future_dates=beg_date, calendar=calendar
            )
        term_disc = term_disc_num / term_disc_den

        return term_disc


class SRDeterministicVolDiffusionMC(DeterministicVolDiffusionMC):
    """
    Monte Carlo engine for stochastic short rate products.

    Parameters
    ----------
    short_rate : pricing.ir_models.ShortRateModel
        Short rate

    Attributes
    ----------
    sr_paths : numpy.ndarray
        Array of simulated paths

    short_rate : pricing.ir_models.ShortRateModel
        Short rate
    """

    def __init__(self, short_rate):
        DeterministicVolDiffusionMC.__init__(self)
        self.sr_paths = None
        self.short_rate = short_rate
        self.draws = None

    def simulate_short_rate(
        self, start_dates, simulation_dates, no_sims, calendar, forward_measure=True
    ):
        """
        Simulates the short rate using given parameters and stores the paths.

        This method simulates the evolution of the short rate over specified dates. It uses the provided parameters to
        generate random draws and constructs the paths of the short rate based on these draws and the underlying short
        rate model. The computed short rate paths are stored in the `sr_paths` attribute of the object.

        Parameters
        ----------
        start_dates : pandas.DatetimeIndex, list of strings, pandas.Timestamp, or string
            The starting dates for each simulation. Can be a single date or a list of dates.

        simulation_dates : datetime-like, list of datetime-like
            The dates on which the short rate will be simulated. Can be a single date or a list of dates.

        no_sims : int
            Number of simulation paths to be generated.

        calendar : data.calendars.DayCountCalendar
            Calendar used for the day count convention.

        forward_measure : bool, optional, default=True
            Indicator for whether to use forward measure. Note: the function always uses forward measure regardless of this input.

        Returns
        -------
        numpy.ndarray
            An array containing the simulated paths of the short rate.
        """
        start_dates = afsfun.dates_formatting(start_dates)
        simulation_dates = afsfun.dates_formatting(simulation_dates)

        corr = np.identity(self.short_rate.no_factors)
        intervals = self.compute_intervals(
            start_dates=start_dates, step_dates=simulation_dates, calendar=calendar
        )
        # draws = self.generate_draws(corr=corr, no_obsdates=intervals.shape[0], no_valuation_dates=intervals.shape[1], no_sims=no_sims)
        draws = self.short_rate.generate_draws(
            no_obsdates=intervals.shape[0],
            no_valuation_dates=intervals.shape[1],
            no_sims=no_sims,
        )

        self.sr_paths = self.short_rate.generate_paths(
            start_dates=start_dates,
            step_dates=simulation_dates,
            short_rate_draws=draws,
            intervals=intervals,
            forward_measure=forward_measure,
        )
        return self.sr_paths

    def simulate_asset(
        self,
        assets,
        start_dates,
        simulation_dates,
        no_sims,
        calendar,
        discount_curve=None,
        quanto_correction=0,
        forward_measure=True,
        for_pricing=False,
        reuse_draws=False,
    ):
        """
        Simulate asset paths given start dates, simulation dates, and the number of simulations.

        Assumes independence between assets and the short rate, always works in the forward measure. It doesn't work in the risk-neutral measure as it doesn't keep track
        of the money-market account.

        Parameters
        ----------
        assets : pricing.ratecurves.MultiRate or data.underlyings.MultiAsset
            The assets object which may or may not have specific attributes such as 'components' or 'generate_rates'.

        start_dates : pandas.DatetimeIndex, list of strings, pandas.Timestamp, or string
            Starting dates for the simulations.

        simulation_dates : pandas.DatetimeIndex, list of strings, pandas.Timestamp, or string
            Dates for the simulation intervals.

        no_sims : int
            Number of simulations.

        calendar : data.calendars.DayCountCalendar
            Calendar used for the day count convention.

        discount_curve : pricing.discount_curves.DiscountCurve, optional
            Curve used for discounting purposes.

        quanto_correction : float, default=0
            Quanto correction factor.

        forward_measure : bool, default=True
            Indicator for whether to use forward measure. Note: the function always uses forward measure regardless of this input.

        for_pricing : bool, default=False
            Argument added for consistency; not used in the function.

        reuse_draws : bool, default=False
            If True, the method uses previously computed draws. Otherwise, new draws are generated.
            This option can be useful for computing greeks in Monte Carlo.

        Returns
        -------
        numpy.ndarray
            Array of simulated asset paths.

        Notes
        -----
        The function handles assets with 'components' differently, by computing a correlation matrix and generating draws.
        If assets have a method 'generate_rates', the function will simulate rates using the short rate and use these to adjust the asset paths.
        In the presence of 'components' in assets, the function will adjust asset paths with respect to a discount curve and concatenate them with other paths.

        """
        # """
        # Assumes independence between assets and the short rate;
        # it always works in the forward measure (despite appearing to give the option not to);
        # in particular it does not work in the risk-neutral measure (it never keeps track of the money-market account)
        # [ARS] "for_pricing" argument added for consistency, although it is not used.
        # """
        intervals = self.compute_intervals(
            start_dates=start_dates, step_dates=simulation_dates, calendar=calendar
        )
        if hasattr(assets, "components"):
            corr_assets = DeterministicVolDiffusionMC.compute_correlation_matrix(
                self, assets, start_dates
            )
            corr = np.identity(
                corr_assets.shape[-1] + self.short_rate.no_factors
            ) * np.ones(
                (corr_assets.shape[0], corr_assets.shape[1], corr_assets.shape[2])
            )
            corr[:, : -self.short_rate.no_factors, : -self.short_rate.no_factors] = (
                corr_assets
            )
            draws = assets.generate_draws(
                corr=corr,
                no_obsdates=intervals.shape[0],
                no_valuation_dates=intervals.shape[1],
                no_sims=no_sims,
            )  # TODO: this assumes that they
            # independent, but can be easily generalized to a given correlation structure using Cholesky as in MultiAsset.generate_draws.

        else:
            if reuse_draws:
                draws = self.draws
            else:
                draws = self.short_rate.generate_draws(
                    no_obsdates=intervals.shape[0],
                    no_valuation_dates=intervals.shape[1],
                    no_sims=no_sims,
                )
                self.draws = draws

        self.sr_paths = self.short_rate.generate_paths(
            start_dates=start_dates,
            step_dates=simulation_dates,
            short_rate_draws=draws[:, :, :, -self.short_rate.no_factors :],
            intervals=intervals,
        )

        if hasattr(assets, "generate_rates"):
            # dates1, tenor_dic = assets.get_forward_dates(simulation_dates)
            # dates_prev = pd.DatetimeIndex(list(dates1.values()))
            # dates_prev = pd.Series(dates_prev).dt.date
            # total_dates = simulation_dates.union(dates_prev)
            #
            # intervals_sr = self.compute_intervals(start_dates=start_dates, step_dates=total_dates, calendar=calendar)
            # sr_sims = self.short_rate.generate_paths(start_dates=start_dates, step_dates=total_dates,
            #                                                short_rate_draws=draws[:, :, :, -self.short_rate.no_factors:],
            #                                                intervals=intervals_sr)
            # sr_sims = sr_sims[:, 1:]
            # forward_bonds = {}
            # discount_bonds = {}
            # for j, date in enumerate(simulation_dates):
            #     forward_bonds[date] = self.short_rate.compute_forward_bond(start_dates, dates1[date], tenor_dic[date], calendar, sr_sims[:, [j]])
            #     discount_bonds[date] = self.short_rate.compute_future_bond(start_dates, dates1[date], tenor_dic[date], calendar, sr_sims[:, [j]])
            # paths = assets.generate_rates(dates1, forward_bonds=forward_bonds, discounts=discount_bonds, intervals=intervals)

            sr_sims = self.sr_paths[
                :, 1:
            ]  # The first value in the second index corresponds to the starting dates
            tenor_dic = assets.get_forward_dates(simulation_dates)
            forward_bonds = {}
            discount_bonds = {}
            for j, date in enumerate(simulation_dates):
                forward_bonds[date] = self.short_rate.compute_forward_bond(
                    start_dates, date, tenor_dic[date], calendar, sr_sims[:, [j]]
                )
                discount_bonds[date] = self.short_rate.compute_future_bond(
                    start_dates, date, tenor_dic[date], calendar, sr_sims[:, [j]]
                )

            paths = assets.generate_rates(
                forward_bonds=forward_bonds, discounts=discount_bonds
            )

        else:
            paths = np.array([])

        if hasattr(assets, "components"):
            equity_paths = assets.generate_paths(
                start_dates=start_dates,
                step_dates=simulation_dates,
                draws=draws[:, :, :, : -self.short_rate.no_factors],
                intervals=intervals,
                discount_curve=discount_curve,
                forward_measure=True,
            )
            # numeraire part (forward measure): # TODO: test this part and put the arguments properly, change fbond. See my handwritten notes
            bonds_0 = discount_curve.get_value(
                dates=start_dates, future_dates=simulation_dates[-1], calendar=calendar
            )
            bonds_0 = bonds_0.reshape((start_dates.size, 1))
            bonds = self.compute_fbond(
                dates=simulation_dates,
                beg_dates=simulation_dates,
                end_dates=simulation_dates[-1],
                calendar=calendar,
                sr_path=paths[:, :, :, -self.short_rate.no_factors],
            )
            bonds = bonds.reshape(bonds.shape + (1,))
            equity_paths = equity_paths * bonds / bonds_0

            paths = np.concatenate((equity_paths, paths), axis=3)

        return paths

    def compute_fbond(self, dates, beg_dates, end_dates, calendar, sr_path=None):
        if sr_path is None:
            sr_path = self.sr_paths
        disc = self.short_rate.compute_future_bond(
            dates, beg_dates, end_dates, calendar, sr_path
        )
        return disc

    def compute_simulated_bonds(
        self, dates, beg_dates, end_dates, discount_curve, calendar
    ):
        """
        Compute bond prices based on simulated paths of short rates.

        Given valuation dates, beginning and end dates, this function determines the bond prices by utilizing the short rate paths simulated in a previous step.

        Parameters
        ----------
        dates : pandas.DatetimeIndex, list of strings, pandas.Timestamp, or string
            Valuation dates, the "calibration" time, :math:`t`, for calculating the bond prices based on the short rate paths.
        beg_dates : pandas.DatetimeIndex, list of strings, pandas.Timestamp, or string
            Beginning dates, representing the start time, :math:`T`, for the bond's validity.
        end_dates : pandas.DatetimeIndex, list of strings, pandas.Timestamp, or string
            Final dates, signifying the maturity or end time, :math:`S`, of the bond.
        discount_curve : pricing.discount_curves.DiscountCurve, optional
            Curve used for discounting purposes, although not directly used in this function, incorporated for consistency.
        calendar : data.calendars.DayCountCalendar
            Calendar utilized for day count conventions.

        Returns
        -------
        numpy.ndarray
            Returns bond prices based on simulated paths of short rates, represented as :math:`P_{t}(T,S)` where the bond pricing function is determined by the simulated
            short rates.

        Notes
        -----
        The function derives the bond prices primarily from the short rate's future bond computation which depends on the simulated paths of the short rates, `srsim`.
        """
        srsim = self.sr_paths[:, 1:]
        bond = self.short_rate.compute_future_bond(
            dates, beg_dates, end_dates, calendar, srsim
        )[:, :, :, 0]

        return bond


class RegressionMC(DiffusionMC):
    def __init__(self):
        DiffusionMC.__init__(self)

    def compute_fdiscount(self, discount_curve, dates, beg_dates, end_dates, calendar):
        disc = discount_curve.compute_fbond(dates, beg_dates, end_dates, calendar)
        return disc

    def compute_price_from_simulations_artur(
        self, product, discount_curve, simulation_data, statistics_gatherer=None
    ):
        """
        Computes the price for a given financial product based on simulation data.

        This function iterates over the provided simulation data, computes payoff and discount rates at each step,
        determines early exercise opportunities for the product, and calculates the mean price over all simulations.

        Parameters
        ----------
        product : object
            A financial product object with attributes and methods for calculating payoff, determining early exercise
            opportunities, remaining coupons and so on.

        discount_curve : pricing.discount_curves.DiscountCurve
            An object that represents a discount curve, which can be used to compute discount factors for different dates.

        simulation_data : iterable
            An iterable object that provides the simulation data to be used for the computation. Each item in the iterable
            should provide dates, paths, a scalar n, and a dictionary of dates.

        statistics_gatherer : object, optional
            An object used for gathering statistics from the simulation. If provided, its 'load' method will be called
            with the final payoff data.

        Returns
        -------
        pandas.Series
            A pandas series with dates as the index and the estimated prices as the values. The name of the series
            indicates the number of simulations used for the computation.

        Notes
        -----
        This method assumes the product has methods for computing payoffs, determining early exercise opportunities,
        and computing remaining coupons. The discount curve object is expected to have a method for getting the value
        for a range of dates.

        This function uses a LinearRegression model to predict continuation values at each step. The predictors are
        constructed as powers of the path values at each step. The specific details of this regression model are
        commented out in the code.
        """

        payoffs_lists = []
        dates = pd.to_datetime([])
        for valuation_dates, paths, n, dates_dic in simulation_data:
            valuation_dates = pd.DatetimeIndex.union(dates, valuation_dates)
            payoffs = product.payoff(paths, dates_dic, n)
            N = product.obsdates.size
            n = paths.shape[1]  # [ARS] n is renamed, which is stupid
            if n > 1:
                step_dates = pd.to_datetime(list(dates_dic.keys())).sort_values()
                # DO NOT REMOVE paths[:, 1:, :, -1] BELOW !!!!!!!!!!!!!!!!!
                # the paths[:, 1:, :, -1] is for it to work with short rates
                # at this level, it does not correspond to short rate, but it also has no effect on fdiscounts
                disc = self.compute_fdiscount(
                    discount_curve=discount_curve,
                    dates=valuation_dates,
                    beg_dates=step_dates[:-1],
                    end_dates=step_dates[1:],
                    calendar=product.calendar,
                )
                for i in range(n - 1):
                    if disc.ndim == 2:
                        payoffs = payoffs * disc[n - (i + 2)]
                    elif disc.ndim == 3:
                        payoffs = payoffs * disc[:, n - (i + 2)]
                    # If SR is simulated, column 0 will be short rate
                    values = paths[:, n - (i + 1)]
                    continuation_values = np.full(payoffs.shape, np.nan)
                    no_polys = 21
                    for j in range(values.shape[1]):
                        temp_values = values[:, j, :]
                        basis_values = np.full(temp_values.shape + (no_polys,), np.nan)
                        for d in range(basis_values.shape[2]):
                            basis_values[:, :, d] = temp_values**d
                        X = np.full(
                            (temp_values.shape[0], no_polys * temp_values.shape[1]),
                            np.nan,
                        )
                        for k in range(X.shape[0]):
                            X[k] = np.concatenate(basis_values[k])
                        # twovars = [(0, 0),
                        #            (1, 0), (0, 1),
                        #            (2, 0), (0, 2), (1, 1),
                        #            (3, 0), (0, 3), (2, 1), (1, 2),
                        #            (4, 0), (0, 4), (3, 1), (1, 3), (2, 2),
                        #            (5, 0), (0, 5), (4, 1), (1, 4), (3, 2), (2, 3),
                        #            (6, 0), (0, 6), (5, 1), (1, 5), (4, 2), (2, 4), (3, 3)
                        #            ]
                        #
                        # threevars = [(0, 0, 0),
                        #              (1, 0, 0), (0, 1, 0), (0, 0, 1),
                        #              (2, 0, 0), (0, 2, 0), (0, 0, 2), (1, 1, 0), (1, 0, 1), (0, 1, 1),
                        #              (3, 0, 0), (0, 3, 0), (0, 0, 3), (2, 1, 0), (2, 0, 1), (1, 2, 0), (0, 2, 1),
                        #              (1, 0, 2), (0, 1, 2), (1, 1, 1),
                        #              (4, 0, 0), (0, 4, 0), (0, 0, 4), (3, 1, 0), (3, 0, 1), (1, 3, 0), (0, 3, 1),
                        #              (1, 0, 3), (0, 1, 3), (2, 1, 1), (1, 2, 1), (1, 1, 2),
                        #              (5, 0, 0), (0, 5, 0), (0, 0, 5), (4, 1, 0), (4, 0, 1), (1, 4, 0), (0, 4, 1)
                        #              ]
                        # if temp_values.shape[1] == 2:
                        #     exponents = twovars
                        # elif temp_values.shape[1] == 3:
                        #     exponents = threevars
                        # else:
                        #     print("Number of underlyings must be <4")
                        #     return None
                        # X = np.full((temp_values.shape[0], len(exponents)), np.nan)
                        # for k in range(len(exponents)):
                        #     # print(exponents[k])
                        #     # print(temp_values[:,0])
                        #     # print(temp_values[:,0]**exponents[k][0])
                        #     X[:,k] = (temp_values[:,0]**exponents[k][0])*(temp_values[:,1]**exponents[k][1])
                        #     if len(exponents[k]) == 3:
                        #         X[:,k] = X[:,k] * (temp_values[:, 2]**exponents[k][2])
                        model = LinearRegression()
                        model.fit(X, payoffs[:, j])
                        continuation_values[:, j] = model.predict(X)
                    # N-i-2 because the nth date is the (n-1)th entry of vector of dates due to python convetions (starting at 0)
                    pay_dates = product.pay_dates
                    pay_dates = pay_dates[
                        (pay_dates > product.obsdates[N - i - 2])
                        & (pay_dates <= product.obsdates[N - i - 1])
                    ]
                    # DO NOT REMOVE SRSIM BELOW !!!!!!!!!!!!!!
                    # the paths[:, 1:, :, -1] is for it to work with short rates
                    # at this level, it does not correspond to short rate, but it also has no effect on fdiscounts
                    srsim = paths[:, n - (i + 2), :, -1]
                    # srsim = srsim.reshape((srsim.shape[0], 1, srsim.shape[1]))
                    discounts = self.compute_fdiscount(
                        discount_curve,
                        valuation_dates,
                        product.obsdates[N - i - 2],
                        pay_dates,
                        product.calendar,
                    )
                    payoffs = product.early_exercise(
                        continuation_values, N - i - 2, discounts
                    )
                    # print(np.mean(payoffs, axis=0))
            previous_call = product.call_dates[product.call_dates <= valuation_dates[0]]
            if previous_call.size != 0:
                previous_date = previous_call[-1]
            else:
                previous_date = product.effective_date
            next_obs = product.obsdates[product.obsdates >= valuation_dates[-1]][0]
            pay_dates = product.pay_dates[
                (product.pay_dates > previous_date) & (product.pay_dates <= next_obs)
            ]
            discounts = np.zeros((pay_dates.size, valuation_dates.size))
            for i in range(valuation_dates.size):
                boolean = pay_dates > valuation_dates[i]
                discounts[:, i][boolean] = discount_curve.get_value(
                    dates=valuation_dates[i],
                    future_dates=pay_dates[boolean],
                    calendar=product.calendar,
                )
            remaining_coupons = product.remaining_coupons(valuation_dates, discounts)
            # print(remaining_coupons)
            discount = discount_curve.get_value(
                dates=valuation_dates, future_dates=next_obs, calendar=product.calendar
            )
            # print("payoffs:", np.mean(payoffs, axis=0))
            # print("discounted payoffs:", np.mean(discount * payoffs))
            # print("remaining coupons:", remaining_coupons)
            # print(np.mean(remaining_coupons + discount * payoffs))
            payoffs_lists.append(remaining_coupons + discount * payoffs)

        # joining everything together (don't forget payoffs_lists is a list of numpys)
        all_payoffs = np.concatenate(payoffs_lists, axis=1)
        # loading the statistics gatherer, if any
        if hasattr(statistics_gatherer, "load"):
            statistics_gatherer.load(all_payoffs)
        # computing price estimates
        prices = np.mean(all_payoffs, axis=0)
        no_sims = all_payoffs.shape[0]
        price_series = pd.Series(
            prices,
            index=valuation_dates,
            dtype="float64",
            name="Price ({} sims)".format(no_sims),
        )
        return price_series

    def compute_exercise_value(self, product, discount_curve, simulation_data):
        """
        Compute the exercise value of a financial product based on the european counterpart.

        Parameters
        ----------
        product : pricing.callable.AmericanFromEuropean
            The american financial product.

        discount_curve : pricing.discount_curves.DiscountCurve
            The discount curve used for valuing the product.

        simulation_data : list
            A list containing simulation details. It should be a tuple containing:
                - valuation_dates: A pandas DatetimeIndex indicating the dates of each valuation.
                - paths: A numpy array of shape (num_simulations, num_obs_dates, num_val_dates, num_assets) representing the simulated asset prices.
                - n: An integer (not used in this function).
                - dates_dic: A dictionary mapping pandas Timestamps to integer indices, typically indicating the position of each date in the `paths` array.

        Returns
        -------
        numpy.ndarray
            An array containing the discounted payoff for each simulation path and each payment date.

        Raises
        ------
        AttributeError
            Raised if the `pastmatters` attribute of the `eur_prod` object within the product is True, which is not implemented.

        Notes
        -----
        - The method allows the modification of the `eur_prod` attributes like `maturity`, `obsdates`, and `pay_dates` on the fly for computational efficiency.
        - Only works for one simulation datum.

        """
        valuation_dates, paths, n, dates_dic = simulation_data[
            0
        ]  # TODO: only one simulation datum
        prices = paths
        pay_dates = product.pay_dates
        disc_payoff = np.full(prices.shape[:3], np.nan)
        assert prices.shape[1] == pay_dates.size
        for pay_date in pay_dates:
            j = dates_dic[pay_date]
            if not product.eur_prod.pastmatters:
                prices_temp = prices[:, j : j + 1]
                pay_dates_temp = pd.DatetimeIndex([pay_date])
                product.eur_prod.maturity = pay_date
                product.eur_prod.obsdates = pay_dates_temp
                product.eur_prod.pay_dates = pay_dates_temp
                dates_dic_temp = {
                    k: ind
                    for ind, (k, v) in enumerate(list(dates_dic.items())[j : j + 1])
                }  # Slicing items from the dictionary and resetting the values
            else:
                raise AttributeError(
                    "Not implemented."
                )  # Allows the possibility of multiple payments

            disc_payoff_temp = self.compute_disc_payoff_from_simulations(
                product.eur_prod,
                discount_curve,
                [[pd.DatetimeIndex([pay_date]), prices_temp, n, dates_dic_temp]],
            )[0]
            disc_payoff[:, j] = disc_payoff_temp

        return disc_payoff

    def compute_price_from_simulations(
        self,
        product,
        discount_curve,
        simulation_data,
        statistics_gatherer=None,
        method=("LS", 2),
    ):
        """
        Compute prices of a financial product based on given simulation data.

        The method uses regression as part of the Longstaff-Schwartz algorithm to estimate
        continuation values for an American option based on simulated paths. Specifically, the
        continuation value at a given time :math:`t` for a certain path is approximated using
        regression on some polynomial basis functions of the state at that time :math:`t`.
        These basis functions transform the raw state information into a set of explanatory
        variables for the regression, where the dependent variable is the discounted expected
        future payoffs.

        Mathematically, this is represented as, see Chapter 3 of [Andersen and Piterbarg, 2010] for details:

        .. math::
           C_t(x) = \\mathbb{E}\\left[ \\frac{N(t)}{N(T)} V_T| X_t=x \\right] \\approx f_t(x) = \\beta_0 + \\sum_{i=1}^N\\beta_i \\phi_i(x)\\,,

        where:

        - :math:`C_t(S)` is the continuation or hold value at time :math:`t` given state :math:`x`.
        - :math:`V_T(x)` is the products value at :math:`T` given state :math:`x`.
        - :math:`N(t)` is the numeraire factor at time :math:`t`.
        - :math:`f_t(x)` is the estimated continuation value function.
        - :math:`\\phi_i` is the i-th basis function.
        - :math:`\\beta_i` is the regression coefficient for the i-th basis function.

        In this method, polynomial basis functions of the form :math:`\phi(x)=\\prod_{l=0}^L x_l^{i_l}` are constructed.
        The maximum degree of the polynomial (``method[1]``) determines the number of these basis functions. The regression is then fit using these polynomial
        features of the state to approximate the continuation value. A term representing the
        payoff is concatenated to these polynomial features to construct the regression matrix :math:`X`,
        meaning the regression is also based on the simulated payoffs, adding a richer set of
        variables for predicting the continuation value, see Example 8.6.1 of [Glasserman, 2004].

        Parameters
        ----------
        product : pricing.GeneralCallable
            The financial product to be valued.
        discount_curve : pricing.discount_curves.DiscountCurve
            The curve required to determine the rate used for discounting.
        simulation_data : list of tuple
            Data from the Monte Carlo simulations. Each tuple should consist of valuation dates, paths, the number of paths,
            and a dictionary of dates.
        statistics_gatherer : callable, optional
            A function or method to gather and process statistics based on the simulation results.
        method : tuple, optional
            Method to be used for the price computation. Defaults to ("LS", 2). First element of the tuple can be either "LS"
            (Longstaff and Schwartz) or "TvR" (Tsitsiklis and van Roy), and the second element represents the degree of the polynomial used in regression. See Section 8.6.1
            of [Glasserman, 2004] for more details.

        Returns
        -------
        pandas.Series
            A pandas Series containing the computed prices with the respective dates as its index.

        Notes
        -----
        The choice of the number of polynomial basis functions for the regression is critical in accurately approximating the continuation values. While increasing the number
        of basis functions can provide a more flexible model that closely fits the simulated data, it also comes with several challenges:

        1. **Overfitting**: A model with too many basis functions is more likely to overfit the simulated data. Overfitting occurs when the model fits the noise in the data
        rather than the underlying structure. This means that the model might perform well on the given set of simulated data, but may not generalize well to new data or
        scenarios. Overfitting can lead to suboptimal exercise strategies when valuing American options.

        2. **Oscillatory Behavior**: Polynomial regression with a high degree tends to exhibit oscillatory behavior, especially near the boundaries of the data. This means
        that even if there are small deviations, the estimated continuation value can have significant swings. This oscillatory behavior is particularly
        problematic for financial derivatives, where smoothness is often more realistic and desired.

        3. **Computational Complexity**: The complexity of regression increases with the number of basis functions. This means that not only does the computation take longer,
        but it's also more memory-intensive. For large-scale Monte Carlo simulations, this can become a bottleneck.

        Given these challenges, it's essential to strike a balance between flexibility and stability. Some techniques, such as cross-validation or out-of-sample validation,
        can be employed to determine the optimal number of basis functions. It's also worth exploring non-polynomial basis functions or regularization techniques to mitigate
        some issues associated with high-degree polynomial regression.

        Raises
        ------
        ValueError
            If the length of simulation_data is not 1.
        AttributeError
            If the provided method is not supported.

        References
        ----------
        - [Andersen and Piterbarg, 2010] Andersen, L. B. G., and Piterbarg, V.V. (2010). Interest Rate Modeling.
        - [Glasserman, 2004] Glasserman, P. (2004). Monte Carlo Methods in Financial Engineering. Springer-Verlag New York.
        - [QAB, 2021] Quantitative Analytics Bloomberg L.P. (January 31, 2021). Overview of American Monte Carlo Pricing in DLIB.

        """

        no_polys = method[1]
        method = method[0]

        dates = pd.to_datetime([])
        pay_dates = product.pay_dates
        Jp = pay_dates.size
        values_list = []
        for (
            valuation_dates,
            paths,
            n,
            dates_dic,
        ) in (
            simulation_data
        ):  # TODO: does this work for several simulation_datum? See below
            dates = pd.DatetimeIndex.union(dates, valuation_dates)
            if hasattr(product, "payoff"):
                payoffs = product.payoff(paths, dates_dic, n)
            else:  # Payoff must be taken from European product
                payoffs = self.compute_exercise_value(
                    product, discount_curve, simulation_data
                )
            numeraire = self.compute_simulated_bonds(
                dates, pay_dates, pay_dates[-1], discount_curve, product.calendar
            )  # We are in the forward measure
            cont_values = np.full(
                (paths.shape[0], pay_dates.size, paths.shape[2]), np.nan
            )  # This is the same as hold values in the references
            pay_index = [dates_dic[pay_date] for pay_date in pay_dates]
            ex_values = payoffs[:, pay_index]
            values = np.full(cont_values.shape, np.nan)
            for jp in reversed(range(0, Jp)):
                if jp == Jp - 1:
                    cont_values[:, jp] = 0
                    values[:, jp] = ex_values[:, jp]
                else:
                    temp_paths = paths[:, jp, :]
                    # Old method to create the basis functions. Maybe useful if polynomials are not considered
                    # functions = [(lambda x, d=d: x ** d) for d in range(1, no_polys + 1)]
                    # # functions.append(lambda x: product.payoff(x, dates_dic, n))
                    # basis_values = np.array([function(temp_paths)for function in functions])  # Otherwise, several ones would be included for multiple assets
                    # assert np.all(basis_values[0, :, :, :] == temp_paths)
                    # basis_values = basis_values.transpose((0, 3, 1, 2))
                    # X = np.concatenate(basis_values, axis=0).transpose((1, 2, 0))
                    # ones = np.ones((X.shape[0], X.shape[1], 1))
                    # X = np.concatenate((ones, X), axis=2)
                    # payoff = product.payoff(paths[:, [jp], :], dates_dic, n).transpose((0, 2, 1))  # Only one date, and not an index for X. We use that index for the third axis
                    # X = np.concatenate((X, payoff), axis=2)

                    poly = PolynomialFeatures(degree=no_polys, include_bias=True)
                    for k in range(values.shape[2]):
                        X = poly.fit_transform(temp_paths[:, k, :])
                        # payoff = product.payoff(paths[:, [jp], :, :], dates_dic, n)[:, :, k]  # Only one date, and not an index for X. We use that index for the third axis
                        payoff = ex_values[
                            :, [jp], k
                        ]  # Only one date, and not an index for X. We use that index for the third axis
                        # assert np.allclose(ex_values[:, [jp], k], product.payoff(paths[:, [jp], :, :], dates_dic, n)[:, :, k])
                        X = np.concatenate((X, payoff), axis=1)
                        model = LinearRegression(fit_intercept=False)
                        if method == "LS":  # See Glasserman, below (8.52)
                            mask = values[:, jp + 1, k] >= 0
                        else:
                            mask = values[:, jp + 1, k] >= -np.inf
                        X_masked = X[mask]
                        y_masked = (
                            numeraire[:, jp, k] / numeraire[:, jp + 1, k]
                        ) * values[mask, jp + 1, k]
                        model.fit(X_masked, y_masked)
                        cont_values[:, jp, k] = model.predict(X)
                        # coef = model.coef_  # To see the coefficients
                    if method == "LS":
                        mask_hold = cont_values[:, jp] >= ex_values[:, jp]
                        values[:, jp] = np.where(
                            mask_hold,
                            (numeraire[:, jp] / numeraire[:, jp + 1])
                            * values[:, jp + 1],
                            ex_values[:, jp],
                        )
                    elif method == "TvR":
                        values[:, jp] = np.maximum(
                            ex_values[:, jp], cont_values[:, jp]
                        )  # np.max DOES NOT give the max of two arrays
                    else:
                        raise AttributeError(f"Method {method} is not implemented.")
            numeraire0 = self.compute_simulated_bonds(
                dates, dates, pay_dates[-1], discount_curve, product.calendar
            )[
                :, 0
            ]  # The price is numeraire adjusted, forward measure
            values_list.append(
                values[:, 0] * numeraire0 / numeraire[:, 0]
            )  # P(t, T_n)/P(T_0, T_n) is the adjustment, following Andersen and Piterbarg's notation.

        if not len(simulation_data) == 1:
            raise ValueError("Method assumes a single simulation datum.")
        prices = np.mean(values_list[0], axis=0)
        if pay_dates[0] == valuation_dates:
            assert np.allclose(
                prices, cont_values[0, 0], cont_values[-1, 0]
            )  # Comment on p.825 of [Andersen and Piterbarg, 2010]
        no_sims = values_list[0].shape[0]
        price_series = pd.Series(
            prices, index=dates, dtype="float64", name="Price ({} sims)".format(no_sims)
        )
        return price_series


class SRRegressionMC(
    RegressionMC, SRDeterministicVolDiffusionMC, DiffusionMC
):  # So method resolution order works properly
    def __init__(self, short_rate):
        # HW always simulated in forward measure (to avoid having to compute the money market account)
        RegressionMC.__init__(self)
        SRDeterministicVolDiffusionMC.__init__(self, short_rate=short_rate)


# ----------------------------------------------------------------------------------------------------------------------
# Stochastic volatility
# ----------------------------------------------------------------------------------------------------------------------


class StochasticVolDiffusionMC(DeterministicVolDiffusionMC):
    """
    Class implementing Monte Carlo simulation techniques to determine the price of a financial product using Stochastic Volatility models for the underlyings.

    Attributes
    ----------
    time_step : float
        time_step for the discretization in days.
    discretization_method : string, default = 'unbiased'
        Discretization method for the simulation. By default, an unbiased generate_paths is used but generate_paths_euler ('euler') is also implemented in some cases.
    randno_gen : function object
        Distribution for generate_draws (used in DeterministicVolDiffusionMC but not in this class).
    """

    def __init__(
        self,
        time_step,
        discretization_method="unbiased",
        randno_gen=multivariate_normal.rvs,
    ):
        DeterministicVolDiffusionMC.__init__(self, randno_gen=randno_gen)
        self.time_step = time_step  # Compared to DeterministicVolDiffusionMC, now we need to discretize.
        self.discretization_method = discretization_method

    def simulate_asset(
        self,
        assets,
        discount_curve,
        start_dates,
        simulation_dates,
        no_sims,
        calendar,
        quanto_correction=0,
        forward_measure=False,
        for_pricing=False,
    ):
        """
        This method automatically compiles from assets the parameters necessary for the MC hardcore
        for_pricing if True, disables the adjustment by numéraire, delaying it so the routine for IR sensitivity works
        """
        # matrix intervals
        start_dates, simulation_dates = afsfun.dates_formatting(
            start_dates, simulation_dates
        )
        intervals = self.compute_intervals(
            start_dates=start_dates, step_dates=simulation_dates, calendar=calendar
        )
        draws = assets.generate_draws(
            no_obsdates=intervals.shape[0],
            no_valuation_dates=intervals.shape[1],
            no_sims=no_sims,
        )
        if self.discretization_method == "unbiased":
            paths = assets.generate_paths(
                start_dates=start_dates,
                step_dates=simulation_dates,
                draws=draws,
                intervals=intervals,
                discount_curve=discount_curve,
                forward_measure=forward_measure,
            )
        else:
            method_name = "generate_paths_" + self.discretization_method
            if hasattr(assets, method_name):
                paths = getattr(assets, method_name)(
                    start_dates=start_dates,
                    step_dates=simulation_dates,
                    draws=draws,
                    intervals=intervals,
                    forward_measure=forward_measure,
                )
            else:
                methods_names = [
                    str(method)
                    for method in dir(assets)
                    if method.startswith("generate_paths")
                ]
                raise AttributeError(
                    f"{type(assets)} does not have the method {method_name}. The available methods are: {methods_names}."
                )

        # elif self.discretization_method == 'euler':
        #     if hasattr(assets, 'generate_paths_euler'):
        #         paths = assets.generate_paths(start_dates=start_dates, step_dates=simulation_dates, draws=draws, intervals=intervals, forward_measure=forward_measure)
        #     else:
        #         raise AttributeError(f"{assets.__name__} does not have the method 'generate_paths_euler'. The avaible methods are: {}")
        # else:
        #     raise AttributeError(f"{assets.__name__} does not have this method.")

        if forward_measure:
            to_steps = np.full((simulation_dates.size + 1, start_dates.size), np.nan)
            to_steps[0] = discount_curve.get_value(
                dates=start_dates,
                future_dates=simulation_dates[-1],
                calendar=calendar,
            )
            for i in range(simulation_dates.size):
                to_steps[i + 1] = discount_curve.compute_fbond(
                    dates=start_dates,
                    beg_dates=simulation_dates[i],
                    end_dates=simulation_dates[-1],
                    calendar=calendar,
                )
            to_steps = to_steps.reshape(to_steps.shape + (1,))
            paths = paths * to_steps

            # # Numeraire adjustments, same idea as above, but general formulation, valid for stochastic short rates, see below. #TODO: implement
            # N_t = self.compute_terminal_discounts(start_dates, simulation_dates, simulation_dates[-1], discount_curve, calendar)
            # N_t = N_t[:, :, np.newaxis]
            # N_0 = discount_curve.get_value(start_dates, simulation_dates[-1], calendar)
            # N_0 = N_0[np.newaxis, :, np.newaxis]
            # paths = paths * N_t/N_0
            #
            # print("Same numeraire adjustment? ", np.all(paths == paths_old))

        # adjust for currency effects
        if quanto_correction != 0:
            I = np.ones(draws.shape)
            quanto_correction = (quanto_correction * intervals).reshape(
                (intervals.shape + (1,))
            ) * I
            quanto_correction_big = np.ones((draws.shape[0],) + intervals.shape)
            quanto_correction_big[:, 1:, :] = np.exp(
                np.add.accumulate(quanto_correction, 1)
            )
            paths = quanto_correction_big * paths

        return paths

    def generate_paths_for_pricing(self, product, dates, discount_curve, no_sims):
        """
        TODO Create proper docstring.
        We override this method in order to introduce more simulation dates (we need to dicretize).
        Parameters
        ----------
        product
        dates
        discount_curve
        no_sims

        Returns
        -------

        Notes

        """
        dates = afsfun.dates_formatting(dates)
        underlying = product.underlying
        # if np.asarray(underlyings).shape == (): underlyings = [underlyings]
        calendar = product.calendar

        # for i in range(product.obsdates.size):
        #     # once get_vol is fully working on a surface, it will always have values...
        #     tenors = calendar.interval(dates, product.obsdates[i])
        #     temp = underlying.get_vol(dates, tenors, product.strike)
        #     dates = temp.index
        if dates.size == 0:
            print("No data available for any of the given dates; aborting simulation.")
            return pd.Series(name="Price ({} sims)".format(no_sims))
        # checking if there are data for past observation dates
        obs_dates = product.obsdates
        if product.pastmatters:
            tempobspast = obs_dates[obs_dates < dates[-1]]
            if tempobspast.size != 0:
                # n = min([pd.DatetimeIndex.intersection(tempobspast, underlying[i].get_dates()).size for i in range(underlying.no_components)])
                n = pd.DatetimeIndex.intersection(
                    tempobspast, underlying.get_dates()
                ).size
                if n != tempobspast.size:
                    print(
                        "Data for past observation dates insufficient for simulation."
                    )
                    pass
        # sorting cases of valuation between observation dates
        # adding [] just simplifies the filtering expression
        # We define the simulation dates and change obs_dates -> sim_dates (indeed, the code in DeterministicVolDiffusionMC is a particular case of
        # this one, in which simulation_dates = obs_dates).
        dt = self.time_step
        inter_dates = pd.date_range(
            dates[0], obs_dates[-1], freq=pd.DateOffset(days=dt), inclusive="neither"
        )  # We generate intermediate dates between the first
        # valuation date and the last observation date.
        sim_dates = inter_dates.union(
            obs_dates
        )  # We add all the obs_dates (removing possible duplicated dates).
        L = [pd.to_datetime([])] + [dates[dates <= simdate] for simdate in sim_dates]
        L = [
            (L[i + 1][~L[i + 1].isin(L[i])], sim_dates[i]) for i in range(len(L) - 1)
        ]  # The filtered sub-list only contains elements that are not in the previous sub-list.
        L = [l for l in L if l[0].size != 0]
        # if len(L) == 0:  # L is empty so there are no past obs_dates
        #     obs_past = []
        # else:
        #     obs_past = obs_dates[obs_dates < L[-1][1]]

        # evaluating between each two observation dates
        simulations_data = []
        for d, simdate in L:
            # simplifying, since interest rates effect is very small
            step_dates = sim_dates[sim_dates >= simdate]
            if hasattr(product, "is_quanto"):
                quanto_correction = product.compute_correction(dates)
            else:
                quanto_correction = 0
            paths = self.simulate_asset(
                assets=underlying,
                discount_curve=discount_curve,
                start_dates=d,
                simulation_dates=step_dates,
                no_sims=no_sims,
                calendar=calendar,
                quanto_correction=quanto_correction,
                forward_measure=True,
                for_pricing=True,
            )
            paths = paths[:, 1:]  # We remove the first date (valuation date).
            mask = step_dates.isin(
                obs_dates
            )  # Boolean indexing to get a boolean array representing observation dates.
            indices = np.where(mask)[
                0
            ]  # Positions (integer indices) of observation dates.
            paths = paths[:, indices]  # We only need the prices for observation dates.
            # computing payoffs
            if product.pastmatters:
                # dealing with past
                # adding in prices of past observation dates if they matter
                dates_dic = dict(zip(obs_dates, np.arange(obs_dates.size)))
                tempobspast = obs_dates[
                    obs_dates < simdate
                ]  # This should not change, we are only obtaining the value of previous dates (we don't need to simulate).
                n = tempobspast.size
                if n != 0:
                    paths_new = np.zeros(
                        (
                            paths.shape[0],
                            paths.shape[1] + tempobspast.size,
                            paths.shape[2],
                            paths.shape[3],
                        )
                    )
                    obs_data = underlying.get_value(tempobspast).values
                    if obs_data.ndim == 1:
                        obs_data = obs_data.reshape((obs_data.size, 1))
                    obs_data = obs_data * np.ones(
                        (d.size, obs_data.shape[0], obs_data.shape[1])
                    )
                    obs_data = obs_data.transpose((1, 0, 2))
                    paths_new[:, : tempobspast.size] = obs_data
                    paths_new[:, tempobspast.size :] = paths
                    paths = paths_new
            else:
                n = 0
                obs_dates_2 = obs_dates[
                    obs_dates >= simdate
                ]  # This is the definition of step_dates in generate_paths_for_pricing of DeterministicVolDiffusionMC.
                dates_dic = dict(
                    zip(obs_dates_2, np.arange(obs_dates_2.size))
                )  # We also need to change step_dates here and use obs_dates_2.
            simulations_data.append((d, paths, n, dates_dic))
        return simulations_data


# ----------------------------------------------------------------------------------------------------------------------
# For educational purposes only
# ----------------------------------------------------------------------------------------------------------------------


class MC1d:
    def __init__(self, randno_gen=norm.rvs):
        self.dynamics = "lognormal"
        self.randno_gen = randno_gen

    def generate_paths(self, prices, drift, vol, intervals, no_sims):
        if self.dynamics == "lognormal":
            """
            Generates N Monte-Carlo simulations of underlying, with steps spaced out as in deltas.
            We may input an array of initial prices, possibly a times series with intervals as in d.

            In detail:
            Suppose prices=[s_1 s_2 ... s_n] is an array of prices at times t_1<t_2<...<t_n, equally spaced by d
            Let deltas=[delta_1 delta_2 delta_3 ...delta_m] be the consecutive time intervals we want to evolve
            the last price s_n in the price series.
            Function outputs a 3D array S where:
            (1) on a given page S[k], the i-th row (namely, S[k,i,:]) corresponds to
                price s_1 evolved by a time interval = intervals[1,1]+...+intervals[i,1]
                price s_2 evolved by a time interval = intervals[1,2]+...+intervals[i,2]
                price s_3 evolved by a time interval = intervals[1,2]+...+intervals[i,3]
                ...
            (2) on a given page S[k], the i-th column (S[k,i,:]) corresponds to path simulation starting a t_i, s_i;
                concretely price s_i is consecutively evolved by (n-i)d+Delta_1, (n-i)d+Delta_1+Delta_2,  ...
            (3) each page is an independent simulation of the above

            prices: price or array of prices of underlying (possibly, equally spaced time series)
            drift: array of drifts of underlying
                    if 1d array, must be same length as prices (interpreted as time series of drift constant
                        for simulation steps);
                    if 2d array, i-th row corresponds to drift for the i-th step in the simulation
            vol: volatility or array of volatilities of underlying;
                 if 1d array, must be same length as prices (interpreted as time series);
                 if 2d array, i-th row corresponds to volatility for the i-th step in the simulation
            intervals: 2d array with dimensions (no. steps in simulation, number of initial prices)
                        Each column intervals[:,i] represents the distance between successive steps in the simulation,
                        starting from date corresponding to prices[i]
            no_sims: number of simulations
            """
            prices = np.asarray(prices)
            drift = np.asarray(drift)
            vol = np.asarray(vol)
            intervals = np.asarray(intervals)
            draw = self.randno_gen(size=(no_sims, intervals.shape[0], prices.size))
            I = np.ones(draw.shape)
            exponent_all = (
                drift * I - vol**2 * I / 2
            ) * intervals + vol * draw * np.sqrt(intervals)
            S = np.ones((no_sims, intervals.shape[0] + 1, prices.size))
            S[:, 1:, :] = np.exp(np.add.accumulate(exponent_all, 1))
            return prices * S
        else:
            print("Dynamics not supported")
            pass

    def simulate_asset(
        self,
        asset,
        start_dates,
        step_dates,
        no_sims,
        calendar,
        discount_curve=None,
        quanto_correction=0,
    ):
        """
        date format: %Y%m%d
        """
        start_dates = afsfun.dates_formatting(start_dates)
        if np.asarray(step_dates).shape == ():
            start_dates = [step_dates]
        step_dates = pd.to_datetime(step_dates)
        # this method just converts real world info to numerical format
        # if asset.dynamics != self.dynamics:
        #     print("Warning: simulating with a dynamics different from the asset's")
        # getting data and filtering dates for which there is none
        temp = asset.get_value(start_dates)
        prices = temp.values
        start_dates = temp.index

        # drifts
        discounts = discount_curve.get_value(
            dates=start_dates, future_dates=step_dates[-1], calendar=calendar
        )
        taus = calendar.interval(start_dates, step_dates[-1])
        interest_rates = -np.log(discounts) / taus
        divrate = asset.get_divrate(start_dates).values
        drift = interest_rates + quanto_correction - divrate

        # volatilities
        vol = asset.get_vol(start_dates).values
        # should change so that it computes local vols for each step
        # vol = np.zeros((start_dates.size, step_dates.size))
        # for i in range(start_dates.size):
        #     # tenors = calendar.interval(start_dates, step_dates[i])
        #     tenors = calendar.interval(start_dates[i], step_dates)
        #     # temp_vol = asset.get_vol(start_dates[i], tenors)
        #     # temp_vol[1:] = np.sqrt(tenors[1:]*temp_vol[1:]**2 - tenors[:-1]*temp_vol[:-1]**2)/(tenors[1:] - tenors[:-1])
        #     vol[:, i] = asset.get_vol(start_dates[i], tenors)
        # --------------------------------------------------------------------------------------------------------------
        # trial for a product
        # vol = np.array([0.6182]+[0.3960]*3+[0.4618]*5+[0.3469]*7+[0.3971]*3)
        # vol = vol.reshape(vol.size, 1)
        # --------------------------------------------------------------------------------------------------------------

        # matrix intervals
        intervals = np.zeros((start_dates.size, step_dates.size))
        for i in range(start_dates.size):
            taus = calendar.interval(start_dates[i], step_dates)
            if taus.size > 1:
                taus[1:] = taus[1:] - taus[:-1]
            intervals[i] = taus
        intervals = intervals.transpose((1, 0))

        # generating paths
        paths = self.generate_paths(prices, drift, vol, intervals, no_sims)
        return paths

    def generate_paths_for_pricing(self, product, dates, discount_curve, no_sims):
        """
        date format: %Y%m%d
        """
        underlying = product.underlying
        calendar = product.calendar
        dates = pd.to_datetime(dates)
        # filtering dates for which there are no data
        dates = pd.DatetimeIndex.intersection(dates, underlying.get_dates())
        # for i in range(product.obsdates.size):
        #     # once get_vol is fully working on a surface, it will always have values...
        #     tenors = calendar.interval(dates, product.obsdates[i])
        #     temp = underlying.get_vol(dates, tenors, product.strike)
        #     dates = temp.index
        if underlying.yieldsdiv:
            dates = pd.DatetimeIndex.intersection(
                dates, underlying.get_divrate(dates).index
            )
        if dates.size == 0:
            print("Data unavailable; aborting simulation.")
            pass
        # checking if there are data for past observation dates
        obs_dates = product.obsdates
        if product.pastmatters:
            obspast = obs_dates[obs_dates < dates[-1]]
            if obspast.size != 0:
                n = pd.DatetimeIndex.intersection(obspast, underlying.get_dates()).size
                if n != obspast.size:
                    print(
                        "Data for past observation dates insufficient for simulation."
                    )
                    pass
        # sorting cases of valuation between observation dates
        # adding [] just simplifies the filtering expression
        L = [pd.to_datetime([])] + [dates[dates <= obsdate] for obsdate in obs_dates]
        L = [(L[i + 1][~L[i + 1].isin(L[i])], obs_dates[i]) for i in range(len(L) - 1)]
        L = [l for l in L if l[0].size != 0]
        # obs_past = obs_dates[obs_dates < L[-1][1]]
        # if obs_past.size != 0:
        #     if np.min(obs_past.isin(underlying.px.keys())) == 0:
        #         print("Missing price data for past observation dates")
        #         return None

        # evaluating between each two observation dates
        simulations_data = []
        for d, obsdate in L:
            # simplifying, since interest rates effect is very small
            step_dates = obs_dates[obs_dates >= obsdate]
            if hasattr(product, "is_quanto"):
                quanto_correction = product.compute_correction(dates)
            else:
                quanto_correction = 0
            paths = self.simulate_asset(
                asset=underlying,
                discount_curve=discount_curve,
                start_dates=d,
                step_dates=step_dates,
                no_sims=no_sims,
                calendar=calendar,
                quanto_correction=quanto_correction,
            )
            paths = paths[:, 1:]
            # computing payoffs
            if product.pastmatters:
                # dealing with past
                # adding in prices of past observation dates if they matter
                dates_dic = dict(zip(obs_dates, np.arange(obs_dates.size)))
                tempobspast = obs_dates[obs_dates < obsdate]
                n = tempobspast.size
                if n != 0:
                    paths_new = np.zeros(
                        (
                            paths.shape[0],
                            paths.shape[1] + tempobspast.size,
                            paths.shape[2],
                        )
                    )
                    obs_data = underlying.get_value(tempobspast).values
                    # obs_data = (obs_data * np.ones((d.size, obs_data.size))).transpose()
                    # simpler:
                    obs_data = obs_data.reshape((obs_data.size, 1))
                    paths_new[:, : tempobspast.size, :] = obs_data
                    paths_new[:, tempobspast.size :, :] = paths
                    paths = paths_new
            else:
                n = 0
                dates_dic = dict(zip(step_dates, np.arange(step_dates.size)))
            simulations_data.append((d, paths, n, dates_dic))
        return simulations_data

    def compute_price_from_simulations(
        self, product, discount_curve, simulation_data, statistics_gatherer=None
    ):
        payoffs_lists = []
        dates = pd.to_datetime([])
        for d, paths, n, dates_dic in simulation_data:
            dates = pd.DatetimeIndex.union(dates, d)
            step_dates = pd.to_datetime(list(dates_dic.keys())).sort_values()
            discount = discount_curve.get_value(
                dates=d, future_dates=product.maturity, calendar=product.calendar
            )
            if product.pastmatters:
                disc = np.zeros((step_dates.size, d.size))
                for i in range(step_dates.size):
                    disc[i] = discount_curve.get_value(
                        dates=d, future_dates=step_dates[i], calendar=product.calendar
                    )
                disc = discount / disc
            else:
                disc = 0
            payoffs = product.payoff(paths, dates_dic, n, disc)
            # Add credit discount
            payoffs_lists.append(discount * payoffs)

        # joining everything together (don't forget payoffs_lists is a list of numpys)
        all_payoffs = np.concatenate(payoffs_lists, axis=1)
        # loading the statistics gatherer, if any
        if hasattr(statistics_gatherer, "load"):
            statistics_gatherer.load(all_payoffs)
        # computing price estimates
        prices = np.mean(all_payoffs, axis=0)
        no_sims = all_payoffs.shape[0]
        price_series = pd.Series(
            prices, index=dates, dtype="float64", name="Price ({} sims)".format(no_sims)
        )
        return price_series

    def price(self, product, dates, discount_curve, no_sims, statistics_gatherer=None):
        # separated path generation from pricing so we can use the same paths for sensitivity computations (e.g., tols)
        dates = afsfun.dates_formatting(dates)
        simulations_data = self.generate_paths_for_pricing(
            product, dates, discount_curve, no_sims
        )
        prices = self.compute_price_from_simulations(
            product, discount_curve, simulations_data, statistics_gatherer
        )
        return prices
