from abc import ABC
from scipy import interpolate
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

try:
    from .discount_curves import CRDC
except (ImportError, ModuleNotFoundError, ValueError):
    from pricing.discount_curves import CRDC
try:
    from ..data.underlyings import LognormalAsset
except (ImportError, ModuleNotFoundError, ValueError):
    from data.underlyings import LognormalAsset
try:
    from .structured import Call, Put
except (ImportError, ModuleNotFoundError, ValueError):
    from pricing.structured import Call, Put


def _strike_to_moneyness(strike, spot):
    """
    Convert strike prices to moneyness values as a percentage.

    Parameters
    ----------
    strike : numpy.ndarray or pandas.Series
        The strike price(s) to convert to moneyness, :math:`K`.
    spot : numpy.ndarray or pandas.Series
        The price of the underlying asset, :math:`S_t`.

    Returns
    -------
    moneyness : numpy.ndarray or pandas.Series
        The moneyness value(s) expressed as a percentage.

    Notes
    -----
    This function calculates moneyness from the strike, which is defined as the ratio of the strike price
    to the price of the spot, expressed as a percentage:

    .. math::
           K\mapsto M_t = \\dfrac{K}{S_t}\\cdot 100.

    """
    moneyness = strike / spot * 100  # Percentage
    return moneyness


def _moneyness_to_strike(moneyness, spot):
    """
    Convert moneyness values to strike prices.

    Parameters
    ----------
    moneyness : numpy.ndarray or pandas.Series
        The moneyness value(s) to convert to strike prices, :math:`M_t`.
    spot : float
        The price of the underlying asset, :math:`S_t`.

    Returns
    -------
    numpy.ndarray or pandas.Series
        The strike price(s) corresponding to the provided moneyness value(s).

    Notes
    -----
    This function calculates strike prices based on moneyness values expressed as
    a percentage, which represent the ratio of the strike to the spot:

    .. math::
           M_t\mapsto K = \\dfrac{S_t}{100}\\cdot M_t.

    """
    strike = spot * moneyness / 100
    return strike


class VolatilitySmile(ABC):
    """
    This class deals with a volatility smile, :math:`\\Sigma_t(\\cdot,\\tau)` for
    :math:`\\tau:=T-t` for fixed :math:`t, T`. We will often supress the subscript `t`.

    Parameters
    ----------
    dtime : pandas.Timestamp or string
        Date at which the smile is computed, :math:`t`.
    maturity : pandas.Timestamp or string
        Common maturity for all options in the smile, :math:`T`.
    underlying_ticker : str
        Ticker of the underlying.
    spot : float, optional
        Value of the underlying at dtime, :math:`S_t`. Default is None.
    strikes : numpy.ndarray or pandas.Series
        Strikes in the smile, :math:`\\{ K_i \\}_{i=0}^{N}`.
    implied_vols : numpy.ndarray or pandas.Series
        Implied volatilities in the smile, :math:`\\{ \\Sigma_t(K_i,\\tau) \\}_{i=0}^N`.
    direction : str, optional
        "C" for call (default), "P" for put.
    deltas : pandas.Series or numpy.ndarray, optional
        Deltas for the corresponding options, :math:`\\{ \\Delta_i \\}_{i=0}^N`.
        Default=None.
    calendar : data.calendars.DayCountCalendar, optional
        The calendar system (Day Count Convention) to use when computing intervals.
        Calendar should be understood as an element of the classes defined in calendar.py.

    Attributes
    ----------
    dtime : pandas.Timestamp
        Date at which the smile is computed, :math:`t`.
    maturity : pandas.Timestamp
        Common maturity for all options in the smile, :math:`T`.
    underlying_ticker : str
        Ticker of the underlying.
    spot : float
        Value of the underlying at dtime, :math:`S_t`.
    strikes : numpy.ndarray or pandas.Series
        Strikes in the smile, :math:`\\{ K_i \\}_{i=0}^{N}`.
    implied_vols : numpy.ndarray or pandas.Series
        Implied volatilities in the smile, :math:`\\{ \\Sigma_t(K_i,\\tau) \\}_{i=0}^N`.
    direction : str
        "C" for call, "P" for put.
    polynomial : scipy.interpolate.PchipInterpolator
        Piecewise interpolation polynomial that fits the smile, i.e.,
        :math:`\\hat{\\Sigma}_t(\\cdot,\\tau)`.
    deltas : pandas.Series or numpy.ndarray
        Deltas for the corresponding options, :math:`\\{ \\Delta_i \\}_{i=0}^N`.
    discount_curve : pricing.discount_curves.DiscountCurve
        Discount curve to use during computations.
    dividends : float
        Dividend rate.
    calendar : data.calendars.DayCountCalendar
        The calendar system (Day Count Convention) to use when computing intervals.
        Calendar should be understood as an element of the classes defined in calendar.py.
    Note
    ----
    We use :math:`\\Sigma` for the market data and :math:`\\hat{\\Sigma}` for the interpolated one.

    Note
    ----
    ``strikes`` and ``implied_vols`` must be arrays (or pandas.Series) of the same size, :math:`N`, as each tuple
    :math:`(K, \\Sigma=\\Sigma_t(K,\\tau))` represents an observed point in the surface. As ``deltas``
    refers to the same set of options, it also has the same size.
    """

    def __init__(
        self,
        dtime,
        maturity,
        underlying_ticker,
        strikes,
        implied_vols,
        direction="C",
        spot=None,
        deltas=None,
        calendar=None,
    ):
        self.dtime = pd.to_datetime(dtime)
        self.maturity = pd.to_datetime(maturity)
        self.underlying_ticker = underlying_ticker
        self.spot = spot
        self.strikes = strikes
        self.implied_vols = implied_vols.astype(float)
        self.direction = direction
        self.polynomial = None
        self.deltas = deltas
        self.discount_curve = None
        self.dividends = None
        self.calendar = calendar

    def fit(self):
        """
        Method to fit the volatility smile to a given set of options.

        Parameters
        ----------
        None

        Returns
        -------
        None

        Notes
        -----
        This method is expected to update the attribute ``polynomial``.
        """
        self.polynomial = interpolate.PchipInterpolator(self.strikes, self.implied_vols)

    def evaluate_strike(self, strike):
        """
        Method to evaluate the implied volatility for a given strike using the previous
        fit. That is, given :math:`K`, it returns :math:`\\hat{\\Sigma}_t(K, \\tau)`.

        Parameters
        ----------
        strike : numpy.ndarray or pandas.Series
            Value(s) to evaluate the implied volatility.

        Returns
        -------
        Implied Volatility : numpy.ndarray or pandas.Series
            Value(s) of the implied volatility.
        """
        if self.polynomial is None:
            raise ValueError("You need to fit first")
        else:
            return self.polynomial(strike)

    def evaluate_moneyness(self, moneyness):
        """
        Evaluate implied volatility for a given moneyness using a fitted polynomial.
        That is, given :math:`M`, it returns :math:`\\hat{\\Sigma}_t(K(M), \\tau)`, being
        :math:`K(M)` the function defined in
        :py:meth:`_moneyness_to_strike <pricing.implied_volatility._moneyness_to_strike>`.


        Parameters
        ----------
        moneyness : numpy.ndarray or pandas.Series
            The moneyness value for which to calculate implied volatility.

        Returns
        -------
        numpy.ndarray or pandas.Series
            The implied volatility corresponding to the given moneyness.

        Raises
        ------
        ValueError
            If the polynomial has not been fitted or if no underlying value is provided.
        """
        if self.polynomial is None:
            raise ValueError("You need to fit first")
        elif self.spot is None:
            raise ValueError("No underlying value provided")
        else:
            strike = _moneyness_to_strike(moneyness, self.spot)
            return self.polynomial(strike)

    def compute_deltas(
        self,
        calendar=None,
        interest_rate=None,
        discount_curve=None,
        dividends=0,
        maturities=None,
    ):
        """
        Compute option deltas based on provided parameters using Black-Scholes formula.

        Parameters
        ----------
        interest_rate : float, optional
            The interest rate :math:`r` for discounting. If provided, this overrides ``discount_curve``.
        discount_curve : discount_curves.DiscountCurve, optional
            Gives the discounts for the computation of delta.
        dividends : float, optional
            Dividend yield. Default is 0.
        calendar : data.calendars.DayCountCalendar, optional
            The calendar system (Day Count Convention) to use when computing intervals.
            Calendar should be understood as an element of the classes defined in calendar.py.
        maturities : numpy.ndarray or pandas.Series, optional
            Array of maturities for the options, :math:`\\{T_j\\}_{i=1}^{N}`.
            If not provided use maturity of the smile.

        Notes
        -----
        This method calculates option deltas based on provided interest rate (or discount curve), dividends,
        and calendar. It updates the ``discount_curve``, ``dividends``, and ``calendar``
        attributes and populates ``deltas``.
        """
        if calendar is not None:
            self.calendar = calendar
        elif self.calendar is None:
            raise ValueError("No calendar provided.")
        if interest_rate is None and discount_curve is None:
            raise ValueError(
                "At least one of 'discount_curve' or 'interest_rate' must be provided."
            )
        if interest_rate is not None:
            self.discount_curve = CRDC(interest_rate)
        else:
            self.discount_curve = discount_curve

        self.dividends = dividends

        self.deltas = self._get_delta(
            self.strikes, self.implied_vols, maturities=maturities
        )

    def _get_delta(self, strikes, implieds, maturities=None):
        """
        Calculate and return option deltas for the given strikes, implied volatilities
        and maturities using Black-Scholes formula.

        Parameters
        ----------
        strikes : pandas.Series or numpy.ndarray of float
            Array of option strikes.
        implieds : pandas.Series or numpy.ndarray
            Array of implied volatilities as percentages.
        maturities : numpy.ndarray or pandas.Series, optional
            Array of maturities for the options, :math:`\\{T_i\\}_{i=1}^{N}`.
            If not provided use maturity of the smile.


        Returns
        -------
        pandas.Series or numpy.ndarray :
            Array of option deltas corresponding to the provided strikes.

        Notes
        -----
        This private method calculates and returns option deltas based on the provided
        strike prices and implied volatilities. It uses ``LognormalAsset``, ``Call``, and ``Put``
        objects to perform the calculations.

        Mathematically, it computes the greek delta as a function of

        .. math::
            (K,\\Sigma)\mapsto\\Delta(K,\\Sigma, r, q\\ldots)\\,.

        Note that the implicit parameters, :math:`r,q\\ldots,` are attributes of the class.



        """
        deltas = []
        asset = LognormalAsset(self.underlying_ticker)
        if maturities is None:
            maturities = [self.maturity for i in range(0, len(strikes))]
        for strike, implied, maturity in zip(strikes, implieds, maturities):
            vol_df = pd.DataFrame(
                data=[np.array([self.spot, implied / 100, self.dividends])],
                index=[self.dtime],
                columns=["Price", "Volatility", "Dividend Rate"],
            )
            asset.set_data(vol_df)
            if self.direction == "C":
                option = Call(
                    asset, strike, maturity, nominal=1, calendar=self.calendar
                )
            else:
                option = Put(asset, strike, maturity, nominal=1, calendar=self.calendar)
            delta = option.get_delta(self.dtime, self.discount_curve)
            deltas.append(delta[0])
        return deltas

    def plot_strike(self, provide_fig=False):
        """
        Plot implied volatility against strike prices, :math:`K \\mapsto \\hat{\\Sigma}(K,\\tau)`.

        Parameters
        ----------
        provide_fig : bool, optional
            If True, return the plot figure. Default is False.

        Returns
        -------
        matplotlib.figure.Figure or None
            The plot figure if 'provide_fig' is True, otherwise, None.

        Raises
        ------
        ValueError
            If the polynomial has not been fitted.

        Notes
        ----
        This method generates a plot of implied volatility against strike prices. It is
        used to visualize the relationship between implied volatility and strike prices.

        """
        if self.polynomial is None:
            raise ValueError("You need to fit first")
        xaxis = "Strike"
        if provide_fig:
            figure = self._plot(xaxis, provide_fig=True)
            return figure
        else:
            self._plot(xaxis)

    def plot_moneyness(self, provide_fig=False):
        """
        Plot implied volatility against moneyness, :math:`M \\mapsto \\hat{\\Sigma}(K(M),\\tau)`,
        being :math:`K(M)` the function defined in
        :py:meth:`_moneyness_to_strike <pricing.implied_volatility._moneyness_to_strike>`.

        Parameters
        ----------
        provide_fig : bool, optional
            If True, return the plot figure. Default is False.

        Returns
        -------
        matplotlib.figure.Figure or None
            The plot figure if 'provide_fig' is True, otherwise, None.

        Raises
        ------
        ValueError
            If the polynomial has not been fitted.

        Notes
        -----
        This method generates a plot of implied volatility against moneyness, where
        moneyness is expressed as a percentage. It is used to visualize the relationship
        between implied volatility and moneyness values.

        """
        if self.polynomial is None:
            raise ValueError("You need to fit first")
        xaxis = "Moneyness (%)"
        if provide_fig:
            figure = self._plot(xaxis, provide_fig=True)
            return figure
        else:
            self._plot(xaxis)

    def plot_delta(self, provide_fig=False):
        """
        Plot implied volatility against delta.

        Parameters
        ----------
        provide_fig : bool, optional
            If True, return the plot figure. Default is False.

        Returns
        -------
        matplotlib.figure.Figure or None
            The plot figure if 'provide_fig' is True, otherwise, None.

        Raises
        ------
        ValueError
            If deltas have not been computed or if the polynomial has not been fitted.

        Notes
        -----
        This method generates a plot of option deltas against the corresponding strike prices.
        It is used to visualize the relationship between option deltas and strike prices.
        Technically, given the strikes of the grid, it plots
        :math:`K`, it plots :math:`\\hat{\\Sigma}(K,\\tau)` against
        :math:`\\Delta(K, \\hat{\\Sigma}(K,\\tau))`, being :math:`\\Delta` the function defined in
        :py:meth:`_get_delta <pricing.implied_volatility.VolatilitySmile._get_delta>`.

        """
        if self.deltas is None:
            raise ValueError("You need to compute deltas first")
        if self.polynomial is None:
            raise ValueError("You need to fit first")

        xaxis = "Delta"
        if provide_fig:
            figure = self._plot(xaxis, provide_fig=True)
            return figure
        else:
            self._plot(xaxis)

    def _plot(self, xaxis, provide_fig=False):
        """
        Generate a plot of implied volatility data.

        Parameters
        ----------
        xaxis : str
            The label for the x-axis ('Strike', 'Moneyness (%)', 'Delta').
        provide_fig : bool, optional
            If True, return the plot figure. Default is False.

        Returns
        -------
        plotly.graph_objs.Figure or None
            The plot figure if 'provide_fig' is True, otherwise, None.

        Notes
        -----
        This private method generates a plot of implied volatility data against different
        x-axis labels, such as 'Moneyness (%)' or 'Delta'. The plot includes option data
        and a spline curve. It is used to visualize the volatility smile.

        """
        x_curve = np.linspace(
            float(self.strikes[0]), float(self.strikes[len(self.strikes) - 1]), 100
        )
        y_curve = self.polynomial(x_curve)
        if xaxis == "Moneyness (%)":
            x_curve = _strike_to_moneyness(x_curve, self.spot)
            x = _strike_to_moneyness(self.strikes, self.spot)
        elif xaxis == "Delta":
            x_curve = self._get_delta(x_curve, y_curve)
            x = self.deltas
        else:
            x_curve = x_curve
            x = self.strikes
        data_line = pd.DataFrame({"x": x_curve, "y": y_curve})
        data = pd.DataFrame({"x": x, "y": self.implied_vols})
        fig = px.scatter(
            data,
            x="x",
            y="y",
            title="Volatility Smile",
            labels={"x": xaxis, "y": "Implied Volatility"},
        )
        fig.add_trace(px.line(data_line, x="x", y="y").data[0])
        fig.data[1].line.color = "#EF553B"
        fig.data[0].name = "Options Data"
        fig.data[1].name = "Fit"
        title = f"Volatility Smile. Underlying: {self.underlying_ticker}. Direction: {self.direction}."
        sub_title = f"Underlying Value: {self.spot}. Date: {self.dtime.strftime('%Y-%m-%d')}. Maturity: {self.maturity.strftime('%Y-%m-%d')}."
        fig.update_layout(title=title, title_x=0.5)
        fig.add_annotation(
            text=sub_title,
            xref="paper",
            yref="paper",
            x=0.5,
            y=1.11,
            showarrow=False,
            font=dict(size=14),
        )
        fig.update_traces(showlegend=True)
        if provide_fig:
            return fig
        else:
            fig.show()

    @staticmethod
    def plot_range_smiles_strike(smile_dict):
        """
        Plot a range of volatility smiles with the same maturity against strike prices
        for different dates using a slider plot.

        Parameters
        ----------
        smile_dict : dict
        A dictionary of ``VolatilitySmile`` objects, where keys are dates and values are instances of the ``VolatilitySmile`` class.

        Returns
        -------
        None

        Notes
        -----
        This function creates an interactive plot that displays a range of volatility smiles
        against strike prices for different dates. It uses Plotly to generate the plot and
        allows you to navigate through the smiles using a slider.

        Examples
        --------
        >>> import afslibrary as afs
        >>> ticker = "SX5EIDX"
        >>> maturity = "2023-09-15"
        >>> from_date = "2023-06-08"
        >>> to_date ="2023-06-15"
        >>> smile_dict = afs.DataFactoryBeautifulData.import_volatility_smile(ticker, maturity, from_date=from_date, to_date=to_date, call_put="P")
        >>> afs.VolatilitySmile.plot_range_smiles_strike(smile_dict)

        """
        fig = go.Figure()
        keys = list(smile_dict.keys())
        titles = []
        subtitles = []
        vol_max = 0
        vol_min = 100
        for step, date in enumerate(keys):
            smile = smile_dict[date]
            temp_fig = smile.plot_strike(provide_fig=True)
            for trace in temp_fig.data:
                trace.visible = False
            title = temp_fig.layout.title.text
            subtitle = temp_fig.layout.annotations[0].text
            titles.append(title)
            subtitles.append(subtitle)
            fig.add_traces(temp_fig.data)
            vol_max_smile = np.max(smile.implied_vols)
            vol_max = np.maximum(vol_max_smile, vol_max)
            vol_min_smile = np.min(smile.implied_vols)
            vol_min = np.minimum(vol_min_smile, vol_min)
        y_range = [vol_min * 0.9, vol_max * 1.1]
        # Make 10th trace visible
        fig.data[0].visible = True
        fig.data[1].visible = True

        # Create and add slider
        steps = []
        for i in range(len(keys)):
            step = dict(
                method="update",
                args=[
                    {"visible": [False] * len(fig.data)},
                    {
                        "title": titles[i],
                        "xaxis.title": "Strike",
                        "yaxis.title": "Implied Volatility",
                        "annotations": [
                            dict(
                                text=subtitles[i],  # Add the annotation as a subtitle
                                x=0.5,
                                y=1.11,
                                xref="paper",
                                yref="paper",
                                showarrow=False,
                                font=dict(size=14),
                            )
                        ],
                    },
                ],  # layout attribute
            )
            step["args"][0]["visible"][2 * i] = True  # Toggle i'th trace to "visible"
            step["args"][0]["visible"][2 * i + 1] = (
                True  # Toggle i+1'th trace to "visible"
            )
            step["args"][1]["yaxis.range"] = y_range  # Toggle i+1'th trace to "visible"
            steps.append(step)

        sliders = [
            dict(
                active=10,
                currentvalue={"prefix": "Date: "},
                pad={"t": 50},
                steps=steps,
            )
        ]

        fig.update_layout(sliders=sliders)

        # Edit slider labels
        for i, date in enumerate(keys):
            fig["layout"]["sliders"][0]["steps"][i]["label"] = str(date)

        fig.show()


class VolatilitySurface(VolatilitySmile):
    """
    This class represents a volatility surface, :math:`\\Sigma_t(\\cdot,\\cdot)`,
    for a fixed time, , :math:`t`.

    Parameters
    ----------
    dtime : pandas.Timestamp, or string
        Date at which the surface is computed, :math:`t`.
    maturities : pandas.Series
        Maturities for options in the surface, :math:`\\{T_i\\}_{i=1}^{N}`.
    underlying_ticker : str
        Ticker of the underlying.
    spot : float, optional
        Value of the underlying at dtime, :math:`S_t`, default is None.
    strikes : pandas.Series or numpy.ndarray
        Strikes in the surface, :math:`\\{K_i\\}_{i=1}^{N}`.
    implied_vols : pandas.Series or numpy.ndarray
        Implied volatilities in the surface, :math:`\\{\\Sigma_i\\}_{i=1}^{N}`.
    direction : str, optional
        "C" for call (default), "P" for put.
    calendar : data.calendars.DayCountCalendar, optional
        The calendar system (Day Count Convention) to use when computing intervals.
        Calendar should be understood as an element of the classes defined in calendar.py

    Attributes
    ----------
    dtime : pandas.Timestamp
        Date at which the surface is computed, :math:`t`.
    maturities : pandas.Series or numpy.ndarray
        Maturities for options in the surface, :math:`\\{T_i\\}_{i=1}^{N}`.
    underlying_ticker : str
        Ticker of the underlying.
    spot : float
        Value of the underlying at dtime, :math:`S_t`.
    strikes : pandas.Series or numpy.ndarray
        Strikes in the surface, :math:`\\{K_i\\}_{i=1}^{N}`.
    implied_vols : pandas.Series or numpy.ndarray
        Implied volatilities in the surface, :math:`\\{\\Sigma_i\\}_{i=1}^{N}`.
    direction : str
        "C" for call, "P" for put.
    calendar : data.calendars.DayCountCalendar, optional
        The calendar system (Day Count Convention) to use when computing intervals.
        Calendar should be understood as an element of the classes defined in calendar.py
    polynomial : scipy.interpolate.RegularGridInterpolator or scipy.interpolate.LinearNDInterpolator
        Piecewise interpolation polynomial that fits the surface, i.e., :math:`\\hat{\\Sigma}_t(\\cdot,\\cdot)`.
    deltas : pandas.Series or numpy.ndarray
        Array of deltas for the corresponding options, :math:`\\{\\Delta_i\\}_{i=1}^N`.
    discount_curve : pricing.discount_curves.DiscountCurve, optional
        Discount curve to use during computations.
    dividends : float, optional
        Dividends rate.
    tenors : pandas.Series or numpy.ndarray
        Tenors, computed from maturities and dtime, :math:`\\{\\tau_i=T_i-t\\}_{i=1}^N`.
    is_rectangular : bool
        True if the surface is defined on a rectangular grid, otherwise False.

    Note
    --------
    ``maturities``, ``strikes`` and ``implied_vols`` must be arrays (or pandas.Series) of the same size, :math:`N`, as each tuple
    :math:`(T-t, K, \\Sigma=\\Sigma_t(K,\\tau=T-t))` represents an observed point in the surface. As ``deltas`` and ``tenors``
    refer to the same set of options, they also have the same size.

    """

    def __init__(
        self,
        dtime,
        maturities,
        underlying_ticker,
        strikes,
        implied_vols,
        calendar,
        direction="C",
        spot=None,
    ):
        self.dtime = pd.to_datetime(dtime)
        self.maturities = pd.to_datetime(maturities)
        self.underlying_ticker = underlying_ticker
        self.spot = spot
        self.strikes = strikes
        self.implied_vols = implied_vols.astype(float)
        self.direction = direction
        self.polynomial = None
        self.deltas = None
        self.discount_curve = None
        self.dividends = None
        self.calendar = calendar
        self.tenors = self._get_tenors(self.maturities)
        self.is_rectangular = None

    def _get_tenors(self, maturities):
        """
        Calculate tenors from maturities for the volatility surface.
        Tenors are computed in unit of years using the corresponding calendar
        for the day-count convention.

        Parameters
        ----------
        maturities :  pandas.Series or numpy.ndarray
            Maturities to compute tenors, :math:`\\{T_i\\}_{i=1}^{N}`.

        Returns
        -------
         pandas.Series or numpy.ndarray
            Tenors calculated from maturities, :math:`\\{\\tau_i=T_i-t\\}_{i=1}^{N}`.

        Note
        -----
        This method calculates tenors, which represent the time intervals between the
        reference date (``dtime``) and the maturity dates.
        It uses the specified calendar to compute these intervals. Thus, technically, it returns

        .. math::

            \\text{DC}(t, T) = \\tau(t,T)\\,,

        being :math:`\\text{DC}` the day-count convention.

        """
        starting_time = self.dtime
        finishing_time = pd.DatetimeIndex(maturities)
        tenors = self.calendar.interval(
            starting_time, finishing_time, sort_values=False
        )
        return tenors

    def fit(self, verbose=False, method="pchip"):
        """
        Fit a volatility surface with an interpolation method.

        Parameters
        ----------
        verbose : bool, optional
            If True, print fitting information. Default is False.
        method : str, optional
            The interpolation method to use when grid is rectangular.
            Options include "pchip" for Piecewise Cubic Hermite Interpolating Polynomial (default)
            and  “linear”, “nearest”, “slinear”, “cubic” and “quintic”.

        Returns
        -------
        None

        Notes
        -----
        This method fits the volatility surface with the specified interpolation method.
        It allows for both regular grid and non-regular grid data (in this case always linear),
        and selects the appropriate interpolation method accordingly. The fitted result
        is stored in the ``polynomial`` attribute for later use.
        """
        data = np.array([self.strikes, self.tenors, self.implied_vols]).transpose()
        strike_values = np.unique(data[:, 0])
        tenor_values = np.unique(data[:, 1])

        # Check if the values form a rectangular grid
        is_rectangular = len(strike_values) * len(tenor_values) == len(data)
        self.is_rectangular = is_rectangular
        if verbose:
            print(f"Data corresponds to regular grid: {is_rectangular}.")
            if is_rectangular:
                print(f"Data fitted using {method} method.")
            else:
                print("Data fitted using linear method.")

        if is_rectangular:
            X, Y = np.meshgrid(strike_values, tenor_values, indexing="ij")
            function_values = np.zeros(X.shape)
            for i in range(len(data)):
                x_index = np.where(strike_values == data[i, 0])
                y_index = np.where(tenor_values == data[i, 1])
                function_values[x_index, y_index] = data[i, 2]

            self.polynomial = interpolate.RegularGridInterpolator(
                (strike_values, tenor_values),
                function_values,
                fill_value=None,
                method=method,
            )
        else:
            self.polynomial = interpolate.LinearNDInterpolator(
                list(zip(self.strikes, self.tenors)), self.implied_vols, fill_value=20
            )

    def evaluate_strike(self, strike, tenor):
        """
        Evaluate the implied volatility for a given strike and tenor using the fitting.
        That is, given :math:`(K,\\tau)`, it returns :math:`\\hat{\\Sigma}_t(K, \\tau)`.

        Parameters
        ----------
        strike :  pandas.Series or numpy.ndarray
            Value(s) to evaluate the implied volatility at.
        tenor :  pandas.Series or numpy.ndarray
            Tenor value(s) to evaluate the implied volatility at.

        Returns
        -------
        pandas.Series or numpy.ndarray
            Implied Volatility value(s).

        Notes
        -----
        This method evaluates the implied volatility for a specific strike and tenor using
        the fitted surface. It takes into account whether the data represents a regular grid or
        not and uses the appropriate interpolation method.
        """
        if self.polynomial is None:
            raise ValueError("You need to fit first")
        else:
            if self.is_rectangular:
                return self.polynomial((strike, tenor))
            else:
                return self.polynomial(strike, tenor)

    def evaluate_moneyness(self, moneyness, tenor):
        """
        Evaluate the implied volatility for a given moneyness and tenor using the fitted smile.
        That is, given :math:`(M,\\tau)`, it returns :math:`\\hat{\\Sigma}_t(K(M), \\tau)`,
        being :math:`K(M)` the function defined in
        :py:meth:`_moneyness_to_strike <pricing.implied_volatility._moneyness_to_strike>`.

        Parameters
        ----------
        moneyness :  pandas.Series or numpy.ndarray
            Moneyness value(s) (percentage) to evaluate the implied volatility at.
        tenor :  pandas.Series or numpy.ndarray
            Tenor value(s) to evaluate the implied volatility at.

        Returns
        -------
         pandas.Series or numpy.ndarray
            Implied Volatility value(s).

        Notes
        -----
        This method evaluates the implied volatility for a specific moneyness and tenor using
        the fitted surface. It requires a fitted smile and an underlying value to perform the
        calculation.
        """
        if self.polynomial is None:
            raise ValueError("You need to fit first")
        elif self.spot is None:
            raise ValueError("No underlying value provided")
        else:
            strike = _moneyness_to_strike(moneyness, self.spot)
            return self.evaluate_strike(strike, tenor)

    def plot_strike(self, provide_fig=False):
        """
        Plot the volatility surface against strike prices and tenors,
        :math:`(K, \\tau) \\mapsto \\hat{\\Sigma}(K,\\tau)`.


        Parameters
        ----------
        provide_fig : bool, optional
            If True, return the plot as a Plotly Figure. Default is False.

        Returns
        -------
        None or plotly.graph_objs._figure.Figure
            If provide_fig is True, returns a Plotly Figure.

        Notes
        -----
        It requires a fitted surface to be available.
        """
        if self.polynomial is None:
            raise ValueError("You need to fit first")

        xaxis = "Strike"
        if provide_fig:
            figure = self._plot(xaxis, provide_fig=True)
            return figure
        else:
            self._plot(xaxis)

    def plot_moneyness(self, provide_fig=False):
        """
        Plot the volatility surface against moneyness and tenors,
        :math:`(M, \\tau) \\mapsto \\hat{\\Sigma}(K(M),\\tau)`, being :math:`K(M)` the function defined in
        :py:meth:`_moneyness_to_strike <pricing.implied_volatility._moneyness_to_strike>`.

        Parameters
        ----------
        provide_fig : bool, optional
            If True, return the plot as a Plotly Figure. Default is False.

        Returns
        -------
        None or plotly.graph_objs._figure.Figure
            If provide_fig is True, returns a Plotly Figure.

        Notes
        -----
        It requires a fitted surface to be available.
        """
        if self.polynomial is None:
            raise ValueError("You need to fit first")

        xaxis = "Moneyness (%)"
        if provide_fig:
            figure = self._plot(xaxis, provide_fig=True)
            return figure
        else:
            self._plot(xaxis)

    def plot_delta(self, provide_fig=False):
        """
        Plot the volatility surface against deltas and tenors, but only
        market data, not fitting:
        :math:`(\\Delta, \\tau) \\mapsto \\Sigma(K(\\Delta),\\tau)`, being :math:`K(\\Delta)` the function defined in
        :py:meth:`_get_delta <pricing.implied_volatility.VolatilitySmile._get_delta>`.

        Parameters
        ----------
        provide_fig : bool, optional
            If True, return the plot as a Plotly Figure. Default is False.

        Returns
        -------
        None or plotly.graph_objs._figure.Figure
            If provide_fig is True, returns a Plotly Figure.

        Notes
        -----
        It requires a fitted surface to be available and having pre-computed the deltas. See
        :py:meth:`plot_delta <pricing.implied_volatility.VolatilitySmile.plot_delta>` for more
        details.
        """
        if self.deltas is None:
            raise ValueError("You need to compute deltas first")
        if self.polynomial is None:
            raise ValueError("You need to fit first")

        xaxis = "Delta"
        if provide_fig:
            figure = self._plot(xaxis, provide_fig=True)
            return figure
        else:
            self._plot(xaxis)

    def _plot(self, xaxis, provide_fig=False):
        """
        Plot the volatility surface in 3D space.

        Parameters
        ----------
        xaxis : str
            The variable to use as the x-axis of the 3D plot ("Strike," "Moneyness (%)," or "Delta").
        provide_fig : bool, optional
            If True, return the plot as a Plotly Figure. Default is False.

        Returns
        -------
        None or plotly.graph_objs._figure.Figure : If provide_fig is True, returns a Plotly Figure.

        Notes
        -----
        This method plots the volatility surface in a 3D space, with customizable x-axis variables
        (strike, moneyness, or delta). It uses the fitted smile and the specified x-axis variable to
        generate the 3D plot. The plot can be displayed or returned as a Plotly Figure for further
        analysis or visualization.
        """
        data_line = {}
        x_curve = np.linspace(
            float(self.strikes[0]),
            float(self.strikes[len(self.strikes) - 1]),
            100,
        )
        y_curve = np.linspace(
            float(min(self.tenors)),
            float(max(self.tenors)),
            100,
        )
        data_line["y"] = y_curve
        X, Y = np.meshgrid(x_curve, y_curve)
        z_curve = self.evaluate_strike(X, Y)
        if xaxis == "Strike":
            data_x = self.strikes
        elif xaxis == "Moneyness (%)":
            x_curve = _strike_to_moneyness(x_curve, self.spot)
            data_x = _strike_to_moneyness(self.strikes, self.spot)
        elif xaxis == "Delta":
            data_x = self.deltas

        data_line["x"] = x_curve
        data_line["z"] = z_curve
        data = pd.DataFrame(
            {
                "x": data_x,
                "y": self.tenors,
                "z": self.implied_vols,
            }
        )
        fig = go.Figure()
        fig.add_trace(
            go.Scatter3d(
                x=data["x"],
                y=data["y"] * 365,
                z=data["z"],
                mode="markers",
                marker=dict(size=3),
            )
        )
        if xaxis == "Delta":
            pass
        else:
            surface_trace = go.Surface(
                x=data_line["x"],
                y=data_line["y"] * 365,
                z=data_line["z"],
            )
            fig.add_trace(surface_trace)
        title = f"Underlying: {self.underlying_ticker}. Value: {self.spot}. Direction: {self.direction}. Date: {self.dtime.strftime('%Y-%m-%d')}."
        fig.update_layout(title=title, title_x=0.5)
        fig.update_scenes(
            xaxis_title=xaxis,
            yaxis_title="Tenor (Days)",
            zaxis_title="Implied Volatility",
        )
        fig.update_layout(width=800, height=700)
        if provide_fig:
            return fig
        else:
            fig.show()


class VolatilitySurfaceDelta(VolatilitySurface):
    """
    This class is a volatility surface that can generate a regular grid
    in delta for all maturities. To do so, fit the smile for all maturities
    and evaluate for each of them at different deltas.

    This class inheritates all methods and attributes from the ``VolatilitySurface`` class.

    Parameters
    ----------
    dtime : pandas.Timestamp, or string
        Date at which the surface is computed, :math:`t`.
    maturities : numpy.ndarray or pandas.Series
        Maturities for options in the surface, :math:`\\{T_i\\}_{i=1}^{N}`.
    underlying_ticker : str
        Ticker of the underlying.
    spot : float, optional
        Value of the underlying at dtime, :math:`S_t`, default is None.
    strikes : numpy.ndarray or pandas.Series
        Strikes in the surface, :math:`\\{K_i\\}_{i=1}^{N}`.
    implied_vols : numpy.ndarray or pandas.Series
        Implied volatilities in the surface, :math:`\\{\\Sigma_i\\}_{i=1}^{N}`.
    direction : str, optional.
        "C" for call (default), "P" for put
    calendar : data.calendars.DayCountCalendar, optional
        The calendar system (Day Count Convention) to use when computing intervals.
        Calendar should be understood as an element of the classes defined in calendar.py.


    The additional attributes this class has are the following.

    Attributes
    ----------
    polynomial_delta : scipy.interpolate.RegularGridInterpolator
        Piecewise interpolation polynomial that fits the surface in delta representation,
        i.e., :math:`\\hat{\\Sigma}_t(\\cdot,\\cdot)`.
    delta_min : float
        Minimum common delta in all the smiles contained in this surface.
    delta_max : float
        Maximum common delta in all the smiles contained in this surface.
    dict_smiles : dict
        Dictionary where keys are maturities and values the corresponding smiles.
        Smiles are instances of the :py:meth:`VolatilitySmile <pricing.implied_volatility.VolatilitySmile>` class.

    Note
    --------
    ``maturities``, ``strikes`` and ``implied_vols`` must be arrays (or pandas.Series) of the same size, :math:`N`, as each tuple
    :math:`(T-t, K, \\Sigma=\\Sigma_t(K,\\tau=T-t))` represents an observed point in the surface.
    """

    def __init__(
        self,
        dtime,
        maturities,
        underlying_ticker,
        strikes,
        implied_vols,
        calendar,
        direction="C",
        spot=None,
    ):
        super().__init__(
            dtime,
            maturities,
            underlying_ticker,
            strikes,
            implied_vols,
            calendar,
            direction=direction,
            spot=spot,
        )
        self.polynomial_delta = None
        self.delta_min = None
        self.delta_max = None
        self.dict_smiles = None
        self.synthetic_vols = None

    def _split_smiles(self, interest_rate=None, discount_curve=None, dividends=0):
        """
        Split the options in the surface into their corresponding smiles (equivalent to split
        them by maturities) and stores them in the attribute ``dict_smiles``.

        Parameters
        ----------
        interest_rate : float, optional
            The interest rate :math:`r` for discounting. If provided, this overrides ``discount_curve``.
        discount_curve : discount_curves.DiscountCurve, optional
            Gives the discounts for the computation.
        dividends : float, optional
            Dividends rate used for delta calculations. Default is 0.

        Returns
        -------
        None

        Notes
        -----
        This method updates the attributes ``delta_min``, ``delta_max`` and ``dict_smiles``.
        """
        self.compute_deltas(
            interest_rate=interest_rate,
            discount_curve=discount_curve,
            dividends=dividends,
            maturities=self.maturities,
        )
        d = {
            "Maturity": self.maturities,
            "Strike": self.strikes,
            "Implied Vol": self.implied_vols,
            "Delta": self.deltas,
        }
        df = pd.DataFrame(data=d)
        grouped = df.groupby("Maturity")
        dict_smiles = {}
        delta_max = 1
        delta_min = -1
        for name, group in grouped:
            df = group.sort_values(by="Delta", ascending=True)
            deltas = df["Delta"].astype(float)
            smile = VolatilitySmile(
                dtime=name,
                maturity=df["Maturity"][0],
                underlying_ticker=self.underlying_ticker,
                spot=self.spot,
                strikes=df["Strike"].astype(float),
                implied_vols=df["Implied Vol"].astype(float),
                direction=self.direction,
                deltas=deltas,
            )
            dict_smiles[name] = smile
            delta_min = np.maximum(delta_min, np.min(deltas))
            delta_max = np.minimum(delta_max, np.max(deltas))
        self.delta_min = delta_min
        self.delta_max = delta_max
        self.dict_smiles = dict_smiles

    def _generate_delta_grid(
        self, interest_rate=None, discount_curve=None, dividends=0
    ):
        """
        This method interpolates the smiles in the surface to generate a synthetic,
        rectangular grid in the representation :math:`\\Sigma_t(\\Delta,\\tau)`. It stores it in the
        attributes ``delta_grid`` and ``synthetic_vols``. The limits of the grid are defined
        by ``delta_min`` and ``delta_max``.

        Parameters
        ----------
        interest_rate : float, optional
            The interest rate :math:`r` for discounting. If provided, this overrides ``discount_curve``.
        discount_curve : discount_curves.DiscountCurve, optional
            Gives the discounts for the computation.
        dividends : float, optional
            Dividends rate used for delta calculations. Default is 0.

        Returns
        -------
        None

        Notes
        -----
        This method updates the attributes ``delta_grid`` and ``synthetic_vols``. ``delta_grid``
        is a grid of 100 points between the values ``delta_min`` and ``delta_max``, and ``synthetic_vols``
        are the value of each smile interpolation on these points:
        :math:`\\hat{\\Sigma}_t(\\Delta,\\tau)\\,, \\forall~\\tau\\in\\{T_i-t\\}_{i=1}^{N}` and
        :math:`\\forall~\\Delta\\in\\lbrace\\Delta_\\min + \dfrac{i\cdot(\\Delta_\\max - \\Delta_\\min)}{99}\\rbrace_{i=0}^{99}`.
        """
        if self.delta_min is None:
            self._split_smiles(
                interest_rate=interest_rate,
                discount_curve=discount_curve,
                dividends=dividends,
            )
        self.delta_grid = np.linspace(self.delta_min, self.delta_max, 100)
        volatilities = []
        for maturity, smile in self.dict_smiles.items():
            synthetic_vols = interpolate.pchip_interpolate(
                smile.deltas, smile.implied_vols, self.delta_grid
            )
            volatilities.append(synthetic_vols)
        self.synthetic_vols = np.array(volatilities).transpose()

    def fit_delta(
        self, interest_rate=None, discount_curve=None, dividends=0, method="pchip"
    ):
        """
        This method interpolates the surface using the synthetic grid previously
        generated, using the method provided.

        Parameters
        ----------
        interest_rate : float, optional
            The interest rate :math:`r` for discounting. If provided, this overrides ``discount_curve``.
        discount_curve : discount_curves.DiscountCurve, optional
            Gives the discounts for the computation.
        dividends : float, optional
            Dividends rate used for delta calculations. Default is 0.
        method : str, optional
            Method used to interpolate the surface. Default is "pchip", the available
            methods are the same as in the fit method.

        Returns
        -------
        None

        Notes
        -----
        This method updates the attribute ``polynomial_delta``.
        """
        self._generate_delta_grid(
            interest_rate=interest_rate,
            discount_curve=discount_curve,
            dividends=dividends,
        )
        self.polynomial_delta = interpolate.RegularGridInterpolator(
            (self.delta_grid, np.unique(self.tenors)),
            self.synthetic_vols,
            fill_value=None,
            method=method,
        )

    def get_implied_from_delta(self, delta, tenor):
        """
        Evaluate the implied volatility for a given delta and tenor using the fitting.
        That is, given :math:`(\\Delta,\\tau)`, it returns :math:`\\hat{\\Sigma}_t(\\Delta, \\tau)`

        Parameters
        ----------
        delta : numpy.ndarray or pandas.Series
            Delta value(s) (percentage) to evaluate the implied volatility at, :math:`\\Delta`.
        tenor : numpy.ndarray or pandas.Series
            Tenor value(s) to evaluate the implied volatility at, :math:`\\tau`.

        Returns
        -------
        float
            Implied Volatility value(s).

        Notes
        -----
        This method evaluates the implied volatility for a specific delta and tenor using
        the fitted surface. It requires a fitted surface and an underlying value to perform the
        computation.
        """
        implied = self.polynomial_delta((delta, tenor))
        return implied

    def plot_delta(self, provide_fig=False):
        """
        Plot the volatility surface against deltas and tenors,
        :math:`(\\Delta, \\tau) \\mapsto \\hat{\\Sigma}(\\Delta,\\tau)`, abussing of the notation.
        See :py:meth:`plot_delta <pricing.implied_volatility.VolatilitySmile.plot_delta>` for more
        details.


        Parameters
        ----------
        provide_fig : bool, optional
            If True, return the plot as a Plotly Figure. Default is False.

        Returns
        -------
        None or plotly.graph_objs._figure.Figure
            If provide_fig is True, returns a Plotly Figure.

        Notes
        -----
        It requires a fitted surface to be available.
        """
        if self.polynomial_delta is None:
            raise ValueError("You need to fit first")

        data_line = {}
        x_curve = self.delta_grid
        y_curve = np.linspace(
            float(min(self.tenors)),
            float(max(self.tenors)),
            100,
        )
        data_line["x"] = x_curve
        data_line["y"] = y_curve
        X, Y = np.meshgrid(x_curve, y_curve)
        z_curve = self.get_implied_from_delta(X, Y)

        data_line["z"] = z_curve

        data = {"x": self.deltas, "y": self.tenors, "z": self.implied_vols}
        data = {
            "x": [x for x in data["x"] if self.delta_min <= x <= self.delta_max],
            "y": [
                y * 365
                for i, y in enumerate(data["y"])
                if self.delta_min <= data["x"][i] <= self.delta_max
            ],
            "z": [
                z
                for i, z in enumerate(data["z"])
                if self.delta_min <= data["x"][i] <= self.delta_max
            ],
        }
        fig = go.Figure()

        fig.add_trace(
            go.Scatter3d(
                x=data["x"],
                y=data["y"],
                z=data["z"],
                mode="markers",
                marker=dict(size=3),
            )
        )

        surface_trace = go.Surface(
            x=data_line["x"],
            y=data_line["y"] * 365,
            z=data_line["z"],
        )
        fig.add_trace(surface_trace)
        title = f"Underlying: {self.underlying_ticker}. Value: {self.spot}. Direction: {self.direction}. Date: {self.dtime.strftime('%Y-%m-%d')}."
        fig.update_layout(title=title, title_x=0.5)
        fig.update_scenes(
            xaxis_title="Delta",
            yaxis_title="Tenor (Days)",
            zaxis_title="Implied Volatility",
        )
        fig.update_layout(width=800, height=700)
        if provide_fig:
            return fig
        else:
            fig.show()
