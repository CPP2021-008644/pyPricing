import numpy as np
import pandas as pd
from pandas.tseries.offsets import BDay
from scipy import optimize, interpolate
from itertools import product
from abc import ABC, abstractmethod
import statsmodels.api as sm
import plotly.graph_objects as go

try:
    from . import (
        functions as afsfun,
    )  # (Relative) import needed for the workspace. In this case __package__ is 'pypricing.pricing'
except (ImportError, ModuleNotFoundError, ValueError):
    import pricing.functions as afsfun  # (Absolute) local import


# -------------------------------------------------------------------------------------------------------
# Parent classes
# -------------------------------------------------------------------------------------------------------


class DiscountCurve(ABC):
    def __init__(self, calendar=None):
        self.calendar = calendar  # TODO : unused attribute
        self.interpolation_data = None

    def compute_fbond(
        self, dates, beg_dates, end_dates, calendar
    ):  # TODO : the name should be, compute_forward_bonds, to distinguish from future bonds.
        """
        Compute forward bond price between beginning and ending dates, given a certain calendar.

        Parameters
        ----------
        dates : pandas.DatetimeIndex, list of strings, pandas.Timestamp, or string
            Dates at which to compute the forward bond prices. It can be a single date (as a pandas.Timestamp or its string representation) or an array-like object
            (as a pandas.DatetimeIndex or a list of its string representation) containing dates.
        beg_dates : pandas.DatetimeIndex, list of strings, pandas.Timestamp, or string
            Beginning dates of the bond period. If only one date is provided, it will be converted to a list. It can be a single date (as a pandas.Timestamp or its string
            representation) or an array-like object (as a pandas.DatetimeIndex or a list of its string representation) containing dates..
        end_dates : pandas.DatetimeIndex, list of strings, pandas.Timestamp, or string
            Ending dates of the bond period. If only one date is provided, it will be converted to a list. It can be a single date (as a pandas.Timestamp or its string
            representation) or an array-like object (as a pandas.DatetimeIndex or a list of its string representation) containing dates.
        calendar : Calendar
            The calendar system (Day Count Convention) to use when computing intervals. Calendar should be understood as an element of the classes defined in calendar.py.

        Returns
        -------
        ndarray
            The forward bond prices. A 2-D array of shape (``len(beg_dates)``, ``len(dates)``) containing the
            computed forward bond prices for each date and each beginning-ending period.

        Raises
        ------
        ValueError
            If ``beg_dates`` and ``end_dates`` cannot be broadcasted to a common shape. They should either have the same number
            of elements, or one of them should be a single date that applies to all elements of the other.

        Notes
        -----
        This function uses the ``get_value`` method of the current class to calculate discount factors
        at the beginning and ending dates, and then computes forward bond prices by dividing the
        discount factor at the ending date by the one at the beginning date.

        In mathematical terms, if :math:`D(t, T)` denotes the discount factor at time :math:`t` for maturity :math:`T`,
        the forward bond price for a bond with beginning date :math:`T_1` and ending date :math:`T_2`, computed at a
        date :math:`t`, is given by:

        .. math::

           \\text{FP}(t, T_1, T_2) = \\frac{D(t, T_2)}{D(t, T_1)}

        where:

        - :math:`t` corresponds to ``dates`` in the function parameters.
        - :math:`T_1` corresponds to ``beg_dates`` in the function parameters.
        - :math:`T_2` corresponds to ``end_dates`` in the function parameters.
        """
        
        beg_dates, end_dates = afsfun.dates_formatting(beg_dates, end_dates)

        disc_end = np.full((end_dates.size, dates.size), np.nan)
        for i in range(end_dates.size):
            disc_end[i] = self.get_value(
                dates=dates, future_dates=end_dates[i], calendar=calendar
            )

        disc_beg = np.full((beg_dates.size, dates.size), np.nan)
        for i in range(beg_dates.size):
            disc_beg[i] = self.get_value(
                dates=dates, future_dates=beg_dates[i], calendar=calendar
            )
        disc = disc_end / disc_beg
        if disc.shape == (1, 1):  # To avoid issues if we just want one number
            return disc.flatten()
        return disc

    def _get_value_formatting(
        self, dates, future_dates=None, tenors=None, calendar=None
    ):
        """
        Prepare and return the formatted input for the proper implementation of get_value in child classes.

        This method takes in dates and either future dates or tenors and returns them formatted along with the calculated time intervals (tau).
        If future dates are provided, tau is calculated as the interval between dates and future dates using the provided calendar.
        If future dates are not provided, tau is taken to be the tenors.

        Parameters
        ----------
        dates : pandas.DatetimeIndex, list of strings, pandas.Timestamp, or string
            Dates at which to compute the discounted values. It can be a single date (as a pandas.Timestamp or its string representation) or an array-like object
            (as a pandas.DatetimeIndex or a list of its string representation) containing dates.
        future_dates : pandas.DatetimeIndex, list of strings, pandas.Timestamp, or string, optional
            Future dates for which the discounted values are computed. If provided, this overrides tenors. It can be a single date (as a pandas.Timestamp or its string
            representation) or an array-like object
            (as a pandas.DatetimeIndex or a list of its string representation) containing dates.
        tenors : array_like, optional
            Tenors for which the discounted values are computed. Only used if future_dates is not provided.
        calendar : Calendar, optional
            The calendar system (Day Count Convention) to use when computing intervals. If None, the discount curve calendar is used.
            Calendar should be understood as an element of the classes defined in calendar.py.

        Returns
        -------
        pandas.DatetimeIndex
            A pandas DatetimeIndex object containing the formatted dates in ascending order.
        pandas.DatetimeIndex
            A pandas DatetimeIndex object containing the formatted dates in ascending order.
        ndarray
            The calculated time intervals.

        Raises
        ------
        ValueError
            - If both future_dates and tenors are None. At least one must be provided.

        Notes
        -----
        This function uses the `interval` method of the Calendar class to calculate time intervals between dates and future dates
        or tenors. If future_dates is provided, it is used. Otherwise, tenors is used.
        """
        if future_dates is None and tenors is None:
            raise ValueError(
                "At least one of 'future_dates' or 'tenors' must be provided."
            )

        dates = afsfun.dates_formatting(dates)

        if calendar is None:
            calendar = self.calendar

        if future_dates is not None:
            future_dates = afsfun.dates_formatting(future_dates)
            tau = calendar.interval(dates, future_dates)
        else:
            tau = tenors
            future_dates = dates

        if not np.all(tau >= 0):
            raise Exception(f"taus are not all positive: {tau}")

        return dates, future_dates, tau

    @abstractmethod
    def get_value(self, dates, future_dates=None, tenors=None, calendar=None):
        """
        Abstract method to compute the discounted value at given dates for a specific type of discount curve.

        The exact formula for the discounted value depends on the specific discount curve used, and should be
        implemented in a child class. The discounted value is generally a function of the time intervals between
        the provided dates and future_dates (or ``tenors`` if ``future_dates`` is not provided).

        In more precise terms, this corresponds to :math:`D(t,T)` of Definition 1.1.2 of [Brigo and Mercurio, 2006]. As is standard, if rates :math:`r` are deterministic,
        then :math:`D` is deterministic as well and necessarily :math:`D(t, T) = P(t, T)`, being the latter the price of a zero-coupon bond for each pair :math:`(t, T)`.
        However, if rates are stochastic, :math:`D(t, T)` is a random quantity at time :math:`t` depending on the future evolution of rates :math:`r` between :math:`t` and
        :math:`T`.


        Parameters
        ----------
        dates : pandas.DatetimeIndex, list of strings, pandas.Timestamp, or string
            Dates at which to compute the discounted values. It can be a single date (as a pandas.Timestamp or its string representation) or an array-like object
            (as a pandas.DatetimeIndex or a list of its string representation) containing dates.
        future_dates : pandas.DatetimeIndex, list of strings, pandas.Timestamp, or string
            Future dates for which the discounted values are computed. If provided, this overrides tenors. It can be a single date (as a pandas.Timestamp or its string
            representation) or an array-like object (as a pandas.DatetimeIndex or a list of its string representation) containing dates .
        tenors : array_like, optional
            Tenors for which the discounted values are computed. Only used if ``future_dates`` is not provided.
        calendar : Calendar, optional
            The calendar system to use when computing intervals. If None, the discount curve calendar is used. Calendar should be a class of the
        calendars.py module.

        Returns
        -------
        values : ndarray
            The computed discounted values at each date.

        Raises
        ------
        ValueError
            - If both ``future_dates`` and ``tenors`` are None. At least one must be provided.

        Notes
        -----
        This function uses the ``interval`` method of the Calendar class to calculate time intervals between dates
        and ``future dates`` or ``tenors``. If ``future_dates`` is provided, it is used. Otherwise, ``tenors`` is used.

        The actual computation of discounted values should be implemented in a child class, depending on the
        specific type of discount curve used. For example, for a constant rate r, the discounted value might be
        calculated as :math:`e^{-r \\tau}`, where :math:`\\tau` is the time interval.

        References
        ----------
        - Brigo, D., & Mercurio, F. (2006). Interest rate models: theory and practice. Berlin: Springer.


        """
        pass

    @abstractmethod
    def fit(self, data):
        """
        Abstract method to fit the curve parameters to a given data.

        This method should be implemented in a child class, according to the specific type of discount curve used.
        The exact process of fitting the curve parameters will depend on the specific curve and the structure of the data.

        Parameters
        ----------
        data : DataFrame or similar
            The data to which the curve parameters will be fitted. The structure and content of this data
            will depend on the specific type of discount curve and should be described in the child class implementation.

        Returns
        -------
        None

        Raises
        ------
        NotImplementedError
            If the method is not implemented in a child class.

        Notes
        -----
        This method is expected to update the internal state of the instance, to reflect the fitted curve parameters.
        The exact details of this process will depend on the child class implementation.
        """
        pass


class ShiftedCurve(DiscountCurve):
    """
    A class used to represent a shifted discount curve, inherited from the DiscountCurve class.

    This class provides methods for calculating the value of a discount curve that has been shifted in time.
    It supports three methods of curve shifting: 'refit', 'shift', and 'scaling'.

    Parameters
    ----------
    curve : DiscountCurve
        The base curve from which the shifted curve is derived.
    date_0 : pandas.DatetimeIndex, pandas.Timestamp, or string
        The date when the curve is calibrated :math:`t_0`.
    delay : int, optional
        The number of periods by which to shift the interpolation data of the curve. Defaults to None.
    method : str, optional
        The method to use for shifting the curve. Can be 'refit', 'shift', or 'scaling'. Defaults to "scaling".

    Attributes
    ----------
    curve : DiscountCurve
        The original discount curve that will be shifted.
    date_0 : pandas.DatetimeIndex
        The start date for the curve shift.
    offset : pandas.DateOffset
        Delay in days.
    method : str
        The method of curve shifting, one of 'refit', 'shift', or 'scaling'. Default is 'scaling'.
    interpolation_data : DataFrame
        The data used for interpolation in the 'refit' method, shifted by 'delay' periods.
    curve_copy : DiscountCurve
        A copy of the original curve, fitted with the shifted interpolation data (only used in 'refit' method).

    Raises
    ------
    AttributeError
        If 'delay' attribute is not an integer in the 'refit' method.

    Notes
    -----
    This method is mainly used to calculate theta using Monte Carlo simulations.
    It's a technical detail, but theta is the temporal derivative. To perform the finite difference,
    one only has to vary the time, while keeping everything else constant, particularly the discount curve,
    which needs to be shifted. There are several ways to achieve this, and this method supports
    three of them: 'refit', 'shift', and 'scaling'. Neverthless, the scaling method has
    the advantage that gives the discount curve as a function of :math:`(t,T)`, but calibrated
    at a different date, :math:`t_0`, which can be useful for other purposes.
    """

    def __init__(self, curve, date_0, delay=None, method="scaling"):
        super().__init__(curve.calendar)
        self.date_0 = afsfun.dates_formatting(date_0)
        self.curve = curve
        self.method = method
        if delay is not None:
            self.offset = pd.DateOffset(days=delay)
        if self.method == "refit":
            if delay >= 0:
                self.interpolation_data = curve.interpolation_data.shift(
                    delay, freq=None
                ).dropna(axis="rows", how="all")
            elif delay < 0:
                self.interpolation_data = curve.interpolation_data.shift(
                    delay, freq=None
                ).dropna(axis="rows", how="all")
            else:
                raise AttributeError("Delay must be an integer.")
            self.curve_copy = type(curve)(calendar=curve.calendar)
            self.curve_copy.fit(curve.interpolation_data)

    def get_value(self, dates, future_dates=None, tenors=None, calendar=None):
        """
        Get the value of the shifted discount curve at specified dates and tenors.

        This method calculates the value of the shifted curve based on the provided dates and tenors.
        The calculation method depends on the 'method' attribute of the class, and can
        be 'refit', 'shift' and 'scaling'. If the subscript in :math:`P_{t_c}` represents the calibration date.

        - If the method is 'refit', the value of the curve copy (fitted with shifted interpolation data) is returned.
        - If the method is 'shift', the value of the original curve at the shifted dates is returned.
        - If the method is 'pure_scaling', the value of the original curve at the dates is divided by the value of the curve at the shifting dates as follows:\
         :math:`(t,T)\\mapsto \\frac{P_{t_0}(t_0, T)}{P_{t_0}(t_0, t)}=P_{t_0}(t, T)`, i.e., the forward discount using the curve calibrated at :math:`t_0`. For :math:`t<t_0`\
         it uses the following extension,  :math:`P^\\text{shifted}(t,T):=P_{t_0}(t_0, T)\cdot P_{t}(t, t_0)`.

        Parameters
        ----------
        dates : pandas.DatetimeIndex, list of strings, pandas.Timestamp, or string
            The dates at which to calculate the value of the discount curve. It can be a single date (as a pandas.Timestamp or its string representation) or an array-like object
            (as a pandas.DatetimeIndex or a list of its string representation) containing dates.
        future_dates : pandas.DatetimeIndex, list of strings, pandas.Timestamp, or string
            Future dates used in the calculation. Defaults to None. It can be a single date (as a pandas.Timestamp or its string representation) or an array-like object (as a
            pandas.DatetimeIndex or a list of its string representation) containing dates.
        tenors : float or array-like, optional
            The tenors at which to calculate the value of the discount curve. Defaults to None.
        calendar : Calendar, optional
            The calendar to consider for date calculations. Defaults to None.

        Returns
        -------
        numpy.array
            The calculated value(s) of the shifted discount curve.
        """
        if self.method == "refit":
            return self.curve_copy.get_value(dates, future_dates, tenors, calendar)
        elif self.method == "shift":
            dates = afsfun.dates_formatting(dates)
            return self.curve.get_value(
                dates + self.offset, future_dates, tenors, calendar
            )
        elif self.method == "scaling":  # See page 20 of Bloomberg documentation, Greeks
            if dates >= self.date_0:
                return self.curve.compute_fbond(
                    self.date_0, dates, future_dates, calendar=calendar
                )
            else:  # Valid for one date
                if future_dates < self.date_0:
                    return 1
                else:
                    return self.curve.get_value(
                        self.date_0, future_dates, calendar=calendar
                    ) * self.curve.get_value(dates, self.date_0, calendar=calendar)

    def fit(self, data):
        pass


class CRDC(DiscountCurve):
    """
    This class represents a constant rate discount curve. It inherits from the DiscountCurve class.

    Parameters
    ----------
    r : float
        The constant discount rate to use.
    calendar : Calendar object, optional
        A calendar object to use for date calculations, Day Count Convention used. If not provided, the discount curve calendar is used. Calendar should be a class of the
        calendars.py module.
    is_ir_curve : bool, optional
        If True, the curve is treated as an interest rate curve. If False, it's treated as a discount curve.
        Default is True.

    Attributes
    ----------
    rate : float
        The constant discount rate of the curve.
    p : int
        This attribute can take three different values,
            - -1 : negative perturbation in `get_value`,
            - 0 : no perturbation in `get_value`,
            - +1 : positive perturbation in `get_value`.
    is_ir_curve : bool
        If True, the curve is treated as an interest rate curve. If False, it's treated as a discount curve.
    calendar : Calendar object
        The calendar object used for date calculations.

    Notes
    -----
    The constant rate discount curve represents a special case of a discount curve where the discount rate
    is constant over time. This means that the present value of a future cash flow can be computed directly
    by applying the constant discount rate to the future value of the cash flow.

    The constant rate discount curve is most appropriate in settings where interest rates are stable and
    are expected to remain stable over the forecast period.

    Examples
    --------
    To create a constant rate discount curve with a rate of 0.05:

    >>> curve = CRDC(0.05)
    >>> print(curve.rate)
    0.05
    """

    # date format: %Y%m%d
    # annualized rate

    def __init__(self, r, calendar=None, is_ir_curve=True):
        self.rate = r
        self.p = 0
        self.is_ir_curve = is_ir_curve
        DiscountCurve.__init__(self, calendar=calendar)

    def fit(self, data):  # Added for coherence
        """
        Fits the curve parameters to the given data. This method assumes that the first element of the
        data list/array is the constant discount rate (r) and the second element is the constant p.

        Parameters
        ----------
        data : list, array
            The list or array of parameters to fit the curve. The first element should be the constant
            discount rate (r) and the second element should be the constant p.

        Notes
        -----
        In the context of this class, "fitting" is rather trivial. It only involves assigning the provided
        values to the class attributes ``rate`` and ``p``.

        The fit method modifies the internal state of the instance by changing the ``rate`` and ``p`` attributes.

        Examples
        --------
        >>> curve = CRDC(0.05)
        >>> curve.fit([0.06, 1])
        >>> print(curve.rate, curve.p)
        0.06 1
        """
        self.rate = data[0]
        self.p = data[1]

    def get_value(
        self, dates, future_dates=None, tenors=None, calendar=None, ty="continuous"
    ):
        """
        Compute the discounted value at given dates for a specified interest rate and compounding type.

        The discounted value is calculated based on the time intervals between the provided dates and future_dates (or
        tenors if future_dates is not provided). For the "continuous" compounding type, the value is calculated as

        .. math::
            D^\\textnormal{c}(t, T)=e^{-(r+p\\cdot 0.0005)\\cdot\\tau},

        and for the "simple" compounding type, it is calculated as

        .. math::
            D^\\textnormal{s}(t, T)=\\dfrac{1}{(1+(r+p\\cdot0.0005)\\cdot\\tau)},

        where :math:`r` is the interest rate, :math:`p` is a parameter (perturbation), and :math:`\\tau` is the time interval. :math:`\\tau(t, T)` is computed using the
        calendar if :math:`t` (``dates``) and :math:`T` (``future_dates``) are given. Recall that :math:`p`, as mentioned in the class docstring, represents a perturbation.
        It can take on the values :math:`\\pm 1` or 0. By convention, the absolute value of the total perturbation for the fixed rate is set to 5 basis points.

        Parameters
        ----------
        dates : array_like
            Dates at which to compute the discounted values.
        future_dates : array_like, optional
            Future dates for which the discounted values are computed. If provided, this overrides tenors.
        tenors : array_like, optional
            Tenors for which the discounted values are computed. Only used if future_dates is not provided.
        calendar : Calendar, optional
            The calendar system (Day Count Convention) to use when computing intervals. If None, the discount curve calendar is used.
        ty : str, optional
            The type of compounding to use ("continuous" or "simple"). Default is "continuous".

        Returns
        -------
        ndarray
            The computed discounted values at each date.

        Raises
        ------
        ValueError
            If both future_dates and tenors are None. At least one must be provided.
            If ty is not "continuous" or "simple".

        Notes
        -----
        This function uses the `interval` method of the Calendar class to calculate time intervals between dates and future dates
        or tenors. If future_dates is provided, it is used. Otherwise, tenors is used.
        """
        if self.rate == 0:
            p = 0
        else:
            p = (
                self.p
            )  # TODO : as self.p=0 in the _init_, p is always 0 unless modified externally before calling the method. Include as an argument?

        dates, future_dates, tau = DiscountCurve._get_value_formatting(
            self,
            dates=dates,
            future_dates=future_dates,
            tenors=tenors,
            calendar=calendar,
        )

        if ty == "continuous":
            values = np.exp(-(self.rate + p * 0.0005) * tau)
        elif ty == "simple":
            values = 1 / (1 + (self.rate + p * 0.0005) * tau)
        else:
            raise ValueError("Compounding type not well defined")
        # values = values * (dates <= future_dates)
        return values


class LCExpCurve(DiscountCurve):
    """
    A subclass of DiscountCurve that represents a locally constant exponential discount curve. That is, basically (below for more details),

    .. math::
        D(t, T) = \\exp\\left(\\int_0^{\\tau=T-t} \\lambda(s) ds\\right)

    for a locally constant function :math:`\\lambda`.

    Parameters
    ----------
    calendar : Calendar object, optional
        A calendar object to use for date calculations, Day Count Convention used. If not provided, the discount curve calendar is used. Calendar should be a class of the
        calendars.py module.
    perturbations : tuple of float, optional
        The perturbations used for computing risk sensitivities, default is (-0.0020, 0.0020). This is the convention for credit risk.
    is_ir_curve : bool, optional
        If True, the curve represents an interest rate curve. Default is False.
    is_credit_curve : bool, optional
        If True, the curve represents a credit curve. Default is True.

    Attributes
    ----------
    p : int
        This attribute can take three different values,
            - -1 : negative perturbation, :math:`p_{-1}`,
            - 0 : no perturbation, :math:`p_{0}=0`,
            - +1 : positive perturbation, :math:`p_{1}`.
    interpolation_data : pandas.DataFrame
        A DataFrame that contains parameters to fit the curve.
    perturbations : tuple of float
        The perturbations used for computing risk sensitivities.
    is_ir_curve : bool
        Flag indicating if the curve is an interest rate curve.
    is_credit_curve : bool
        Flag indicating if the curve is a credit curve.
    params : dict
        Dictionary to hold parameters related to the curve. It is a dictionary of perturbations where keys are -1, 0, and 1, and values are pandas DataFrame of parameters.
    """

    def __init__(
        self,
        calendar,
        perturbations=(-0.0020, 0.0020),
        is_ir_curve=False,
        is_credit_curve=True,
    ):
        DiscountCurve.__init__(self, calendar=calendar)
        self.p = 0
        self.interpolation_data = None
        self.perturbations = perturbations
        self.is_ir_curve = is_ir_curve
        self.is_credit_curve = is_credit_curve
        self.params = {}

    def fit(self, params_df):
        """
        Fit the LCExpCurve model to the given data.

        This method prepares the parameters data for the calculation of the curve values.
        It sets up three versions of the parameters DataFrame (``params_df``), corresponding to a perturbation down (-1), no perturbation (0), and a perturbation up (+1).


        Parameters
        ----------
        params_df : pandas.DataFrame
            The input parameters as a DataFrame. Each row should correspond to a date and each column to a tenor.

        Notes
        -----
        The interpolation data is set as the input ``params_df``. The fitted parameters are stored in the ``params`` attribute of the object.
        The perturbations are added to the ``params_df`` values according to:

        :math:`\\hspace{1.5cm}\\lambda^{(t,~p)}` = ``params_df.loc[date][l]`` with :math:`s\\in [T_l, T_{l+1})` and :math:`t` = ``date``.

        Also,

        ``params[i]`` = ``params_df`` + :math:`p_i\\cdot\\Delta_i`

        where :math:`p_i` are the perturbation values defined at the object initialization, ``self.perturbations`` = :math:`(\\Delta_{-1}, \\Delta_{1})`   and ``params[-1]``,
        ``params[0]``, and ``params[1]`` are the parameters values stored in the ``params`` dictionary attribute under the keys `-1`, `0`, and `1` respectively.

        The value ``p`` is set to 0 but currently has no effect on the method's operation.

        See Also
        ----------
        discount_curves.LCExpCurve.get_value
        """
        self.interpolation_data = params_df
        self.params = {
            -1: params_df + self.perturbations[0],
            0: params_df,
            1: params_df + self.perturbations[1],
        }

    @staticmethod
    def compute_relative_values(
        params, tenors, tau
    ):  # As a static method so it appears in the documentation, although could be private
        """
        This function calculates the curve value at a specific tenor.

        Parameters
        ----------
        params : ndarray
            Array of parameter values for the curve.

        tenors : ndarray
            Array of tenors for the curve.

        tau : float
            Specific tenor at which to calculate the curve value.

        Returns
        -------
        float
            The calculated curve value at the tenor ``tau``.

        Notes
        -----
        This function calculates the curve value as:

        .. math::
            V =
            \\begin{cases}
            0, & \\text{if}\\ \\tau \\leq 0, \\\\
            \\sum_{l=0}^{\\text{I}-1} \\rho_l \cdot \\Delta T_{l} + \\rho_{\\text{I}} \cdot \\Delta T_{\\text{I}}, & \\text{otherwise},
            \\end{cases}

        where:

        - :math:`V` is the calculated value,
        - :math:`\\rho_l` are the parameter values in `params`,
        - :math:`\\Delta T_{l}:=T_l-T_{l-1}` with :math:`T_{-1}:=0` are the relative tenors,
        - :math:`\\text{I}` is the index where `tenors` becomes greater than or equal to :math:`\\tau`,
        - :math:`\\Delta T_{\\text{I}}` is the difference between :math:`\\tau` and :math:`T_{\\text{I}-1}`,
        - :math:`\\tau` is the specific tenor at which to calculate the curve value.

        See Also
        ----------
        discount_curves.LCExpCurve.get_value

        Examples
        --------
        Suppose we have ``params = numpy.array([0.1, 0.2, 0.3, 0.4])`` and ``tenors = numpy.array([1, 2, 3, 4])``, and we want to calculate the curve value at
        :math:`\\tau:=` ``tau = 2.5``.

        Step 1: Check if :math:`\\tau \le 0`. In this case, tau is 2.5 so we proceed to the next steps.

        Step 2: Calculate :math:`\\text{I}` as the index where ``tenors >= tau``. In this case, :math:`\\text{I}` would be 2 as the value at index 2 in `tenors` (which is 3)
        is the first value that is greater than `tau`.

        Step 3: Create ``rel_tenors`` as the concatenation of ``tenors`` up to :math:`\\text{I}` and :math:`\\tau`. This would give ``rel_tenors = numpy.array([1, 2, 2.5])``.

        Step 4: Subtract the preceding tenor from each tenor in ``rel_tenors`` to get the relative tenors: ``rel_tenors = numpy.array([1, 1, 0.5])``.

        Step 5: Add zeros to the end of ``rel_tenors`` to make it the same length as ``tenors``. This gives ``rel_tenors = numpy.array([1, 1, 0.5, 0])``.

        Step 6: Calculate the curve value :math:`V` as the sum of the element-wise product of ``params`` and ``rel_tenors``. This gives`V = 0.1*1 + 0.2*1 + 0.3*0.5 + 0.4*0 = 0.25`.

        """

        if tau <= 0:
            return 0
        else:
            idx = np.argmax(
                tenors >= tau
            )  # This is a boolean sequence, so it return the first time the index is 1
            rel_tenors = np.concatenate((tenors[:idx], [tau]))
            rel_tenors -= np.concatenate(([0], rel_tenors[:-1]))
            rel_tenors = np.concatenate(
                (rel_tenors, np.zeros(max(tenors.size - idx - 1, 0)))
            )
            return np.sum(params * rel_tenors)

    def get_value(self, dates, future_dates=None, tenors=None, calendar=None):
        """
        Computes curve values at specified dates and tenors.

        Parameters
        ----------

        dates : pandas.DatetimeIndex, list of strings, pandas.Timestamp, or string
            Dates at which to compute the discounted values. It can be a single date (as a pandas.Timestamp or its string representation) or an array-like object
            (as a pandas.DatetimeIndex or a list of its string representation) containing dates.
        future_dates : pandas.DatetimeIndex, list of strings, pandas.Timestamp, or string
            Future dates for which the discounted values are computed. If provided, this overrides tenors. It can be a single date (as a pandas.Timestamp or its string
            representation) or an array-like object (as a pandas.DatetimeIndex or a list of its string representation) containing dates .
        tenors : array-like, optional
            The tenors at which to compute the curve values. Either this or ``future_dates`` must be provided.
        calendar : Calendar object, optional
            A calendar object to use for date calculations, Day Count Convention used. If not provided, the discount curve calendar is used. Calendar should be a class of the
            calendars.py module.

        Returns
        -------
        numpy.ndarray
            The curve values at specified dates and tenors.

        Raises
        -------
        ValueError
            If ``dates`` and ``end_dates`` cannot be broadcasted to a common shape. They should either have the same number of elements,
            or one of them should be a single date that applies to all elements of the other. See ``_get_value_formatting``.

        Notes
        -----
        The calculation of discounted values is based on the formula:

        .. math::
            D(t, T) = \\exp\\left(\\int_0^{\\tau(t, T)} \\lambda^{(p,t)}(s) ds\\right)

        where :math:`\\lambda^{(p,t)}` is a locally constant function with parameters associated with each date (for the given perturbation level, :math:`p`). If the time interval
        :math:`\\tau` is less or equal to zero, the corresponding value is set to zero before the exponential calculation. :math:`\\tau(t, T)` is computed using the calendar
         if :math:`t` and :math:`T` are given.

        See Also
        ----------
        discount_curves.LCExpCurve.compute_relative_values

        """
        p = (
            self.p
        )  # TODO : This is zero unless modified externally before calling the method. Include as an argument?
        dates, future_dates, taus = DiscountCurve._get_value_formatting(
            self,
            dates=dates,
            future_dates=future_dates,
            tenors=tenors,
            calendar=calendar,
        )

        values = np.full(taus.shape, np.nan)
        if future_dates is None:
            size = taus.size
        else:
            size = future_dates.size
        for i in range(max(dates.size, size)):
            if np.all(np.isin(dates, self.params[0].index)):
                pass
            else:
                raise ValueError(f"Not all dates ({dates}) are in the fitting dates.")
            temp_params = self.params[p].loc[
                dates[min(i, dates.size - 1)]
            ]  # If dates.size == 1, then always 0. Otherwise, params depend on chosen date.
            temp_tenors = self.params[p].columns
            values[i] = LCExpCurve.compute_relative_values(
                temp_params, temp_tenors, taus[i]
            )

        return np.exp(values)


class YieldCurve(DiscountCurve):
    """
    Implements a discount curve using cubic spline interpolation.

    Parameters
    ----------
    calendar : Calendar object
        A calendar object to use for date calculations, Day Count Convention used. If not provided, the discount curve calendar is used. Calendar should be a class of the
        calendars.py module.
    perturbations : tuple, optional
        Perturbations to apply in the curve fitting, by default (-0.0005, 0.0005), namely :math:`\\pm 5` basis points.
    is_ir_curve : bool, optional
        If True, this is an interest rate curve, by default True.
    is_credit_curve : bool, optional
        If False, this is a credit curve, by default False.

    Attributes
    ----------
    fitting_dates : pandas.DatetimeIndex, list of strings, pandas.Timestamp, or string
        The dates used for fitting the curve. It can be a single date (as a pandas.Timestamp or its string representation) or an array-like object (as a pandas.DatetimeIndex
        or a list of its string representation) containing dates.
    interpolation_data : pandas.DataFrame
        The DataFrame holding the data for interpolation.
    interpolations_dic : dict
        Dictionary holding interpolations for different dates and perturbations.
    p : int
        The index of the current perturbation used. This attribute can take three different values,
            - -1 : negative perturbation, :math:`p_{-1}`,
            - 0 : no perturbation, :math:`p_{0}=0`,
            - +1 : positive perturbation, :math:`p_{1}`.
    is_ir_curve : bool
        Boolean flag to indicate if this is an interest rate curve.
    is_credit_curve : bool
        Boolean flag to indicate if this is a credit curve.
    perturbations : tuple
        Perturbations to apply in the curve fitting. Mathematically, :math:`(p_{-1}, p_1)`.
    zc_yields : pandas.Dataframe
        Bond yields, (6.5) of Andersen and Piterbarg, of interpolation data.

    Notes
    ----------
    Do not extrapolate (beyond 30y)
    """

    def __init__(
        self,
        calendar,
        perturbations=(-0.0005, 0.0005),
        is_ir_curve=True,
        is_credit_curve=False,
    ):
        self.zc_yields = None
        self.fitting_dates = None
        self.interpolation_data = None
        self.interpolations_dic = None
        self.p = 0
        self.is_ir_curve = is_ir_curve
        self.is_credit_curve = is_credit_curve
        self.perturbations = perturbations
        DiscountCurve.__init__(self, calendar=calendar)

    def fit(self, data, delta_ind=None, method="bond_spline"):
        """
        Fits the interpolation data to a given data set using either cubic spline, yield spline, flat forwards or Piecewise cubic Hermite interpolating polynomial methods.

        The method sets the object's interpolation data and fitting dates based on the provided data. It then fits the
        specified type of interpolation for each date in the data and for perturbations at levels -1, 0, and 1.

        For each date and perturbation pair, an interpolation is fit to the data. For perturbation levels -1 and 1,
        the data values are multiplied by a discount factor before fitting the interpolation.

        The interpolation for a given date `t` and perturbation `i` is given by:

        .. math::
            S_{p_i,t}(\\tau) = \\text{InterpolationMethod}\\left(\\textbf{T}, \\textbf{P}_t \\cdot e^{+p_i \\cdot\\boldsymbol{\\delta}_j \\textbf{T}}\\right)(\\tau)

        where:

        - :math:`\\textbf{T}` is the vector of tenors,
        - :math:`\\textbf{P}_t` are the data values for date `t`,
        - :math:`p_i` is the perturbation for level `i`.
        - :math:`\\boldsymbol{\\delta}_j` (a vector of zeros except for the :math:`j`-th entry) where :math:`j` is the value of ``delta_ind``. If ``delta_ind=None``,
          :math:`\\boldsymbol{\\delta}_j=\\textbf{1}`, a vector of ones.


        Parameters
        ----------
        data : DataFrame
            A DataFrame with dates as the index and tenors as columns. The values are the discount values.

        delta_ind : int, optional
            Index of the tenor to use for perturbation. By default, perturbation is applied to all tenors.

        method : str, optional
            The interpolation method to use. Can be "bond_spline", "yield_spline", "flat_forwards" or "yield_pchip". Default is "bond_spline", which perform the spline directly on bonds.
            "yield_spline" follows Section 6.2.3 of [Andersen and Piterbarg, 2010] , being the yields the interpolated object. "flat_forwards" follows Section 6.2.1.2 of
            [Andersen and Piterbarg, 2010]. "yield_pchip" is performed by scipy.interpolate.PchipInterpolator class.

        Notes
        -----
        The fitted curve is stored internally in the `interpolations_dic` attribute for later retrieval.

        Returns
        -------
        None

        References
        ----------
        - Andersen, L. B. G., and Piterbarg, V.V. (2010). Interest Rate Modeling.

        Warnings
        --------
        - The input `data` of this method can differ from that of the corresponding method in child classes.
        """

        if (
            type(self) == YieldCurve
        ):  # Do not enter here if it is coming from a child class with already interpolation_data
            self.interpolation_data = data
        tenor_len = data.columns.size
        self.fitting_dates = self.interpolation_data.index
        pairs = product(
            self.interpolation_data.index, [-1, 0, 1]
        )  # Creates a new iterable of 2-tuples (pairs), the first element correspond to a date and
        # the second is either -1, 0 or 1.
        self.interpolations_dic = {}
        perturbations = np.array([self.perturbations[0], 0, self.perturbations[1]])
        pert_ind_dict = {-1: 0, 0: 1, 1: 2}
        if delta_ind is None:
            delta_ind = 1
        elif isinstance(delta_ind, int):
            delta_ind = np.eye(tenor_len, dtype=int)[delta_ind, :]
        else:
            raise AttributeError("delta_ind must be an integer.")

        # for pair in pairs:
        #     if pair[1] == 0:
        #         self.interpolations_dic[pair] = interpolate.CubicSpline(x=self.interpolation_data.loc[pair[0]].dropna().index,
        #                                                                 y=self.interpolation_data.loc[pair[0]].dropna().values)
        #     else:
        #         self.interpolations_dic[pair] = interpolate.CubicSpline(x=self.interpolation_data.loc[pair[0]].dropna().index,
        #                                                                 y=self.interpolation_data.loc[pair[0]].dropna().values *
        #                                                                 np.exp(-self.perturbations[pair[1]] *  TODO : check, pair[1] is either +1 or -1, so it is always
        #                                                                                                              perturbations[1], as this has size 2. check also the
        #                                                                                                              + sign and delete
        #                                                                        self.interpolation_data.loc[pair[0]].dropna().index.astype("float64")))
        if method == "bond_spline":
            for pair in pairs:
                self.interpolations_dic[pair] = interpolate.CubicSpline(
                    x=data.loc[pair[0]].dropna().index,
                    y=data.loc[pair[0]].dropna().values
                    * np.exp(
                        +perturbations[pert_ind_dict[pair[1]]]
                        * delta_ind
                        * data.loc[pair[0]].dropna().index.astype("float64")
                    ),
                )  # TODO : bc_type=natural
        # TODO : the coefficients should be stored using the attribute CubicSpline.c
        elif method == "yield_spline":
            self.zc_yields = -np.log(data) / data.columns
            self.zc_yields[0] = 0  # The degenerate case of tenor = 0

            def exponential_yield(f):
                def exponential_wrapper(*args, **kwargs):
                    return np.exp(
                        -1 * f(*args, **kwargs) * args[0]
                    )  # Note the minus sign and that it multiplies by the tenor

                return exponential_wrapper

            for pair in pairs:
                self.interpolations_dic[pair] = exponential_yield(
                    interpolate.CubicSpline(
                        x=self.zc_yields.loc[pair[0]].dropna().index,
                        y=(
                            self.zc_yields.loc[pair[0]].dropna().values
                            - perturbations[pert_ind_dict[pair[1]]] * delta_ind
                        ),
                    )
                )
                # Minus sign for the perturbation for convention considerations

            # To check that is being done properly,
            curve_spline = self.interpolations_dic[
                (pair[0], 0)
            ]  # Check with the last pair
            series = self.zc_yields.loc[pair[0]]
            value_temp = self.interpolation_data.columns[-1]
            assert (
                round(curve_spline(value_temp), 5)
                == round(np.exp(-series.loc[value_temp] * value_temp), 5)
                == round(data.loc[pair[0]][value_temp], 5)
            )
            # # Plots, see also Code Examples - Discount Curves  #ToDO: erase after this version is stable, this is already in Code Examples
            # x_values = np.linspace(0, 30, 1000)  # Change the range and number of points as needed
            # y_values = curve_spline(x_values)
            # y_knots = np.exp(-series*series.index.values)
            # fig = go.Figure(data=go.Scatter(x=x_values, y=y_values))
            # fig.add_trace(go.Scatter(x=series.index.values, y=y_knots, mode="markers", name='Knot Points'))
            # fig.update_layout(title='Curve Spline Plot', xaxis_title='x', yaxis_title='curve_spline(x)')
            # fig.show()
            # # Plot, now a perturbation
            # curve_spline = self.interpolations_dic[(pair[0], 1)]
            # series = self.zc_yields.loc[pair[0]]
            # x_values = np.linspace(0, 30, 1000)  # Change the range and number of points as needed
            # y_values = curve_spline(x_values)
            # y_knots = np.exp(-series*series.index.values)
            # fig = go.Figure(data=go.Scatter(x=x_values, y=y_values))
            # fig.add_trace(go.Scatter(x=series.index.values, y=y_knots, mode="markers", name='Knot Points'))
            # fig.update_layout(title='Curve Spline Plot, perturbation', xaxis_title='x', yaxis_title='curve_spline(x)')
            # fig.show()
        elif method == "flat_forwards":
            yT = -np.log(data)  # This is y(T)T, see p.237 of Andersen and Piterbarg.
            for pair in pairs:
                # The `pair=pair` in the lambda function arguments creates a new scope capturing the current value of `pair`.
                # This is necessary due to Python's late binding behavior which would otherwise cause all functions in the
                # dictionary to use the final value of `pair` from the loop, rather than the specific value of `pair` at the time
                # the function was created.
                self.interpolations_dic[pair] = lambda x, pair=pair: np.exp(
                    -np.interp(
                        x,
                        xp=yT.loc[pair[0]].dropna().index,
                        fp=yT.loc[pair[0]].dropna().values
                        - perturbations[pert_ind_dict[pair[1]]]
                        * delta_ind
                        * data.columns,
                    )
                )

            curve_ff = self.interpolations_dic[(pair[0], 0)]  # Check with the last pair
            value_temp = self.interpolation_data.columns[-1]
            assert round(curve_ff(value_temp), 5) == round(
                data.loc[pair[0]][value_temp], 5
            )
            # # Plot derivative #ToDO: erase after this version is stable, this is already in Code Examples
            # curve_log_ff = lambda x: -np.log(curve_ff(x))
            #
            # def curve_forward(x):
            #     h = 0.00001  # a small delta value to calculate the derivative, adjust if necessary
            #     return (curve_log_ff(x + h) - curve_log_ff(x)) / h  # finite difference method
            # x_values = np.linspace(0, 30, 1000)  # Change the range and number of points as needed
            # y_values = curve_forward(x_values)
            # fig = go.Figure(data=go.Scatter(x=x_values, y=y_values))
            # fig.update_layout(title='Forwards Plot')
            # fig.show()
            #
            # # Plots, see also Code Examples - Discount Curves
            # series = data.loc[pair[0]]
            # x_values = np.linspace(0, 30, 1000)  # Change the range and number of points as needed
            # y_values = curve_ff(x_values)
            # y_knots = series
            # fig = go.Figure(data=go.Scatter(x=x_values, y=y_values))
            # fig.add_trace(go.Scatter(x=series.index.values, y=y_knots, mode="markers", name='Knot Points'))
            # fig.update_layout(title='Flat forwards Plot', xaxis_title='x', yaxis_title='flat_forwards(x)')
            # fig.show()
            # # Plot, now a perturbation
            # curve_ff = self.interpolations_dic[(pair[0], 1)]
            # x_values = np.linspace(0, 30, 1000)  # Change the range and number of points as needed
            # y_values = curve_ff(x_values)
            # fig = go.Figure(data=go.Scatter(x=x_values, y=y_values))
            # fig.add_trace(go.Scatter(x=series.index.values, y=y_knots, mode="markers", name='Knot Points'))
            # fig.update_layout(title='Flat forwards Plot, perturbation', xaxis_title='x', yaxis_title='flat_forwards(x)')
            # fig.show()
        elif method == "yield_pchip":
            for pair in pairs:
                self.interpolations_dic[pair] = interpolate.PchipInterpolator(
                    x=data.loc[pair[0]].dropna().index,
                    y=data.loc[pair[0]].dropna().values
                    * np.exp(
                        +perturbations[pert_ind_dict[pair[1]]]
                        * delta_ind
                        * data.loc[pair[0]].dropna().index.astype("float64")
                    ),
                )
        else:
            raise AttributeError(method + "is not implemented.")

    def get_value(self, dates, future_dates=None, tenors=None, calendar=None):
        """
        Computes and returns the values of the fitted curves at given dates and tenors.

        For each date in ``dates``, the method tries to find the corresponding cubic spline interpolation.
        If the interpolation is found, it's evaluated at the corresponding tenors. If no interpolation
        is found for a given date, NaN values are returned for that date.

        The method expects either ``future_dates`` or ``tenors`` to be provided. If ``future_dates`` are
        provided, the tenors are computed as the intervals between ``dates`` and ``future_dates`` using
        the specified ``calendar``. If only ``tenors`` are provided, they are used as is.

        The computation of curve values for a date `t` and future date `T` is given by:

        .. math::
            D(t, T) = S_{p,t}(\\tau) \cdot \mathbb{1}_{\\tau \geq 0},

        where:

        - :math:`S_{p,t}(T)` is the cubic spline interpolation for date `t` evaluated at `T`,
        - :math:`\mathbb{1}_{\\tau \geq 0}` is the indicator function that is 1 if :math:`\\tau \ge 0` and 0 otherwise. :math:`\\tau=\\tau(t, T)` is computed using the calendar
          if :math:`t` (``dates``) and :math:`T` (``future_dates``) are given.

        Parameters
        ----------
        dates : pandas.DatetimeIndex, list of strings, pandas.Timestamp, or string
            Dates at which to compute the discounted values. It can be a single date (as a pandas.Timestamp or its string representation) or an array-like object
            (as a pandas.DatetimeIndex or a list of its string representation) containing dates.
        future_dates : pandas.DatetimeIndex, list of strings, pandas.Timestamp, or string
            Future dates for which the discounted values are computed. If provided, this overrides tenors. It can be a single date (as a pandas.Timestamp or its string
            representation) or an array-like object (as a pandas.DatetimeIndex or a list of its string representation) containing dates.

        tenors : array_like of float, optional
            The tenors at which to compute the curve values. If provided, it should be the same size as ``dates``.

        calendar : Calendar object, optional
            A calendar object to use for date calculations, Day Count Convention used. If not provided, the discount curve calendar is used. Calendar should be a class of the
            calendars.py module.

        Returns
        -------
        numpy.ndarray
            The computed curve values. It has the same shape as ``dates``.

        Raises
        ------
        ValueError
            If neither ``future_dates`` nor ``tenors`` is provided.
        ValueError
            If ``dates``, ``future_dates`` (resp. ``tenors``) do not follow broadcasting rules, with ``dates.size`` not being greater than ``future_dates.size``
            (resp. ``tenors.size``).
        """
        p = self.p
        # if np.asarray(dates).shape == ():
        #     dates = [dates]
        # dates = pd.to_datetime(dates)
        # if tenors is not None:
        #     tenors = np.array(tenors)
        # if future_dates is not None:
        #     if np.asarray(future_dates).shape == ():
        #         future_dates = [future_dates]
        #     future_dates = pd.to_datetime(future_dates)
        #     tenors = self.calendar.interval(dates, future_dates)
        #     tenors = tenors*(tenors >= 0)
        dates, future_dates, taus = DiscountCurve._get_value_formatting(
            self,
            dates=dates,
            future_dates=future_dates,
            tenors=tenors,
            calendar=calendar,
        )
        tenors = taus
        if dates.size == 1:
            try:
                curve_values = self.interpolations_dic[(dates[0], p)](tenors) * (
                    tenors >= 0
                )  # + (tenors < 0)
            except KeyError:
                curve_values = np.full(tenors.shape, np.nan)
        else:
            curve_values = np.full(tenors.shape, np.nan)
            for i in range(dates.size):
                try:
                    curve_values[i] = self.interpolations_dic[(dates[i], p)](
                        tenors[i]
                    ) * (
                        tenors[i] >= 0
                    )  # TODO : added indicator function here
                except KeyError:
                    continue
        return curve_values


# -------------------------------------------------------------------------------------------------------
# Interest Rates Curves
# -------------------------------------------------------------------------------------------------------
class ZCCurve(YieldCurve):
    """
    Implements a Zero Coupon Curve using cubic spline interpolation.

    This class extends the YieldCurve class to fit a discount curve to zero coupon bond yield data.

    Attributes
    ----------
    calendar : Calendar
        The calendar system (Day Count Convention) to use when computing intervals. Calendar should be understood as an element of the classes defined in calendar.py.
    interest_type : str, optional
        The type of interest rate ("simple" or "compound"), by default "simple".
    semiannual : bool, optional
        If True, assume semiannual compounding, by default False.

    interpolation_data : DataFrame
        A DataFrame containing zero coupon yield data.

    Raises
    ------
    ValueError
        If the interest_type is not "simple" or "compound".
    """

    def __init__(self, calendar, interest_type="simple", semiannual=False):
        self.calendar = calendar
        if interest_type in ["simple", "compound"]:
            self.interest_type = interest_type
        else:
            raise ValueError(f"Interest type {interest_type} not known.")
        self.semiannual = semiannual
        YieldCurve.__init__(
            self,
            calendar=self.calendar,
            perturbations=(-0.0005, 0.0005),
            is_ir_curve=True,
            is_credit_curve=False,
        )

    def fit(self, data, delta_ind=None):
        """
        Fits the zero coupon curve to the provided yield data.

        The method will first compute bond prices from the yield data. For simple interest, the bond price :math:`P` for
        a zero coupon bond with yield :math:`y` and maturity :math:`T` is given by:

        .. math::
            P = \\frac{1}{{1 + yT}}

        For compound interest, the bond price :math:`P` is given by:

        .. math::
            P = \\frac{1}{{(1 + \\frac{y}{1 + s})^{T(1 + s)}}}

        where :math:`s` is 1 if semiannual compounding is assumed and 0 otherwise.

        After computing the bond prices, the method calls the fit method of the YieldCurve superclass to fit a
        cubic spline to the bond prices, taking into account the perturbations and a delta shift if given by ``delta_ind``.

        Parameters
        ----------
        data : DataFrame
            A DataFrame with dates as index and maturities as columns, :math:`T`. The values are the zero coupon bond yields, :math:`y`.

        delta_ind : int, optional
            Index of the tenor to use for perturbation. By default, perturbation is applied to all tenors.

        Returns
        -------
        None
        """
        self.interpolation_data = data
        if self.interest_type == "simple":
            bond_prices = 1 / (
                1 + self.interpolation_data.columns * self.interpolation_data
            )
        else:
            bond_prices = 1 / (1 + self.interpolation_data / (1 + self.semiannual)) ** (
                self.interpolation_data.columns * (1 + self.semiannual)
            )
        bond_prices[0] = 1
        bond_prices = bond_prices.sort_index(axis=1)
        YieldCurve.fit(self, bond_prices, delta_ind)


class DepositCurve(YieldCurve):
    """
    A class used to represent a Deposit Curve.

    This class is a specific implementation of the YieldCurve (Cubic Discount Curve) superclass for handling deposit rates.
    It supports both simple and compound interest calculations.

    Attributes
    ----------
    calendar : Calendar
        The calendar system (Day Count Convention) to use when computing intervals. Calendar should be understood as an element of the classes defined in calendar.py.

    interest_type : str, optional
        A string representing the type of interest (either "simple" or "compound"). Default is "simple".

    semiannual : bool, optional
        A boolean indicating whether compounding is semiannual. Default is False.

    interpolation_data : DataFrame
        A DataFrame containing deposit rates data.

    Raises
    ------
    ValueError
        If the interest_type is not "simple" or "compound".
    """

    def __init__(self, calendar, interest_type="simple", semiannual=False):
        self.calendar = calendar
        if interest_type in ["simple", "compound"]:
            self.interest_type = interest_type
        else:
            raise ValueError(f"Interest type {interest_type} not known.")
        self.semiannual = semiannual
        YieldCurve.__init__(
            self,
            calendar=self.calendar,
            perturbations=(-0.0005, 0.0005),
            is_ir_curve=True,
            is_credit_curve=False,
        )

    def fit(self, data, delta_ind=None):
        """
        Fits the deposit curve to the provided deposit rate data.

        The method will first compute bond prices from the deposit rate data. For simple interest, the bond price :math:`P` for
        a bond with deposit rate :math:`d` and maturity :math:`T` is given by:

        .. math::
            P = 1 + d\\cdot T

        For compound interest, the bond price :math:`P` is given by:

        .. math::
            P = \\left(1 + \\frac{d}{1 + s}\\right)^{T(1 + s)}

        where :math:`s` is 1 if semiannual compounding is assumed and 0 otherwise.

        After computing the bond prices, the method calls the fit method of the YieldCurve superclass to fit a
        cubic spline to the bond prices, taking into account the perturbations and a delta shift if given by ``delta_ind``.

        Parameters
        ----------
        data : DataFrame
            A DataFrame with dates as index and tenors (year fraction using calendar) as columns. The values are the deposit rates.

        delta_ind : int, optional
             Index of the tenor to use for perturbation. By default, perturbation is applied to all tenors.

        Returns
        -------
        None
        """
        self.interpolation_data = data
        if self.interest_type == "simple":
            bond_prices = 1 + self.interpolation_data.columns * self.interpolation_data
        else:
            bond_prices = (1 + self.interpolation_data / (1 + self.semiannual)) ** (
                self.interpolation_data.columns * (1 + self.semiannual)
            )  # False + 1 == 0, True + 1 == 2
        bond_prices[0] = 1
        bond_prices = bond_prices.sort_index(axis=1)
        YieldCurve.fit(self, bond_prices, delta_ind)


class CubicSplineSwapCurve(YieldCurve):
    """
    A class for computing the swap curve using cubic spline interpolation.

    This class is a specific implementation of the YieldCurve (Cubic Discount Curve) superclass
    for handling swap rates. The curve is fitted based on the provided swap rates using cubic
    spline interpolation.

    Attributes
    ----------
    calendar : Calendar
        The calendar system (Day Count Convention) to use when computing intervals.
        Calendar should be understood as an element of the classes defined in calendar.py.

    """

    def __init__(self, calendar, perturbations=(-0.0005, 0.0005)):
        YieldCurve.__init__(
            self,
            calendar=calendar,
            perturbations=perturbations,
            is_ir_curve=True,
            is_credit_curve=False,
        )
        self.zc_yields = None

    def bootstrapping(self, swap_rates):
        """
        Computes the discount factors from given swap rates via bootstrapping methods.

        This function uses recursive computation to generate discount factors from swap rates. In a typical plain vanilla interest rate swap, two parties agree to exchange
        cash flows, where one set of cash flows is based on a variable interest rate (like LIBOR), and the other set of cash flows is based on a fixed interest rate.
        The swap rate refers to the fixed interest rate in this contract.

        Mathematically, the swap rate :math:`S` is determined such that the present value of fixed leg payments equals the
        present value of the floating leg payments. If we assume annual fixed payments, the swap rate can be computed using
        the formula (more details in [Brigo and Mercurio, 2006], Definition 1.5.3, and [Andersen and Piterbarg, 2010], p.231):

        .. math::
            S_T = \\frac{1 - P(T)}{\\sum_{i=1}^{n} P(t_i)}

        where:

        - :math:`S_T` is the swap rate,
        - :math:`T` is the swap's maturity (:math:`t_n=T`),
        - :math:`P(t_i)=P(0,t_i)` is the discount factor at time :math:`t_i` (the present value of a single payment of 1 unit of currency received at time :math:`t_i`),
        - The denominator represents the present value of a series of 1 unit payments (an annuity) made at times :math:`1` through :math:`T`. That is, we are assuming that
          :math:`\\tau_i=1` for every index.

        Thus, if there's only one swap rate provided, the discount factor :math:`D` is calculated using the simple formula:

        .. math::
            P = \\frac{1}{{1 + S}}

        where :math:`S` is the swap rate.

        For a series of swap rates, the discount factor :math:`P(t_i)` at time :math:`t_i` is computed using the formula:

        .. math::
            P(t_i) = \\frac{1 - S(t_i) \\sum_{j=1}^{i-1} P(t_j)}{1 + S(t_i)}

        where, as above, :math:`S(t_i)` is the swap rate at time :math:`t_i`. The computation is done recursively, where  :math:`P(t_i)` is dependent
        on  :math:`P(t_{j<i})`. This process is repeated until all discount factors for the swap rates are calculated.

        Parameters
        ----------
        swap_rates : DataFrame
            A DataFrame with dates as index and maturities as columns. The values are the swap rates.

        Returns
        -------
        DataFrame
            A DataFrame with dates as index and maturities as columns. The values are the discount factors computed from
            swap rates.

        References
        ----------
        - Brigo, D., & Mercurio, F. (2006). Interest rate models: theory and practice. Berlin: Springer.
        - Andersen, L. B. G., and Piterbarg, V.V. (2010). Interest Rate Modeling.


        """
        swap_rates = swap_rates.sort_index(axis=1)
        if len(swap_rates.columns) == 1:
            # We assume fixed_freq = annual, floating_freq doesn't matter, bc floating leg is always 1-P(T)
            discs = 1 / (1 + swap_rates)
            return discs
        else:
            tenors = swap_rates.columns
            discs = self.bootstrapping(swap_rates[tenors[:-1]])
            n = swap_rates.columns[-1]
            # Again, annual fixed payments, and using floating leg = 1-P(T)
            discs[n] = (1 - swap_rates[tenors[-1]] * np.sum(discs, axis=1)) / (
                1 + swap_rates[tenors[-1]]
            )
            return discs

    def fit(self, data, delta_ind=None, method="bond_spline"):
        """
        Fits the zero coupon curve to the provided data by separating tenors less than 1 year and greater than or equal to 1 year.

        This method separates the data into `shorts` and `yearlies` based on the tenor. The `shorts` represent the short term rates where tenor is less than 1 year,
        whereas `yearlies` represent the longer term rates with tenors greater than or equal to 1 year. See, for instance, below formula (6.3) in [Andersen and Piterbarg,
        2010] and Definition 1.3.1 and (1.9) of [Brigo and Mercurio, 2006].

        For short-term rates, the bond prices are calculated as:

        .. math::
            P = \\frac{1}{{1 + y\\cdot T}}

        where:

        - :math:`P` is the bond price,
        - :math:`y` is the yield for each tenor,
        - :math:`T` is the respective tenor.

        For long-term rates, the method performs bootstrapping using the ``bootstrapping`` method to calculate the discount factors. For more information on the bootstrapping
        method, refer to its docstring.

        After the calculations, this method calls the `fit` method from the `YieldCurve` superclass to fit a cubic spline to the bond prices.

        Parameters
        ----------
        data : DataFrame
            A DataFrame with dates as index and maturities as columns. The values are the yields.

        delta_ind : int, optional
            Index of the tenor to use for perturbation. By default, perturbation is applied to all tenors.

        method : str, optional
            The interpolation method to use. Can be "bond_spline", "yield_spline", or "flat_forwards". Default is "bond_spline", which perform the spline directly on bonds.
            "yield_spline" follows Section 6.2.3 of [Andersen and Piterbarg, 2010] , being the yields the interpolated object. "flat_forwards" follows Section 6.2.1.2 of
            [Andersen and Piterbarg, 2010].


        Returns
        -------
        None

        Notes
        -----
        For computing the discount factors we are using the swap rate 1-year (e.g., USSA1 Currency) instead of the short rate 1-year (e.g., USD LIBOR 1-Year).
        It could make sense to use the short rate for this tenor if these contracts are more liquid.

        References
        ----------
        - Brigo, D., & Mercurio, F. (2006). Interest rate models: theory and practice. Berlin: Springer.
        - Andersen, L. B. G., and Piterbarg, V.V. (2010). Interest Rate Modeling.

        Warnings
        --------
        - The input `data` of this method differs from that of the corresponding method in the parent class.
        """
        self.interpolation_data = data
        shorts = data[
            [column for column in data.columns if column < 1]
        ]  # Tenor smaller than one year. Usually, month, quarter and 1/2 year
        shorts = 1 / (1 + shorts * shorts.columns)
        shorts[float(0)] = float(1)
        shorts = shorts.sort_index(axis=1)
        yearlies = data[[column for column in data.columns if column >= 1]]
        yearlies = yearlies

        yearlies = self.bootstrapping(yearlies)
        zc_bonds = pd.merge(shorts, yearlies, left_index=True, right_index=True).astype(
            "float64"
        )
        # self.zc_yields = 1 - zc_bonds ** (1 / zc_bonds.columns)  # This in an approximation, see my notes
        self.zc_yields = -np.log(zc_bonds) / zc_bonds.columns
        self.zc_yields[0] = 0  # The degenerate case of tenor = 0
        YieldCurve.fit(self, zc_bonds, delta_ind, method)


class SWICurve(YieldCurve):
    """
    SWICurve is a subclass of YieldCurve for managing seasonal adjustment of inflation swaps.

    The class handles the seasonality adjustment of inflation swaps and fits cubic discount curves
    to this adjusted data.

    # TODO : **there is a conceptual error in this class.**
    This was originally intended as if they were bonds :math:`P`, but they are actually FORWARD bonds, :math:`\\text{FP}`.
    This must be corrected so the `get_value` gives the inflation bond. This is a misconception I have corrected elsewhere.

    Attributes
    ----------
    cpi : DataFrame
        A DataFrame with dates as index and consumer price index (CPI) as columns.

    seasonal_factors : DataFrame
        A DataFrame with dates as index and months as columns. The values are the seasonal adjustment factors computed
        from a regression using historical CPI data.

    delay_months : int
        Month delay for fit method.


    """

    def __init__(self, calendar, cpi):
        YieldCurve.__init__(
            self,
            calendar=calendar,
            perturbations=(-0.0005, 0.0005),
            is_ir_curve=False,
            is_credit_curve=False,
        )
        self.cpi = cpi
        self.seasonal_factors = pd.DataFrame(columns=np.arange(1, 13))
        self.delay_months = None

    def fit_seasonality_adjustment(self, regression_dates, no_regression_years=20):
        """
        Fits a seasonality adjustment to the CPI data using a regression method.

        The adjustment is performed on a rolling basis using a specified number of years of data. The method employs
        ordinary least squares regression and applies a log return transformation to the CPI data.

        The CPI log return is calculated as follows:

        .. math::
            r_t = \\log\\left(\\frac{P_t}{P_{t-1}}\\right) = \\log\\left(1+\\pi_t\\right)

        where:

        - :math:`r_t` is the log return at time :math:`t`,
        - :math:`P_t` is the CPI at time :math:`t`,
        - :math:`P_{t-1}` is the CPI at time :math:`t-1`,
        - :math:`\\pi_t` is the inflation rate at time :math:`t`.

        The seasonality adjustment factor for a given month is calculated by taking the exponential of the regression
        coefficients corresponding to that month minus the mean of the regression coefficients. See also Fleckenstein et al., the appendix. That is, if

        .. math::
            r_t = \\sum_{i=1}^{12} \\beta_i\\cdot d_i(t) + \\varepsilon_t ,

        with :math:`d_i` a dummy equal to one if the month of the date :math:`t` is the `i`-th month. Then,

        .. math::
            a_i = e^{-\\beta_i + \\bar{\\beta}} ~~\\text{ such that }~~ \\left( 1+\\tilde{\\pi}_t\\right)=\\left( 1+{\\pi}_t\\right)a_{i_t}

        where:

        - :math:`a_i` is the seasonality adjustment factor for month `i`,
        - :math:`\\beta_i` is the regression coefficient for month `i`,
        - :math:`\\bar{\\beta}` is the mean of the regression coefficients,
        - :math:`\\tilde{\\pi}_t` is the deseasonalized inflation rate.

        By construction,

        .. math::
            \\prod_{i=1}^{12} a_i = 1 .

        Finally, ``self.seasonal_factors`` is filled as a data frame with index ``regression_dates`` and values :math:`a_i`.

        Parameters
        ----------
        regression_dates : array-like
            Dates at which the seasonality adjustment is computed.
        no_regression_years : int, optional
            Number of years of historical data to use for the regression. Default is 20.

        Returns
        -------
        None

        References
        ----------
        - Fleckenstein, M., Longstaff, F. A., & Lustig, H. (2014). The TIPS‐treasury bond puzzle. `Journal of Finance`, 69(5), 2151-2197.

        """

        regression_dates = afsfun.dates_formatting(regression_dates)

        # [ARS] New code using pandas vectorization
        # Retrieve the CPI (Consumer Price Index) values, convert them to a DataFrame, sort by index (which are the dates), and rename the first column to "Price"
        df = (
            self.cpi.get_value(self.cpi.get_dates())
            .to_frame()  # Convert to DataFrame
            .sort_index()  # Sort by date
            .resample("M")
            .last()  # va a la última fecha del mes aunque no aparezca. Sin embargo, devuelve un NaN si no hay dato para ese mes.
        )
        assert not df.isna().any().any(), "NaN values found in the DataFrame of CPI."
        df = df.rename(columns={df.columns[0]: "Price"})  # Rename the column to "Price"
        # Calculate the log returns: the log of the ratio of the current price to the previous price. Use the .shift() function to get the previous price
        df["Return"] = np.log(df["Price"] / df["Price"].shift())
        # Create dummy variables for each month:
        # Use .assign() to add new columns to the DataFrame, the new columns represent each month (1 to 12)
        # df.index.month == i creates a boolean Series where True (1) indicates the current month equals 'i', and False (0) otherwise
        df = df.assign(
            **{f"month_{i}": (df.index.month == i).astype(int) for i in range(1, 13)}
        )
        # Convert column names back to integers, assign only works with strings
        df.columns = [
            int(col.replace("month_", "")) if "month_" in col else col
            for col in df.columns
        ]
        self.seasonal_factors = self.seasonal_factors.reindex(
            self.seasonal_factors.index.union(regression_dates)
        )
        for regression_date in regression_dates:
            X = (
                df[:regression_date]
                .iloc[-no_regression_years * 12 :, 2:]
                .values.astype("int")
            )
            y = df[:regression_date].iloc[-no_regression_years * 12 :, 1].values
            betas = sm.OLS(y, X).fit().params
            self.seasonal_factors.loc[regression_date] = np.exp(-betas + betas.mean())

    def adjust_cpi(self, month_cpi, no_regression_years=10, last=False):
        """
        Adjust the returns based on the month CPI. The method first fits a seasonality adjustment to the CPI based on the month,
        and then adjusts the returns based on these seasonal factors.

        Parameters
        ----------
        month_cpi : pandas.Series
            The consumer price index (CPI) data for each month. It must have one, and only one, date per month.
        no_regression_years : int, optional
            The number of years to consider in the regression. By default, 10.
        last : boolean, optional
            Takes the last date for the seasonal adjustment. By default, it is `False`.

        Returns
        -------
        pandas.Series
            The adjusted CPI data.

        """
        prev_cpi = month_cpi.shift().iloc[1:]
        month_cpi_short = month_cpi.iloc[1:]
        returns = month_cpi_short / prev_cpi
        self.fit_seasonality_adjustment(
            month_cpi_short.index, no_regression_years=no_regression_years
        )
        if last:
            adjusted_returns = returns.mul(
                returns.index.to_series().apply(
                    lambda x: self.seasonal_factors[x.month].loc[returns.index[-1]]
                )
            ).squeeze()  # It works for series
        else:
            adjusted_returns = returns.mul(
                returns.index.to_series().apply(
                    lambda x: self.seasonal_factors[x.month].loc[x]
                )
            ).squeeze()  # It works for series
        cpi_adj = pd.Series(index=month_cpi.index)
        cpi_adj.iloc[0] = month_cpi.iloc[0]
        cpi_adj.iloc[1:] = (
            adjusted_returns.cumprod() * month_cpi.iloc[0]
        )  # Initial value is the month value so both coincide at the beginning.

        return cpi_adj

    def fit_old(self, swi_rates, delay_months=3, no_regression_years=10):
        """
        # TODO : **Delete. The deseasonalized procedure in the original code was sloppy and incorrect. It is better to rewrite this method from scratch.**
        Fit the model to the provided inflation swap rates (SWI).
        """
        #
        # This method performs the fitting procedure for the SWICurve model. The process includes the calculation of bond prices
        # using the inflation swap rates and CPI (Consumer Price Index), adjusting the data for seasonality, and calculating returns.
        # The fitting is performed by the YieldCurve (Cubic Discount Curve) class.
        #
        # Forward CPI are calculated with the formula:
        #
        # .. math::
        #     \\mathcal{I}(t, t+\\tau) = I(t) \\cdot (1 + r(t,t+\\tau))^{\\tau}
        #
        # where:
        #
        # - :math:`\\mathcal{I}` is the forward CPI, Remark on p.10 of Waldenberger,
        # - :math:`I(t)` is the Consumer Price Index value at :math:`t`,
        # - :math:`r` is the inflation swap rate (a DataFrame indexed by :math:`t` and columns :math:`\\tau`), see Defintion 1.4 of Waldenberger,
        # - :math:`\\tau` is the tenor of the rate.
        #
        # Returns are calculated with the formula:
        #
        # .. math::
        #     R = [CPI_{current} , \\frac{AdjFactor_{t+1} \\cdot V_{t+1}}{V_{t}}]
        #
        # where:
        #
        # - :math:`r_t` is the return,
        # - :math:`CPI_{current}` is the current Consumer Price Index value,
        # - :math:`AdjFactor_{t+1}` is the adjustment factor at time :math:`t+1`,
        # - :math:`V_{t+1}` and :math:`V_{t}` are the values at time :math:`t+1` and :math:`t`, respectively.
        #
        # Parameters
        # ----------
        # swi_rates : DataFrame
        #     A DataFrame containing the inflation swap rates. The DataFrame should have dates as index and maturities as columns.
        # delay_months : int, optional
        #     The number of months of delay to consider in the fitting process. Defaults to 3.
        # no_regression_years : int, optional
        #     The number of years to consider for the regression process in seasonality adjustment. Defaults to 10.
        #
        # Returns
        # -------
        # None.
        #
        # References
        # ----------
        #
        # - Waldenberger, Stefan. Inflation market models. `Technische Universität Graz`. (2011).

        ...
        self.delay_months = delay_months
        self.interpolation_data = swi_rates
        date_range = pd.date_range(
            swi_rates.index.min() - (self.delay_months + 1) * pd.offsets.MonthEnd(),
            swi_rates.index.max(),
            freq="1D",
        )
        # Not the same as pd.DateOffset, pd.offsets.MonthEnd() generates an offset that ensures the shifted date falls on the last day of the month.
        cpis_df = self.cpi.get_value(
            date_range
        ).ffill()  # In case of NaN ffill propagates last valid observation forward to next valid.
        # cpis_df = self.cpi.get_value(date_range)  # In case of NaN ffill propagates last valid observation forward to next valid.
        base_cpis = cpis_df.loc[
            swi_rates.index - self.delay_months * pd.offsets.MonthEnd()
        ]
        base_cpis = base_cpis.values
        base_cpis = base_cpis.reshape(base_cpis.size, 1)
        bond_prices = (
            base_cpis * (1 + swi_rates) ** swi_rates.columns
        )  # Forward CPI (not a bond), by definition of ISS Rate, see below (1.4) of [Waldenberger, 2011].
        bond_prices.columns -= (
            self.delay_months / 12
        )  # Adjust bond prices columns to reflect delay in months
        bond_prices[0] = cpis_df.loc[
            swi_rates.index
        ].values  # Set bond price at delay month to the CPI value at the swap rate date
        bond_prices = bond_prices.sort_index(axis=1)  # Sort bond prices by maturity
        YieldCurve.fit(self, bond_prices)

        # Seasonal adjustment
        max_maturity = np.max(swi_rates.columns)
        adjusted_fitting_data = pd.DataFrame(
            index=self.fitting_dates, columns=np.arange(12 * max_maturity + 1) / 12
        )  # Tenor for every month
        for fitting_date in self.fitting_dates:
            if (
                fitting_date not in self.seasonal_factors.index
            ):  # Check if seasonality adjustment has already been performed for the fitting date
                self.fit_seasonality_adjustment(
                    regression_dates=fitting_date,
                    no_regression_years=no_regression_years,
                )
            monthlies = pd.date_range(
                start=fitting_date,
                freq=pd.DateOffset(months=1),
                periods=12 * max_maturity + 1,
            )  # Generate monthly dates from fitting_date up to maximum
            # maturity
            adjustment_factors = np.array(
                [
                    self.seasonal_factors.loc[fitting_date, date.month]
                    for date in monthlies[1:]
                ]
            )  # Calculate adjustment factors for each month
            adjustment_factors[: 12 - self.delay_months] = (
                1  # Set adjustment factor to 1 for delay_months
            )
            values = YieldCurve.get_value(
                self, dates=fitting_date, tenors=np.arange(12 * max_maturity + 1) / 12
            )
            curr_cpi = cpis_df.loc[fitting_date]
            adjusted_returns = np.concatenate(
                ([curr_cpi], adjustment_factors * values[1:] / values[:-1])
            )
            adjusted_fitting_data.loc[fitting_date] = np.multiply.accumulate(
                adjusted_returns
            )
        YieldCurve.fit(self, adjusted_fitting_data)

    def fit(self, swi_rates, no_regression_years=10, deseasonalized=True):
        """
        Fit the model to the provided inflation swap rates (SWI).

        This method performs the fitting procedure for the SWICurve model. The process includes the calculation of bond prices
        using the inflation swap rates and CPI (Consumer Price Index), adjusting the data for seasonality, and calculating returns.
        The fitting is performed by the YieldCurve (Cubic Discount Curve) class.

        Forward CPI are calculated with the formula:

        .. math::
            \\mathcal{I}(t, t+\\tau) = I(t) \\cdot (1 + r(t,t+\\tau))^{\\tau}

        where:

        - :math:`\\mathcal{I}` is the forward CPI, Remark on p.10 of Waldenberger,
        - :math:`I(t)` is the Consumer Price Index value at :math:`t`,
        - :math:`r` is the inflation swap rate (a DataFrame indexed by :math:`t` and columns :math:`\\tau`), see Defintion 1.4 of Waldenberger,
        - :math:`\\tau` is the tenor of the rate.

        If deseasonalized is True, then :math:`I(t)` is substituted by :math:`\\tilde{I}(t)` as in ``fit_seasonality_adjustment``.

        Parameters
        ----------
        swi_rates : DataFrame
            A DataFrame containing the inflation swap rates. The DataFrame should have dates as index and maturities as columns.
        no_regression_years : int, optional
            The number of years to consider for the regression process in seasonality adjustment. Defaults to 10.
        deseasonalized : bool, optional
            If deseasonalized is True, then :math:`I(t)` is substituted by :math:`\\tilde{I}(t)` as in ``fit_seasonality_adjustment``.

        Returns
        -------
        None.

        References
        ----------

        - Waldenberger, Stefan. Inflation market models. `Technische Universität Graz`. (2011).

        """
        self.interpolation_data = swi_rates
        date_range = pd.date_range(
            swi_rates.index.min() - pd.offsets.MonthEnd(),
            swi_rates.index.max(),
            freq="1D",
        )
        # Not the same as pd.DateOffset, pd.offsets.MonthEnd() generates an offset that ensures the shifted date falls on the last day of the month.
        cpis_df = self.cpi.get_value(
            date_range
        ).ffill()  # In case of NaN ffill propagates last valid observation forward to next valid.
        base_cpis = cpis_df.loc[swi_rates.index]
        if deseasonalized:
            month_cpi = cpis_df[cpis_df.index >= swi_rates.index.min()]
            month_cpi = (
                month_cpi[month_cpi.index <= swi_rates.index.max()].resample("M").last()
            )  # va a la última fecha del mes aunque no aparezca. Sin embargo,
            # devuelve un NaN si no hay dato para ese mes.
            assert (
                not month_cpi.isna().any().any()
            ), "NaN values found in the DataFrame of CPI."
            base_cpis_adj = self.adjust_cpi(month_cpi)
            base_cpis_adj = base_cpis_adj.to_frame()
            base_cpis_adj = base_cpis_adj.reindex(
                swi_rates.index.union(base_cpis_adj.index)
            ).ffill()  # Fill to have the same value within a month, it forwards the last value
            base_cpis_adj = base_cpis_adj.reindex(
                swi_rates.index
            )  # Only interested in values dt swi_rates.index
            cpi = base_cpis_adj[0].values
            # Create figure
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=base_cpis_adj.index, y=cpi, mode="lines", name="base_cpis_adj"
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=base_cpis.index,
                    y=base_cpis.values,
                    mode="lines",
                    name="base_cpis",
                )
            )
            fig.update_layout(
                title="Comparison of base_cpis_adj and base_cpis",
                xaxis_title="X-axis label",
                yaxis_title="Y-axis label",
            )
            fig.show()

        else:
            cpi = base_cpis.values

        cpi = cpi.reshape(cpi.size, 1)
        bond_prices = (
            cpi * (1 + swi_rates) ** swi_rates.columns
        )  # Forward CPI (not a bond), by definition of ISS Rate, see below (1.4) of [Waldenberger, 2011].
        bond_prices[0] = cpis_df.loc[
            swi_rates.index
        ].values  # Set bond price at delay month to the CPI value at the swap rate date
        bond_prices = bond_prices.sort_index(axis=1)  # Sort bond prices by maturity
        YieldCurve.fit(self, bond_prices)

    def get_value(self, dates):
        """TODO : **as a byproduct of the conceptual error, the signature does not match the base method. This gives the CPI, not the inflation bond**"""
        return self.cpi.get_value(dates)

    def get_forward_value(self, dates, future_dates=None, tenors=None):
        return YieldCurve.get_value(
            self, dates=dates, future_dates=future_dates, tenors=tenors
        )


# -------------------------------------------------------------------------------------------------------
# Credit curves
# -------------------------------------------------------------------------------------------------------


class CDSCurve(LCExpCurve):
    """
    Class for constructing a full term structure of survival probabilities (credit curve) from a finite number of CDS market quotes (spreads).

    Attributes
    ----------
    name : String
        Name of the curve.
    seniority : boolean
        If True, senior bonds are considered.  If not, subordinated ones with recovery rate 0.2.
    recovery :  float
        Deterministic recovery rate. For senior bonds 0.4 is assumed and 0.2 otherwise. This is, for instance, the values assumed in Bloomberg platform.
    calendar : Calendar
        The calendar system (Day Count Convention) to use when computing intervals.
        Calendar should be understood as an element of the classes defined in calendar.py. For CDSs, typically Act360 [O'Kane].
    discount_curve : pricing.discount_curves.DiscountCurve
            Curve needed for the interest free rate :math:`r_s` in [Privault, 2022].
    market_spreads : DataFrame
        CDS spreads (annual amount that the protection buyer must pay the protection seller) for different dates (index) and tenors in years (columns).
        The data is assumed to be per unit (100 basis points = 0.01 per unit).
    payment_dates : pandas.DatetimeIndex
        Premium payment dates of the CDS contract. By default, IMM dates (20 March, 20 June, 20 September and 20 December of each year) are used. Note that the effective date
        is not included here.


    Notes
    ------
    - We are assuming J.P. Morgan model described in [Castellacci, 2008].
    - For market_spreads, Mid spread data from Bloomberg is usually used.
    - Note that the official IMM date is the 3rd Wednesday of the month but the CDS market always chooses the 20th date of these months.
    - For computing the default leg we are approximating the integral with one month steps.

    References
    ----------
        - Privault, N. (2022). Notes on Financial Risk and Analytics (version June 9, 2023).
        - O’Kane, D. (2008). Modelling Single-name and Multi-name Credit Derivatives (first edition).
        - Castellacci, G. (2008). Bootstrapping Credit Curves from CDS Spread Curves http://dx.doi.org/10.2139/ssrn.2042177
    """

    def __init__(self, name, seniority, calendar, discount_curve, payment_dates=None):
        self.name = name
        self.seniority = seniority
        self.recovery = 0.4 if self.seniority is True else 0.2
        self.discount_curve = discount_curve
        self.market_spreads = None
        self.fitting_dates = (
            None  # Note that we always have fitting_dates = market_spreads.index.
        )
        self.hazard_rates = None
        self.payment_dates = payment_dates
        LCExpCurve.__init__(
            self, calendar=calendar, is_ir_curve=False, is_credit_curve=True
        )

    def _bootstrap_cds_all(
        self, cds_values, tenors, discounts, recovery
    ):  # TODO : This method should be (eventually) removed
        """
        IT DOES NOT WORK PROPERLY .
        It is also not used (at least as private method).
        """
        tenors = np.array(tenors)

        def error(hazards):
            all_hazards = np.full(tenors[-1] + 1, np.nan)
            all_hazards[0] = 0
            all_hazards[tenors] = hazards
            idx = np.where(~np.isnan(all_hazards), np.arange(all_hazards.size), np.inf)
            out = np.flip(np.minimum.accumulate(np.flip(idx))).astype("int")
            all_hazards = all_hazards[out]
            all_q = np.exp(np.add.accumulate(-all_hazards))
            integral = np.full(tenors.size, np.nan)
            rpvbp = np.full(tenors.size, np.nan)
            for i in range(tenors.size):
                temp_z = discounts[: tenors[i] + 1]
                temp_q = all_q[: tenors[i] + 1]
                integral = np.sum(temp_z[1:] * (temp_q[:-1] - temp_q[1:]))
                rpvbp[i] = (1 / 2) * np.sum(temp_z[1:] * (temp_q[:-1] + temp_q[1:]))
            errors = cds_values * rpvbp - (1 - recovery) * integral
            return errors

        hazards = optimize.root(error, 0.001 * np.ones(cds_values.size))
        return hazards

    @staticmethod
    def compute_hazard_rates(tenors, survival):
        """
        Compute hazard rates from survival probabilities. As usual, we assume piecewise constant risk neutral hazard rates.

        Parameters
        ----------
        tenors : numpy.ndarray or list
            Maturities of the CDS contracts.
        survival : numpy.ndarray or list
            Survival probabilities at the given maturities.

        Returns
        -------
        numpy.ndarray
            Hazard rates at the given maturities.
        Notes
        -----
            The hazard rate at t=0 is set as that of to the one corresponding to the first tenor.
        Example
        -------
        >>> tenors = [0.25, 0.5, 1, 2]
        >>> survival = [0.9925, 0.9851, 0.9705,0.9418]
        >>> CDSCurve.compute_hazard_rates(tenors, survival)
        array([0.03011307, 0.03011307, 0.02993541, 0.02986351, 0.03001846])
        """
        tenors = np.array(tenors)
        tenors = np.insert(tenors, 0, 0)  # We include the effective date
        survival = np.array(survival)
        survival = np.insert(
            survival, 0, 1
        )  # We include the survival at the effective date.
        h = np.zeros(len(survival))
        for t in range(1, len(h)):
            h[t] = -np.log(survival[t] / survival[t - 1]) / (tenors[t] - tenors[t - 1])
        h[0] = h[1]  # We set the initial hazard rate as the second one.
        return h

    @staticmethod
    def get_imm_dates(start_date: str, end_date: str) -> np.ndarray:
        """
        Returns an array containing IMM dates between two dates. IMM dates are 20 March, 20 June, 20 September, or 20 December.

        If these dates fall on a weekend, the function selects the next business day.

        Parameters
        ----------
        start_date : str
            The start date in 'YYYY-MM-DD' format.
        end_date : str
            The end date in 'YYYY-MM-DD' format.

        Returns
        -------
        numpy.ndarray
            Numpy array containing the selected IMM dates.

        Notes
        -----
            The maturity date is calculated as the first ‘IMM’ date T years after the effective date. See examples below.

        Examples
        --------
        >>> start_date = pandas.Timestamp('2022-10-24')
        >>> end_date = pandas.Timestamp('2022-12-25')
        >>> print(CDSCurve.get_imm_dates(start_date, end_date))
        [Timestamp('2022-12-20 00:00:00') Timestamp('2023-03-20 00:00:00')]

        >>> start_date = pandas.Timestamp('2022-03-21')
        >>> end_date = pandas.Timestamp('2022-03-22')
        >>> print(CDSCurve.get_imm_dates(start_date, end_date))
        [Timestamp('2022-03-21 00:00:00') Timestamp('2022-06-20 00:00:00')]

        >>> start_date = pandas.Timestamp('2020-03-20')
        >>> end_date = pandas.Timestamp('2020-06-21')
        >>> print(CDSCurve.get_imm_dates(start_date, end_date))
        [Timestamp('2020-03-20 00:00:00') Timestamp('2020-06-22 00:00:00')]

        >>> start_date = pandas.Timestamp('2019-12-20')
        >>> end_date = pandas.Timestamp('2020-06-21')
        >>> print(CDSCurve.get_imm_dates(start_date, end_date))
        [Timestamp('2019-12-20 00:00:00') Timestamp('2020-03-20 00:00:00')
         Timestamp('2020-06-22 00:00:00')]
        """
        start_date = pd.Timestamp(start_date)
        end_date = pd.Timestamp(end_date)
        imm_months = np.array([3, 6, 9, 12])
        if not np.isin(np.array([end_date.month]), imm_months):
            imm_month_maturity = imm_months[np.array([end_date.month]) <= imm_months][0]
            end_date_enlarged = end_date.replace(month=imm_month_maturity, day=25)
        else:
            if end_date.day < 20:
                end_date_enlarged = end_date.replace(day=25)
            else:
                if end_date.day > 22:
                    end_date_enlarged = pd.to_datetime(end_date) + pd.DateOffset(
                        months=3
                    )
                else:
                    twentieth_of_month = end_date.replace(day=20)
                    imm_date_of_month = (twentieth_of_month - BDay(1)) + BDay(
                        1
                    )  # If twentieth_of_month is a business day imm_date = twentieth_of_month
                    if end_date <= imm_date_of_month:
                        end_date_enlarged = imm_date_of_month
                    else:
                        end_date_enlarged = pd.to_datetime(end_date) + pd.DateOffset(
                            months=4
                        )

        dates = pd.date_range(start=start_date, end=end_date_enlarged)
        imm_dates = dates[(dates.day == 20) & (dates.month.isin([3, 6, 9, 12]))]
        # We adjust the dates so if they fall on a weekend, they are changed to the next business day
        imm_dates = imm_dates.to_series().map(
            lambda d: d if d.weekday() < 5 else d + BDay(1)
        )
        imm_dates = np.array(imm_dates, dtype=object)
        if np.isin(end_date, imm_dates):
            imm_dates = imm_dates[imm_dates <= end_date]
        else:
            last_date = imm_dates[end_date < imm_dates][0]
            imm_dates = imm_dates[
                imm_dates <= last_date
            ]  # We remove a possible additional date.
        if (
            start_date.day == 21 or start_date.day == 22
        ):  # In case start_date is a IMM date but start_date.day not 20.
            twentieth_of_month = start_date.replace(day=20)
            imm_day = ((twentieth_of_month - BDay(1)) + BDay(1)).day
            if imm_day < start_date.day:
                pass
            else:
                imm_dates = np.sort(np.insert(start_date, 0, imm_dates))
        return imm_dates

    def fit_single_date_artur(
        self, date, discount_curve, cds_values, tenors=np.arange(10) + 1
    ):  # TODO : This method should be (eventually) removed
        """
        Return the failure (hazard) rate for a single date and several tenors.

        Parameters
        ----------
        date : pandas.Timestamp
            First date of the tenor structure :math:`t = T_0` in [Privault, 2022]. Effective date in Chapter 5 of [O`Kane].
        discount_curve : pricing.discount_curves.DiscountCurve
            Curve needed for the interest free rate :math:`r_s` in [Privault, 2022].
        cds_values : Series
            CDS spread values for a single date.
        tenors : numpy.ndarray, default = [ 1  2  3  4  5  6  7  8  9 10]
            Array with the tenors in years.

        Returns
        -------
        numpy.ndarray
            Failure (hazard) rate.

        Notes
        ------
            The failure (hazard) rate is indeed the forward default rate as defined in Eq. (7.2) of [O`Kane].
        """
        # remember to keep the calibration and CDS pricing consistent [AFA comment].
        calendar = self.calendar
        cds_values = np.asarray(cds_values)
        if np.sum(cds_values > 0.5) >= 1:
            print(
                "Spread values are high: divide by 10000?"
            )  # This seems to be a warning when the CDS spreads units are not correct (basis points instead of per unit).
        #     Now the appropriate units are indicated in the docstring.
        date = pd.to_datetime(date)
        tenors = np.asarray(tenors)
        maturities = np.array(
            [date] + [date + pd.DateOffset(months=tenor * 12) for tenor in tenors]
        )
        all_dates = np.array(
            [date + pd.DateOffset(months=i * 12) for i in range(int(tenors[-1]) + 1)]
        )
        Z = discount_curve.get_value(
            dates=date, future_dates=all_dates, calendar=calendar
        )
        R = self.recovery

        def bootstrap_cds(cds_values_temp, tenors_temp):
            """
            Return (forward) hazard rates for each date.

            Parameters
            ----------
            cds_values_temp : Series
                CDS spread values for a single date.
            tenors_temp : numpy.ndarray
                Array with the tenors in years.

            Returns
            -------
            numpy.ndarray
                Hazard rates
            """
            tenors_temp = np.array(tenors_temp)
            if cds_values_temp.size == 1:  # Single CDS spread.
                if np.asarray(cds_values_temp).shape == ():
                    cds_values_temp = np.array([cds_values_temp])
                # end_date = date + pd.DateOffset(months=tenors_temp[0])
                integral_steps = pd.date_range(
                    start=date, end=maturities[1], freq=pd.DateOffset(days=1)
                )  # Is this high frequency needed? According to O'Kane it is NOT
                # It suffices with monthly time steps (see end of section 6.6.0)
                # date + pd.timedelta_range(start="0", periods=no_steps + 1, freq="{}D".format(f))
                integral_steps = integral_steps[1:]
                epsilon = (
                    calendar.interval(maturities[0], maturities[1])
                    / integral_steps.size
                )
                k = np.arange(integral_steps.size) + 1
                disc_steps = discount_curve.get_value(
                    dates=date, tenors=integral_steps, calendar=calendar
                )

                def f(hazard_temp):
                    """
                    CDS price (from the point of view of the seller of protection) as a function of the (forward) hazard rate (see) Eq. (7.2) in [O`Kane].

                    Parameters
                    ----------
                    hazard_temp : float
                        Forward hazard rate.

                    Returns
                    -------
                    float
                        Premium PV - Protection PV

                    Notes
                    ------
                    The proper definitions of these two legs are described in Chapter 6 of [O`Kane].

                    """
                    integral = (np.exp(hazard_temp * epsilon) - 1) * np.sum(
                        disc_steps * np.exp(-hazard_temp * k * epsilon)
                    )  # (First) Protection PV in page 106 of [O`Kane].
                    # Use the implementation described before section 6.6.1. (significantly improve the accuracy for a very small additional cost).
                    q = np.exp(
                        -hazard_temp * np.arange(tenors_temp[0] + 1)
                    )  # Vector of Q(0,t_n), being t_n the dates for which we have market quotes.
                    Z_local = Z[all_dates <= maturities[1]]
                    rpvbp = (1 / 2) * np.sum(
                        Z_local[1:] * (q[:-1] + q[1:])
                    )  # Risky present value of a basis point for a CDS. See equation after Eq. (6.5) of [O`Kane].
                    # Is this right? The sum should over the payment dates. If the maturity is one year we usually have one payment each 3 months (IMM dates).
                    # Note that we do assume that the hazard rate is piecewise constant between the maturities of the liquid CDS contracts.
                    value = cds_values_temp * rpvbp - (1 - R) * integral
                    return value

                hazard = optimize.newton(f, 0.05)
                return hazard
            else:
                n = cds_values_temp.size
                hazards = np.zeros(n)
                hazards[:-1] = bootstrap_cds(
                    cds_values_temp[:-1], tenors_temp[:-1]
                )  # Recursive implementation, until we have the first CDS spread (shortest dated instrument).
                h = np.zeros(
                    tenors_temp[-2] + 1
                )  # Does not work for non-integer values.
                h[1 : tenors_temp[0] + 1] = hazards[0]
                for i in range(tenors_temp.size - 2):
                    h[tenors_temp[i] + 1 : tenors_temp[i + 1] + 1] = hazards[i + 1]
                q = np.exp(-np.sum(np.tril(np.ones((h.size, h.size))) * h, axis=1))
                Z_local = Z[all_dates <= maturities[tenors_temp.size - 1]]
                rp_previous = np.sum(Z_local[1:] * (q[1:] + q[:-1]))

                start = date + pd.DateOffset(months=tenors_temp[-2] * 12)
                end = date + pd.DateOffset(months=tenors_temp[-1] * 12)
                integral_steps = pd.date_range(
                    start=start, end=end, freq=pd.DateOffset(days=1)
                )
                integral_steps = integral_steps[1:]
                epsilon = (
                    calendar.interval(
                        maturities[tenors_temp.size - 1], maturities[tenors_temp.size]
                    )
                    / integral_steps.size
                )
                k = np.arange(integral_steps.size) + 1
                disc_steps = discount_curve.get_value(
                    dates=date, tenors=integral_steps, calendar=calendar
                )

                missing = tenors_temp[-1] - tenors_temp[-2]
                z = Z[
                    (all_dates > maturities[tenors_temp.size - 1])
                    * (all_dates <= maturities[tenors_temp.size])
                ]

                def f(hazard_temp):
                    """
                    Parameters
                    ----------
                    hazard_temp : float
                        Hazard rate.

                    Returns
                    -------

                    """
                    exponent = -hazard_temp * (np.arange(missing) + 1)
                    a1 = (
                        2
                        * (1 - R)
                        * (1 - np.exp(hazard_temp * epsilon))
                        * np.sum(disc_steps * np.exp(-hazard_temp * k * epsilon))
                    )
                    a2 = (
                        cds_values_temp[-1]
                        * (1 + np.exp(hazard_temp))
                        * np.sum(z * np.exp(exponent))
                    )
                    b = cds_values_temp[-1] - cds_values_temp[-2]
                    value = q[-1] * (a1 + a2) + b * rp_previous
                    return value

                hazards[-1] = optimize.newton(f, 0.01)
                return hazards

        # self.hazard_rates = bootstrap_cds(cds_values, tenors)
        # self.hazard_rates = self._bootstrap_cds(cds_values=cds_values, tenors=tenors, discounts=Z, recovery=R)
        # params_df = pd.DataFrame(-self.hazard_rates.reshape((1, tenors.size)), index=[date], columns=tenors)
        return bootstrap_cds(cds_values, tenors)

    def fit_single_date(self, date, discount_curve, cds_values):
        """
        Return the failure (hazard) rates for a single date and several tenors.

        Parameters
        ----------
        date : pandas.Timestamp
            First date of the tenor structure :math:`t = T_0` in [Privault, 2022]. Effective date in Chapter 5 of [O`Kane].
        discount_curve : pricing.discount_curves.DiscountCurve
            Curve needed for the interest free rate :math:`r_s` in [Privault, 2022].
        cds_values : Series
            CDS spread values for a single date.

        Returns
        -------
        numpy.ndarray
            Failure (hazard) rates.

        Notes
        ------
            The failure (hazard) rate is indeed the forward default rate as defined in Eq. (7.2) of [O`Kane].
        """
        tenors = np.asarray(cds_values.index)
        calendar = self.calendar
        cds_values = np.asarray(cds_values)
        effective_date = pd.to_datetime(
            date
        )  # It can fall on a Saturday or on a holiday. It does not need to be a business day.
        maturities = np.array(
            [effective_date]
            + [effective_date + pd.DateOffset(months=tenor * 12) for tenor in tenors]
        )
        if self.payment_dates is None:
            payment_dates = self.get_imm_dates(effective_date, maturities[-1])
        else:
            payment_dates = self.payment_dates
        payment_dates_0 = np.sort(
            np.insert(payment_dates, 0, effective_date)
        )  # We include the effective date
        n_months = len(
            pd.date_range(start=effective_date, end=payment_dates[-1], freq="MS")
        )
        # We include the effective and payment dates in the integral dates
        integral_dates = np.array(
            [effective_date]
            + [effective_date + pd.DateOffset(months=i) for i in range(1, n_months + 1)]
        )
        integral_dates = pd.concat(
            [pd.Series(integral_dates), pd.Series(payment_dates)]
        ).sort_values()
        integral_dates = np.array(integral_dates.to_list())  # np.array of pd.TimeStamps
        integral_dates = np.unique(
            integral_dates
        )  # We remove (possible) duplicated dates
        z_payment = discount_curve.get_value(
            dates=effective_date, future_dates=payment_dates, calendar=calendar
        )
        z_integral = discount_curve.get_value(
            dates=effective_date, future_dates=integral_dates, calendar=calendar
        )
        intervals_payment = np.array(
            [
                self.calendar.interval(
                    payment_dates_0[i - 1], payment_dates_0[i]
                ).item()
                for i in range(1, len(payment_dates_0))
            ]
        )

        def bootstrap_cds(cds_values_temp, tenors_temp):
            """
            Return (forward) hazard rates for a set of CDS spreads.

            Parameters
            ----------
            cds_values_temp : Series
                CDS spread values for a single date.
            tenors_temp : numpy.ndarray
                Array with the tenors in years.

            Returns
            -------
            numpy.ndarray
                Hazard rates
            """
            tenors_temp = np.array(tenors_temp)
            if cds_values_temp.size == 1:  # First CDS spread.
                if np.asarray(cds_values_temp).shape == ():
                    cds_values_temp = np.array([cds_values_temp])
                payment_dates_temp = payment_dates[payment_dates <= maturities[1]]
                payment_dates_temp_0 = np.insert(
                    payment_dates_temp, 0, effective_date
                )  # We include the effective_date for computing the intervals
                payment_dates_temp_0 = np.append(
                    payment_dates_temp_0,
                    payment_dates[maturities[1] < payment_dates][0],
                )  # We include the next payment date as maturity.
                payment_dates_temp = np.append(
                    payment_dates_temp, payment_dates[maturities[1] < payment_dates][0]
                )  # We include the next payment date as maturity.
                integral_dates_temp = integral_dates[
                    integral_dates <= payment_dates_temp[-1]
                ]
                z_integral_temp = z_integral[: integral_dates_temp.size]
                z_payment_temp = z_payment[: payment_dates_temp.size]
                intervals_payment_temp = intervals_payment[: payment_dates_temp.size]
                intervals_q_payment = np.array(
                    [
                        self.calendar.interval(
                            effective_date, payment_dates_temp_0[i]
                        ).item()
                        for i in range(len(payment_dates_temp_0))
                    ]
                )
                intervals_q_integral = np.array(
                    [
                        self.calendar.interval(
                            effective_date, integral_dates_temp[i]
                        ).item()
                        for i in range(len(integral_dates_temp))
                    ]
                )
                # intervals_integral_temp = np.array([self.calendar.interval(integral_dates_temp[i-1], integral_dates_temp[i]).item() for i in range(1, len(integral_dates_temp))])

                def f(hazard_temp):
                    """
                    CDS price (from the point of view of the seller of protection) as a function of the (forward) hazard rate (see) Eq. (7.2) in [O`Kane].
                    This is the price for the first CDS spread.

                    Parameters
                    ----------
                    hazard_temp : float
                        Forward hazard rate.

                    Returns
                    -------
                    float
                        Premium PV - Protection PV

                    Notes
                    ------
                    The proper definitions of these two legs are described in Chapter 6 of [O`Kane].

                    """
                    q_payment_temp1 = 1 * np.exp(
                        -hazard_temp * intervals_q_payment
                    )  # Interpolation based on Section 7.4.1 of [0' Kane].
                    q_integral_temp1 = 1 * np.exp(
                        -hazard_temp * intervals_q_integral
                    )  # Interpolation based on Section 7.4.1 of [0' Kane].
                    integral = (
                        1
                        / 2
                        * np.sum(
                            (z_integral_temp[:-1] + z_integral_temp[1:])
                            * (q_integral_temp1[:-1] - q_integral_temp1[1:])
                        )
                    )  # Prot. PV in page 106 of [O`Kane].
                    rpvbp = (
                        1
                        / 2
                        * np.sum(
                            intervals_payment_temp
                            * z_payment_temp
                            * (q_payment_temp1[:-1] + q_payment_temp1[1:])
                        )
                    )
                    # Risky present value of a basis point for a CDS. See equation after Eq. (6.5) of [O`Kane].
                    value = cds_values_temp * rpvbp - (1 - self.recovery) * integral
                    return value

                hazard = afsfun.find_root(f, 0.05, [0, 1])
                q_payment_f = 1 * np.exp(
                    -hazard * intervals_q_payment
                )  # Interpolation based on Section 7.4.1 of [0' Kane].
                q_integral_f = 1 * np.exp(-hazard * intervals_q_integral)
                self.q_payment = q_payment_f  # We use these values when n > 1.
                self.q_integral = q_integral_f  # We use these values when n > 1.
                return hazard
            else:
                n = cds_values_temp.size
                hazards = np.zeros(n)
                hazards[:-1] = bootstrap_cds(
                    cds_values_temp[:-1], tenors_temp[:-1]
                )  # Recursive implementation, until we have the first CDS spread (shortest dated instrument).
                # Previous dates
                payment_dates_prev = payment_dates[: (self.q_payment.size - 1)]
                integral_dates_prev = integral_dates[: self.q_integral.size]
                q_payment_prev = self.q_payment
                q_integral_prev = self.q_integral
                # New dates np.append(payment_dates_temp, payment_dates[maturities[1] < payment_dates][0])
                integral_dates_new = integral_dates[
                    integral_dates_prev.size :
                ]  # First we remove previous dates, except the last one (new effective date).
                payment_dates_new = payment_dates[payment_dates_prev.size :]
                try:
                    integral_dates_new = integral_dates_new[
                        integral_dates_new
                        <= payment_dates[maturities[n] < payment_dates][0]
                    ]
                    payment_dates_new = payment_dates_new[
                        payment_dates_new
                        <= payment_dates[maturities[n] < payment_dates][0]
                    ]
                except (
                    IndexError
                ):  # In this case payment_dates[maturities[n] < payment_dates] is empty (last spread)
                    pass
                integral_dates_temp = integral_dates[
                    integral_dates <= integral_dates_new[-1]
                ]
                payment_dates_temp = payment_dates[
                    payment_dates <= payment_dates_new[-1]
                ]
                z_integral_temp = z_integral[: integral_dates_temp.size]
                z_payment_temp = z_payment[: payment_dates_temp.size]
                # Intervals from the new effective date (integral_dates_new[0]) for the payments and integral
                intervals_payment_new = np.array(
                    [
                        self.calendar.interval(
                            payment_dates_prev[-1], payment_dates_new[i]
                        ).item()
                        for i in range(len(payment_dates_new))
                    ]
                )
                intervals_integral_new = np.array(
                    [
                        self.calendar.interval(
                            payment_dates_prev[-1], integral_dates_new[i]
                        ).item()
                        for i in range(len(integral_dates_new))
                    ]
                )
                intervals_payment_temp = np.array(
                    intervals_payment[: payment_dates_temp.size]
                )

                def f(hazard_temp):
                    """
                    CDS price (from the point of view of the seller of protection) as a function of the (forward) hazard rate (see) Eq. (7.2) in [O`Kane].

                    Parameters
                    ----------
                    hazard_temp : float
                        Forward hazard rate.

                    Returns
                    -------
                    float
                        Premium PV - Protection PV

                    Notes
                    ------
                    - The proper definitions of these two legs are described in Chapter 6 of [O`Kane].
                    - Note that we use previous data for computing :math:`Q(t, t')`

                    """
                    q_payment_new = q_payment_prev[-1] * np.exp(
                        -hazard_temp * intervals_payment_new
                    )  # Interpolation based on Section 7.4.1 of [0' Kane].
                    q_integral_new = q_integral_prev[-1] * np.exp(
                        -hazard_temp * intervals_integral_new
                    )  # Interpolation based on Section 7.4.1 of [0' Kane].
                    q_payment_temp = np.concatenate((q_payment_prev, q_payment_new))
                    q_integral_temp = np.concatenate((q_integral_prev, q_integral_new))
                    integral = (
                        1
                        / 2
                        * np.sum(
                            (z_integral_temp[:-1] + z_integral_temp[1:])
                            * (q_integral_temp[:-1] - q_integral_temp[1:])
                        )
                    )  # Protection PV in page 106 of [O`Kane].
                    rpvbp = (
                        1
                        / 2
                        * np.sum(
                            intervals_payment_temp
                            * z_payment_temp
                            * (q_payment_temp[:-1] + q_payment_temp[1:])
                        )
                    )
                    # Risky present value of a basis point for a CDS. See equation after Eq. (6.5) of [O`Kane].
                    value = cds_values_temp[-1] * rpvbp - (1 - self.recovery) * integral
                    return value

                hazards[-1] = afsfun.find_root(f, 0.05, [0, 1])
                q_payment_new_f = q_payment_prev[-1] * np.exp(
                    -hazards[-1] * intervals_payment_new
                )
                q_integral_new_f = q_payment_prev[-1] * np.exp(
                    -hazards[-1] * intervals_integral_new
                )
                q_payment = np.concatenate((q_payment_prev, q_payment_new_f))
                q_integral = np.concatenate((q_integral_prev, q_integral_new_f))
                self.q_payment = q_payment  # We use these values when n > 1.
                self.q_integral = q_integral  # We use these values when n > 1.
                # if n == cds_values.size:
                #     print(q_payment)
                return hazards

        return bootstrap_cds(cds_values, tenors)

    def fit(self, market_data):
        """
        Compute the failure (hazard) rate given the CDS spreads. These values are saved as an attribute, together with the
        corresponding hazard rates for computing risk sensitivities :math:`\\lambda^{(p,t)}`.


        Parameters
        ----------
        market_data : pandas.DataFrame
            CDS spreads (annual amount that the protection buyer must pay the protection seller). The data is assumed to be in basis points (one hundredth of a percentage point).

        Returns
        -------
        None

        Notes
        ------
            - The failure (hazard) rate is indeed the forward default rate as defined in Eq. (7.2) of [O`Kane].
            - IMM dates assumed as payment dates.

        """
        self.market_spreads = market_data
        self.fitting_dates = market_data.index
        params_df = pd.DataFrame(index=market_data.index, columns=market_data.columns)
        for date in market_data.index:
            cds_values = market_data.loc[date]
            params_df.loc[date] = self.fit_single_date(
                date, self.discount_curve, cds_values
            )
        self.hazard_rates = params_df
        LCExpCurve.fit(self, -params_df)


class CDSCurve_Allmats(LCExpCurve):  # TODO : This class can probably be removed.
    def __init__(self, name, seniority):
        self.name = name
        self.seniority = seniority
        self.recovery = 0.4 if self.seniority is True else 0.2
        LCExpCurve.__init__(self)

    def fit(
        self,
        date,
        discount_curve,
        cds_values,
        calendar,
        tenors=np.arange(10) + 1,
        no_steps=365,
    ):
        # remember to keep the calibration and CDS pricing consistent
        self.calendar = calendar
        date = pd.to_datetime(date)
        self.fitting_date = date
        tenors = np.asarray(tenors)
        maturities = np.array(
            [date] + [date + pd.DateOffset(months=tenor * 12) for tenor in tenors]
        )
        maturities = date + pd.timedelta_range(start="0", periods=11, freq="365D")
        Z = discount_curve.get_value(
            dates=date, future_dates=maturities, calendar=calendar
        )
        R = self.recovery

        no_steps = 365

        def bootstrap_cds(cds_values):
            cds_values = np.asarray(cds_values)
            if cds_values.size == 1:
                if np.asarray(cds_values).shape == ():
                    cds_values = np.array([cds_values])
                integral_steps = pd.date_range(
                    start=date, periods=no_steps + 1, freq=pd.DateOffset(days=1)
                )
                # date + pd.timedelta_range(start="0", periods=no_steps + 1, freq="{}D".format(f))
                integral_steps = integral_steps[1:]
                epsilon = 1 / no_steps
                k = np.arange(no_steps) + 1
                disc_steps = discount_curve.get_value(
                    dates=date, tenors=integral_steps, calendar=self.calendar
                )

                def f(hazard):
                    a = cds_values * Z[1] * (1 + np.exp(-hazard))
                    b = (
                        2
                        * (1 - R)
                        * (1 - np.exp(hazard * epsilon))
                        * np.sum(disc_steps * np.exp(-hazard * k * epsilon))
                    )
                    value = a + b
                    return value

                hazard = optimize.newton(f, 0.05)
                return hazard
            else:
                h = bootstrap_cds(cds_values[:-1])
                q = np.ones(h.size + 1)
                q[1:] = np.exp(-np.sum(np.tril(np.ones((h.size, h.size))) * h, axis=1))
                n = cds_values.size
                hazards = np.zeros(n)
                hazards[:-1] = h

                # the factor of 1/2 is already in the formula below
                rp_previous = np.sum(Z[1:n] * (q[:-1] + q[1:]))

                t1 = date + pd.DateOffset(months=12 * (n - 1))
                integral_steps = pd.date_range(
                    start=t1, periods=no_steps + 1, freq=pd.DateOffset(days=1)
                )
                integral_steps = integral_steps[1:]
                epsilon = 1 / no_steps
                k = np.arange(no_steps) + 1
                disc_steps = discount_curve.get_value(
                    dates=date, tenors=integral_steps, calendar=calendar
                )

                # previous_integral = 0
                # for i in range(n-1):
                #     t = date + pd.DateOffset(months=i)
                #     steps = pd.date_range(start=t1, periods=no_steps+1, freq=pd.DateOffset(days=1))
                #     steps = steps[1:]

                def f(hazard):
                    a1 = (
                        2
                        * (1 - R)
                        * (1 - np.exp(hazard * epsilon))
                        * np.sum(disc_steps * np.exp(-hazard * k * epsilon))
                    )
                    a2 = cds_values[-1] * Z[n] * (1 + np.exp(-hazard))
                    b = cds_values[-1] - cds_values[-2]
                    value = q[-1] * (a1 + a2) + b * rp_previous
                    return value

                tol = 10 ** (-12)
                hazard = optimize.newton(f, 0.01, tol=tol)

                # sigma1 = np.sum(Z[1:n-1]*(q[:-1]-q[1:]))
                # num1 = 2*(1-R)*sigma1 -cds_values[-1]*rp_previous
                # num2 = Z[n]*q[-1]*(cds_values[-1]-2*(1-R))
                # denom = Z[n]*q[-1]*(cds_values[-1]+2*(1-R))
                # hazard = -np.log((num1-num2)/denom)

                hazards[-1] = hazard
                return hazards

        self.hazard_rates = bootstrap_cds(cds_values)
        # h = self.hazard_rates
        # minus sign necessary because LCExp directly inputs the parameters without changing sign
        LCExpCurve.fit(self, maturities, -self.hazard_rates)


class CorporateCurve(LCExpCurve):
    def __init__(self, name):
        self.name = name
        LCExpCurve.__init__(self)

    def fit(self, date, bonds, calendar):
        #:param date: date of fitting, corresponding to data in bonds
        #:param bonds: data frame, indexed by maturities; columns: "PX_LAST", "CPN", "CPN_FREQ"
        #:param calendar:
        #:return:
        date = pd.to_datetime(date)
        self.calendar = calendar
        bonds = bonds.sort_index()

        def bootstrap_bonds(bonds):
            if len(bonds) == 1:
                maturity = bonds.index[0]
                coupon_rate = bonds.iloc[0]["CPN"]
                price = bonds.iloc[0]["PX_LAST"]
                freq = int(12 / bonds.iloc[0]["CPN_FREQ"])
                coupon_dates = pd.date_range(
                    start=maturity, end=date, freq=-pd.DateOffset(months=freq)
                ).sort_values()
                taus = self.calendar.interval(date, coupon_dates)
                deltas = np.copy(taus)
                deltas[1:] = deltas[1:] - deltas[:-1]

                def f(rate):
                    px = 100 * (
                        np.sum(deltas * coupon_rate * np.exp(-rate * taus))
                        + np.exp(-rate * taus[-1])
                    )
                    diff = price - px
                    return diff

                tol = 10 ** (-12)
                rate = optimize.newton(f, 0, tol=tol)
                return rate
            else:
                rates = np.zeros(len(bonds))
                rates[:-1] = bootstrap_bonds(bonds[:-1])
                maturities_intervals = self.calendar.interval(date, bonds.index)
                maturity = bonds.index[-1]
                coupon_rate = bonds.iloc[-1]["CPN"]
                price = bonds.iloc[-1]["PX_LAST"]
                freq = int(12 / bonds.iloc[-1]["CPN_FREQ"])
                coupon_dates = pd.date_range(
                    start=maturity, end=date, freq=-pd.DateOffset(months=freq)
                ).sort_values()
                taus = self.calendar.interval(date, coupon_dates)
                deltas = np.copy(taus)
                deltas[1:] = deltas[1:] - deltas[:-1]

                def f(rate):
                    temp_rates = np.copy(rates)
                    temp_rates[-1] = rate
                    # nominal
                    px = np.exp(-np.sum(temp_rates * maturities_intervals))
                    # coupons up to first maturity
                    bool = coupon_dates <= bonds.index[0]
                    px = px + np.sum(
                        coupon_rate * deltas[bool] * np.exp(-temp_rates[0] * taus[bool])
                    )
                    # remaining coupons
                    for i in range(bonds.index[1:-1].size):
                        bool = (coupon_dates <= bonds.index[i + 1]) * (
                            coupon_dates > bonds.index[i + 1]
                        )
                        px = px + np.sum(
                            coupon_rate
                            * deltas[bool]
                            * np.exp(-temp_rates[i] * taus[bool])
                        )
                    diff = price - 100 * px
                    return diff

                tol = 10 ** (-12)
                rates[-1] = optimize.newton(f, 0, tol=tol)
                return rates

        self.short_rates = bootstrap_bonds(bonds)
        # minus sign necessary because LCExp directly inputs the parameters without changing sign
        maturities = bonds.index.union([date]).sort_values()
        LCExpCurve.fit(self, maturities, -self.short_rates)
