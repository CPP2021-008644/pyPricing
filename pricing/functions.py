import numpy as np
import math
from scipy import optimize
import plotly.graph_objects as go
import pandas as pd

try:
    from .discount_curves import CubicSplineSwapCurve
except (ImportError, ModuleNotFoundError, ValueError):
    from pricing.discount_curves import CubicSplineSwapCurve


def check_all_nonnumeric(arr):
    """
    Check if all elements in a numpy array or Python list are non-numeric.

    This function tries to convert each element in the array or list to a float. If the
    conversion raises a ValueError or TypeError, or if the value is nan, it means the
    element is non-numeric, so the function continues to the next element. If the
    conversion does not raise an exception and the value is not nan, it means the element
    is numeric, so the function immediately returns False. If the function finishes
    checking all elements without finding a numeric one, it returns True.

    Parameters
    ----------
    arr : numpy.ndarray or list
        The array or list to check.

    Returns
    -------
    bool
        True if all elements are non-numeric, False otherwise.
    """

    for i in arr:
        try:
            val = float(i)
            if not math.isnan(val):
                return False
        except (ValueError, TypeError):
            continue
    return True


def find_root(func, x0, interval, tolerance=10**-8, fprime=None):
    """
    Function to find the root of a given function, using several methods.
    Each method is tried in turn until one succeeds.

    If none succeeds, we plot the function in interval.

    Parameters
    ----------
    func : callable
        The function for which the root is to be computed.
    x0 : float
        Initial guess for the root.
    interval : list
        Interval [a,b] for ridder, bisecction and brentq.
    tolerance : float
        If func(solution)>tolerance an exception is raised.
    fprime : callable, optional
        The derivative of the function. If not provided, the Newton method will use the secant method.

    Returns
    -------
    float
        The root found by the successful method (unless all methods failed).
    """
    methods = [
        ("newton (Secant)", optimize.newton),
        ("newton (Newton-Raphson)", optimize.newton),
        ("fixed_point", optimize.fixed_point),
        ("bisection", optimize.bisect),
        ("brentq", optimize.brentq),
        ("ridder", optimize.ridder),
    ]

    root = None

    for name, method in methods:
        try:
            if name == "fixed_point":
                root = method(lambda x: x - func(x), x0)
            elif name == "newton (Secant)":
                root = method(func, x0, fprime=None)
            elif name == "newton (Newton-Raphson)":
                root = method(func, x0, fprime=fprime)
            else:
                root = method(func, interval[0], interval[1])
            # print(f"Method {name} succeeded with root {root}")
            break  # if method succeeded, stop trying the rest
        except Exception:
            pass
    if root is None:
        x_vals = np.linspace(interval[0], interval[1], 200)
        y_vals = [
            func(x) for x in x_vals
        ]  # Note that this code snippet is intentionally not vectorized.
        fig = go.Figure(data=go.Scatter(x=x_vals, y=y_vals))
        fig.update_layout(title="Plot of func", xaxis_title="x", yaxis_title="y")
        fig.show()
        raise Exception("All methods failed")
    else:
        if np.abs(func(root)) > tolerance:  # Maybe another tolerance can be chosen.
            raise Exception("The numerical error is too large.")
        else:
            return root


def dates_formatting(*date_sets, sort_values=True):
    """
    Convert dates to a consistent format and sort them in ascending order.

    Parameters
    ----------
    dates : pandas.DatetimeIndex, list of strings, pandas.Timestamp, or string
        Dates to be formatted. It can be a single date (as a pandas.Timestamp or its string representation)
        or an array-like object (as a pandas.DatetimeIndex or a list of its string representation) containing dates.

    Returns
    -------
    pandas.DatetimeIndex
        A pandas DatetimeIndex object containing the formatted dates in ascending order.

    Examples
    --------
    >>> dates_formatting('2022-01-01')
    DatetimeIndex(['2022-01-01'], dtype='datetime64[ns]', freq=None)

    >>> dates_formatting(['2022-01-03', '2022-01-01', '2022-01-02'])
    DatetimeIndex(['2022-01-01', '2022-01-02', '2022-01-03'], dtype='datetime64[ns]', freq=None)

    >>> dates_formatting(['2022-01-03', '2022-01-01', '2022-01-02'], '2022-01-01')
    [DatetimeIndex(['2022-01-01', '2022-01-02', '2022-01-03'], dtype='datetime64[ns]', freq=None), DatetimeIndex(['2022-01-01'], dtype='datetime64[ns]', freq=None)]

    """

    formatted_dates = []
    for date_set in date_sets:
        if np.asarray(date_set).shape == ():
            dates = [date_set]
        else:
            dates = date_set
        if sort_values:
            formatted_dates.append(pd.to_datetime(dates).sort_values())
        else:
            formatted_dates.append(pd.to_datetime(dates))
    if len(formatted_dates) == 1:
        formatted_dates = formatted_dates[0]

    return formatted_dates


def contains_word(string_list, word):
    """
    Check if strings in the given list contain the specified word. Words in each string are separated by underscores.

    Parameters
    ----------
    string_list : list of str
        The list of strings where each string has words separated by underscores.
    word : str
        The word to search for within the strings.

    Returns
    -------
    list of str
        A list of strings from the input `string_list` that contain the specified `word`.

    Examples
    --------
    >>> contains_word(['word1_word2', 'word3_word4', 'word2_word5'], 'word2')
    ['word1_word2', 'word2_word5']

    """
    return [s for s in string_list if word in s.split("_")]


def set_past_dates(arr, a, dates1, dates2):
    """
    Returns a modified array where the value 'a' is set at positions where ``dates1`` < ``dates2``.

    Parameters
    ----------
    arr : numpy.ndarray
        The original array to be modified.
    a : float
        The value to be set in the array at the specified positions.
    dates1 : pandas.DatetimeIndex, list of strings, pandas.Timestamp, or string
             First set of dates. It can be a single date (as a pandas.Timestamp or its string representation) or an array-like object
            (as a pandas.DatetimeIndex or a list of its string representation) containing dates.
    dates2 : pandas.DatetimeIndex, list of strings, pandas.Timestamp, or string
             Second set of dates. It can be a single date (as a pandas.Timestamp or its string representation) or an array-like object
            (as a pandas.DatetimeIndex or a list of its string representation) containing dates.
    Returns
    -------
    numpy.ndarray
        The modified array where at positions dictated by ``dates1`` being less than ``dates2``, the value equals 'a'.

    Examples
    --------
    >>> dates1 = pandas.DatetimeIndex(["20211229", "20221231", "20231231"])
    >>> dates2 = pandas.DatetimeIndex(["20211229", "20211230"])
    >>> arr = numpy.full((dates1.size, dates2.size), 1)
    >>> set_past_dates(arr, 0, dates1, dates2)
    numpy.array([[1, 0], [1, 1], [1, 1]])
    """
    indices = np.indices(arr.shape)
    locations = np.array(dates1)[indices[0]] < np.array(dates2)[indices[1]]
    arr[locations] = a
    return arr


def accumulate_index(arr, ind, op):
    """

    Parameters
    ----------
    arr : numpy.ndarray
        Array for the particular indexing.
    ind : numpy.array, dtype=int or list
        List of integers for the indexing.
    op : str
        Operation for the cumulative operation. Either "sum" or "prod".

    Returns
    -------
    numpy.ndarray
        Returns the same array but in the last dimension an indexing has occured and the missing positions has been "accumulated" in the next remaining position.
    Examples
    -------
    >>> arr = numpy.full((2, 8), 2)
    >>> accumulate_index(arr, [0, 1, 2, 4, 7], "prod")
    numpy.array([[2., 2., 2., 4., 8.],
              [2., 2., 2., 4., 8.]])
    >>> arr = numpy.full((2, 8), 1)
    >>> accumulate_index(arr, [0, 1, 2, 4, 7], "sum")
    array([[1, 1, 2, 3],
           [1, 1, 2, 3]], dtype=int32)
    """
    if op == "sum":
        arr_cum = arr.cumsum(axis=-1)
    elif op == "prod":
        arr_cum = arr.cumprod(axis=-1)
    else:
        raise AttributeError("Invalid cumulative operation.")

    arr_cum_ind = arr_cum[..., ind]
    shape_f = list(arr.shape)
    shape_f.pop()
    shape_f.append(len(ind))
    shape = tuple(shape_f)
    arr_f = np.full(shape, np.nan)
    arr_f[..., 0] = arr_cum_ind[..., 0]
    if op == "sum":
        arr_f = arr_cum_ind[..., 1:] - arr_cum_ind[..., :-1]
    else:
        arr_f[..., 1:] = arr_cum_ind[..., 1:] / arr_cum_ind[..., :-1]

    return arr_f


def number_to_array(*args, for_broadcasting=False):
    """
    Convert the input arguments to NumPy arrays.

    Parameters
    ----------
    *args : scalar or array_like
        Input arguments to be converted to NumPy arrays.
    for_broadcasting : bool, optional
        If ``True``, reshape the arrays for broadcasting. Default is ``False``.

    Returns
    -------
    list of ndarrays
        A list of NumPy arrays where each input argument is converted to an array. If an input argument is already a NumPy array, it is returned as is.

    Examples
    --------
    >>> import numpy as np
    >>> number_to_array(1.0, 2, numpy.array([3.0, 4.0]), for_broadcasting=True)
    [array([[[1.]]]), array([[[2]]]), array([[[3.]],
    <BLANKLINE>
           [[4.]]])]
    Note
    -------
    We need to add the BLANKLINE for format considerations, https://docs.python.org/3/library/doctest.html.
    """
    lis = []
    for elem in args:
        if not isinstance(elem, np.ndarray):
            elem = np.array([elem])
        if for_broadcasting:
            elem = elem.reshape((elem.size, 1, 1))

        lis.append(elem)

    return lis


def delta_indexing(arr, dates_0, dates_1, num, second_index=False):
    """

    Parameters
    ----------
    arr : np.ndarray
        Array for the particular indexing.
    dates_0 : pandas.DatetimeIndex, list of strings, pandas.Timestamp, or string
             First set of dates. It can be a single date (as a pandas.Timestamp or its string representation) or an array-like object
            (as a pandas.DatetimeIndex or a list of its string representation) containing dates.
    dates_1 : pandas.DatetimeIndex, list of strings, pandas.Timestamp, or string
             Second set of dates. It can be a single date (as a pandas.Timestamp or its string representation) or an array-like object
            (as a pandas.DatetimeIndex or a list of its string representation) containing dates.
    num : int
        Either 0 or 1
    second_index : Boolean
        If ``True``, operation perform in the second index instead of the first one.

    Returns
    -------
    np.ndarray
        Returns the same array but in with zeros in the indices corresponding to the dates of subscript num + 1 (mod 2).

    Examples
    -------
    >>> arr = np.full(6, 2)
    >>> dates_0 = pd.date_range(start="20200101", end="20250101", freq=pd.DateOffset(months=6))
    >>> dates_1 = pd.date_range(start="20200101", end="20250101", freq=pd.DateOffset(years=1))
    >>> delta_indexing(arr, dates_0, dates_1, 1)
    array([2., 0., 2., 0., 2., 0., 2., 0., 2., 0., 2.])
    >>> arr = np.full(6, 2)
    >>> dates_1 = pd.date_range(start="20200101", end="20250101", freq=pd.DateOffset(months=6))
    >>> dates_0 = pd.date_range(start="20200101", end="20250101", freq=pd.DateOffset(years=1))
    >>> delta_indexing(arr, dates_0, dates_1, 0)
    array([2., 0., 2., 0., 2., 0., 2., 0., 2., 0., 2.])

    It must leave the array intact if ``dates_0`` equals ``dates_1``.

    >>> arr = np.full(6, 2)
    >>> dates_1 = pd.date_range(start="20200101", end="20250101", freq=pd.DateOffset(months=12))
    >>> dates_0 = pd.date_range(start="20200101", end="20250101", freq=pd.DateOffset(years=1))
    >>> delta_indexing(arr, dates_0, dates_1, 0)
    array([2., 2., 2., 2., 2., 2.])

    """

    total_dates = dates_0.union(dates_1)
    dates_dic0 = {}
    dates_dic1 = {}
    for date in dates_0:
        dates_dic0[date] = total_dates.get_loc(date)
    for date in dates_1:
        dates_dic1[date] = total_dates.get_loc(date)
    indices0 = [dates_dic0[date] for date in dates_0]
    indices1 = [dates_dic1[date] for date in dates_1]

    if num == 0:
        indices = indices0
    elif num == 1:
        indices = indices1
    else:
        raise AttributeError("num is either 0 or 1.")

    if second_index:
        shape_ext = list(arr.shape)
        shape_ext[1] = total_dates.size
        shape = tuple(shape_ext)
        arr_ext = np.full(shape, 0, dtype=float)
        arr_ext[:, indices] = arr
    else:
        arr_ext = np.full(total_dates.size, 0, dtype=float)
        arr_ext[indices] = arr

    return arr_ext


def disc_curve_reconstruction(
    dates, discount_table_df, ticker_discount_curve, calendar, method="bond_spline"
):
    dates = dates_formatting(dates)
    disc_curve_df = discount_table_df[
        discount_table_df["discount_curve"] == ticker_discount_curve
    ]
    dcs_x_nobump = disc_curve_df.loc[:, ["dtime", "ppoly_breakpoints"]].set_index(
        "dtime"
    )
    dcs_c_nobump = disc_curve_df.loc[:, ["dtime", "ppoly_coefficients"]].set_index(
        "dtime"
    )

    new_cssc = CubicSplineSwapCurve(calendar)
    x_date = dcs_x_nobump.loc[:, "ppoly_breakpoints"].values[0][1:]
    yields = []
    for i in range(len(dates)):
        df_temp = dcs_c_nobump.loc[:, "ppoly_coefficients"].values[i][3][1:]
        last_df = (
            dcs_c_nobump.loc[:, "ppoly_coefficients"].values[i][0][-1]
            * (x_date[-1] - x_date[-2]) ** 3
            + dcs_c_nobump.loc[:, "ppoly_coefficients"].values[i][1][-1]
            * (x_date[-1] - x_date[-2]) ** 2
            + dcs_c_nobump.loc[:, "ppoly_coefficients"].values[i][2][-1]
            * (x_date[-1] - x_date[-2]) ** 1
            + dcs_c_nobump.loc[:, "ppoly_coefficients"].values[i][3][-1]
        )
        df_temp.append(last_df)
        df_temp = np.array(df_temp)
        yields_temp = -np.log(df_temp) / x_date
        yields.append(yields_temp)

    data = pd.DataFrame(data=yields, index=dates, columns=x_date)
    new_cssc.fit(data, method=method)
    return new_cssc
