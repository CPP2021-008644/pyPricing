import numpy as np
import pandas as pd
# from scipy.stats import norm
import matplotlib.pyplot as plt

try:
    from ..data.underlyings import Basis  # Import needed for the workspace. In this case we need the parent package since underlyings
    # is in a different subpackage (data)
except (ImportError, ModuleNotFoundError, ValueError):
    from data.underlyings import Basis  # (Absolute) local import
try:
    from .risk_factors import *
except (ImportError, ModuleNotFoundError, ValueError):
    from risk.risk_factors import *


def plot_margin_backtest(scan_range, prices, price_col=None, steps_ahead=2,
                         title="Margin Backtest: +/- Margin vs. Forward Price Change",
                         dates=None):
    """
    Plot margin backtest with symmetric margin bands and forward price changes.

    Parameters
    ----------
    scan_range : pandas.Series or pandas.DataFrame
        Series with a DateTimeIndex representing the margin for each date.
        If a DataFrame is provided, it must contain exactly one column with margin values.
    prices : pandas.Series or pandas.DataFrame
        Series with a DateTimeIndex representing price levels.
        If a DataFrame is provided, specify the column to use via `price_col`.
    price_col : str, optional
        Column name to extract from `prices` if it is a DataFrame with multiple columns.
        Ignored if `prices` is already a Series or has only one column.
    steps_ahead : int, default=2
        Number of index steps ahead used to compute forward price change.
        For example, with `steps_ahead=2`, the change is calculated as
        `price[t+2] - price[t]`. This respects the index frequency
        (e.g., business days).
    title : str, default="Margin Backtest: +/- Margin vs. Forward Price Change"
        Title for the plot.
    dates : list-like of datetime-like, optional
        Specific subset of dates to include in the plot. If provided,
        both margin and forward changes are filtered to these dates.
        If None, all available dates are used.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The matplotlib axis containing the plot.
    df : pandas.DataFrame
        DataFrame containing aligned margin and forward changes, with
        an additional boolean coverage column.
    """
    if isinstance(scan_range, pd.DataFrame):
        if scan_range.shape[1] != 1:
            raise ValueError("scan_range DataFrame must have exactly one column.")
        margin = scan_range.iloc[:, 0].copy()
        margin.name = "margin"
    else:
        margin = scan_range.copy()
        margin.name = "margin"

    if isinstance(prices, pd.DataFrame):
        if price_col is None:
            if prices.shape[1] != 1:
                raise ValueError("prices has multiple columns. Specify price_col.")
            price = prices.iloc[:, 0].copy()
        else:
            price = prices[price_col].copy()
    else:
        price = prices.copy()

    margin = margin.sort_index()
    price = price.sort_index()

    fwd_change = price.shift(-steps_ahead) - price
    fwd_change.name = f"fwd_change_{steps_ahead}"

    df = pd.concat([margin, fwd_change], axis=1).dropna()

    if dates is not None:
        dates_idx = pd.DatetimeIndex(pd.to_datetime(list(dates)))
        df = df.loc[df.index.intersection(dates_idx)]

    if df.empty:
        raise ValueError("No data to plot after alignment/filtering. "
                         "Check `steps_ahead`, input indices, and `dates` filter.")

    covered = df[fwd_change.name].abs() <= df["margin"]
    colors = np.where(covered, "green", "red")

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df.index, df["margin"], label="+margin", linewidth=1.5)
    ax.plot(df.index, -df["margin"], label="-margin", linewidth=1.5)

    ax.scatter(df.index, df[fwd_change.name], c=colors, s=30,
               label=f"{steps_ahead}-step change", zorder=3)

    ax.axhline(0, linewidth=0.8, alpha=0.6)
    ax.set_ylabel("€")
    ax.set_title(title)
    ax.legend()

    cov_rate = float(covered.mean())
    ax.text(0.01, 0.95, f"Coverage: {cov_rate:.1%}", transform=ax.transAxes, va="top")

    plt.tight_layout()
    return ax, df


def correlation_quantile(rhos, N=255, M=100000, quantile=0.1, memory=0.94):
    if np.asarray(rhos).ndim == 0:
        rhos = [rhos]
    rhos = np.asarray(rhos)
    draws = norm.rvs(size=(2, M, N))
    x = draws[0]
    y = draws[1]
    adjusted_corr = np.full(rhos.size, np.nan)
    for i in range(rhos.size):
        z = rhos[i] * x + np.sqrt(1 - rhos[i] ** 2) * y
        weights = memory ** np.arange(N)
        vol_x = np.sqrt(np.sum(weights * x**2 / np.sum(weights), axis=1))
        vol_z = np.sqrt(np.sum(weights * z**2 / np.sum(weights), axis=1))
        cov = np.sum(weights * x * z / np.sum(weights), axis=1)
        corr = cov / (vol_x * vol_z)
        adjusted_corr[i] = np.quantile(corr, quantile)
    return adjusted_corr if adjusted_corr.size > 1 else adjusted_corr[0]


def correlation_interpolation_correction(rho,values=None):
    """
    Apply empirical correlation correction (ECC) through interpolation.

    Parameters
    ----------
    rho : float
        Target correlation coefficient, constrained between -1 and 1.
    values : array-like, optional
        Pre-computed adjusted correlation values for interpolation. Default
        is None.

    Returns
    -------
    float
        Corrected correlation value interpolated from the empirical grid.
    """
    if values is None:
        values = np.zeros(10)
    N=len(values)
    assert -1<=rho<=1
    rhos = (np.arange(N)-(N-1)/2)/((N-1)/2)
    if rho == 1: return 1
    for i in range(N):
        if rhos[i]>=rho:
            break
        pos = i
    return values[pos] + (values[i+1]-values[i])/(rhos[i+1]-rhos[i]) * (rho-rhos[i])


def scan_range_white_noise(risk_factor, dates, w=5, a=0.2, days=2, quantile_choice="average"):
    """
    Margin calculation using white noise quantiles (0.01 and 0.99), volatilities and prices.
    Applies stress/countercyclicality factors and returns the scan range in monetary units.

    Parameters
    ----------
    risk_factor : RiskFactor
        Risk factor object containing underlying prices and volatility methods.
    dates : pandas.DatetimeIndex
        Dates for which to compute the scan range.
    w : float, optional
        Scaling factor for stressed volatility. Default is 5.
    a : float, optional
        Weighting parameter for critical volatility threshold. Default is 0.2.
    days : int, optional
        Horizon in days for scaling the scan range. Default is 2.
    quantile_choice : {"average", "maximum"}, optional
        Method to compute the risk parameter from white-noise quantiles. 
        Default is "average".

    Returns
    -------
    pandas.Series
        Time series of scan range values indexed by `dates`.
    """
    prices = np.abs(risk_factor.underlying.get_value(dates))
    dates = prices.index
    vols = risk_factor.get_vol(dates)
    dates = dates[dates.isin(vols.index)]
    quantile_1 = risk_factor.get_white_noise_quantile(dates, 0.01, window=250)
    quantile_99 = risk_factor.get_white_noise_quantile(dates, 0.99, window=250)
    if quantile_choice == "average":
        parameter = (np.abs(quantile_1)+np.abs(quantile_99))/2
    else:
        parameter = np.maximum(np.abs(quantile_1),np.abs(quantile_99))
    bounds = risk_factor.get_vol_bounds(dates)
    sigma_min = bounds["Min"]
    sigma_max = bounds["Max"]
    sigma_stressed = w*(sigma_max-vols)/(255*vols)
    sigma_crit = (1-a)*sigma_min + a*sigma_max
    den = sigma_max-sigma_crit
    den.replace(0, np.nan, inplace=True)
    b = 0.25*(1-(vols-sigma_crit)/den*(vols > sigma_crit))
    vol_weight = 1 + np.maximum(b, sigma_stressed)
    scan_range = parameter*vols*vol_weight*prices*np.sqrt(days)
    return scan_range


def scan_range_bilateral_var(risk_factor, dates, w=5, a=0.2, days=2, window=250, quantile_choice="average"):
    """
    Margin calculation using bilateral quantiles, volatilities and prices.
    Applies stress/countercyclicality factors and returns the scan range in monetary units.

    Parameters
    ----------
    risk_factor : RiskFactor
        Risk factor object containing underlying prices and volatility methods.
    dates : pandas.DatetimeIndex
        Dates for which to compute the scan range.
    w : float, optional
        Scaling factor for stressed volatility. Default is 5.
    a : float, optional
        Weighting parameter for critical volatility threshold. Default is 0.2.
    days : int, optional
        Horizon in days for scaling the scan range. Default is 2.
    window : int, optional
        Rolling window size for variance estimation. Default is 250.
    quantile_choice : {"average", "maximum"}, optional
        Aggregation method for bilateral variance. Default is "average".

    Returns
    -------
    pandas.Series
        Time series of scan range values indexed by `dates`.
    """
    var = risk_factor.get_bilateral_var(dates, 0.01, window=window, position='long')
    if quantile_choice == "average":
        base_scan = np.abs(var).mean(axis=1)
    elif quantile_choice == "maximum":
        base_scan = np.abs(var).max(axis=1)

    vols = risk_factor.get_vol(dates)
    bounds = risk_factor.get_vol_bounds(dates)
    index = pd.DatetimeIndex.intersection(vols.index, bounds.index)
    vols = vols.loc[index]
    bounds = bounds.loc[index]
    sigma_min = bounds["Min"]
    sigma_max = bounds["Max"]
    sigma_stressed = w * (sigma_max - vols) / (255 * vols)
    sigma_crit = (1 - a) * sigma_min + a * sigma_max
    den = sigma_max-sigma_crit
    den.replace(0, np.nan, inplace=True)
    b = 0.25 * (1 - (vols - sigma_crit) / den * (vols > sigma_crit))
    antiprociclicality = 1 + np.maximum(b, sigma_stressed)
    scan_range = base_scan * antiprociclicality * np.sqrt(days)
    return scan_range


def scan_range_var(risk_factor, dates, w=5, a=0.2, days=2, quantile_choice="average"):
    """
    Margin calculation using var, volatilities and prices.
    Applies stress/countercyclicality factors and returns the scan range in monetary units.

    Parameters
    ----------
    risk_factor : RiskFactor
        Risk factor object containing underlying prices and volatility methods.
    dates : pandas.DatetimeIndex
        Dates for which to compute the scan range.
    w : float, optional
        Scaling factor for stressed volatility. Default is 5.
    a : float, optional
        Weighting parameter for critical volatility threshold. Default is 0.2.
    days : int, optional
        Horizon in days for scaling the scan range. Default is 2.
    quantile_choice : {"average", "maximum"}, optional
        Method to compute the risk parameter from VaR quantiles. 
        Default is "average".

    Returns
    -------
    pandas.Series
        Time series of scan range values indexed by `dates`.
    """
    vols = risk_factor.get_vol(dates)
    dates = dates[dates.isin(vols.index)]
    quantile_1 = risk_factor.get_var(dates, 0.01, window=250)
    quantile_99 = risk_factor.get_var(dates, 0.99, window=250)
    if quantile_choice == "average":
        parameter = (np.abs(quantile_1)+np.abs(quantile_99))/2
    else:
        parameter = np.maximum(np.abs(quantile_1),np.abs(quantile_99))
    bounds = risk_factor.get_vol_bounds(dates)
    sigma_min = bounds["Min"]
    sigma_max = bounds["Max"]
    sigma_stressed = w*(sigma_max-vols)/(255*vols)
    sigma_crit = (1-a)*sigma_min + a*sigma_max

    aux = sigma_max-sigma_crit
    aux = aux.replace(['0', 0], np.nan)  # Added for occasional errors in dividing by zero.

    b = 0.25*(1-(vols-sigma_crit)/(aux)*(vols > sigma_crit))
    vol_weight = 1 + np.maximum(b, sigma_stressed)
    scan_range = parameter*vol_weight*np.sqrt(days)
    scan_range.name = 'Scan range'
    return scan_range


def basis_scan_range(risk_factor1, risk_factor2, correlation_model, dates, w=5, a=0.2, days=2, quantile_choice="average"):
    """
    Scan the basis risk between two risk factors.
    Calculate each individual scan range (with scan_range_white_noise), 
    adjust conditional correlation and combine them.

    Parameters
    ----------
    risk_factor1 : RiskFactor
        First risk factor.
    risk_factor2 : RiskFactor
        Second risk factor.
    correlation_model : object
        Model used to estimate conditional correlations.
    dates : pandas.DatetimeIndex
        Dates for which to compute the scan range.
    w : float, optional
        Scaling factor for stressed volatility. Default is 5.
    a : float, optional
        Weighting parameter for critical volatility threshold. Default is 0.2.
    days : int, optional
        Horizon in days for scaling the scan range. Default is 2.
    quantile_choice : {"average", "maximum"}, optional
        Method to compute the base scan range. Default is "average".

    Returns
    -------
    pandas.Series
        Time series of basis scan range values.
    """
    basis = Basis(risk_factor1.underlying, risk_factor2.underlying)
    scan_range1 = scan_range_white_noise(risk_factor1, dates, w=w, a=a, days=days, quantile_choice = quantile_choice)
    scan_range2 = scan_range_white_noise(risk_factor2, dates, w=w, a=a, days=days, quantile_choice = quantile_choice)
    correlation_model.fit(risk_factor1, basis)
    correlation_1 = correlation_model.get_conditional_correlation(dates)
    correlation_model.fit(risk_factor2, basis)
    correlation_2 = correlation_model.get_conditional_correlation(dates)
    return scan_range1*correlation_1-scan_range2*correlation_2


def spread_credit_scan_range(risk_factor1, risk_factor2, correlation_model, dates, w=5, a=0.2, days=2, window=250,
                      adjust_correlation=False):
    """
    Scans the spread risk between two risk factors. 
    Calculates returns for both, volatility, 
    conditional correlation, and a normalised VaR. 
    Correlation correction can be applied if requested.

    Parameters
    ----------
    risk_factor1 : RiskFactor
        First risk factor.
    risk_factor2 : RiskFactor
        Second risk factor.
    correlation_model : object
        Model used to estimate conditional correlations.
    dates : pandas.DatetimeIndex
        Dates for which to compute the spread credit.
    w : float, optional
        Scaling factor for stressed volatility. Default is 5.
    a : float, optional
        Weighting parameter for critical volatility threshold. Default is 0.2.
    days : int, optional
        Horizon in days for scaling the spread credit. Default is 2.
    window : int, optional
        Rolling window size for quantile estimation. Default is 250.
    adjust_correlation : bool, optional
        Whether to apply empirical correlation correction. Default is False.

    Returns
    -------
    pandas.Series
        Time series of spread credit scan range values.
    """
    px1 = risk_factor1.underlying.get_value(risk_factor1.get_dates()) 
    returns1 = px1[1:]-px1[:-1].values
    returns1 = returns1[returns1 != 0] 
    # we could replace the lines above by: returns1 = risk_factor1.underlying.get_absolute_return(risk_factor1.get_dates())
    px2 = risk_factor2.underlying.get_value(risk_factor2.get_dates()) 
    returns2 = px2[1:]-px2[:-1].values
    returns2 = returns2[returns2 !=0]
    # Analog observation
    index = pd.DatetimeIndex.intersection(px1.index, px2.index) 

    correlation_model.fit(risk_factor1.underlying, risk_factor2.underlying)
    vol1 = risk_factor1.get_vol().loc[index]
    vol2 = risk_factor2.get_vol().loc[index]
    rho = correlation_model.get_conditional_correlation(index)
    if adjust_correlation:
        rho = rho.apply(correlation_interpolation_correction)
    vol = np.sqrt((vol1*px1)**2+(vol2*px2)**2-2*rho*vol1*px1*vol2*px2)

    normalized_returns = (returns1-returns2)/vol
    normalized_returns = normalized_returns[~normalized_returns.isnull()]
    quantile99 = pd.Series(np.nan, index=dates)
    quantile01 = pd.Series(np.nan, index=dates)
    
    for date in dates:
        temp = normalized_returns[:date - pd.Timedelta("1D")]
        if window is not None:
            temp = temp[-window:]
        quantile99.loc[date] = temp.quantile(0.99)
        quantile01.loc[date] = temp.quantile(0.01)
    quantile = (np.abs(quantile99)+np.abs(quantile99))/2
    return np.sqrt(2)*quantile*vol


def scan_range_returns_normalized(risk_factor, dates, memory = 0.94, window = 250):
    """
    Compute scan range based on absolute returns.
    It calculates volatility with exponential memory, 
    normalises returns, and obtains quantiles.

    Parameters
    ----------
    risk_factor : RiskFactor
        Risk factor object containing underlying prices and volatility methods.
    dates : pandas.DatetimeIndex
        Dates for which to compute the scan range.
    memory : float, optional
        Exponential weighting factor for volatility estimation. Default is 0.94.
    window : int, optional
        Rolling window size for quantile estimation. Default is 250.

    Returns
    -------
    pandas.Series
        Time series of scan range values based on absolute returns normalized by volatility.
    """
    absolute_returns = risk_factor.underlying.get_return(risk_factor.get_dates(), raw=True)
    vol = pd.Series(np.nan, index=dates)
    for date in dates:
        temp = absolute_returns[:date - pd.Timedelta("1D")]
        if window is not None:
            temp = temp[-window:]
        # mean = np.mean(temp)
        weights = memory ** np.flip(np.arange(len(temp)))
        vol.loc[date] = np.sqrt(np.sum(weights*(temp)**2)/np.sum(weights))

    normalized_returns = absolute_returns/vol
    normalized_returns = normalized_returns[~normalized_returns.isnull()]
    quantile99 = pd.Series(np.nan, index=dates)
    quantile01 = pd.Series(np.nan, index=dates)
    
    for date in dates:
        temp = normalized_returns[:date - pd.Timedelta("1D")]
        if window is not None:
            temp = temp[-window:]
        quantile99.loc[date] = temp.quantile(0.99)
        quantile01.loc[date] = temp.quantile(0.01)
    quantile = (np.abs(quantile99)+np.abs(quantile99))/2
    return np.sqrt(2)*quantile*vol


def hybrid_var_es_scan_range(risk_factor, dates, window="1Y", confidence=0.01, alpha=0.75, quantile_interpolation="linear", 
                             position="both", rounding_decimals=None, tick_size=None, buffer=0.0, bounds=None):
    """
    Hybrid scan range combining quantile VaR and tail Expected Shortfall (generic CPP-style).

    This procedure builds a scan range as a convex combination of a short-window
    quantile risk measure (``R_var``) and a long-history tail measure
    based on Expected Shortfall (``R_es``):

        scan_range = alpha * R_var + (1 - alpha) * max(R_es, R_var)

    Where:
      - ``R_var`` is the max absolute VaR (long/short) estimated over a
        recent window (``window``).
      - ``R_es`` is a long-history ES computed at the tail defined by the
        absolute VaR from the full available history (per side), averaged across
        long/short tails by the number of tail observations.

    Parameters
    ----------
    risk_factor : RiskFactor
        Risk factor with methods ``get_var`` and ``get_expected_shortfall``.
    dates : array-like of datetime-like
        Valuation dates for which the scan range is computed.
    window : int or str, optional
        Window used for ``R_var`` VaR estimation. If int, it is a number
        of observations; if str, any pandas offset alias (e.g. "1Y"). Default "1Y".
    confidence : float, optional
        VaR tail probability in (0, 1). Default 0.01.
    alpha : float, optional
        Weight for the quantile component in [0, 1]. Default 0.75.
    quantile_interpolation : str, optional
        Quantile interpolation method passed through to the VaR estimator. Default "linear".
    position : {"both", "long", "short"}, optional
        Whether to use both sides (max abs of long/short) or a single side for VaR/ES. Default "both".
    rounding_decimals : int or None, optional
        If provided, rounds **up** to the given number of decimals after applying ``buffer``.
        Default None.
    tick_size : float or None, optional
        If provided, rounds **up** to the nearest multiple of ``tick_size`` after applying ``buffer``.
        (Applied after ``rounding_decimals`` if both are given.) Default None.
    buffer : float, optional
        Multiplicative buffer applied as ``Range *= (1 + buffer)``. Default 0.0.
    bounds : tuple or None, optional
        Optional bounds to clip the final range. If provided, it must be a tuple
        ``(floor, cap)`` where each element can be a scalar or a pandas Series
        indexed by ``dates``. Clipping is applied after rounding. Default None.

    Returns
    -------
    pandas.Series
        Time series of hybrid scan range values.
    """
        
    if np.asarray(dates).shape == ():
        dates = [dates]
    dates = pd.to_datetime(dates)

    var_long_w = risk_factor.get_var(
        dates, confidence, window=window, position="long",
        quantile_interpolation=quantile_interpolation
    )
    var_short_w = risk_factor.get_var(
        dates, confidence, window=window, position="short",
        quantile_interpolation=quantile_interpolation
    )
    var_long_all = risk_factor.get_var(
        dates, confidence, position="long",
        quantile_interpolation=quantile_interpolation
    )
    var_short_all = risk_factor.get_var(
        dates, confidence, position="short",
        quantile_interpolation=quantile_interpolation
    )

    if position == "long":
        R_var = np.abs(var_long_w)
        q_extreme = np.abs(var_long_all)
        es_long, npts_long = risk_factor.get_expected_shortfall(
            dates, threshold=-q_extreme, position="long",
            return_no_points=True
        )
        R_es = np.abs(es_long)
    elif position == "short":
        R_var = np.abs(var_short_w)
        q_extreme = np.abs(var_short_all)
        es_short, npts_short = risk_factor.get_expected_shortfall(
            dates, threshold=-q_extreme, position="short",
            return_no_points=True
        )
        R_es = np.abs(es_short)
    else:  # "both"
        R_var = np.maximum(np.abs(var_long_w), np.abs(var_short_w))
        q_extreme = np.maximum(np.abs(var_long_all), np.abs(var_short_all))
        es_long, npts_long = risk_factor.get_expected_shortfall(
            dates, threshold=-q_extreme, position="long",
            return_no_points=True
        )
        es_short, npts_short = risk_factor.get_expected_shortfall(
            dates, threshold=-q_extreme, position="short",
            return_no_points=True
        )
        denom = (npts_long + npts_short).replace(0, np.nan)
        R_es = (np.abs(es_long) * npts_long + np.abs(es_short) * npts_short) / denom

    scan_range = alpha * R_var + (1 - alpha) * np.maximum(R_es, R_var)

    if buffer:
        scan_range = scan_range * (1.0 + buffer)
    if rounding_decimals is not None:
        scan_range = np.ceil(scan_range * (10 ** rounding_decimals)) / (10 ** rounding_decimals)
    if tick_size is not None and tick_size > 0:
        scan_range = np.ceil(scan_range / tick_size) * tick_size
    if bounds is not None:
        floor, cap = bounds
        if floor is not None:
            if hasattr(floor, "reindex"):
                floor = floor.reindex(scan_range.index)
            scan_range = np.maximum(scan_range, floor)
        if cap is not None:
            if hasattr(cap, "reindex"):
                cap = cap.reindex(scan_range.index)
            scan_range = np.minimum(scan_range, cap)

    return scan_range
