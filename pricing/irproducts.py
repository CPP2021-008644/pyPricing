import numpy as np

try:
    from .structured import (
        Call,
        Put,
        Structured,
        Derivative,
        Forward,
    )  # (Relative) import needed for the workspace. In this case __package__ is 'pypricing.pricing'
except (ImportError, ModuleNotFoundError, ValueError):
    from pricing.structured import (
        Call,
        Put,
        Structured,
        Derivative,
        Forward,
    )  # (Absolute) local import
try:
    from .ratecurves import (
        SwapRate,
        LognormalSwapRate,
        NormalSwapRate,
        DifferenceRate,
        GSCFRate,
        TARNRate,
    )
except (ImportError, ModuleNotFoundError, ValueError):
    from pricing.ratecurves import (
        SwapRate,
        LognormalSwapRate,
        NormalSwapRate,
        DifferenceRate,
        GSCFRate,
        TARNRate,
    )
try:
    from .mc_engines import SRDeterministicVolDiffusionMC
except (ImportError, ModuleNotFoundError, ValueError):
    from pricing.mc_engines import SRDeterministicVolDiffusionMC
try:
    from .fixedincome import *
except (ImportError, ModuleNotFoundError, ValueError):
    from pricing.fixedincome import *

try:
    from . import functions as afsfun
except (ImportError, ModuleNotFoundError, ValueError):
    import pricing.functions as afsfun

zero_credit_risk = CRDC(0)


# ------------------------------------------------------------------------------------
# swap
# ------------------------------------------------------------------------------------


class IRProduct(Derivative):
    """
    Represents an interest rate product.

    Parameters
    ----------
    rate : pricing.ratecurves.MultiRate
       Underlying interest rate of the product.
    obsdates : pandas.DatetimeIndex or list
       Observation dates.
    calendar : tuple or data.calendars.DayCountCalendar
       Calendar used for calculating observation dates.
    credit_curve : pricing.discount_curves.DiscountCurve
       The credit curve for the swap. Default is ``zero_credit_risk``.
    """

    def __init__(self, rate, obsdates, calendar, credit_curve):
        Derivative.__init__(
            self,
            underlying=rate,
            obsdates=obsdates,
            pastmatters=True,
            calendar=calendar,
            credit_curve=zero_credit_risk,
        )

    def get_px(self, dates, discount_curve, short_rate=None, no_sim=10**6):
        # short_rate is a new argument, w.r.t. structured.py
        """
        Calculate the price of the interest rate product using a Monte Carlo simulation.

        Parameters
        ----------
        dates : pandas.DatetimeIndex, list of strings, pandas.Timestamp, or string
            Dates at which the price is calculated. It can be a single date (as a pandas.Timestamp or its string representation) or an array-like object (as a pandas.DatetimeIndex
            or a list of its string representation) containing dates.

        discount_curve : pricing.discount_curves.DiscountCurve
            Provides the discounts for the computation of the product.

        short_rate : pricing.ir_models.ShortRateModel
            The short-term interest rate.

        no_sim : int, optional
            Number of simulations to perform. Default is 10^6.

        Returns
        -------
        pandas.Series
            The calculated price of the interest rate product.

        See Also
        --------
        SRDeterministicVolDiffusionMC : Monte Carlo simulation class.

        """
        if short_rate is None:
            raise AttributeError("Short rate must be specified.")
        mc = SRDeterministicVolDiffusionMC(short_rate)
        price_mc = mc.price(self, dates, discount_curve, no_sim)

        return price_mc

    def payoff(self, prices, dates_dic, n):
        """
        Calculate the payoff of the interest rate product based on prices and dates.

        Parameters
        ----------
        prices: np.ndarray
            Prices of the underlying. The first index of price denotes the simulation, the second one the (full) observation dates and the third one the valuation dates.

        dates_dic: dict
            Dictionary (or pandas.Series) which assigns an index to each date.
        n: int
            Number of observation dates previous to a fixed observation date.

        Returns
        -------
        np.ndarray
            Array of calculated payoffs for each observation.
        """
        pass


class Swap(SwapRate, IRProduct):
    """
    Represents an interest rate swap with fixed and floating legs, providing methods
    to calculate the par rate and price of the swap.

    Inherits from `SwapRate` and `IRProduct` to utilize the swap rate modeling
    framework and interest rate product features, respectively.

    Parameters
    ----------
    fixed_rate : float
        The fixed interest rate for the swap, expressed as a decimal (e.g., 0.05 for 5%).
    curve : DiscountCurve
        The discount curve, :math:`\\tilde{P}`, used for forward bonds.
    effective_date : pandas.Timestamp or string
        The effective start date of the swap, :math:`T_\\alpha^{\\text{float}}=T_\\alpha^{\\text{fixed}}`. It is assumed that both floating and fixed legs have the same start date.
    end_date : pandas.Timestamp or string
        The termination date of the swap, :math:`T_\\beta^{\\text{float}}=T_\\beta^{\\text{fixed}}`. It is assumed that both floating and fixed legs have the same end date.
    floating_freq : int
        The frequency of the floating rate payments, :math:`12\\cdot(T_l^{\\text{float}}-T_{l-1}^{\\text{float}})` for :math:`l\\in\\{\\alpha^{\\text{float}}+1, \\ldots, \\beta^{\\text{float}}\\}`.
        Assumes a constant period between payments. For example, if `floating_freq=6`, then, approximately, :math:`(T_l^{\\text{float}}-T_{l-1}^{\\text{float}})=0.5`.
    fixed_freq : int
        The frequency of the fixed rate payments per year, similar to `floating_freq` but for the fixed leg.
    legs_calendars : list of DayCountCalendar or DayCountCalendar
        The day count conventions for the floating and fixed legs, :math:`[\\text{DC}^{\\text{float}}, \\text{DC}^{\\text{fixed}}]` or a single :math:`\\text{DC}` if the same calendar is used for both legs.
    nominal : float, optional
        The nominal or principal amount of the swap. Default is 100.
    credit_curve : pricing.discount_curves.DiscountCurve, optional
        The credit curve used for adjusting cash flows for credit risk. Default assumes no credit risk.

    Attributes
    ----------
    nominal : float
        The nominal or principal amount that interest payments are calculated against.
    fixed_rate : float
        The annual fixed interest rate agreed upon at the inception of the swap.
    credit_curve : pricing.discount_curves.DiscountCurve
        The curve representing the credit risk of the swap's counterparty.
    floating_calendar : DayCountCalendar
        The day count calendar for the floating leg, derived from `legs_calendars`.
    fixed_calendar : DayCountCalendar
        The day count calendar for the fixed leg, derived from `legs_calendars`.
    floating_dates : pandas.DatetimeIndex
        The scheduled payment dates for the floating leg, calculated from the `effective_date` to the `end_date` at intervals defined by `floating_freq`.
    fixed_dates : pandas.DatetimeIndex
        The scheduled payment dates for the fixed leg, similar to `floating_dates` but calculated using `fixed_freq`.
    tau_floating : numpy.ndarray
        Time intervals between consecutive floating rate payment dates, calculated using `floating_calendar`.
    tau_fixed : numpy.ndarray
        Time intervals between consecutive fixed rate payment dates, calculated using `fixed_calendar`.
    total_dates : pandas.DatetimeIndex
        A union of `floating_dates` and `fixed_dates`, representing all scheduled dates in the swap.
    indices_float : list
        Indices of `floating_dates` within `total_dates`.
    indices_fixed : list
        Indices of `fixed_dates` within `total_dates`.
    tenor_length : float
        The total tenor length of the swap in years, explicitly specified or derived from `effective_date` and `end_date`.
    settlement : str
        The settlement type of the swap, specifying how the swap will be settled, either physically or in cash.

    Note
    -----
    The `Swap` class allows for the modeling and valuation of standard interest rate swaps,
    facilitating the analysis of cash flows, swap rates, and valuation under different
    market conditions. It supports both plain vanilla swaps with fixed-to-floating rate
    exchanges and variations thereof, depending on the configuration of parameters.

    References
    ----------
    - [Brigo and Mercurio, 2006]: Brigo, D., & Mercurio, F. (2006). Interest Rate Models - Theory and Practice.
    """

    def __init__(
        self,
        fixed_rate,
        curve,
        effective_date,
        end_date,
        floating_freq,
        fixed_freq,
        legs_calendars=None,
        nominal=100,
        credit_curve=zero_credit_risk,
    ):
        self.nominal = nominal
        self.fixed_rate = fixed_rate
        self.credit_curve = credit_curve
        SwapRate.__init__(
            self,
            curve,
            effective_date,
            end_date,
            floating_freq,
            fixed_freq,
            legs_calendars=legs_calendars,
        )

    def get_parrate(self, dates, discount_curve):
        """
        Calculate the par rate of the swap at specified dates.

        Parameters
        ----------
        dates : pandas.DatetimeIndex, list of strings, pandas.Timestamp, or string
            Dates at which to compute the price. It can be a single date (as a pandas.Timestamp or its string representation) or an array-like object (as a pandas.DatetimeIndex or
            a list of its string representation) containing dates.
        discount_curve : pricing.discount_curves.DiscountCurve , default = None
            Gives the discounts for the computation of the strikes if these are not given.

        Returns
        -------
        pandas.Series
            The calculated par rate of the swap as a pandas Series.
        """
        parrate = self.get_value(dates, discount_curve)
        parrate = pd.Series(parrate, index=dates)
        return parrate

    def get_px(
        self, dates, discount_curve, short_rate=None, no_sims=None
    ):  # The last arguments are not needed for this analytic price
        """
        Calculate the price of the swap at specified dates using analytic methods.

        Parameters
        ----------
        dates : pandas.DatetimeIndex, list of strings, pandas.Timestamp, or string
            Dates at which to compute the price. It can be a single date (as a pandas.Timestamp or its string representation) or an array-like object
            (as a pandas.DatetimeIndex or a list of its string representation) containing dates.
        discount_curve : pricing.discount_curves.DiscountCurve , default = None
            Gives the discounts for the computation of the strikes if these are not given.
        short_rate : pricing.ir_models.ShortRateModel
            The short-term interest rate.
        no_sims : int or None, optional
            Number of simulations. Not used for this analytic price calculation.

        Returns
        -------
        pandas.Series
            The calculated price of the swap as a pandas Series.
        """
        parrate = self.get_value(dates, discount_curve)
        pvbp = self.get_pvbp(dates, discount_curve)
        px = self.nominal * (parrate - self.fixed_rate) * pvbp
        return px


class GeneralSwap(SwapRate, IRProduct):
    """
    Represents a generalized interest rate swap capable of modeling both fixed and floating rates,
    including exotic swaps and those with amortizing or accreting structures.

    Parameters
    ----------
    rate : MultiRate
        The underlying rate object, potentially combining several rates. Mathematically represented by :math:`R`.
    obsdates : pandas.DatetimeIndex, list of strings, pandas.Timestamp, or string
        Observation (fixing) dates for the floating rate. Mathematically represented by :math:`\\{t^1_j\\}_{j=0}^J`.
    curve : DiscountCurve
        The discount curve used to compute forward rates and discounts, denoted as :math:`\\tilde{P}`.
    effective_date : pandas.Timestamp or string
        The start date of the swap, :math:`T_\\alpha`, where floating and fixed legs commence.
    end_date : pandas.Timestamp or string
        The end date of the swap, :math:`T_\\beta`, where floating and fixed legs terminate.
    floating_freq : int
        Frequency of the floating leg payments in months, denoted as :math:`12 \\cdot (T_l^{\\text{float}} - T_{l-1}^{\\text{float}})`
        for :math:`l\\in\\{\\alpha^{\\text{float}}+1, \\ldots, \\beta^{\\text{float}}\\}` (it is assumed that this is constant).
        For instance, if floating_freq=6, then, approximately (depends on the day count convention), :math:`\\left(T_l^{\\text{float}}-T_{l-1}^{\\text{float}}\\right)=0.5`,
    fixed_freq : int
        Frequency of the fixed leg payments in months, similar to `floating_freq`.
    legs_calendars : list of DayCountCalendar or DayCountCalendar
        Day count conventions for each leg, represented as :math:`[\\text{DC}^{\\text{float}}, \\text{DC}^{\\text{fixed}}]`.
    nominal : ndarray or float, optional
        Nominal amounts for the swap, potentially variable over time. Default is 100.
    strike : ndarray or float, optional
        Strike rates for the fixed leg, potentially variable over time. Default is 0.

    Attributes
    ----------
    pay_dates : pandas.DatetimeIndex
        Combined payment dates for both legs of the swap, excluding the effective date. See below for detailed explanation.
    nominal : np.ndarray
        Array containing the nominal values applicable for each payment date. Mathematically represented, after possibly extending it to all the necessary dates, as  :math:`[N_j]_{j=0}^{J^p}`.
    strike : np.ndarray
        Array containing the strike rates applicable for each payment date.  Mathematically represented, after possibly extending it to all the necessary dates, as  :math:`[K_j]_{j=0}^{J^\\text{fixed}}`.

    Note
    -----
    The `GeneralSwap` class is designed to handle a wide array of swap contracts through its flexible structure,
    including but not limited to vanilla, amortizing, accreting, and exotic swaps.

    Given the tenor structure for each leg, :math:`\\mathcal{T}^f=\\{T_\\alpha,\\ldots, T^f_l,\\ldots,T_\\beta\\}` for :math:`f\\in\\{\\text{float, fixed}\\}` with:

    - *payment dates*: :math:`\\{t_j^{p,f}\\}_{j=0}^{J^f}\\subset \\mathcal{T^f},` payment dates for each leg. The union gives :math:`\\{t_j^{p}\\}_{j=0}^{J^p}\\subset \\mathcal{T},`
    - *fixing dates* (observation dates): :math:`\\{t_j^1\\}_{j=0}^{J}\\subset \\mathcal{T}^\\text{float},`

    where :math:`\\mathcal{T}:=\\bigcup_f\\mathcal{T}^f=\\mathcal{T}^\\text{float}\\cup\\mathcal{T}^\\text{fixed}`, and the underlying rate :math:`R`, the payoff (long position) is given by

    .. math::
        P_j(t_j^p) = f_j\\left(\\tilde{R}(t_j^p)\\right)=N_j\\cdot\\left(\\delta_{t^p_j, \\mathcal{T}^\\text{float}}\\left(\\tau^\\text{float}_j\\cdot\\tilde{R}(t_j^p)\\right)  - \\delta_{t^p_j, \\mathcal{T}^\\text{fixed}}\\left(\\tau^\\text{fixed}_j\\cdot K_j\\right)\\right),

    being :math:`N_j` the, possibly varying, nominal. Obviously,

    .. math::
        \\delta_{t^p_j, \\mathcal{T}^f} = \\sum_{l=0}^{L^f} \\delta_{t^p_j, {T}^f_l}.

    This function allows us to extend trivially arrays that are defined only for one leg to the union of both legs, as :math:`K`. With some *abuse of notation*, we will use the same names for both vectors. Also, we assume there is a bijection between :math:`\\{t^1_j\\}_{j=0}^J` and :math:`\\{t^{p,\\text{float}}_j\\}_{j=0}^J` so

    .. math::
        \\tilde{R}(t^{p,\\text{float}}_j):=R(t^{1}_j).

    The class `SwapRate` is used to define some attributes. Also, we assume that:

    - `pay_dates` :math:`\\rightarrow \\{t^p_j\\}_{j=0}^{J^p}=\\mathcal{T}\\backslash \\{T_\\alpha\\}`, which cover the "*in-arrears*" and "*fixed-in-advance*" (standard) cases. \
    Note that the first date is needed for :math:`\\tau`.

    Several remarks are in order:

    - The strike and nominal are not necessarily constant, so *amortizing* and *accreting* swaps are included.
    - :math:`R` can be the difference of two rates with :math:`K_j=0` using the `DifferenceRate` class, so *exotic* *swaps* are included. E.g., the difference between a CMS and Libor. \
      See Section 5.13 of Andersen and Piterbarg.
    - Payment dates and observation (fixing) dates are not necessarily the same, so the "*in-arrears*" and "*fixed-in-advance*" (standard) cases are included.
    - The function :math:`\\delta_{t^1_j, \\mathcal{T}^f}` is implemented in the code under the name of :py:meth:`delta_indexing<pricing.functions.delta_indexing>`.

    More specifically,

    - **(Vanilla) Libor Swap**: In this case, :math:`t^1_j=t^{p, \\text{ float}}_{j-1}` and the rate equal to the Libor rate.
    - **Libor-in-arrears Swap**: In this case, :math:`t^1_j=t^{p, \\text{ float}}_j` and the rate equal to the Libor rate.
    - **Constant Maturity Swap (CMS Swap)**: In this special case :math:`R` is a swap rate of the form :math:`S_{j,j+c}`, where :math:`c` is the constant maturity.
    - **Amortizing swaps**: :math:`N_j` is no longer constant, allowing for varying nominal amounts over the life of the swap.
    - **TARN(GeneralSwap)**: :math:`R` is an element of the ``TARNRate``, indicating a Targeted Amortization Redemption Note where the rate and payment schedules may vary based on predefined conditions.
    """

    def __init__(
        self,
        rate,
        obsdates,
        effective_date,
        end_date,
        floating_freq,
        fixed_freq,
        legs_calendars=None,
        nominal=100,
        strike=0,
        curve=None,
    ):
        SwapRate.__init__(
            self,
            curve,
            effective_date,
            end_date,
            floating_freq,
            fixed_freq,
            legs_calendars=legs_calendars,
        )  # This is for the payment dates
        Derivative.__init__(
            self,
            underlying=rate,
            obsdates=obsdates,
            pastmatters=True,
            calendar=self.floating_calendar,
            credit_curve=zero_credit_risk,
        )
        self.calendar = self.floating_calendar  # TODO: general case

        # Nominal and strikes as arrays
        if not isinstance(nominal, np.ndarray):
            self.nominal = np.array([nominal])
        else:
            self.nominal = nominal
        if not isinstance(strike, np.ndarray):
            self.strike = np.array([strike])
        else:
            self.strike = strike

        # Pay_dates from SwapRate, overwrite attribute
        self.pay_dates = self.total_dates[1:]

    def payoff(self, prices, dates_dic, n):
        """
        Calculate the payoff of the general swap based on prices and dates.

        Parameters
        ----------
        prices : np.ndarray
            Prices of the underlying. Mathematically represented by :math:`\\textrm{price}[i,j,k,l]=R_{i, t_k^0}^l(t^s_j)`. The first index of the price denotes the simulation,
            the second one the simulation dates, the third one the valuation dates and the last the l-th component. More details below.
        dates_dic : dict or pandas.Series
            Assigns an index to each date, effectively mapping :math:`t^1_j \\mapsto j'` such that :math:`t^1_j = t^s_{j'}`, being the latter the simulation dates.
            It serves to identify the simulation dates that correspond to the required observation dates. More details below.
        n : int
            The number of observation dates previous to a fixed observation date. Used in Monte Carlo engine for path-dependent products to get the value of the underlying for the previous `n` observation dates and simulate it for the new observation dates.

        Returns
        -------
        np.ndarray
            A 3-dimensional array of calculated payoffs for each observation. The shape is :math:`(|\\mathcal{I}|,|\\mathcal{J}|-n,|\\mathcal{K}|)` where :math:`|\\mathcal{I}|:= I+1`, :math:`|\\mathcal{J}|:= J+1` and :math:`|\\mathcal{K}|:=K+1`.

        Note
        -----
        Explicitly,

        - For the simulation dates :math:`\\{t^s_j\\}_{j=0}^{J'}` and the valuation dates :math:`\\{t^0_k\\}_{k=0}^{K}`, the *rate* associated with the i-th simulation of the `l`-th component\
          is a numpy.ndarray denoted by
        .. math::
            \\textrm{price}[i,j,k,l]=R_{i, t_k^0}^l(t^s_j)\\,

        - `dates_dic` is a dictionary (or pandas.Series) which assigns an index to each date. The observation dates :math:`\\{t^1_j\\}_{j=0}^{J}` satisfy :math:`\\{t^1_j\\}_{j=0}^{J}\\subseteq \\{t^s_{j'}\\}_{j'=0}^{J'}`,\
          where :math:`\\{t^s_{j'}\\}_{j'=0}^{J'}` are the simulation dates. That is to say, we generally simulate the prices for more dates than the ones we need for a particular product. In this sense,
          `dates_dic` can be understood as the map: :math:`t^s_{j'} \\mapsto j'`. In particular, :math:`t^1_j \\mapsto j'` such that :math:`t^1_j = t^s_{j'}\, `.

        - `n` is an integer indicating the number of observation dates previous to a fixed observation date. This is needed in the Monte Carlo engine for exotic (path-dependent) products. \
        For a given  `n` we get the value of the underlying for the previous `n` observation dates and simulate it for the new observation dates. For more details see the method `generate_paths_for_pricing` in *mc_engines.py*.

        The payoff function returns a numpy.ndarray of **3 dimensions** and shape :math:`(|\\mathcal{I}|,|\\mathcal{J}|-n,|\\mathcal{K}|)`, where :math:`|\\mathcal{I}|:= I+1`, :math:`|\\mathcal{J}|:= J+1` and :math:`|\\mathcal{K}|:=K+1` are the numbers of elements in the indexes :math:`i`, :math:`j` and :math:`k`. Note that we give the payoff excluding the previous observation dates. The output satisfies

        .. math::
            \\textrm{payoff}[i,j,k]=f_j(R_{i, t_k^0}(t^p_j))\\,

        where :math:`f_j(R)` was defined above.

        **Remark**: For every product, a long position is assumed. For a short position, the payoff would be :math:`P_T = -f(S)`.
        """
        
        payoffs_matrix = np.zeros(
            (prices.shape[0], prices.shape[1] - n, prices.shape[2])
        )
        indices = [dates_dic[date] for date in self.obsdates]
        prices = prices[
            :, indices, :
        ]  # Prices correspond to obsdates, not pay_dates, but see documentation, bijection with float_pày_dates
        prices = prices[:, :, :, 0]  # Only the first index
        dates_0 = self.floating_dates[1:]  # T_0 is not a payment date
        dates_1 = self.fixed_dates[1:]
        prices = afsfun.delta_indexing(
            prices, dates_0, dates_1, 0, second_index=True
        )  # By hypothesis, there is a bijection between dates_0 and obsdates
        taus_float = afsfun.delta_indexing(self.tau_floating, dates_0, dates_1, 0)
        taus_float = taus_float.reshape(
            (taus_float.size, 1)
        )  # For broadcasting considerations
        taus_fixed = afsfun.delta_indexing(self.tau_fixed, dates_0, dates_1, 1)
        taus_fixed = taus_fixed.reshape(
            (taus_fixed.size, 1)
        )  # For broadcasting considerations
        if self.strike.size > 1:  # If it is constant, it is not going to work
            strike = afsfun.delta_indexing(
                self.strike, self.floating_dates, self.fixed_dates, 1
            )
        else:
            strike = self.strike
        strikes = strike.reshape((strike.size, 1))  # For broadcasting considerations
        payoffs = taus_float * prices - taus_fixed * strikes
        indices = [ind - n for ind in indices]
        payoffs_matrix[:, indices] = payoffs

        nominal = self.nominal.reshape(self.nominal.size, 1)

        return nominal * payoffs_matrix


# ---------------------------------------------------------------------------------------------
# options
# ---------------------------------------------------------------------------------------------


class Swaption(IRProduct):
    """
    Represents a swaption, which is an option to enter into an interest rate swap.

    Parameters
    ----------
    swap_rate : pricing.ratecurves.SwapRate
        The underlying swap rate associated with the swaption. Mathematically represented as :math:`\\rightarrow [S_{\\alpha,\\beta},\\mathcal{A}]`, as a `SwapRate` (as in `MultiRate`).
    maturity : pandas.DatetimeIndex or str, optional
        The maturity date of the swaption. If not provided, it defaults to the ``effective_date`` of the :py:meth:`SwapRate <pricing.ratecurves.SwapRate>` .
        Mathematically represented as :math:`T_\\alpha`.

    Attributes
    ----------
    strike : float, optional
        The strike rate of the swaption. The rate at which the holder can enter into the swap. To be defined in child classes, None here. Mathematically represented as :math:`K`.
    kind : str, optional
        The type of swaption, either 'call' for a payer swaption or 'put' for a receiver swaption. To be defined in child classes, None here.
    nominal : float, optional
        The nominal amount of the swaption. To be defined in child classes, None here. Mathematically represented as :math:`N`.
    pastmatters : bool
        Indicates whether the past rates (path) matter for the pricing of the product. For European swaptions, this is `False`.

    Note
    -----
    If the swap has a tenor structure given by :math:`\\{T_\\alpha,\\ldots,T_\\beta\\}`, the payoff of a European payer swaption is given by, [Brigo and Mercurio, 2006] p.19 ss,

    .. math::
        N\\cdot \\mathcal{A}(T_\\alpha)(S_{\\alpha,\\beta}(T_\\alpha)-K)^+\\,,

    being :math:`S_{\\alpha,\\beta}` the swap rate and :math:`\\mathcal{A}` defined below. For a receiver swaption,

    .. math::
        N\\cdot \\mathcal{A}(T_\\alpha)(K-S_{\\alpha,\\beta}(T_\\alpha))^+\\,.

    In general,

    .. math::
        \\frac{f^{\\text{kind}}(S_{\\alpha,\\beta}(T_\\alpha), \\mathcal{A}(T_\\alpha))}{N\\cdot \\mathcal{A}(T_\\alpha)}:=\\left({S_{\\alpha,\\beta}(T_\\alpha)}-K\\right)\\mathbb{1}_{\\{S_{\\alpha,\\beta}(T_\\alpha)\\geq K\\}} -\\left({S_{\\alpha,\\beta}(T_\\alpha)}-K\\right)\\delta_{\\text{kind,put}}\\,.

    For

    - *Physical-settlement:* :math:`\\mathcal{A}(t):=A(t)`, the annuity.
    - *Cash-settlement:* :math:`\\mathcal{A}(t):=a\\left(S_{\\alpha,\\beta}(t)\\right)`, see p.206 of Andersen and Piterbarg.
    """

    def __init__(self, swap_rate, maturity):
        if maturity is None:
            maturity = swap_rate.effective_date
            self.maturity = maturity
        Derivative.__init__(
            self,
            underlying=swap_rate,
            obsdates=maturity,
            pastmatters=False,
            calendar=None,
            credit_curve=zero_credit_risk,
        )
        if pd.to_datetime(maturity) > swap_rate.effective_date:
            print("Maturity past swap's start date; object not created.")
            pass
        self.strike = None  # To be defined in child classes
        self.kind = None
        self.nominal = None
        self.pastmatters = False

    def payoff(self, prices, dates_dic, n):
        """
        Calculate the payoff of the swaption based on prices and dates.

        Parameters
        ----------
        prices : np.ndarray
            Prices of the underlying. Mathematically represented by :math:`\\textrm{price}[i,j,k,l]=R_{i, t_k^0}^l(t^s_j)`. The first index of the price denotes the simulation,
            the second one the simulation dates, the third one the valuation dates and the last the l-th component. More details below.
        dates_dic : dict or pandas.Series
            Assigns an index to each date, effectively mapping :math:`t^1_j \\mapsto j'` such that :math:`t^1_j = t^s_{j'}`, being the latter the simulation dates.
            It serves to identify the simulation dates that correspond to the required observation dates. More details below.
        n : int
            The number of observation dates previous to a fixed observation date. Used in Monte Carlo engine for path-dependent products to get the value of the underlying for the previous `n` observation dates and simulate it for the new observation dates.

        Returns
        -------
        np.ndarray
            A 3-dimensional array of calculated payoffs for each observation. The shape is :math:`(|\\mathcal{I}|,|\\mathcal{J}|-n,|\\mathcal{K}|)` where :math:`|\\mathcal{I}|:= I+1`, :math:`|\\mathcal{J}|:= J+1` and :math:`|\\mathcal{K}|:=K+1`.

        Note
        -----

        We analyze in detail the input and output of the `payoff` method:

        - For the observation dates :math:`\\{t^1_j\\}_{j=0}^{J}` and the valuation dates :math:`\\{t^0_k\\}_{k=0}^{K}`, the *swap rate* associated with the :math:`i`-th simulation of the :math:`l`-th component \
          is a numpy.ndarray denoted by\

        .. math::
             \\text{price}[i,j,k,l]=S_{i, t_k^0}^l(t^1_j)\,,\

        where, for the sake of simplicity, we have removed the subscripts :math:`\\alpha,\\beta`. Recall that :math:`l=0` corresponds to :math:`S_{\\alpha,\\beta}` and :math:`l=1` with the "rate" :math:`\\mathcal{A}`.

        - `dates_dic` is a dictionary (or pandas.Series) which assigns an index to each date. The observation dates :math:`\\{t^1_j\\}_{j=0}^{J}` satisfy :math:`\\{t^1_j\\}_{j=0}^{J}\\subseteq \\{\\tau_{j'}\\}_{j'=0}^{J'}`, where :math:`\\{\\tau_{j'}\\}_{j'=0}^{J'}` are the simulation dates. This means we generally simulate the prices for more dates than the ones needed for a particular product. Thus, `dates_dic` maps :math:`\\tau_{j'}` to :math:`j'`. Specifically, :math:`t^1_j \\mapsto j'` such that :math:`t^1_j = \\tau_{j'}`.

        - `n` is an integer indicating the number of observation dates before a fixed observation date. This is necessary in the Monte Carlo engine for exotic (path-dependent) products. For a given `n`, we obtain the value of the underlying for the previous `n` observation dates and simulate it for the new observation dates. For more details, see the method `generate_paths_for_pricing` in *mc_engines.py*.

        The payoff function returns a numpy.ndarray of **3 dimensions** with shape :math:`(|\\mathcal{I}|,|\\mathcal{J}|-n,|\\mathcal{K}|)`, where :math:`|\\mathcal{I}|:= I+1`, :math:`|\\mathcal{J}|:= J+1` and :math:`|\\mathcal{K}|:=K+1` are the number of elements in the indexes :math:`i`, :math:`j`, and :math:`k`. Note that the payoff is provided excluding the previous observation dates. The output satisfies

        .. math::
            \\text{payoff}[i,j,k]=f\\left(S_{i, t_k^0}^{0}(t^1_j), S_{i, t_k^0}^{1}(t^1_j)\\right)\\times\\delta_{j, j_T-n}\,,

        where :math:`\\delta_{i,j}` denotes the Kronecker delta, and :math:`j_T` satisfies :math:`t^1_{j_T} = T`.

        Finally, note that we assume a long position for every product. For a short position, the payoff would be :math:`P_T = -f(S)`.

        """
        payoffs_matrix = np.zeros(
            (prices.shape[0], prices.shape[1] - n, prices.shape[2])
        )
        index = dates_dic[self.maturity]
        prices = prices[:, index]
        payoffs = prices[:, :, 1] * (
            (prices[:, :, 0] - self.strike) * (prices[:, :, 0] >= self.strike)
            - (prices[:, :, 0] - self.strike) * (self.kind == "put")
        )
        payoffs_matrix[:, index - n] = payoffs
        return self.nominal * payoffs_matrix


class PayersSwaption(Swaption, Call):
    # MRO is being used, payoff from Swaption, not Call
    """
    Represents a payer's swaption, which gives the holder the right to enter into a fixed-for-floating interest rate swap as the payer of the fixed rate.

    Parameters
    ----------

    swap_rate : pricing.ratecurves.SwapRate
        The underlying swap rate associated with the swaption.
    strike : float
        The strike rate at which the option holder can enter the swap.
    calendar : tuple or data.calendars.DayCountCalendar
        The calendar used for date calculations.
    maturity : pandas.DatetimeIndex or str, optional
        The maturity date of the swaption. If not provided, it defaults to the ``effective_date`` of the swap rate.
    nominal : float, optional
        The nominal amount of the swaption. Default is 100.
    alpha : float, optional
        Alpha parameter used in the option's payoff calculation. Default is 0.
    """

    def __init__(
        self, swap_rate, strike, calendar, maturity=None, nominal=100, alpha=0
    ):
        if maturity is None:
            maturity = swap_rate.effective_date
        if pd.to_datetime(maturity) > swap_rate.effective_date:
            print("Maturity past swap's start date; object not created.")
            pass
        Call.__init__(
            self, swap_rate, strike, maturity, calendar, nominal=nominal, alpha=alpha
        )

    def get_px(
        self, dates, discount_curve, alpha=0, short_rate=None, no_sims=None
    ):  # The last arguments are not needed for this analytic price
        """
        Calculate the price of the payer swaption using analytic methods.

        Parameters
        ----------
        dates : pandas.DatetimeIndex, list of strings, pandas.Timestamp, or string
            Dates at which to compute the price. It can be a single date (as a pandas.Timestamp or its string representation) or an array-like object
            (as a pandas.DatetimeIndex or a list of its string representation) containing dates.
        discount_curve : pricing.discount_curves.DiscountCurve , default = None
            Gives the discounts for the computation of the strikes if these are not given.
        alpha : float, optional
            Adjustment factor for the swaption payoff. Default is 0.
        short_rate : pricing.ir_models.ShortRateModel
            The short-term interest rate.
        no_sims : int or None, optional
            Number of simulations. Not used for this analytic price calculation.

        Returns
        -------
        pandas.Series
            The calculated price of the payer swaption.

        Note
        -----
        This method calculates the price of the payer swaption using analytic methods.
        """
        pvbp = self.underlying.get_pvbp(dates, discount_curve)
        price = Call.get_px(self, dates, discount_curve)
        if issubclass(type(self.underlying), LognormalSwapRate):
            # for lognormal, option price is per nominal amount; need to multiply by strike
            price = (self.strike + alpha) * pvbp * price
        elif issubclass(type(self.underlying), NormalSwapRate):
            price = pvbp * price
        return price

    # go over the other Call methods


class ReceiversSwaption(Put):
    """
    Represents a receiver's swaption.

    Parameters
    ----------
    swap_rate : pricing.ratecurves.SwapRate
        The underlying swap rate associated with the swaption.
    strike : float
        The strike rate at which the option holder can enter the swap.
    calendar : tuple or data.calendars.DayCountCalendar
        The calendar used for date calculations.
    maturity : pandas.DatetimeIndex or str, optional
        The maturity date of the swaption. If not provided, it defaults to the ``effective_date`` of the swap rate.
    nominal : float, optional
        The nominal amount of the swaption. Default is 100.
    alpha : float, optional
        Alpha parameter used in the option's payoff calculation. Default is 0.
    """

    def __init__(
        self, swap_rate, strike, calendar, maturity=None, nominal=100, alpha=0
    ):
        if maturity is None:
            maturity = swap_rate.effective_date
        if pd.to_datetime(maturity) > swap_rate.effective_date:
            print("Maturity past swap's start date; object not created.")
            pass
        Put.__init__(
            self, swap_rate, strike, maturity, calendar, nominal=nominal, alpha=alpha
        )

    def get_px(self, dates, discount_curve, alpha=0, p=0):
        """
        Calculate the price of the payer swaption using analytic methods.

        Parameters
        ----------
        dates : pandas.DatetimeIndex, list of strings, pandas.Timestamp, or string
            Dates at which to compute the price. It can be a single date (as a pandas.Timestamp or its string representation) or an array-like object
            (as a pandas.DatetimeIndex or a list of its string representation) containing dates.
        discount_curve : pricing.discount_curves.DiscountCurve , default = None
            Gives the discounts for the computation of the strikes if these are not given.
        alpha : float, optional
            Adjustment factor for the swaption payoff. Default is 0.
        p : int
            Probability. Not used for this analytic price calculation.

        Returns
        -------
        pandas.Series
            The calculated price of the payer swaption.

        Note
        -----
        This method calculates the price of the payer swaption using analytic methods.

        """
        pvbp = self.underlying.get_pvbp(dates, discount_curve)
        price = Put.get_px(self, dates, discount_curve)
        if issubclass(type(self.underlying), LognormalSwapRate):
            # for lognormal, option price is per nominal amount; need to multiply by strike
            price = (self.strike + alpha) * pvbp * price
        elif issubclass(type(self.underlying), NormalSwapRate):
            price = pvbp * price
        return price

    # go over the other Put methods


class CapFloor(IRProduct):
    """
    This class represents a Cap or Floor interest rate product with given parameters.

    Parameters
    ----------
    rate : pricing.ratecurves.Rate
        The underlying rate index for the Cap or Floor.
    obsdates : pandas.DatetimeIndex
        Observation dates for the swap, mathematically represented as :math:`\\{t^1_j\\}_{j=0}^J`.
    kind : str
        Indicates whether the product is a 'cap' or 'floor', corresponding to the type of option within the interest rate market.
    calendar : data.calendars.DayCountCalendar
        Utilized for day count conventions and date calculations, with :math:`\\tau_j=\\text{DCC}(t_{j-1}^p,t_j^p)` representing the day count fraction between payment dates.
    strike : float or numpy.ndarray
        The strike rate(s) for the Cap or Floor, mathematically represented as :math:`[K_j]_{j=0}^J`.
    tenor_dates : pandas.DatetimeIndex
        The tenor dates of the product, mathematically represented as :math:`\\{T_j\\}_{j=0}^{J+1}`. It's derived from the union of ``obsdates`` and ``pay_dates`` if not directly specified.
    pay_dates : pandas.DatetimeIndex
        Payment dates for the Cap or Floor, mathematically represented as :math:`\\{t^p_j\\}_{j=0}^J`. By default, it's assumed :math:`t^1_j=t^p_j`.
    nominal : float or numpy.ndarray

    Attributes
    ----------
    underlying : pricing.ratecurves.MultiRate
        The underlying rate index for the Cap or Floor, mathematically represented as :math:`R`.
    obsdates : pandas.DatetimeIndex
        Observation dates for the swap, mathematically represented as :math:`\\{t^1_j\\}_{j=0}^J`.
    kind : str
        Indicates whether the product is a 'cap' or 'floor', corresponding to the type of option within the interest rate market.
    calendar : data.calendars.DayCountCalendar
        Utilized for day count conventions and date calculations, with :math:`\\tau_j=\\text{DCC}(t_{j-1}^p,t_j^p)` representing the day count fraction between payment dates.
    strike : float or numpy.ndarray
        The strike rate(s) for the Cap or Floor, mathematically represented as :math:`[K_j]_{j=0}^J`.
    tenor_dates : pandas.DatetimeIndex
        The tenor dates of the product, mathematically represented as :math:`\\{T_j\\}_{j=0}^{J+1}`. It's derived from the union of ``obsdates`` and ``pay_dates`` if not directly specified.
    pay_dates : pandas.DatetimeIndex
        Payment dates for the Cap or Floor, mathematically represented as :math:`\\{t^p_j\\}_{j=0}^J`. By default, it's assumed :math:`t^1_j=t^p_j`.
    nominal : float or numpy.ndarray

    Raises
    ------
    AttributeError
        If both ``tenor_dates`` and ``pay_dates`` are not specified.

    Note
    -----
    Given a tenor structure :math:`\\{T_\\alpha,\\ldots,T_\\beta\\}` with :math:`\\{t_j^p\\}_{j=0}^J`, :math:`\\{t_j^1\\}_{j=0}^J\\subset \\{T_\\alpha,\\ldots,T_\\beta\\}` being the payment and fixing dates, respectively, and the underlying rate :math:`R`, for a long position, the payoff is given by

    .. math::
        P_j(t_j^p) = f_j\\left(R(t_j^1)\\right)=N\\cdot\\tau_j\\cdot\\left(\\left( R(t_j^1)  - K_j\\right)_+ - \\left( R(t_j^1)  - K_j\\right)\\cdot\\delta_{\\text{kind, floor}}\\right).

    being :math:`N` the nominal, :math:`\\mathcal{T}^f` the tenor structure of the :math:`f`-leg of the swap, either float or fixed.

    The parameters and their mathematical representations are as follows:

    - ``underlying`` :math:`\\rightarrow R`, the underlying rate of the swap.
    - ``obsdates`` :math:`\\rightarrow \\{t^1_j\\}_{j=0}^J`, the observation (fixing) dates.
    - ``calendar`` :math:`\\rightarrow` the calendar used for computations, with :math:`\\tau_j=\\text{DCC}(t_{j-1}^p,t_j^p)` representing the day count fraction between payment dates.
    - ``strike`` :math:`\\rightarrow [K_j]_{j=0}^J\\,`, the strike rates for each period.
    - ``nominal`` :math:`\\rightarrow [N_j]_{j=0}^J\\,`, the nominal amounts for each period.
    - ``pay_dates`` :math:`\\rightarrow \\{t^p_j\\}_{j=0}^J`, by default :math:`t^1_j=t^p_j`, the payment dates.
    - ``tenor_dates`` :math:`\\rightarrow \\{T_j\\}_{j=0}^{J+1}`, if it is not specified, it is :math:`\\{t^1_j\\}_{j=0}^J\\cup\\{t^p_j\\}_{j=0}^J`. Either ``pay_dates`` or ``tenor_dates`` must be specified.
    """

    def __init__(
        self,
        rate,
        obsdates,
        kind,
        calendar,
        strike=0,
        tenor_dates=None,
        pay_dates=None,
        nominal=100,
    ):
        Derivative.__init__(
            self,
            underlying=rate,
            obsdates=obsdates,
            pastmatters=True,
            calendar=calendar,
            pay_dates=pay_dates,
            credit_curve=zero_credit_risk,
        )

        if not isinstance(nominal, np.ndarray):
            self.nominal = np.array([nominal])
        else:
            self.nominal = nominal
        if not isinstance(strike, np.ndarray):
            self.strike = np.array([strike])
        else:
            self.strike = strike
        self.kind = kind
        if tenor_dates is None and pay_dates is None:
            raise AttributeError("Either tenor_dates or pay_dates must be specified.")
        if tenor_dates is None:
            self.tenor_dates = self.obsdates.union(self.pay_dates).sort_values()
        else:
            self.tenor_dates = pd.to_datetime(tenor_dates).sort_values()
        self.taus = self.calendar.interval(self.tenor_dates[:-1], self.tenor_dates[1:])

    def payoff(self, prices, dates_dic, n):
        """
        Calculate the payoff of the cap or floor contract.

        For the conventions and details on the price and payoff arrays, refer to the :py:meth:`GeneralSwap.payoff<GeneralSwap.payoff>` or :py:meth:`Swaption.payoff<Swaption.payoff>` methods.
        The payoff function in this case is defined as

        .. math::
            \\textrm{payoff}[i,j,k] = f_j(R_{i, t_k^0}(t^1_j))\\,,

        where :math:`f_j(R)` was defined above. This function computes the payoff for either a cap or floor option based on the difference between the observed rate and the strike rate.

        Parameters
        ----------
        prices : np.ndarray
            Prices of the underlying rate. The first index of the price array denotes the simulation number, the second one corresponds to the (full) observation dates, and the third one to the valuation dates.

        dates_dic : dict
            A dictionary (or pandas.Series) which assigns an index to each date. This index is used to select the relevant observation dates from the `prices` array.

        n : int
            The number of observation dates prior to a given observation date. This parameter is used to adjust the calculation of payoffs for path-dependent features in the cap or floor product.

        Returns
        -------
        np.ndarray
            A 3-dimensional array of calculated payoffs for each observation. The dimensions correspond to the number of simulations, the adjusted number of observation dates (excluding the previous `n` dates), and the number of valuation dates.

        Note
        -----
        The payoff calculation takes into account the type of product (cap or floor) and computes the payoff accordingly. For a cap, the payoff is positive when the rate exceeds the strike, and for a floor, the payoff is positive when the strike exceeds the rate.
        """
        payoffs_matrix = np.zeros(
            (prices.shape[0], prices.shape[1] - n, prices.shape[2])
        )
        indices = [dates_dic[date] for date in self.obsdates]
        prices = prices[:, indices]
        prices = prices[..., 0]  # Do not mix advanced and basic slicing
        taus = self.taus.reshape((self.taus.size, 1))  # For broadcasting considerations
        strikes = self.strike.reshape(
            (self.strike.size, 1)
        )  # For broadcasting considerations
        payoffs = taus * (
            (prices - strikes) * (prices >= strikes)
            - (prices - strikes) * (self.kind == "floor")
        )
        indices = [ind - n for ind in indices]
        payoffs_matrix[:, indices] = payoffs

        return self.nominal * payoffs_matrix

    def payoff_old(self, prices, dates_dic, n):
        """
        Calculate the old version of payoff for the cap or floor contract.

        Parameters
        ----------
        prices: np.ndarray
            Prices of the underlying. The first index of price denotes the simulation, the second one the observation dates and the third one the valuation dates.
        dates_dic: dict
            Dictionary (or pandas.Series) which assigns an index to each date.
        n: int
            Number of observation dates previous to a fixed observation date.

        Returns
        -------
        np.ndarray
             3 dimensional array of calculated payoffs for each observation.
        """
        payoffs_matrix = np.zeros(
            (prices.shape[0], prices.shape[1] - n, prices.shape[2])
        )
        indices = [dates_dic[date] for date in self.obsdates]
        prices = prices[:, indices]
        prices_q = prices[..., 0] / prices[..., 1]
        payoffs = (prices[..., 0] - self.strike * prices[..., 1]) * (
            prices_q >= self.strike
        )
        indices = [ind - n for ind in indices]
        payoffs_matrix[:, indices] = payoffs

        return self.nominal * payoffs_matrix


# ---------------------------------------------------------------------------------------------
# CMS
# ---------------------------------------------------------------------------------------------
class CMSSpreadCoupon(IRProduct):
    """
        Represents a CMS (Constant Maturity Swap) spread coupon, which pays the spread over a CMS rate.
        The CMS spread coupon pays the difference between the rates of two CMS with different tenors,
        adjusted by a gearing factor, and added to a spread. It is subject to a cap and floor which
        limit the maximum and minimum coupon rates paid.See Section 5.13.3, Vol. I, of [Andersen and Piterbarg].
        This is a `toy product` where we have used a forward structure, i.e., just one payment.

        Parameters
        ----------
        cms_rates : ratecurves.MultiRate
            Difference of CMS rates. These are CMS rates
            covering different periods, typically denoted by :math:`S_{n,a}(T_n)` and :math:`S_{n,b}(T_n)` in literature.
        maturity : str or pandas.Timestamp
            The maturity date of the coupon when the CMS spread is observed and the coupon is paid.
        spread : float
            The spread added to the CMS rate.
        gearing : float
            The gearing factor applied to the difference between the long and short CMS rates before adding the spread.
        cap : float
            The maximum rate that the coupon will pay, also known as the cap rate.
        floor : float
            The minimum rate that the coupon will pay, also known as the floor rate.
        calendar : calendars.DayCountCalendar
            The calendar used for date calculations and determining the day count fraction.
        pay_dates : pandas.DatetimeIndex, list of strings, pandas.Timestamp, or string, optional
            Payment dates for the coupon. If not specified, it defaults to the observation dates, here maturity date.
        nominal : float, optional
            The nominal amount of the coupon. Default is 100.

        Attributes
        ----------
        spread : float
            The spread paid over the CMS rate.
        gearing : float
            The gearing factor applied to the spread.
        cap : float
            The cap rate for the coupon.
        floor : float
            The floor rate for the coupon.
        nominal : float
            The nominal amount of the coupon.

        Note
        -----
        The CMS Spread Coupon pays a coupon based on the spread between two CMS rates. Specifically, the coupon for a payment date :math:`T_n` is calculated as:

        .. math::
            C(T_n) = \\text{max}(\\text{min}(g \\times (S_{n,a}(T_n) - S_{n,b}(T_n)) + s, c), f).

        Here, :math:`S_{n,a}(T_n)` and :math:`S_{n,b}(T_n)` represent two CMS rates fixing on :math:`T_n`, covering periods :math:`a` and :math:`b` respectively.
        The coupon :math:`C_n` is then determined by the gearing factor :math:`g`, the spread :math:`s`, and is subject to a cap :math:`c` and floor :math:`f`.
        This structure creates a coupon that reflects the differential movement of two segments of the yield curve over the accrual period,
        adjusted by the contract terms of spread, gearing, cap, and floor.

        References
        -------------
        [Andersen and Piterbarg, 2010] Andersen, L. B. G., and Piterbarg, V.V. (2010). Interest Rate Modeling (Vol. I).

        """
    def __init__(
        self,
        cms_rates,
        maturity,
        spread,
        gearing,
        cap,
        floor,
        calendar,
        pay_dates=None,
        nominal=100,
    ):
        # Following Section 5.13.3, Vol. I, of Andersen and Piterbarg
        maturity = pd.to_datetime(maturity)
        maturity = pd.DatetimeIndex([maturity])
        Derivative.__init__(
            self,
            underlying=cms_rates,
            obsdates=maturity,
            pastmatters=False,
            pay_dates=pay_dates,
            calendar=calendar,
            credit_curve=zero_credit_risk,
        )
        self.spread = spread
        self.gearing = gearing
        self.spread = spread
        self.cap = cap
        self.floor = floor
        self.nominal = nominal

    def payoff(self, prices, dates_dic, n):
        """
        Calculate the payoff of the CMS spread coupon based on prices and dates. The payoff calculation follows the CMS spread coupon formula defined in Section 5.13.3
        of [Andersen and Piterbarg, 2010].

        Parameters
        ----------
        prices : np.ndarray
            Prices of the underlying rates. The first index of the price array denotes the simulation number, the second one corresponds to the (full) observation dates, and the third one to the valuation dates.

        dates_dic : dict
            A dictionary (or pandas.Series) which assigns an index to each date. This index is used to select the relevant observation dates from the `prices` array for `R` and `X`.

        n : int
            The number of observation dates prior to a given observation date. This parameter is used to adjust the calculation of payoffs for path-dependent features in the range accrual product.

        Returns
        -------
        np.ndarray
            A 3-dimensional array of calculated payoffs for each observation. The dimensions correspond to the number of simulations, the adjusted number of observation dates (excluding the previous `n` dates), and the number of valuation dates.

        Note
        -----
        For the conventions and details on the price and payoff arrays, refer to the :py:meth:`GeneralSwap.payoff<GeneralSwap.payoff>` or :py:meth:`Swaption.payoff<Swaption.payoff>` methods.
        The payoff function in this case is defined as

        .. math::
            \\textrm{payoff}[i,j,k] = \\delta_{j,j_T}(C_{i, t_k^0}(t^1_j) -K)\\,,

        where :math:`C`, defined above, is the coupon and :math:`j_T` represents the index associated to the maturity date.

        References
        -------------
        [Andersen and Piterbarg, 2010] Andersen, L. B. G., and Piterbarg, V.V. (2010). Interest Rate Modeling (Vol. I).
        """
        # Following Section 5.13.3, Vol. I, of Andersen and Piterbarg. Here a forward structure (structured.Forward with K=0, see class below)
        payoffs_matrix = np.zeros(
            (prices.shape[0], prices.shape[1] - n, prices.shape[2])
        )
        index = dates_dic[self.maturity]
        price = prices[:, index, :, 0]
        cap_array = self.gearing * price + self.spread
        payoffs = np.maximum(np.minimum(cap_array, self.cap), self.floor)
        payoffs_matrix[:, index - n] = payoffs
        return self.nominal * payoffs_matrix


# ---------------------------------------------------------------------------------------------
# Range Accruals
# ---------------------------------------------------------------------------------------------


class RangeAccrual(IRProduct):
    """
    Represents a range accrual product, which accrues interest based on the underlying rate falling within a specified range on given reference dates.

    Parameters
    ----------
    rates : pricing.ratecurves.MultiRate
        The underlying rates for the accrual. Mathematically represented by :math:`[R, X]` as a `MultiRate`.
    obsdates : pandas.DatetimeIndex, list of strings, pandas.Timestamp, or string
        Observation dates for the accrual. These are the dates on which the underlying rate :math:`X` is observed to determine if it falls within the specified range.
    pay_dates : pandas.DatetimeIndex, list of strings, pandas.Timestamp, or string
        Payment dates for the accrual. These are the dates on which the accrual pays out if the conditions are met.
    low : float
        The lower bound of the interest rate range. Mathematically represented by :math:`l`.
    up : float
        The upper bound of the interest rate range. Mathematically represented by :math:`u`.
    calendar : data.calendars.DayCountCalendar
        The calendar used for date calculations. Used to compute :math:`\\tau_j=\\text{DCC}(t_{j-1}^p,t_j^p)`.
    nominal : float, optional
        The nominal amount of the accrual. Default is 100. Mathematically represented by :math:`N`.
    border : bool, optional
        Indicates whether the rate range includes the upper boundary. Default is ``False``. True if the right end of the interval is included, :math:`\\mid` equals :math:`]`, and False otherwise.

    Attributes
    ----------
    proper_pay_dates : pandas.DatetimeIndex
        Sorted payment dates derived from `pay_dates`. These are the dates on which accrual pays if conditions are met.
    reference_dates_dic : dict
        A dictionary mapping each payment date to its corresponding set of reference dates. Mathematically represented as :math:`\\{t^p_n : \\{t^1_{j}\\}_{j\\in\\mathcal{J}_n} \\}_{n=0}^N`, where :math:`\\mathcal{J}_n:=\\left\\{j\\in\\{0,\\ldots, J\\}\\,\\mid\\, t^1_j\\in[t^p_n, t^p_{n+1}\\mid \\right\\}`.
    nominal : float
        The nominal amount for the accrual. Default is 100.
    low : float
        The lower bound of the interest rate range.
    up : float
        The upper bound of the interest rate range.
    border : bool
        Indicates whether the rate range includes the upper boundary. Default is ``False``.

    Note
    -----
    Given two rates :math:`R, X`, a set of payment dates :math:`\\{t^p_n\\}_{n=0}^N`, a set of reference dates :math:`\\{t^1_{j}\\}_{j=0}^J`, and an interval :math:`[l, u]`, the payoff for a long position is given by:

    .. math::
        P_n(t_{n+1}^p) = f_n\\left(R(t_n^p), \\left(X(t^1_{j})\\right)_{j\\in\\mathcal{J}_n}\\right)=N\\cdot R(t_n^p)\\times \\frac{\\#\\{j\\in\\mathcal{J}_n~\\mid~X(t_j^1)\\in[l,u]\\}}{\\# \\mathcal{J}_n}.

    being :math:`\\mathcal{J}_n:=\\left\\{j\\in\\{0,\\ldots, J\\}\\,\\mid\\, t^1_j\\in[t^p_n, t^p_{n+1}\\mid \\right\\}`, being :math:`\\mid` either :math:`]` or :math:`)`.
    """

    def __init__(
        self, rates, obsdates, pay_dates, low, up, calendar, nominal=100, border=False
    ):
        Derivative.__init__(
            self,
            underlying=rates,
            obsdates=obsdates,
            pastmatters=True,
            calendar=calendar,
            pay_dates=None,
            credit_curve=zero_credit_risk,
        )

        self.proper_pay_dates = pd.to_datetime(pay_dates).sort_values()
        # self.pay_dates = self.obsdates  # TODO: delete. This is implicit as we are saying that pay_dates=None in Derivative.
        ref_dic = {}
        for i, pay_date in enumerate(self.proper_pay_dates):
            if i:
                dates_le_temp = self.obsdates[
                    self.obsdates >= self.proper_pay_dates[i - 1]
                ]
            else:
                dates_le_temp = self.obsdates
            if border:
                ref_dates_temp = dates_le_temp[dates_le_temp <= pay_date]
            else:
                ref_dates_temp = dates_le_temp[dates_le_temp < pay_date]

            ref_dic[pay_date] = ref_dates_temp
        self.reference_dates_dic = ref_dic

        self.nominal = nominal
        self.low = low
        self.up = up

    def payoff(self, prices, dates_dic, n):
        """
        Calculate the payoff of the range accrual contract. The payoff calculation for a range accrual considers the rate `R` observed at payment dates and the reference rate :math:`X` observed at reference dates.
        It calculates the proportion of reference dates where `X` falls within the specified range :math:`[l, u]` and multiplies it by the rate :math:`R` to determine the accrual for each payment date.

        Parameters
        ----------
        prices : np.ndarray
            Prices of the underlying rates. The first index of the price array denotes the simulation number, the second one corresponds to the (full) observation dates, and the third one to the valuation dates.

        dates_dic : dict
            A dictionary (or pandas.Series) which assigns an index to each date. This index is used to select the relevant observation dates from the `prices` array for `R` and `X`.

        n : int
            The number of observation dates prior to a given observation date. This parameter is used to adjust the calculation of payoffs for path-dependent features in the range accrual product.

        Returns
        -------
        np.ndarray
            A 3-dimensional array of calculated payoffs for each observation. The dimensions correspond to the number of simulations, the adjusted number of observation dates (excluding the previous `n` dates), and the number of valuation dates.

        Note
        -----
        For the conventions and details on the price and payoff arrays, refer to the :py:meth:`GeneralSwap.payoff<GeneralSwap.payoff>` or :py:meth:`Swaption.payoff<Swaption.payoff>` methods.
        The payoff function in this case is defined as

        .. math::
            \\textrm{payoff}[i,j,k] = f_j\\left(R_{i, t_k^0}(t^1_j), \\left(X_{i, t_k^0}(t^1_{j'})\\right)_{j'\\in\\mathcal{J}_n}\\right)\\,,

        where :math:`f_j(R, X)` computes the fraction of observation dates within the specified range :math:`[l, u]` that influences the payoff, based on the observed rates :math:`R` and
        reference rates :math:`X`.
        """
        payoffs_matrix = np.zeros(
            (prices.shape[0], prices.shape[1] - n, prices.shape[2])
        )
        indices = [dates_dic[date] for date in self.obsdates]
        indices_pay = [dates_dic[date] for date in self.proper_pay_dates]
        indices_dic = {}
        for date in self.proper_pay_dates:
            indices_dic[date] = [
                dates_dic[date_ref] for date_ref in self.reference_dates_dic[date]
            ]
        prices_ind = prices[:, indices]
        prices_pay = prices[:, indices_pay]
        shape_pay = (prices.shape[0], self.proper_pay_dates.size, prices.shape[2])
        reference_payoff = np.full(shape_pay, np.nan)
        for ind, date in enumerate(self.proper_pay_dates):
            prices_temp = prices_ind[:, indices_dic[date]]
            prices_temp = prices_temp[
                ..., 1
            ]  # Do not mix basic and advance slicing. The reference rate is the second one
            boolean_temp = (prices_temp >= self.low) * (prices_temp <= self.up)
            reference_payoff[:, ind] = np.sum(boolean_temp, axis=1) / len(
                indices_dic[date]
            )
        payoffs = prices_pay[..., 0] * reference_payoff
        indices_pay = [ind - n for ind in indices_pay]
        payoffs_matrix[:, indices_pay] = payoffs

        return self.nominal * payoffs_matrix


# ----------------------------------------------------------------------------------------------
# "Child" products
# ----------------------------------------------------------------------------------------------
class TARN(GeneralSwap, TARNRate):  # Some classes added for the inheritance diagram
    """
    TARN class product derived from ``GeneralSwap`` and ``TARNRate`` for representing Target Accrual Redemption Note (TARN) contracts.

    Parameters
    ----------
    rate : Rate
        The underlying rate process used in the TARN mechanism. Mathematically represented by :math:`R`.
    dates : List[str] or pd.DatetimeIndex
        The dates for which the TARN rate process is generated. Mathematically represented by :math:`(T_j)_j`, the sequence of dates.
    calendar : data.calendars.DayCountCalendar
        The calendar used for computing day count fractions between dates, denoted by :math:`\\text{DC}`, thus :math:`\\tau_i = \\text{DC}(T_{i-1}, T_i)` represents the time intervals between consecutive dates :math:`T_{i-1}` and :math:`T_i`.
    barrier : float
        The barrier level for the up-and-out knockout condition. Mathematically represented by :math:`B`.
    obsdates : pandas.DatetimeIndex, list of strings, pandas.Timestamp, or string
        Observation (fixing) dates for the floating rate. Mathematically represented by :math:`\\{t^1_j\\}_{j=0}^J`.
    curve : DiscountCurve
        The discount curve used to compute forward rates and discounts, denoted as :math:`\\tilde{P}`.
    effective_date : pandas.Timestamp or string
        The start date of the swap, :math:`T_\\alpha`, where floating and fixed legs commence.
    end_date : pandas.Timestamp or string
        The end date of the swap, :math:`T_\\beta`, where floating and fixed legs terminate.
    floating_freq : int
        Frequency of the floating leg payments in months, denoted as :math:`12 \\cdot (T_l^{\\text{float}} - T_{l-1}^{\\text{float}})`
        for :math:`l\\in\\{\\alpha^{\\text{float}}+1, \\ldots, \\beta^{\\text{float}}\\}` (it is assumed that this is constant).
        For instance, if floating_freq=6, then, approximately (depends on the day count convention), :math:`\\left(T_l^{\\text{float}}-T_{l-1}^{\\text{float}}\\right)=0.5`,
    fixed_freq : int
        Frequency of the fixed leg payments in months, similar to `floating_freq`.
    legs_calendars : list of DayCountCalendar or DayCountCalendar
        Day count conventions for each leg, represented as :math:`[\\text{DC}^{\\text{float}}, \\text{DC}^{\\text{fixed}}]`.
    nominal : ndarray or float, optional
        Nominal amounts for the swap, potentially variable over time. Default is 100.
    strike : ndarray or float, optional
        Strike rates for the fixed leg, potentially variable over time. Default is 0.

    Note
    -----
    This class represents a Target Accrual Redemption Note (TARN) contract that pays based on the underlying rate and a barrier level. It is a :py:meth:`swap<irproducts.GeneralSwap>`
    in which the rate it is given by a :py:meth:`TARNRate<ratecurves.TARNRate>`. That is,the payoff function returns a numpy.ndarray of **3 dimensions** and shape :math:`(|\\mathcal{I}|,|\\mathcal{J}|-n,|\\mathcal{K}|)`, where :math:`|\\mathcal{I}|:= I+1`, :math:`|\\mathcal{J}|:= J+1` and :math:`|\\mathcal{K}|:=K+1` are the numbers of elements in the indexes :math:`i`, :math:`j` and :math:`k`. Note that we give the payoff excluding the previous observation dates. The output satisfies

    .. math::
        \\textrm{payoff}[i,j,k]=f_j(R^\\text{TARN}_{i, t_k^0}(t^p_j))\\,,

    where :math:`f_j(R)` was defined in :py:meth:`GeneralSwap<irproducts.GeneralSwap>` and :math:`R^\\text{TARN}` in :py:meth:`TARNRate<ratecurves.TARNRate>`.

    """

    def __init__(
        self,
        rate,
        dates,
        calendar,
        barrier,
        obsdates,
        effective_date,
        end_date,
        floating_freq,
        fixed_freq,
        legs_calendars=None,
        nominal=100,
        strike=0,
        curve=None,
    ):
        tarn_rate = TARNRate(rate, dates, calendar, barrier)
        GeneralSwap.__init__(
            self,
            tarn_rate,
            obsdates,
            effective_date,
            end_date,
            floating_freq,
            fixed_freq,
            legs_calendars,
            nominal,
            strike,
            curve,
        )


class CMSSpreadForward(Forward, DifferenceRate, GSCFRate):
    # Some classes added for the inheritance diagram. Example of product created with rates and a payoff from structured
    """
    Represents a CMS (Constant Maturity Swap) spread coupon, which pays the spread over a CMS rate.
    The CMS spread coupon pays the difference between the rates of two CMS with different tenors,
    adjusted by a gearing factor, and added to a spread. It is subject to a cap and floor which
    limit the maximum and minimum coupon rates paid.See Section 5.13.3, Vol. I, of [Andersen and Piterbarg].
    This is a `toy product` where we have used a forward structure, i.e., just one payment.

    Parameters
    ----------
    rates : Duple of ratecurves.MultiRate
        Duple of CMS rates. These are CMS rates
        covering different periods, typically denoted by :math:`S_{n,a}(T_n)` and :math:`S_{n,b}(T_n)` in literature.
    maturity : str or pandas.Timestamp
        The maturity date of the coupon when the CMS spread is observed and the coupon is paid.
    spread : float
        The spread added to the CMS rate.
    gearing : float
        The gearing factor applied to the difference between the long and short CMS rates before adding the spread.
    cap : float
        The maximum rate that the coupon will pay, also known as the cap rate.
    floor : float
        The minimum rate that the coupon will pay, also known as the floor rate.
    calendar : calendars.DayCountCalendar
        The calendar used for date calculations and determining the day count fraction.
    pay_dates : pandas.DatetimeIndex, list of strings, pandas.Timestamp, or string, optional
        Payment dates for the coupon. If not specified, it defaults to the observation dates, here maturity date.
    nominal : float, optional
        The nominal amount of the coupon. Default is 100.

    Attributes
    ----------
    spread : float
        The spread paid over the CMS rate.
    gearing : float
        The gearing factor applied to the spread.
    cap : float
        The cap rate for the coupon.
    floor : float
        The floor rate for the coupon.
    nominal : float
        The nominal amount of the coupon.

    Note
    -----
    The CMS Spread Coupon pays a coupon based on the spread between two CMS rates. Specifically, the coupon for a payment date :math:`T_n` is calculated as:

    .. math::
        C(T_n) = \\text{max}(\\text{min}(g \\times (S_{n,a}(T_n) - S_{n,b}(T_n)) + s, c), f).

    Here, :math:`S_{n,a}(T_n)` and :math:`S_{n,b}(T_n)` represent two CMS rates fixing on :math:`T_n`, covering periods :math:`a` and :math:`b` respectively.
    The coupon :math:`C_n` is then determined by the gearing factor :math:`g`, the spread :math:`s`, and is subject to a cap :math:`c` and floor :math:`f`.
    This structure creates a coupon that reflects the differential movement of two segments of the yield curve over the accrual period,
    adjusted by the contract terms of spread, gearing, cap, and floor.

    References
    -------------
    [Andersen and Piterbarg, 2010] Andersen, L. B. G., and Piterbarg, V.V. (2010). Interest Rate Modeling (Vol. I).

    See also
    --------
    :py:meth:`CMSSpreadCoupon<irproducts.CMSSpreadCoupon>`. It represents the same product, but with different code implementation, using the
    :py:meth:`Forward<structured.Forward>` class.
    """

    def __init__(
        self,
        rates,
        gearing,
        spread,
        cap,
        floor,
        maturity,
        strike,
        calendar,
        nominal=100,
        credit_curve=zero_credit_risk,
    ):
        self.diff_rate = rates[1] - rates[0]  # See below for a different implementation, which gives a warning
        # self.diff_rate = DifferenceRate(*rates)  # See above for a different implementation, which avoids the warning
        self.gscf_rate = GSCFRate(
            rates=self.diff_rate, gearing=gearing, spread=spread, cap=cap, floor=floor
        )

        Forward.__init__(
            self,
            self.gscf_rate,
            maturity=maturity,
            strike=strike,
            calendar=calendar,
            nominal=nominal,
            credit_curve=credit_curve,
        )


class CMSSpreadSwap(GeneralSwap, DifferenceRate, GSCFRate):
    # Some classes added for the inheritance diagram. Example of product created with rates and a payoff from above
    """
    Represents a CMS (Constant Maturity Swap) spread coupon swap, which pays the spread over a CMS rate.
    The CMS spread coupon pays the difference between the rates of two CMS with different tenors,
    adjusted by a gearing factor, and added to a spread. It is subject to a cap and floor which
    limit the maximum and minimum coupon rates paid.See Section 5.13.3, Vol. I, of [Andersen and Piterbarg].
    This is inserted in a swap structure.

    Parameters
    ----------
    rates : Duple of ratecurves.MultiRate
        Duple of CMS rates. These are CMS rates
        covering different periods, typically denoted by :math:`S_{n,a}(T_n)` and :math:`S_{n,b}(T_n)` in literature.
    spread : float
        The spread added to the CMS rate.
    gearing : float
        The gearing factor applied to the difference between the long and short CMS rates before adding the spread.
    cap : float
        The maximum rate that the coupon will pay, also known as the cap rate.
    floor : float
        The minimum rate that the coupon will pay, also known as the floor rate.
    obsdates : pandas.DatetimeIndex, list of strings, pandas.Timestamp, or string
        Observation (fixing) dates for the floating rate. Mathematically represented by :math:`\\{t^1_j\\}_{j=0}^J`.
    curve : DiscountCurve
        The discount curve used to compute forward rates and discounts, denoted as :math:`\\tilde{P}`.
    effective_date : pandas.Timestamp or string
        The start date of the swap, :math:`T_\\alpha`, where floating and fixed legs commence.
    end_date : pandas.Timestamp or string
        The end date of the swap, :math:`T_\\beta`, where floating and fixed legs terminate.
    floating_freq : int
        Frequency of the floating leg payments in months, denoted as :math:`12 \\cdot (T_l^{\\text{float}} - T_{l-1}^{\\text{float}})`
        for :math:`l\\in\\{\\alpha^{\\text{float}}+1, \\ldots, \\beta^{\\text{float}}\\}` (it is assumed that this is constant).
        For instance, if floating_freq=6, then, approximately (depends on the day count convention), :math:`\\left(T_l^{\\text{float}}-T_{l-1}^{\\text{float}}\\right)=0.5`,
    fixed_freq : int
        Frequency of the fixed leg payments in months, similar to `floating_freq`.
    legs_calendars : list of DayCountCalendar or DayCountCalendar
        Day count conventions for each leg, represented as :math:`[\\text{DC}^{\\text{float}}, \\text{DC}^{\\text{fixed}}]`.
    nominal : ndarray or float, optional
        Nominal amounts for the swap, potentially variable over time. Default is 100.
    strike : ndarray or float, optional
        Strike rates for the fixed leg, potentially variable over time. Default is 0.

    Attributes
    ----------
    spread : float
        The spread paid over the CMS rate.
    gearing : float
        The gearing factor applied to the spread.
    cap : float
        The cap rate for the coupon.
    floor : float
        The floor rate for the coupon.
    pay_dates : pandas.DatetimeIndex
        Combined payment dates for both legs of the swap, excluding the effective date. See below for detailed explanation.
    nominal : np.ndarray
        Array containing the nominal values applicable for each payment date. Mathematically represented, after possibly extending it to all the necessary dates, as  :math:`[N_j]_{j=0}^{J^p}`.
    strike : np.ndarray
        Array containing the strike rates applicable for each payment date.  Mathematically represented, after possibly extending it to all the necessary dates, as  :math:`[K_j]_{j=0}^{J^\\text{fixed}}`.


    Note
    -----
    The CMS Spread Coupon pays a coupon based on the spread between two CMS rates. Specifically, the coupon for a payment date :math:`T_n` is calculated as:

    .. math::
        C(T_n) = \\text{max}(\\text{min}(g \\times (S_{n,a}(T_n) - S_{n,b}(T_n)) + s, c), f).

    Here, :math:`S_{n,a}(T_n)` and :math:`S_{n,b}(T_n)` represent two CMS rates fixing on :math:`T_n`, covering periods :math:`a` and :math:`b` respectively.
    The coupon :math:`C_n` is then determined by the gearing factor :math:`g`, the spread :math:`s`, and is subject to a cap :math:`c` and floor :math:`f`.
    This structure creates a coupon that reflects the differential movement of two segments of the yield curve over the accrual period,
    adjusted by the contract terms of spread, gearing, cap, and floor. This is inserted in our general swap structure, see :py:meth:`General Swap<irproducts.GeneralSwap>` class.

    References
    -------------
    [Andersen and Piterbarg, 2010] Andersen, L. B. G., and Piterbarg, V.V. (2010). Interest Rate Modeling (Vol. I).
    """

    def __init__(
        self,
        rates,
        gearing,
        spread,
        cap,
        floor,
        obsdates,
        effective_date,
        end_date,
        floating_freq,
        fixed_freq,
        legs_calendars=None,
        nominal=100,
        strike=0,
        curve=None,
    ):
        self.diff_rate = rates[1] - rates[0]  # See below for a different implementation, which gives a warning
        # self.diff_rate = DifferenceRate(*rates)  # See above for a different implementation, which avoids the warning
        self.gscf_rate = GSCFRate(
            rates=self.diff_rate, gearing=gearing, spread=spread, cap=cap, floor=floor
        )
        GeneralSwap.__init__(
            self,
            self.gscf_rate,
            obsdates,
            effective_date,
            end_date,
            floating_freq,
            fixed_freq,
            legs_calendars,
            nominal,
            strike,
            curve,
        )
