import numpy as np
import pandas as pd

try:
    from ..data.underlyings import (
        NormalAsset,
        LognormalAsset,
    )

    # Import needed for the workspace. In this case we need the parent package since underlyings
    # is in a different subpackage (data)
except (ImportError, ModuleNotFoundError, ValueError):
    from data.underlyings import NormalAsset, LognormalAsset  # (Absolute) local import
try:
    from . import functions as afsfun
except (ImportError, ModuleNotFoundError, ValueError):
    import pricing.functions as afsfun

try:
    from . import functions as afsfun
except (ImportError, ModuleNotFoundError, ValueError):
    import pricing.functions as afsfun


# ----------------------------------------------------------------------------------------------
# Operation classes
# ----------------------------------------------------------------------------------------------


class MultiRate:  # Same structure as underlyings.MultiAsset
    """
    A class for handling multiple rates, :math:`R_0, \\ldots, R_N`, within the same object.

    This class is designed to aggregate and manage multiple rate objects,
    allowing operations and calculations that involve several rates simultaneously.

    Parameters
    ----------
    *rates : tuple of rates
        A variable number of rate objects to be included in the MultiRate object. Each rate object
        represents a distinct rate, and together they are treated as components of a multi-rate system.\
        Mathematically, :math:`(R_0, \\ldots, R_N)\\,`, a tuple representing the collection of rate objects
        included in the MultiRate object.

    Attributes
    ----------
    rate_components : tuple
        Stores the rate objects passed to the constructor. Each component in the tuple corresponds to one of the
        rates :math:`R_0, \\ldots, R_N` managed by the MultiRate object.

    Notes
    -----
    The `MultiRate` class is akin to handling multiple assets within a `MultiAsset` class in ``underlyings.py``,
    but specifically tailored for interest rates or yield curves. It allows for complex financial modeling
    scenarios where multiple rates may influence the valuation or behavior of financial instruments or portfolios.
    """

    def __init__(self, *rates):
        self.rate_components = rates

    def get_forward_dates(self, dates):
        """
        Get the forward dates for the multi-rate object.

        Parameters
        ----------
        dates : pandas.Timestamp, pandas.DatetimeIndex, list of strings, or string
            Dates for which forward dates are to be computed. Mathematically represented as :math:`\\{t^1_j\\}_{j=0}^J`.

        Returns
        -------
        dict
            A dictionary mapping each input date :math:`t^1_j` to a tuple of tenor dates :math:`(T_l^j)_{l=0}^{L_j}` associated with that date.

        Notes
        -----
        For a given set of input dates :math:`\\{t^1_j\\}_{j=0}^J`, the method returns a dictionary:

        .. math::
            \\text{Dictionary}\\{t^1_j~:~ (T_l^j)_{l=0}^{L_j}\\}\\,,

        where:

        - :math:`\\{T_l^j\\}` represents the complete tenor structure for the date :math:`t^1_j`.
        - :math:`\\{T_l^{j,n}\\}` is the complete tenor structure for the date :math:`t^1_j` and the :math:`n`-th component of the multi-rate object. It is the union of all :math:`\\{T_l^{j,n}\\}` across all components :math:`n` in the multi-rate object, denoted as :math:`\\bigcup_{n=0}^N\\{T_l^{j,n}\\}`.

        This method facilitates the retrieval of forward-looking tenor dates that are relevant for pricing and valuation models which depend on the future structure of interest rates or yields.
        """
        tenor_dic = {}
        for date in dates:
            tenor_dic[date] = self.rate_components[0].get_forward_dates(dates)[date]
            for component in self.rate_components[1:]:
                tenor_dic[date] = tenor_dic[date].union(
                    component.get_forward_dates(dates)[date]
                )

        return tenor_dic

    def get_forward_indices(self, date):
        """
        Get the forward indices for the multi-rate object.

        Parameters
        ----------
        date : pandas.Timestamp or string
            Date for which forward indices are to be computed. Mathematically represented as :math:`t=t^1_j`.

        Returns
        -------
        dict
            A dictionary mapping each rate component `n` to a subset of indices :math:`\\mathcal{L}_n` that correspond to the sorted tenor dates :math:`T^{j}_{\\mathcal{L}_n[l]}=T^{j,n}_l` for this date.

        Notes
        -----
        For a given date :math:`t^1_j`, the method returns a dictionary:

        .. math::
            \\text{Dictionary}\\{n~:~ \\mathcal{L}_n\\}\\,,

        where :math:`\\mathcal{L}_n \\subset \\{0, \\ldots, L_j\\}` is such that, as lists with sorted values,

        .. math::
            T^{j}_{\\mathcal{L}_n[l]}=T^{j,n}_l\\,,

        for :math:`l\\in\\{0, \\ldots, L^n_j\\}`. This dictionary provides the mapping from rate components to the indices of their respective forward-looking tenor dates, aligning each component's tenor structure with the comprehensive tenor structure for the date :math:`t^1_j`.

        This method is particularly useful in multi-rate models where the alignment of tenor dates across different rate components is necessary for computations, valuations, or simulations.
        """
        indices = {}
        date_index = pd.DatetimeIndex([date])
        tenor_dic = self.get_forward_dates(date_index)
        for component in self.rate_components:
            indices[component] = [
                tenor_dic[date].get_loc(d)
                for d in component.get_forward_dates(date_index)[date]
            ]

        return indices

    def generate_rates(self, forward_bonds, discounts):
        """
        Generate rates for the multi-rate object based on given forward bonds and discounts.

        Parameters
        ----------
        forward_bonds : dict
            A dictionary mapping each date :math:`t^1_j` to forward bond prices. Mathematically represented as
            :math:`\\left\\{ t^1_j~:~ \\text{FP}_{i,0,k,l}\\right\\}_{j=0}^J`. In general, this corresponds to
            :math:`\\left\\{ t^1_j~:~ \\left(\\text{FP}_{t^0_k}\\left(t^1_j; T^j_{l-1}, T^j_{l}, \\underline{x}_i\\left(t^0_k, t^1_j\\right)\\right)\\right)_{i, 0, k, l}\\right\\}_{j=0}^J`,
            a dictionary for each date.

        discounts : dict
            A dictionary mapping each date :math:`t^1_j` to discount factors. Mathematically represented as
            :math:`\\left\\{t^1_j~:~ \\left(P_{t^0_k}\\left(t^1_j; T^j_l, \\underline{x}_i\\left(t^0_k, t^1_j\\right)\\right)\\right)_{i,l,k,0}\\right\\}_{j=0}^J`,
            a dictionary for each date.

        Returns
        -------
        np.ndarray
            An array of generated rates :math:`S_{i,j,k,l} = S_{i,j,k,0}^l`, where :math:`S^l` is the fourth-dimensional array given by each component's `generate_rates`.

        Notes
        -----
        For a given set of dates :math:`\\{t^1_j\\}` and associated forward bonds and discounts, this method returns an array structured to reflect the specific rates generated for each component within the multi-rate object. Each component's rates are derived from their respective forward bond and discount information, tailored by the comprehensive tenor structure for each date and the relevant subset of tenor dates needed by each component.

        **Remark:** Note that :math:`T^j_l` represents the complete tenor structure for the date :math:`t^1_j`, denoted as :math:`\\{T_l^j\\}=\\bigcup_{n=0}^N\\{T_l^{j,n}\\}`. However, each component only requires a particular subset of dates from the forward bonds and discounts. This necessitates the use of `get_forward_indices` and the `accumulate_index` function from *ratecurves.py* to properly align and compute the forward rates.
        """

        obs_dates = list(forward_bonds.keys())
        date0 = obs_dates[0]
        rates = np.full(
            (
                forward_bonds[date0].shape[0],
                len(obs_dates) + 1,
                forward_bonds[date0].shape[2],
                len(self.rate_components),
            ),
            np.nan,
        )
        # One aditional observation date for the slicing [:, 1:] in mc_engines
        forwards_dic = {}
        discounts_dic = {}
        for i, component in enumerate(self.rate_components):
            forwards_dic[component] = {}
            discounts_dic[component] = {}
            for date in obs_dates:
                indices = self.get_forward_indices(date)
                forwards_dic[component][date] = afsfun.accumulate_index(
                    forward_bonds[date], indices[component], "prod"
                )
                discounts_dic[component][date] = discounts[date][:, indices[component]]

            rates[..., i] = component.generate_rates(
                forwards_dic[component], discounts_dic[component]
            )[:, :, :, 0]
            # TODO: This assumes that rates are generated from forwards (and discounts) generated from the same model

        return rates


class AdditionRate(MultiRate):
    """
    Represents the aggregated sum of multiple rate components.

    This class takes multiple rate components as input and provides functionality to aggregate them into a single rate by summing them up.
    It primarily overrides the `generate_rates` method from the `MultiRate` class to perform this summation. All other functionalities are inherited from the `MultiRate` class without modification.

    Parameters
    ----------
    *rates : Rate
        Variable length rate arguments. Each rate is an instance of a class derived from `Rate` that represents a component of the overall rate to be summed.

    Notes
    -----
    The main purpose of this class is to facilitate the modeling of financial instruments or contracts where the rate of return or interest rate is determined by the sum of multiple underlying rates.
    """
    def __init__(self, *rates):
        MultiRate.__init__(self, *rates)
        # super().__init__(*rates)  # In general it is a bad praxis to include manually the name of the parent class,
        # better use super() see OOP. But here, it would go to the first parent class, MRO, of self, which is not nec. MultiRate.

    def generate_rates(self, forward_bonds, discounts):
        """
        Aggregates multiple rate components into a single rate by summation.

        Overrides the `generate_rates` method from `MultiRate` to sum the rate components. This method computes the sum across the last axis (representing different rate components) to produce a single aggregated rate.

        Parameters
        ----------
        forward_bonds : dict
            A dictionary mapping each date :math:`t^1_j` to forward bond prices. Mathematically represented as
            :math:`\\left\\{ t^1_j~:~ \\text{FP}_{i,0,k,l}\\right\\}_{j=0}^J`. In general, this corresponds to
            :math:`\\left\\{ t^1_j~:~ \\left(\\text{FP}_{t^0_k}\\left(t^1_j; T^j_{l-1}, T^j_{l}, \\underline{x}_i\\left(t^0_k, t^1_j\\right)\\right)\\right)_{i, 0, k, l}\\right\\}_{j=0}^J`,
            a dictionary for each date.

        discounts : dict
            A dictionary mapping each date :math:`t^1_j` to discount factors. Mathematically represented as
            :math:`\\left\\{t^1_j~:~ \\left(P_{t^0_k}\\left(t^1_j; T^j_l, \\underline{x}_i\\left(t^0_k, t^1_j\\right)\\right)\\right)_{i,l,k,0}\\right\\}_{j=0}^J`,
            a dictionary for each date.

        Returns
        -------
        np.ndarray
            An array of generated rates :math:`S_{i,j,k,l=0} = \\sum_{n=0}^N S_{i,j,k,0}^n`, where :math:`S^l` is the fourth-dimensional array given by each component's `generate_rates`.

        Notes
        -----
        Given :math:`(R_n)_{n=0}^N` as rates, this method computes and returns the rate :math:`\\sum_{n=0}^N R_n`.

        """
        rates = MultiRate.generate_rates(self, forward_bonds, discounts) # In general it is a bad praxis to include manually the name of the parent class,  
        # better use super() see OOP. But here, it would go to the first parent class, MRO, of self, which is not nec. MultiRate.
        # rates = super().generate_rates(forward_bonds, discounts)  
        aggregated_rates = np.sum(rates, axis=3, keepdims=True)
        return aggregated_rates


class DifferenceRate(MultiRate):
    """
    Represents the aggregated sum of multiple rate components.

    This class takes 2 rate components as input and provides functionality to aggregate them into a single rate by subtracting them up.
    It primarily overrides the `generate_rates` method from the `MultiRate` class to perform this summation. All other functionalities are inherited from the `MultiRate` class without modification.

    Parameters
    ----------
    rate_0, rate_1 : Rate
        Each rate is an instance of a class derived from `Rate` that represents a component of the overall rate to be subtracted.

    Notes
    -----
    The main purpose of this class is to facilitate the modeling of financial instruments or contracts where the rate of return or interest rate is determined by the
    difference or spread of two underlying rates.
    """

    def __init__(self, rate_0, rate_1):
        MultiRate.__init__(self, rate_0, rate_1)

    def generate_rates(self, forward_bonds, discounts):
        """
        Subtracts two rate components into a single rate by summation.

        Overrides the `generate_rates` method from `MultiRate` to sum the rate components. This method computes the difference across the last axis (representing different rate components) to
        produce a single aggregated rate.

        Parameters
        ----------
        forward_bonds : dict
            A dictionary mapping each date :math:`t^1_j` to forward bond prices. Mathematically represented as
            :math:`\\left\\{ t^1_j~:~ \\text{FP}_{i,0,k,l}\\right\\}_{j=0}^J`. In general, this corresponds to
            :math:`\\left\\{ t^1_j~:~ \\left(\\text{FP}_{t^0_k}\\left(t^1_j; T^j_{l-1}, T^j_{l}, \\underline{x}_i\\left(t^0_k, t^1_j\\right)\\right)\\right)_{i, 0, k, l}\\right\\}_{j=0}^J`,
            a dictionary for each date.

        discounts : dict
            A dictionary mapping each date :math:`t^1_j` to discount factors. Mathematically represented as
            :math:`\\left\\{t^1_j~:~ \\left(P_{t^0_k}\\left(t^1_j; T^j_l, \\underline{x}_i\\left(t^0_k, t^1_j\\right)\\right)\\right)_{i,l,k,0}\\right\\}_{j=0}^J`,
            a dictionary for each date.

        Returns
        -------
        np.ndarray
            An array of generated rates :math:`S_{i,j,k,l=0} = S_{i,j,k,0}^1-S_{i,j,k,0}^0`, where :math:`S^l` is the fourth-dimensional array given by each component's `generate_rates`.

        Notes
        -----
        Given :math:`(R_n)_{n=0}^1` as rates, this method computes and returns the rate :math:`R_1-R_0`.

        """
        rates = MultiRate.generate_rates(self, forward_bonds, discounts)
        return rates[..., [1]] - rates[..., [0]]


class ProductRate(MultiRate):
    """
    Represents the aggregated product of multiple rate components.

    This class takes multiple rate components as input and provides functionality to aggregate them into a single rate by multiplying them up.
    It primarily overrides the `generate_rates` method from the `MultiRate` class to perform this multiplication. All other functionalities are inherited from the `MultiRate` class without modification.

    Parameters
    ----------
    *rates : Rate
        Variable length rate arguments. Each rate is an instance of a class derived from `Rate` that represents a component of the overall rate to be multiplied.

    Notes
    -----
    The main purpose of this class is to facilitate the modeling of financial instruments or contracts where the rate of return or interest rate is determined by the product of multiple underlying rates.
    """

    def __init__(self, *rates):
        MultiRate.__init__(self, *rates)

    def generate_rates(self, forward_bonds, discounts):
        """
        Aggregates multiple rate components into a single rate by multiplication.

        Overrides the `generate_rates` method from `MultiRate` to sum the rate components. This method computes the product across the last axis (representing different rate components) to produce a single aggregated rate.

        Parameters
        ----------
        forward_bonds : dict
            A dictionary mapping each date :math:`t^1_j` to forward bond prices. Mathematically represented as
            :math:`\\left\\{ t^1_j~:~ \\text{FP}_{i,0,k,l}\\right\\}_{j=0}^J`. In general, this corresponds to
            :math:`\\left\\{ t^1_j~:~ \\left(\\text{FP}_{t^0_k}\\left(t^1_j; T^j_{l-1}, T^j_{l}, \\underline{x}_i\\left(t^0_k, t^1_j\\right)\\right)\\right)_{i, 0, k, l}\\right\\}_{j=0}^J`,
            a dictionary for each date.

        discounts : dict
            A dictionary mapping each date :math:`t^1_j` to discount factors. Mathematically represented as
            :math:`\\left\\{t^1_j~:~ \\left(P_{t^0_k}\\left(t^1_j; T^j_l, \\underline{x}_i\\left(t^0_k, t^1_j\\right)\\right)\\right)_{i,l,k,0}\\right\\}_{j=0}^J`,
            a dictionary for each date.

        Returns
        -------
        np.ndarray
            An array of generated rates :math:`S_{i,j,k,l=0} = \\prod_{n=0}^N S_{i,j,k,0}^n`, where :math:`S^l` is the fourth-dimensional array given by each component's `generate_rates`.

        Notes
        -----
        Given :math:`(R_n)_{n=0}^N` as rates, this method computes and returns the rate :math:`\\prod_{n=0}^N R_n`.

        """
        rates = MultiRate.generate_rates(self, forward_bonds, discounts)  # In general it is a bad praxis to include manually the name of the parent class,  
        # better use super() see OOP. But here, it would go to the first parent class, MRO, of self, which is not nec. MultiRate.
        # rates = super().generate_rates(forward_bonds, discounts)
        return np.prod(rates, axis=3, keepdims=True)


class Rate(MultiRate):  # Syntactic sugar
    """
    Rate class derived from ``MultiRate`` for creating rate objects and enabling syntactic sugar operations.
    """

    def __init__(self):
        pass  # Some common attributes should be here

    def __add__(self, other):
        return AdditionRate(self, other)

    def __radd__(self, other):
        return AdditionRate(other, self)

    def __sub__(self, other):
        return DifferenceRate(other, self)  # r_s-r_o = DifferenceRate(r_o, r_s) as r_1=r_s and r_0=r_o, by definition. See documentation.

    def __rsub__(self, other):
        return DifferenceRate(self, other)

    def __mul__(self, other):
        return ProductRate(self, other)

    def __rmul__(self, other):
        return ProductRate(other, self)


# ----------------------------------------------------------------------------------------------
# Swap curves
# ----------------------------------------------------------------------------------------------


class ForwardRate(Rate):
    """
    ForwardRate class derived from `Rate` to represent forward rates.

    The forward rate is defined based on [Brigo and Mercurio, 2006], Definition 1.4.1 or `Abbreviations and Notation`:

    .. math::
        F(t; T, S) = \\frac{1}{\\text{DC}(T,S)}\\left(\\text{FP}^{-1}(t; T, S)-1\\right)

    where :math:`\\text{FP}(t; T, S):=\\dfrac{P(t, S)}{P(t, T)}` is the forward bond.

    Parameters
    ----------
    curve : pricing.discount_curves.DiscountCurve
        The discount curve represented by :math:`P`.

    effective_date : pandas.DatetimeIndex, list of strings, pandas.Timestamp, or string
        Represents :math:`T`, the effective start date.

    end_date : pandas.DatetimeIndex, list of strings, pandas.Timestamp, or string
        Represents :math:`S`, the end date.

    calendar : data.calendars.DayCountCalendar
        Specifies the calendar used for computation, which, in particular, specifies the function :math:`\\text{DC}`.

    freq : int
        Represents the difference between :math:`T` and :math:`S` in months. This is crucial for calculating the forward rate.
        Mathematically represented by :math:`\\nu .`

    References
    ----------
        - Brigo, D., & Mercurio, F. (2006). Interest rate models: theory and practice. Berlin: Springer.
    """

    def __init__(self, curve, effective_date, end_date, calendar, freq):
        Rate.__init__(self)
        # super().__init__(rate)  # In general it is a bad praxis to include manually the name of the parent class,
        # better use super() see OOP. But here, it would go to the first parent class, MRO, of self, which is not nec. Rate.
        self.curve = curve
        self.effective_date = effective_date
        self.end_date = end_date
        self.calendar = calendar
        # self.taus = calendar.interval(effective_date, end_date)
        # self.tenor_length = pd.to_datetime(end_date)-pd.to_datetime(effective_date)
        self.freq = freq

    def get_forward_dates(self, dates):
        """
        Generate and return a dictionary mapping dates to their corresponding forward dates.

        Given an input set of dates :math:`\\{t^1_j\\}_{j=0}^J`, this method produces a dictionary where each date
        :math:`t^1_j` maps to a tuple containing the date itself and the date incremented by a frequency :math:`\\nu=` ``self.freq``,
        in months. Formally, the resulting dictionary structure is:

        .. math::
            \\text{Dictionary}\\{t^1_j~:~ (t^1_j, t^1_j + \\nu)\\}

        This method essentially yields the "tenor structure" beginning at each :math:`t^1_j`.

        Parameters
        ----------
        dates : pandas.DatetimeIndex, list of strings, pandas.Timestamp, or string
            Represents the set :math:`\\{t^1_j\\}_{j=0}^J`. It can be a single date (as a pandas.Timestamp or its string representation)
            or an array-like object (as a pandas.DatetimeIndex or a list of its string representation) containing dates.

        Returns
        -------
        dict
            Dictionary with keys being the input dates and values as the corresponding forward dates incremented by the frequency :math:`\\nu`.

        Notes
        -----
        The resulting tenor structure provides an intuitive representation of how dates map to their future counterparts based on
        a predefined frequency.
        """
        tenor_dic = {}
        for date in dates:
            end_date_temp = date + pd.DateOffset(months=self.freq)
            dates_temp = pd.DatetimeIndex([date, end_date_temp])
            tenor_dic[date] = dates_temp

        return tenor_dic

    def generate_rates(self, forward_bonds, discounts):
        """
        Calculate and return forward rates based on forward bond data :math:`\\text{FP}` and dates :math:`\\{t^1_j\\}_{j=0}^J`.

        The method determines forward rates using the following formula:

        .. math::
            F_{i,j,k,l=0} :=\\frac1{\\text{DC}\\left(T_{l}^j, T_{l+1}^j\\right)} \\left(\\left(\\text{FP}_{i,j,k,l}\\right)^{-1}-1\\right)\\,.

        Here, :math:`\\{T^j_l\\}_{l=0}^1` represents the tenor structure associated with the date :math:`t^1_j`, given by ``get_forward_dates``. In a more generalized setting:

        .. math::
            \\text{FP}_{i,j,k,l}=\\text{FP}_{t^0_k}\\left(t^1_j,T_{l}^j, T_{l+1}^j, \\underline{x}_i\\left(t^0_k, t^1_j\\right)\\right)

        This forward bond is calculated by the method :py:meth:`compute_future_bond <pricing.ir_models.ShortRateModel.compute_future_bond>`.

        Parameters
        ----------
        forward_bonds : dict
            Dictionary of forward bond data organized for each date. It follows the representation:

            .. math::
                \\left\\{ t^1_j~:~ \\left(\\text{FP}_{t^0_k}\\left(t^1_j; T^j_{l-1}, T^j_{l}, \\underline{x}_i\\left(t^0_k, t^1_j\\right)\\right)\\right)_{i, 0, k, l}\\right\\}_{j=0}^J\\,,

            where :math:`T^j_l` is detailed by the `get_forward_dates` method. It should be noted that the definition of :math:`T_{-1}` aligns with the
            :py:meth:`compute_forward_bond <pricing.ir_models.ShortRateModel.compute_forward_bond>` method, even though it is not utilized here.
        discounts : dict
            Dictionary of discount data, organized per date. The method uses a standardized signature across various rates, thus the discount data is provided, though not directly utilized.

        Returns
        -------
        np.ndarray
            Array containing the computed forward rates. Specifically, the method returns :math:`\\tilde{F}_{i,j,k,l}` with the characteristic that :math:`\\tilde{F}_{i,0,k,l}=0` and the array :math:`\\left(\\tilde{F}_{i,1:,k,l}\\right)` is equivalent to :math:`\\left({F}_{i,j,k,l}\\right)`.

        Notes
        -----
        - The returned forward rates are critical for the *mc_engines* script.
        - Using dictionaries for forwards and discounts might appear as an overcomplication in this scenario, but in general situations (like average swaps), attempting to construct an expansive array might be inefficient. Especially, constructing and handling such an array could result in the computation of bonds with non-essential dates.
        - The dictionary is indispensable for the discounts as they are derived from `compute_future_bond` in *ir_models.py*, and due to broadcasting rules, it mandates :math:`J` to be zero to provide all discounts.
        - The process could be adjusted as the same concept is employed for both the fixed and floating leg via a general function, `accumulate_index` in *ratecurves.py*. The same is applicable for the `MultiRate` class, where no unimportant simulations are conducted. Moreover, unnecessary bond prices could be assigned a value of 1 using the `set_past_dates` method in *ir_models*, though it wouldn't enhance efficiency.
        - ``intervals``, intervals between simulation dates, is not needed for rates. This is needed for the generation of interest rate paths, but this is done in the ``generate_paths`` method of the interest rate models.

        """

        obsdates = list(forward_bonds.keys())
        obsdate0 = obsdates[0]
        forward_rates = np.full(
            (
                forward_bonds[obsdate0].shape[0],
                len(obsdates),
                forward_bonds[obsdate0].shape[2],
                1,
            ),
            np.nan,
        )
        for j, date in enumerate(obsdates):
            fb_temp = forward_bonds[date][..., 1:]
            num = np.reciprocal(fb_temp) - 1
            forward_rates[:, j, :, 0] = num[:, 0, :, 0] / self.calendar.interval(
                date, date + pd.DateOffset(months=self.freq)
            )

        # Path computation
        paths = forward_rates
        shape = np.array(paths.shape)
        shape[1] += 1
        paths_amp = np.full(shape, np.nan)
        paths_amp[:, 0] = np.nan
        # For convention considerations (see mc_engines, the line where paths = paths[:, 1:] right after computing the paths)
        paths_amp[:, 1:] = paths

        return paths_amp


class SwapRate(Rate):
    """
    Represents a swap rate for modeling interest rate swaps.

    Parameters
    ----------
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
    tenor_length : float, optional
        The total tenor length of the swap in years, :math:`T_\\beta-T_\\alpha`. Default is `None`. This parameter is necessary for `get_forward_dates` and adjusts for potential discrepancies in day counting conventions between `effective_date` and `end_date`.
    settlement : str, optional
        The settlement type of the swap, either "physical" or "cash" (not implemented). Default is "physical".

    Attributes
    ----------
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

    Notes
    -----
    Following the notation of [Brigo and Mercurio, 2006] Section 1.5 or Abbreviations and Notation, this class provides a framework for swap rate modeling based on the specified parameters and conventions.
    The swap model encapsulated by this class allows for detailed specification of both floating and fixed legs, including their respective frequencies, day count conventions, and the overall tenor of the swap agreement.
    """

    def __init__(self, curve, effective_date, end_date, floating_freq, fixed_freq,
                 legs_calendars, tenor_length=None, settlement="physical"):
        Rate.__init__(self)
        # super().__init__(rate)  # In general it is a bad praxis to include manually the name of the parent class,
        # better use super() see OOP. But here, it would go to the first parent class, MRO, of self, which is not nec. Rate.        self.curve = curve
        if type(legs_calendars) == tuple:
            self.floating_calendar = legs_calendars[0]
            self.fixed_calendar = legs_calendars[1]
        else:
            self.floating_calendar = legs_calendars
            self.fixed_calendar = legs_calendars
        self.effective_date = pd.to_datetime(effective_date)
        self.end_date = pd.to_datetime(end_date)
        self.floating_freq = floating_freq
        self.fixed_freq = fixed_freq
        self.floating_dates = pd.date_range(
            self.effective_date, self.end_date, freq=pd.DateOffset(months=floating_freq)
        )
        self.fixed_dates = pd.date_range(
            self.effective_date, self.end_date, freq=pd.DateOffset(months=fixed_freq)
        )
        self.tau_floating = self.floating_calendar.interval(
            self.floating_dates[:-1], self.floating_dates[1:]
        )  # Time intervals between floating dates
        self.tau_fixed = self.fixed_calendar.interval(
            self.fixed_dates[:-1], self.fixed_dates[1:]
        )  # Time intervals between fixed dates

        # New attributes
        self.total_dates = self.floating_dates.union(self.fixed_dates)
        # self.obsdates = self.effective_date
        dates_dic_float = {}
        dates_dic_fixed = {}
        for date in self.floating_dates:
            dates_dic_float[date] = self.total_dates.get_loc(date)
        for date in self.fixed_dates:
            dates_dic_fixed[date] = self.total_dates.get_loc(date)
        self.indices_float = [dates_dic_float[date] for date in self.floating_dates]
        self.indices_fixed = [dates_dic_fixed[date] for date in self.fixed_dates]
        self.tenor_length = tenor_length  # Tenor length in years
        self.settlement = settlement

    def get_pvbp(self, dates, discount_curve):
        """
        Calculates the Present Value of a Basis Point (PVBP), also known as the annuity of the swap, for given dates.

        Parameters
        ----------
        dates : pandas.DatetimeIndex, list of strings, pandas.Timestamp, or string
            The dates for which the PVBP is calculated. It can be a single date or a collection of dates. Mathematically, represented by :math:`\\{t^1_j\\}_{j=0}^J`.

        discount_curve : pricing.discount_curves.DiscountCurve
            The discount curve used for calculating present values of cash flows. Mathematically, represented by :math:`P`.

        Returns
        -------
        pandas.Series
            A series where each value corresponds to the PVBP calculated for each input date.

        Notes
        -----
        The method returns a `pandas.Series` representing the annuity of the swap for specified dates. Mathematically, for dates :math:`\\{t^1_j\\}_{j=0}^J`, the series is given by:

        .. math::
            \\text{Series}\\left(j, \\sum_{l=\\alpha^{\\text{fixed}}+1}^{\\beta^{\\text{fixed}}}\\text{DC}^{\\text{fixed}}\\left(T^{\\text{fixed}}_{l-1}, T^{\\text{fixed}}_{l}\\right) P\\left(t^1_j,T^{\\text{fixed}}_{l}\\right)\\text{Ind}\\left({t^1_j\\le T_{l-1}^{\\text{fixed}}}\\right)\\right)\,.

        This series corresponds to the annuity of the swap or its PVBP (Present Value of a Basis Point), and acts as the denominator in formula (1.25) of Brigo and Mercurio (2006), evaluated across different valuation dates.
        The PVBP is crucial for swap valuation, representing the value of fixed leg payments per basis point of the fixed rate.

        References
        ----------
        - [Brigo and Mercurio, 2006] Brigo, D., & Mercurio, F. (2006). Interest Rate Models–Theory and Practice.
        - [Joshi, 2003] Joshi, M. S. (2003). The Concepts and Practice of Mathematical Finance. Page 308.
        - [Privault, 2013] Privault, N. (2013). Stochastic Finance: An Introduction with Market Examples. Chapter 16.
        """

        # Dates formatting
        dates = afsfun.dates_formatting(dates)

        tau = self.tau_fixed.reshape(
            self.tau_fixed.size, 1
        )  # Reshaping for broadcasting

        # Z = np.zeros((self.fixed_dates.size, dates.size))
        # for index, pay_date in np.ndenumerate(np.array(self.fixed_dates)):  # Returns an iterator yielding pairs of array coordinates and values.
        #     Z[index[0]] = discount_curve.get_value(dates, pay_date, self.fixed_calendar) * (dates <= pay_date)

        # To avoid using a loop, better?
        def disc_f(i, j):
            # get_value follows broadcasting rules, so we need this for the general case
            return discount_curve.get_value(
                dates[j], self.fixed_dates[i], self.fixed_calendar
            ) * (dates[j] <= self.fixed_dates[i])

        disc_f = np.vectorize(
            disc_f
        )  # Otherwise there are problems with the calendars and pandas
        Z = np.fromfunction(
            disc_f, shape=(self.fixed_dates.size, dates.size), dtype=int
        )  # Arguments must be of int type for the slicing

        pvbp = np.sum(tau * Z[1:], axis=0)
        pvbp = pd.Series(pvbp, index=dates)
        return pvbp

    def get_value(self, dates, discount_curve):
        """
        Calculate the swap rate for given dates using a specified discount curve.

        Parameters
        ----------
        dates : pandas.DatetimeIndex, list of strings, pandas.Timestamp, or string
            Valuation dates for which the swap rates are to be calculated. Mathematically represented by :math:`\\{t^1_j\\}_{j=0}^J`.
        discount_curve : pricing.discount_curves.DiscountCurve
            The discount curve used for calculating present values of cash flows. Mathematically represented by :math:`P(\\cdot, \\cdot)`.

        Returns
        -------
        numpy.ndarray
            An array of swap rate values for each input date.

        Notes
        -----
        The method calculates swap rates, :math:`S_{\\alpha,\\beta}(t_j^1)`, for specified dates based on the formula:

        .. math::
            F\\left(t^1_j, T^{\\text{float}}_{l-1}, T^{\\text{float}}_{l+1}\\right) :=\\frac{1}{\\text{DC}^{\\text{float}}\\left(T^{\\text{float}}_{l+1}, T^{\\text{float}}_{l-1}\\right)} \\left(\\frac{\\tilde{P}\\left(t^1_j, T^{\\text{float}}_{l}\\right)}{\\tilde{P}\\left(t^1_j, T^{\\text{float}}_{l-1}\\right)}-1\\right)\\text{Ind}\\left({t^1_j\\le T_{l-1}^{\\text{float}}}\\right),

        where :math:`N_j` and :math:`D_j` are defined as:

        .. math::
            N_j=\\sum_{l=\\alpha^{\\text{float}}+1}^{\\beta^{\\text{float}}}\\text{DC}^{\\text{float}}\\left(T^{\\text{float}}_{l-1}, T^{\\text{float}}_{l}\\right) P\\left(t^1_j,T^{\\text{float}}_{l}\\right)F\\left(t^1_j, T^{\\text{float}}_{l-1}, T^{\\text{float}}_{l}\\right),

        .. math::
            D_j=\\sum_{l=\\alpha^{\\text{fixed}}+1}^{\\beta^{\\text{fixed}}}\\text{DC}^{\\text{fixed}}\\left(T^{\\text{fixed}}_{l-1}, T^{\\text{fixed}}_{l}\\right) P\\left(t^1_j,T^{\\text{fixed}}_{l}\\right)\\text{Ind}\\left({t^1_j\\le T_{l-1}^{\\text{fixed}}}\\right),

        and the swap rate is given by :math:`S_{\\alpha,\\beta}(t_j^1)=\\frac{N_j}{D_j}`.

        **Remark (Multicurve Approach):**
        Forwards and discounts are not computed from the same curve, adhering to the multicurve approach outlined in Section 6.5.3 of Andersen and Piterbarg (2010). For a given tenor :math:`\\tau\\approx S-T`,
        two curves are considered: :math:`P` for discounts and :math:`P_\\tau=\\tilde{P}` for computing forwards as follows:

        .. math::
            F_\\tau(t; T, S)=\\frac{1}{\\text{DC}(T,S)}\\left(\\dfrac{P_\\tau(t, T)}{P_\\tau(t, S)}-1\\right)

        This approach distinguishes between the curve used for discounting cash flows (:math:`P`) and the curve used for calculating forward rates (:math:`\\tilde{P}` or :math:`P_\\tau`), allowing for a more accurate representation of market conditions and pricing dynamics.


        References
        ----------
        - [Andersen and Piterbarg, 2010] Andersen, L. B. G., & Piterbarg, V. V. (2010). Interest Rate Modeling. Section 6.5.3.
        - [Joshi, 2003] Joshi, M. S. (2003). The Concepts and Practice of Mathematical Finance.
        """

        # Date formatting
        dates = afsfun.dates_formatting(dates)
        pvbp = self.get_pvbp(dates, discount_curve).values

        tau = self.tau_floating.reshape(self.tau_floating.size, 1)

        # discounts = np.zeros((self.floating_dates.size, dates.size))
        # bonds = np.zeros((self.floating_dates.size, dates.size))
        # boolean = np.zeros((self.floating_dates.size, dates.size))

        # Better to have np.nan in case some values are not properly filled
        discounts = np.full((self.floating_dates.size, dates.size), np.nan)
        bonds = np.full((self.floating_dates.size, dates.size), np.nan)
        boolean = np.full((self.floating_dates.size, dates.size), np.nan)

        if discount_curve == self.curve and np.all(dates <= self.floating_dates[0]):
            # Simpler computation in this case following (1.25) of Brigo and Mercurio.
            num = self.curve.get_value(
                dates, self.floating_dates[0], self.floating_calendar
            ) - self.curve.get_value(
                dates, self.floating_dates[-1], self.floating_calendar
            )
            swap_rate = num / pvbp
        else:
            for index, pay_date in np.ndenumerate(
                self.floating_dates
            ):  # See get_pvbp method for an alternative syntax
                discounts[index[0]] = discount_curve.get_value(
                    dates, pay_date, self.floating_calendar
                ) * (dates <= pay_date)
                bonds[index[0]] = self.curve.get_value(
                    dates, pay_date, self.floating_calendar
                )
                boolean[index[0]] = dates <= pay_date

            w = (
                tau * discounts[1:]
            ) / pvbp  # Computing weights, see (13.12) of [Joshi, 2003].
            f = (
                (bonds[:-1] - bonds[1:]) * boolean[:-1] / (tau * bonds[1:])
            )  # Computing forwards
            swap_rate = np.sum(w * f, axis=0)  # Computing rate

        return swap_rate

    def get_fpx(self, dates, fd, discount_curve, *calendar):
        """
        Return forward prices.

        Parameters
        ----------
        dates : pandas.DatetimeIndex, list of strings, pandas.Timestamp, or string
            Valuation dates. It can be a single date (as a pandas.Timestamp or its string representation) or an array-like object (as a pandas.DatetimeIndex or a list of its
            string representation) containing dates.
        fd :
            Unused, for compatibility with option classes.
        discount_curve : pricing.discount_curves.DiscountCurve
            Curve needed for discount bonds.
        calendar : data.calendars.DayCountCalendar
            Calendar for computation of time intervals. Unused, for compatibility with option classes.

        Returns
        -------
        pandas.Series
            Series of forward values.
        """
        dates = afsfun.dates_formatting(dates)
        px = self.get_value(dates, discount_curve)
        px = pd.Series(px, index=dates)
        return px

    def get_forward_dates(self, dates):
        """
        Return the tenor structure starting at the specified dates.

        Parameters
        ----------
        dates : pandas.DatetimeIndex, list of strings, pandas.Timestamp, or string
            Index of dates for which the tenor structure is to be computed. Mathematically represented by :math:`\\{t^1_j\\}_{j=0}^J`.

        Returns
        -------
        dict
            A dictionary mapping each input date to its corresponding tenor structure. Each tenor structure is represented as a sequence of dates, :math:`(T_l^j)_{l=0}^{L}`, calculated based on the specified frequency.

        Notes
        -----
        For each date :math:`t^1_j` in the set :math:`\\{t^1_j\\}_{j=0}^J`, this method computes the tenor structure, which consists of dates starting at :math:`t^1_j` and spaced according to the specified frequency,
        :math:`\\nu_f`, for both floating and fixed frequencies, :math:`f\\in\\{\\text{float, fixed}\\}.`

        Specifically, the method generates:

        .. math::
            \\text{Dictionary}\\{t^1_j : (T_l^j)_{l=0}^{L_j}\\}

        where:

        - :math:`T_l^{j,f}=t^1_j + l\\cdot\\nu_f`, and :math:`\\nu_f` is the frequency for floating (:math:`f=\\text{float}`) and fixed (:math:`f=\\text{fixed}`) rates, with :math:`l` ranging from 0 to :math:`L_j^f`.
        :math:`L^f` equals the ratio of the tenor length, ``self.tenor_length``, to the monthly frequency, :math:`\\nu_f^m/12`.

        - :math:`\\{T_l^j\\}` combines the sets :math:`\\{T_l^{j,\\text{float}}\\}` and :math:`\\{T_l^{j,\\text{fixed}}\\}`, representing the complete tenor structure for the date :math:`t^1_j`.

        This provides a comprehensive mapping of each date to its subsequent tenor dates, facilitating the calculation of interest rates, cash flows, or pricing structures that depend on the tenor's length and distribution.
        """
        # dates_dic = {date: self.floating_dates for date in dates if date <= self.effective_date}
        # dates_dic.update({date: pd.to_datetime([]) for date in dates if date > self.effective_date})

        tenor_dic = {}
        for date in dates:
            end_date_temp = date + pd.DateOffset(years=self.tenor_length)
            float_temp = pd.date_range(
                date, end_date_temp, freq=pd.DateOffset(months=self.floating_freq)
            )
            fixed_temp = pd.date_range(
                date, end_date_temp, freq=pd.DateOffset(months=self.floating_freq)
            )
            tenor_dic[date] = float_temp.union(fixed_temp)

        return tenor_dic

    def generate_rates(self, forward_bonds, discounts):
        """
        Return paths of the swap rates given an array of simulated forward bonds.

        Parameters
        ----------
        forward_bonds : dict
            A dictionary mapping each date :math:`t^1_j` to forward bond prices. Mathematically represented as
            :math:`\\left\\{ t^1_j~:~ \\text{FP}_{i,0,k,l}\\right\\}_{j=0}^J`. In general, this corresponds to
            :math:`\\left\\{ t^1_j~:~ \\left(\\text{FP}_{t^0_k}\\left(t^1_j; T^j_{l-1}, T^j_{l}, \\underline{x}_i\\left(t^0_k, t^1_j\\right)\\right)\\right)_{i, 0, k, l}\\right\\}_{j=0}^J`,
            a dictionary for each date.
        discounts : dict
            A dictionary mapping each date :math:`t^1_j` to discount factors. Mathematically represented as
            :math:`\\left\\{t^1_j~:~ \\left(P_{t^0_k}\\left(t^1_j; T^j_l, \\underline{x}_i\\left(t^0_k, t^1_j\\right)\\right)\\right)_{i,l,k,0}\\right\\}_{j=0}^J`,
            a dictionary for each date.

        Returns
        -------
        np.ndarray
            Array of swap rates for the different simulations.
        """

        float_leg_ind = self.indices_float
        fixed_leg_ind = self.indices_fixed
        taus_float = self.tau_floating
        taus_fixed = self.tau_fixed

        obs_dates = list(forward_bonds.keys())
        date0 = obs_dates[0]
        swap_rates = np.full(
            (
                forward_bonds[date0].shape[0],
                len(obs_dates),
                forward_bonds[date0].shape[2],
                2,
            ),
            np.nan,
        )
        for j, date in enumerate(obs_dates):
            fb_temp = forward_bonds[date]
            disc_temp = discounts[date][:, :, :, 0]  # Last index is trivial
            fb_float = afsfun.accumulate_index(fb_temp, float_leg_ind, "prod")
            fb_float = fb_float[
                :, 0, :, 1:
            ]  # We do not need the case l=0, see ir_models documentation. Also, only one observation date.
            disc_float = disc_temp[:, float_leg_ind, :][:, 1:].transpose(0, 2, 1)
            disc_fixed = disc_temp[:, fixed_leg_ind, :][:, 1:].transpose(0, 2, 1)
            forwards = (np.reciprocal(fb_float) - 1) / taus_float
            annuity = np.sum(disc_fixed * taus_fixed, axis=-1)
            swap_rates[:, j, :, 0] = (
                np.sum(forwards * disc_float * taus_float, axis=-1) / annuity
            )
            if self.settlement == "physical":
                swap_rates[:, j, :, 1] = annuity
            else:
                raise AttributeError("Given settlement not implemented.")

        # Path computation
        paths = swap_rates
        shape = np.array(paths.shape)
        shape[1] += 1
        paths_amp = np.full(shape, np.nan)
        paths_amp[:, 0] = np.nan
        # For convention considerations (see mc_engines, the line where paths = paths[:, 1:] right after computing the paths)
        paths_amp[:, 1:] = paths

        return paths_amp

        # volatility should be defined by a surface (strike,maturity) [ARS, see BM in Swap Rates, Chapter 1]-------------------------------


class NormalSwapRate(SwapRate, NormalAsset):
    """
    NormalSwapRate class derived from ``SwapRate`` and ``NormalAsset`` for representing normal swap rates.

    Parameters
    ----------
    curve : pricing.discount_curves.DiscountCurve
        Curve needed for discount bonds.

    effective_date : str or pandas.Timestamp
        Starting date for fixed and floating dates.

    end_date : str or pandas.Timestamp
        Final date for fixed and floating dates.

    floating_freq : int
        Floating frequency as the number of months.

    fixed_freq : int
        Fixed frequency as the number of months.

    legs_calendars : tuple or data.calendars.DayCountCalendar
        Calendar for each leg. If there is only one calendar, the same calendar is used for both legs.

    tenor_length : float or None
        Tenor length in years.
    """

    def __init__(
        self,
        curve,
        effective_date,
        end_date,
        floating_freq,
        fixed_freq,
        legs_calendars,
        tenor_length,
    ):
        SwapRate.__init__(
            self,
            curve,
            effective_date,
            end_date,
            floating_freq,
            fixed_freq,
            legs_calendars,
            tenor_length,
        )
        NormalAsset.__init__(self, "", yieldsdiv=False)  # Empty ticker


class LognormalSwapRate(SwapRate, LognormalAsset):
    """
    LognormalSwapRate class derived from ``SwapRate`` and ``LognormalAsset`` for representing lognormal swap rates.

    Parameters
    ----------
    curve : pricing.discount_curves.DiscountCurve
        Curve needed for discount bonds.

    effective_date : str or pandas.Timestamp
        Starting date for fixed and floating dates.

    end_date : str or pandas.Timestamp
        Final date for fixed and floating dates.

    floating_freq : int
        Floating frequency as the number of months.

    fixed_freq : int
        Fixed frequency as the number of months.

    legs_calendars : tuple or data.calendars.DayCountCalendar
        Calendar for each leg. If there is only one calendar, the same calendar is used for both legs.

    tenor_length : float or None
        Tenor length in years.
    """

    def __init__(
        self,
        curve,
        effective_date,
        end_date,
        floating_freq,
        fixed_freq,
        legs_calendars,
        tenor_length,
    ):
        SwapRate.__init__(
            self,
            curve,
            effective_date,
            end_date,
            floating_freq,
            fixed_freq,
            legs_calendars,
            tenor_length,
        )
        LognormalAsset.__init__(self, "", yieldsdiv=False)  # Empty ticker


class CMSRate:
    """
    CMSRate class for representing Constant Maturity Swap (CMS) rates.

    Parameters
    ----------
    curve : pricing.discount_curves.DiscountCurve
        Curve needed for discount bonds.
    tenor : float
        Tenor of the CMS rate in years.
    floating_freq : int
        Floating frequency as the number of months.
    fixed_freq : int
        Fixed frequency as the number of months.
    legs_calendars : tuple or data.calendars.DayCountCalendar
        Calendar for each leg. If there is only one calendar, the same calendar is used for both legs.
    """

    def __init__(self, curve, tenor, floating_freq, fixed_freq, legs_calendars):
        self.curve = curve
        self.tenor = tenor
        self.legs_calendars = legs_calendars
        if type(legs_calendars) == tuple:
            self.floating_calendar = legs_calendars[0]
            self.fixed_calendar = legs_calendars[1]
        else:
            self.floating_calendar = legs_calendars
            self.fixed_calendar = legs_calendars
        self.floating_freq = floating_freq
        self.fixed_freq = fixed_freq

    def get_value(self, dates, discount_curve):
        """
        Calculate CMS rates for the given valuation dates.

        Parameters
        ----------
        dates : pandas.DatetimeIndex, list of strings, pandas.Timestamp, or string
            Valuation dates. It can be a single date (as a pandas.Timestamp or its string representation) or an array-like object (as a pandas.DatetimeIndex or a list of
            its string representation) containing dates.

        discount_curve : pricing.discount_curves.DiscountCurve
            Curve needed for discount bonds.

        Returns
        -------
        pandas.Series
            Returns the CMS rates evaluated for the different dates using a specific discount curve.
        """
        dates = afsfun.dates_formatting(dates)
        rates = pd.Series(index=dates, dtype="float64")
        for date in dates:
            end_date = date + pd.DateOffset(months=self.tenor * 12)
            fixed_dates = pd.date_range(
                start=date, end=end_date, freq=pd.DateOffset(months=self.fixed_freq)
            )
            tau_fixed = self.fixed_calendar.interval(fixed_dates[:-1], fixed_dates[1:])
            z_fixed = discount_curve.get_value(date, fixed_dates, self.fixed_calendar)
            pvbp = np.sum(tau_fixed * z_fixed[1:])

            floating_dates = pd.date_range(
                start=date, end=end_date, freq=pd.DateOffset(months=self.floating_freq)
            )
            tau_float = self.fixed_calendar.interval(
                floating_dates[:-1], floating_dates[1:]
            )
            z_float = discount_curve.get_value(
                date, floating_dates, self.floating_calendar
            )
            w = tau_float * z_float[1:] / pvbp

            bonds = self.curve.get_value(date, floating_dates, self.floating_calendar)
            f = (bonds[:-1] - bonds[1:]) / (tau_float * bonds[1:])

            rates.loc[date] = np.sum(w * f)
        return rates


# ----------------------------------------------------------------------------------------------
# Exotic rates
# ----------------------------------------------------------------------------------------------


class PathDependentRate(Rate):
    """
    Represents a path-dependent interest rate, incorporating mechanisms like spreads and gearing.

    A path-dependent rate varies based on the historical path it takes, making it suitable for financial instruments where the rate depends on its historical performance.
    This class allows for the incorporation of spreads and gearing to modify the path-dependent rate calculation.

    Parameters
    ----------
    rates : Rate or tuple of Rate
        The underlying rate(s) on which the path-dependent rate is based. This can be a single rate or a tuple of multiple rates. Mathematically represented as :math:`(R_n)_{n=0}^N`.
    spread : float or numpy.ndarray
        The spread applied to the rate(s). Can be a single value applied uniformly across all time steps, or an array of values for varying spreads at different times.
        Mathematically represented as :math:`(s_j)_{j=0}^{J_s}`.
    gearing : float or numpy.ndarray
        The gearing factor applied to the rate(s). Similar to `spread`, it can be a uniform value or vary over time. Mathematically represented as :math:`(g_j)_{j=0}^{J_g}`.
    initial_coupon : float, optional
        The initial coupon value at the start of the rate path. Default is 0. Mathematically represented as :math:`C_{-1}`.

    Attributes
    ----------
    rate_components : tuple
        Stores the rate objects passed to the constructor. Each component in the tuple corresponds to one of the
        rates :math:`R_0, \\ldots, R_N` managed by the MultiRate object.
    spread : numpy.ndarray
        Array of spread values, broadcasted for compatibility with the underlying rates' dimensions.
    gearing : numpy.ndarray
        Array of gearing values, similarly broadcasted.
    initial_coupon : float
        The starting value of the coupon in the path-dependent rate calculation.

    Notes
    -----
    This class computes a path-dependent rate, :math:`C_j`, based on the provided underlying rates, spreads, and gearing factors. The rate for each time step, :math:`j`, is given by:

    .. math::
        C_j = C_{j-1} + s_j + g_j \\times \prod_{n=0}^N R_n(t^1_j)

    where :math:`s` is the spread, and :math:`g` is the gearing.

    The parameters and their mathematical representations are as follows:

    - ``rates`` :math:`\\rightarrow (R_n)_{n=0}^N`,
    - ``gearing`` :math:`\\rightarrow (g_j)_{j=0}^{J_g}`,
    - ``spread`` :math:`\\rightarrow (s_j)_{j=0}^{J_s}`,
    - ``initial_coupon`` :math:`\\rightarrow C_{-1}`, by default it equals 0.

    Note that if either `gearing` or `spread` are not constant, their size must match the number of dates in the `generate_rates` method's `forward_bonds` and `discounts` dictionaries. That is:

    .. math::
        J_a = J~~ \lor ~~ \min(J_a, J)=1\,.

    for :math:`a \\in \\{s, g\\}`. Other methods from the `Rate` class are inherited and not overwritten.
    """
    def __init__(self, rates, spread, gearing, initial_coupon=0):
        MultiRate.__init__(self, rates)
        # So get_forward_dates, get_forward_indices are already defined
        self.spread, self.gearing = afsfun.number_to_array(
            spread, gearing, for_broadcasting=True
        )
        self.initial_coupon = initial_coupon

    def generate_rates(self, forward_bonds, discounts):
        """
        Generates path-dependent rates based on simulated forward bonds and discounts.

        The method applies gearing and spread to the underlying rate paths, then adds the initial coupon value to start the path. The spread and gearing are applied uniformly or variably across the rate path based on their input shapes.

        Parameters
        ----------
        forward_bonds : dict
            Dictionary of forward bond data organized for each date. It follows the representation:

            .. math::
                \\left\\{ t^1_j~:~ \\left(\\text{FP}_{t^0_k}\\left(t^1_j; T^j_{l-1}, T^j_{l}, \\underline{x}_i\\left(t^0_k, t^1_j\\right)\\right)\\right)_{i, 0, k, l}\\right\\}_{j=0}^J\\,,

            where :math:`T^j_l` is detailed by the `get_forward_dates` method. It should be noted that the definition of :math:`T_{-1}` aligns with the
            :py:meth:`compute_forward_bond <pricing.ir_models.ShortRateModel.compute_forward_bond>` method, even though it is not utilized here.
        discounts : dict
            A dictionary mapping each date :math:`t^1_j` to discount factors. Mathematically represented as
            :math:`\\left\\{t^1_j~:~ \\left(P_{t^0_k}\\left(t^1_j; T^j_l, \\underline{x}_i\\left(t^0_k, t^1_j\\right)\\right)\\right)_{i,l,k,0}\\right\\}_{j=0}^J`,
            a dictionary for each date.

        Returns
        -------
        np.ndarray
            Array containing the computed path dependent rates. Specifically, the method returns

            .. math::
                {C}_{i,j,k,l=0}=c_{i,j-1,k}+S_j+G_j\\times \\prod_{n=0}^N R_{ijkn}\\,,

            using the same notation as above.


        Raises
        ------
        AttributeError
            If the sizes of `spread` or `gearing` do not match the number of keys in `forward_bonds`.

        Notes
        -----
        The first date in the rate paths is considered irrelevant for path-dependent calculations and is thus skipped. The method ensures that the `spread` and `gearing` arrays are
        compatible with the number of forward bonds provided.
        """
        rate_paths = MultiRate.generate_rates(self, forward_bonds, discounts)[
            :, 1:
        ]  # First date is irrelevant
        rate_paths = np.prod(rate_paths, axis=3, keepdims=True)
        if not (self.spread.size == 1 or self.spread.size == len(forward_bonds.keys())):
            raise AttributeError(
                "spread must be either one or with size equal to forward_bonds.keys()"
            )
        if not (
            self.gearing.size == 1 or self.gearing.size == len(forward_bonds.keys())
        ):
            raise AttributeError(
                "gearing must be either one or with size equal to forward_bonds.keys()"
            )

        paths_diff = self.spread - self.gearing * rate_paths  # Broadcasting applies
        paths = np.cumsum(paths_diff, axis=1) + self.initial_coupon
        # Thus, we can avoid a recursive definition and therefore a loop.

        # Path computation
        shape = np.array(paths.shape)
        shape[1] += 1
        paths_amp = np.full(shape, np.nan)
        paths_amp[:, 0] = np.nan
        # For convention considerations (see mc_engines, the line where paths = paths[:, 1:] right after computing the paths) TODO: this should be the
        # value at the valuation date
        paths_amp[:, 1:] = paths

        return paths_amp


class GSCFRate(Rate):
    """
    Represents rates with Gearing, Spread, Cap, and Floor (GSCF) characteristics.

    Parameters
    ----------
    rates : Rate or tuple of rates
        A collection of underlying rates, :math:`(R_n)_{n=0}^N`.
    gearing : float or numpy.ndarray
        Gearing values, :math:`(g_j)_{j=0}^{J_g}`.
    spread : float or numpy.ndarray
        Spread values, :math:`(s_j)_{j=0}^{J_s}`.
    cap : float or numpy.ndarray
        Cap values, :math:`(c_j)_{j=0}^{J_c}`.
    floor : float or numpy.ndarray
        Floor values, :math:`(f_j)_{j=0}^{J_f}`.

    Attributes
    ----------
    rate_components : tuple
        Stores the rate objects passed to the constructor. Each component in the tuple corresponds to one of the
        rates :math:`R_0, \\ldots, R_N` managed by the MultiRate object.
    gearing : float or numpy.ndarray
        Array of spread values, broadcasted for compatibility with the underlying rates' dimensions.
    spread : float or numpy.ndarray
        Array of gearing values, similarly broadcasted.
    cap : float or numpy.ndarray
        Array of gearing values, similarly broadcasted.
    floor : float or numpy.ndarray
        Array of gearing values, similarly broadcasted.

    Notes
    -----
    This class computes a rate, :math:`C_j`, based on the provided underlying rates, gears, spreads, caps, and floors. The rate for each time step, :math:`j`, is given by:

    .. math::
        C_j = \\max\\left(\\min\\left(s_j + g_j \\times \\prod_{n=0}^N R_n(t^1_j), c_j\\right), f_j\\right)

    where :math:`g` is the gearing, :math:`s` is the spread, :math:`c` is the cap, and :math:`f` is the floor.

    The parameters and their mathematical representations are as follows:

    - ``rates`` :math:`\\rightarrow (R_n)_{n=0}^N`,
    - ``gearing`` :math:`\\rightarrow (g_j)_{j=0}^{J_g}`,
    - ``spread`` :math:`\\rightarrow (s_j)_{j=0}^{J_s}`,
    - ``cap`` :math:`\\rightarrow (c_j)_{j=0}^{J_c}`,
    - ``floor`` :math:`\\rightarrow (f_j)_{j=0}^{J_f}`.

    If either `gearing`, `spread`, `cap`, or `floor` are not constant, their size must match the number of dates in the `forward_bonds` and `discounts` dictionaries. That is:

    .. math::
        J_a = J \\lor \\min(J_a, J)=1

    for :math:`a \\in \\{s, g, c, f\\}`. Other methods from the `Rate` class are inherited and not overwritten.
    """

    def __init__(self, rates, gearing, spread, cap, floor):
        MultiRate.__init__(
            self, rates
        )  # So get_forward_dates, get_forward_indices are already defined
        self.spread, self.gearing, self.cap, self.floor = afsfun.number_to_array(
            spread, gearing, cap, floor
        )

    def generate_rates(self, forward_bonds, discounts):
        """
        Generates rates with Gearing, Spread, Cap, and Floor (GSCF) applied, based on simulated forward bonds and discounts.

        This method combines multiple underlying rate components, applies gearing and spread to the aggregated rate paths, then caps and floors the results based on the input parameters. It effectively transforms the path of rates according to the specified GSCF dynamics.

        Parameters
        ----------
        forward_bonds : dict
            Dictionary of forward bond data organized for each date. It follows the representation:

            .. math::
                \\left\\{ t^1_j~:~ \\left(\\text{FP}_{t^0_k}\\left(t^1_j; T^j_{l-1}, T^j_{l}, \\underline{x}_i\\left(t^0_k, t^1_j\\right)\\right)\\right)_{i, 0, k, l}\\right\\}_{j=0}^J\\,,

            where :math:`T^j_l` is detailed by the `get_forward_dates` method. The forward bonds data structure aligns with the requirements of the
            :py:meth:`compute_forward_bond <pricing.ir_models.ShortRateModel.compute_forward_bond>` method.

        discounts : dict
            A dictionary mapping each date :math:`t^1_j` to discount factors. Mathematically represented as
            :math:`\\left\\{t^1_j~:~ \\left(P_{t^0_k}\\left(t^1_j; T^j_l, \\underline{x}_i\\left(t^0_k, t^1_j\\right)\\right)\\right)_{i,l,k,0}\\right\\}_{j=0}^J`,
            a dictionary for each date.
        Returns
        -------
        np.ndarray
            An array containing the computed rates with Gearing, Spread, Cap, and Floor applied. Specifically, the method returns:

            .. math::
                {C}_{i,j,k,l=0}=\\max\\left(\\min\\left(s_j + g_j \\times \\prod_{n=0}^N R_{ijkn}, c_j\\right), f_j\\right)\\,,

            where :math:`s_j`, :math:`g_j`, :math:`c_j`, and :math:`f_j` represent the spread, gearing, cap, and floor values, respectively, applied to the product of the underlying rates :math:`R_{ijkl}`.

        Raises
        ------
        AttributeError
            If the sizes of `spread`, `gearing`, `cap`, or `floor` do not match the number of keys in `forward_bonds`.

        Notes
        -----
        This method aggregates the effects of multiple rate components to compute the path-dependent rates incorporating gearing, spread, cap, and floor adjustments. The initial date in the computed paths is omitted as it does not contribute to the rate adjustments.
        """
        rate_paths = MultiRate.generate_rates(self, forward_bonds, discounts)[
            :, 1:
        ]  # First date is irrelevant
        rate_paths = np.prod(rate_paths, axis=3, keepdims=True)
        if not (self.spread.size == 1 or self.spread.size == len(forward_bonds.keys())):
            raise AttributeError(
                "spread must be either one or with size equal to forward_bonds.keys()"
            )
        if not (
            self.gearing.size == 1 or self.gearing.size == len(forward_bonds.keys())
        ):
            raise AttributeError(
                "gearing must be either one or with size equal to forward_bonds.keys()"
            )
        if not (self.cap.size == 1 or self.cap.size == len(forward_bonds.keys())):
            raise AttributeError(
                "cap must be either one or with size equal to forward_bonds.keys()"
            )
        if not (self.floor.size == 1 or self.floor.size == len(forward_bonds.keys())):
            raise AttributeError(
                "floor must be either one or with size equal to forward_bonds.keys()"
            )

        paths = np.maximum(
            np.minimum(self.spread + self.gearing * rate_paths, self.cap), self.floor
        )  # Used maximum, not max

        # Path computation
        shape = np.array(paths.shape)
        shape[1] += 1
        paths_amp = np.full(shape, np.nan)
        paths_amp[:, 0] = np.nan
        # For convention considerations (see mc_engines, the line where paths = paths[:, 1:] right after computing the paths)
        paths_amp[:, 1:] = paths

        return paths_amp


class CumulativeRate(Rate):
    """
    Represents cumulative rates calculated over specified time intervals based on simulated forward bonds and discounts. This class accumulates rates over time to generate a path of cumulative rates.

    Parameters
    ----------
    rates : Rate or tuple of Rate
        The set of underlying rates to be accumulated. Mathematically represented by :math:`(R_n)_{n=0}^N`, where each :math:`R_n` is a rate component.
    dates : pandas.DatetimeIndex or list of str
        The dates for which the rates are to be computed. Mathematically represented by :math:`(T_j)_j`, the sequence of dates.
    calendar : data.calendars.DayCountCalendar
        The calendar used for computing day count fractions between dates, denoted by :math:`\\text{DC}`, thus :math:`\\tau_i = \\text{DC}(T_{i-1}, T_i)` represents the time intervals between consecutive dates :math:`T_{i-1}` and :math:`T_i`.

    Notes
    -----
    The cumulative rate :math:`Q_j` is defined as:

    .. math::
        Q_j = \\sum_{i=0}^{j-1} \\tau_i \\prod_{n=0}^N R_n(T_i)\\,,

    where :math:`\\tau_i` are the time intervals between dates determined by the calendar, and :math:`R_n(T_i)` are the rates at time :math:`T_i` for each component :math:`n`.

    Other methods from `MultiRate` are inherited and not overwritten.

    """

    def __init__(self, rates, dates, calendar):
        MultiRate.__init__(self, rates)  # Inheritance of methods
        self.dates = pd.DatetimeIndex(dates)  # TODO, use dates_formatting.
        self.calendar = calendar
        self.intervals = self.calendar.interval(
            self.dates[:-1], self.dates[1:]
        ).reshape(self.dates.size - 1, 1, 1)

    def generate_rates(self, forward_bonds, discounts):
        """
        Generate paths of rates with gearing, spread, cap, and floor given an array of simulated forward bonds.

        Parameters
        ----------
        forward_bonds : dict
            A dictionary mapping each date :math:`t^1_j` to forward bond prices. Mathematically represented as
            :math:`\\left\\{ t^1_j~:~ \\text{FP}_{i,0,k,l}\\right\\}_{j=0}^J`. In general, this corresponds to
            :math:`\\left\\{ t^1_j~:~ \\left(\\text{FP}_{t^0_k}\\left(t^1_j; T^j_{l-1}, T^j_{l}, \\underline{x}_i\\left(t^0_k, t^1_j\\right)\\right)\\right)_{i, 0, k, l}\\right\\}_{j=0}^J`,
            a dictionary for each date.

        discounts : dict
            A dictionary mapping each date :math:`t^1_j` to discount factors. Mathematically represented as
            :math:`\\left\\{t^1_j~:~ \\left(P_{t^0_k}\\left(t^1_j; T^j_l, \\underline{x}_i\\left(t^0_k, t^1_j\\right)\\right)\\right)_{i,l,k,0}\\right\\}_{j=0}^J`,
            a dictionary for each date.

        Returns
        -------
        np.ndarray
            An array containing the cumulative rates for each simulation, following the calculation :math:`Q_{i,j,k,l=0} = \\sum_{j'=0}^{j-1} \\tau_j' \\prod_{n=0}^N R_{ij'kn}`.

        Notes
        -----
        This method accumulates the product of rates over specified time intervals, following the mathematical formula provided in the class description.
        """
        rate_paths = MultiRate.generate_rates(self, forward_bonds, discounts)[:, 1:]
        # First date is irrelevant TODO: change this!
        rate_paths = np.prod(rate_paths, axis=3, keepdims=True)  # By definition
        cumsum = np.cumsum(rate_paths * self.intervals, axis=1)

        paths = np.full(rate_paths.shape, np.nan)
        paths[:, 0] = 0
        paths[:, 1:] = cumsum[:, :-1]

        # Path computation
        shape = np.array(paths.shape)
        shape[1] += 1
        paths_amp = np.full(shape, np.nan)
        paths_amp[:, 0] = np.nan
        # For convention considerations (see mc_engines, the line where paths = paths[:, 1:] right after computing the paths)
        paths_amp[:, 1:] = paths

        return paths_amp


class CMSDiffRate:  # This was a preliminar version of DifferenceRate(MultiRate) class
    """
    CMSDiffRate class for representing the difference between two CMS swap rates.

    Parameters
    ----------
    curve : pricing.discount_curves.DiscountCurve
        Curve needed for discount bonds.
    tenors : list
        List of tenors (in years) for the two CMS swap rates.
    date : str or pandas.Timestamp
        Effective date for the CMS swap rates.
    floating_freq : int
        Floating frequency as number of months.
    fixed_freq : int
        Fixed frequency as number of months.
    legs_calendars : tuple or data.calendars.DayCountCalendar
        Calendar for each leg. If there is only one calendar, same calendar for each leg.
    """

    def __init__(self, curve, tenors, date, floating_freq, fixed_freq, legs_calendars):
        date = pd.to_datetime(date)
        end_dates0 = date + pd.DateOffset(years=tenors[0])
        end_dates1 = date + pd.DateOffset(years=tenors[1])
        end_dates = [end_dates0, end_dates1]
        swap_rate_0 = SwapRate(
            curve,
            date,
            end_dates[0],
            floating_freq,
            fixed_freq,
            legs_calendars,
            tenor_length=tenors[0],
        )
        swap_rate_1 = SwapRate(
            curve,
            date,
            end_dates[1],
            floating_freq,
            fixed_freq,
            legs_calendars,
            tenor_length=tenors[1],
        )
        max_tenor_ind = tenors.index(max(tenors))

        self.swap_rates = [swap_rate_0, swap_rate_1]
        self.swap_rate_max = self.swap_rates[max_tenor_ind]

    def get_forward_dates(self, dates):
        """
        Return dictionary of forward dates.

        Parameters
        ----------
        dates : pandas.DatetimeIndex, list of strings, pandas.Timestamp, or string
            Index of dates.

        Returns
        -------
        dict
            Dictionary with keys the given dates and values the corresponding forward dates.
        """
        return self.swap_rate_max.get_forward_dates(dates)

    def generate_rates(self, forward_bonds, discounts):
        """
        Generate paths of the difference between two CMS swap rates given an array of simulated forward bonds.

        Parameters
        ----------
        forward_bonds : dict
            A dictionary mapping each date :math:`t^1_j` to forward bond prices. Mathematically represented as
            :math:`\\left\\{ t^1_j~:~ \\text{FP}_{i,0,k,l}\\right\\}_{j=0}^J`. In general, this corresponds to
            :math:`\\left\\{ t^1_j~:~ \\left(\\text{FP}_{t^0_k}\\left(t^1_j; T^j_{l-1}, T^j_{l}, \\underline{x}_i\\left(t^0_k, t^1_j\\right)\\right)\\right)_{i, 0, k, l}\\right\\}_{j=0}^J`,
            a dictionary for each date.

        discounts : dict
            A dictionary mapping each date :math:`t^1_j` to discount factors. Mathematically represented as
            :math:`\\left\\{t^1_j~:~ \\left(P_{t^0_k}\\left(t^1_j; T^j_l, \\underline{x}_i\\left(t^0_k, t^1_j\\right)\\right)\\right)_{i,l,k,0}\\right\\}_{j=0}^J`,
            a dictionary for each date.

        Returns
        -------
        np.ndarray
            Array of the difference between two CMS swap rates for the different simulations.
        """
        rate0 = self.swap_rates[0].generate_rates(forward_bonds, discounts)[:, 1:]
        rate1 = self.swap_rates[1].generate_rates(forward_bonds, discounts)[:, 1:]
        diff_rate = rate1[..., [0]] - rate0[..., [0]]

        # Path computation
        paths = diff_rate
        shape = np.array(paths.shape)
        shape[1] += 1
        paths_amp = np.full(shape, np.nan)
        paths_amp[:, 0] = np.nan
        # For convention considerations (see mc_engines, the line where paths = paths[:, 1:] right after computing the paths)
        paths_amp[:, 1:] = paths

        return paths_amp


# ----------------------------------------------------------------------------------------------
# Trade-level "rates"
# ----------------------------------------------------------------------------------------------


class KnockOutRate(Rate):
    """
    Represents a path-dependent knock-out rate, where the knock-out barrier determines whether the process is knocked out.

    Parameters
    ----------
    rates : Rate or tuple of rates
        The underlying rate process. Mathematically represented by :math:`(R_n)_{n=0}^N`.
    barrier : float
        The barrier level for knock-out. Mathematically represented by :math:`B`.
    kind : str
        The kind of knock-out barrier, either 'down-and-out' or 'up-and-out'.

    Notes
    -----
    Given a set of rates :math:`(R_n)_{n=0}^N`, the knock-out rate :math:`\\text{TL}_j` is defined as:

    .. math::
        \\text{TL}_j = \\prod_{i=0}^j \\text{Ind}\\left(\\prod_{n=0}^N R_n(T_i) \\land B\\right),

    where :math:`\\land` is either :math:`>` (for down-and-out) or :math:`<` (for up-and-out). Do not confuse it with :math:`\\min(\\cdot,\\cdot)`.

    Other methods from `MultiRate` are inherited and not overwritten.

    """

    def __init__(self, rates, barrier, kind):
        MultiRate.__init__(self, rates)  # Inheritance of methods
        self.barrier = barrier
        self.kind = kind

    def generate_rates(self, forward_bonds, discounts):
        """
        Generate paths of the knock-out rate process given an array of simulated forward bonds.

        Parameters
        ----------
        forward_bonds : dict
            A dictionary mapping each date :math:`t^1_j` to forward bond prices. Mathematically represented as
            :math:`\\left\\{ t^1_j~:~ \\text{FP}_{i,0,k,l}\\right\\}_{j=0}^J`. In general, this corresponds to
            :math:`\\left\\{ t^1_j~:~ \\left(\\text{FP}_{t^0_k}\\left(t^1_j; T^j_{l-1}, T^j_{l}, \\underline{x}_i\\left(t^0_k, t^1_j\\right)\\right)\\right)_{i, 0, k, l}\\right\\}_{j=0}^J`,
            a dictionary for each date.

        discounts : dict
            A dictionary mapping each date :math:`t^1_j` to discount factors. Mathematically represented as
            :math:`\\left\\{t^1_j~:~ \\left(P_{t^0_k}\\left(t^1_j; T^j_l, \\underline{x}_i\\left(t^0_k, t^1_j\\right)\\right)\\right)_{i,l,k,0}\\right\\}_{j=0}^J`,
            a dictionary for each date.

        Returns
        -------
        np.ndarray
            Array of the knock-out rate process for the different simulations.
            This process is determined by evaluating whether the cumulative product of rates at each point meets the knock-out condition
            defined by the barrier :math:`B` and the specified `kind`. Mathematically represented as:

            .. math::
                \\text{TL}_{i,j,k,l=0} = \\prod_{j'=0}^j \\text{Ind}\\left(\\prod_{n=0}^N R_{ij'kn} \\land B\\right),

        """
        rate_paths = MultiRate.generate_rates(self, forward_bonds, discounts)[
            :, 1:
        ]  # First date is irrelevant TODO: change this!
        rate_paths = np.prod(rate_paths, axis=3, keepdims=True)  # By definition

        if self.kind == "down-and-out":
            boolean = rate_paths > self.barrier
        elif self.kind == "up-and-out":
            boolean = rate_paths < self.barrier
        else:
            raise AttributeError("Invalid kind.")

        paths = np.cumprod(boolean, axis=1)

        # Path computation
        shape = np.array(paths.shape)
        shape[1] += 1
        paths_amp = np.full(shape, np.nan)
        paths_amp[:, 0] = np.nan
        # For convention considerations (see mc_engines, the line where paths = paths[:, 1:] right after computing the paths)
        paths_amp[:, 1:] = paths

        return paths_amp


class GlobalFloorRate(CumulativeRate):
    """
    Represents a path-dependent rate process with a global floor. The global floor level determines the minimum value of the rate process at each point in time.

    Parameters
    ----------
    rates : Rate or tuple of Rate
        The set of underlying rates to be accumulated. Mathematically represented by :math:`(R_n)_{n=0}^N`, where each :math:`R_n` is a rate component.
    floor : float
        The global floor level. Mathematically represented by :math:`F`.
    dates : List[str] or pd.DatetimeIndex
        The dates for which the rates are to be computed. Mathematically represented by :math:`(T_j)_j`, the sequence of dates.
    calendar : data.calendars.DayCountCalendar
        The calendar used for computing day count fractions between dates, denoted by :math:`\\text{DC}`, thus :math:`\\tau_i = \\text{DC}(T_{i-1}, T_i)` represents the time intervals between consecutive dates :math:`T_{i-1}` and :math:`T_i`.

    Notes
    -----
    Given a set of rates :math:`(R_n)_{n=0}^N`, the global floor rate :math:`\\Delta_j` is defined as:

    .. math::
        \\Delta_j = (F - Q_j)_+ \\times \\delta_{jJ},

    where :math:`Q_j` is as in CumulativeRate, and :math:`F` is the global floor level. This formula calculates a coupon paid if the total payments do not reach the given floor :math:`F`. The :math:`\\delta_{jJ}` term ensures that the floor comparison and potential adjustment only occur at the final date :math:`J`.

    Other methods from `CumulativeRate` are inherited and not overwritten.
    """
    def __init__(self, rates, floor, dates, calendar):
        CumulativeRate.__init__(self, rates, dates, calendar)  # Inheritance of methods
        self.floor = floor

    def generate_rates(self, forward_bonds, discounts):
        """
        Generate paths of the rate process with a global floor given an array of simulated forward bonds.

        The method adjusts the cumulative rate paths according to the global floor, ensuring that the rate does not fall below the specified floor level at any point in time.

        Parameters
        ----------
        forward_bonds : dict
            A dictionary mapping each date :math:`t^1_j` to forward bond prices. Mathematically represented as
            :math:`\\left\\{ t^1_j~:~ \\left(\\text{FP}_{t^0_k}\\left(t^1_j; T^j_{l-1}, T^j_{l}, \\underline{x}_i\\left(t^0_k, t^1_j\\right)\\right)\\right)_{i, 0, k, l}\\right\\}_{j=0}^J`,
            a dictionary for each date.

        discounts : dict
            A dictionary mapping each date :math:`t^1_j` to discount factors. Mathematically represented as
            :math:`\\left\\{t^1_j~:~ \\left(P_{t^0_k}\\left(t^1_j; T^j_l, \\underline{x}_i\\left(t^0_k, t^1_j\\right)\\right)\\right)_{i,l,k,0}\\right\\}_{j=0}^J`,
            a dictionary for each date.

        Returns
        -------
        np.ndarray
            An array containing the rate process adjusted by the global floor for each simulation. Specifically, the adjustment is applied as:

            .. math::
                \\Delta_{i,j,k,l=0} = (F - Q_{ijk0})_{+} \\times \\delta_{jJ},

            ensuring that the total payment does not fall below the floor :math:`F`.

        """
        rate_paths = CumulativeRate.generate_rates(self, forward_bonds, discounts)[
            :, 1:
        ]
        paths = (self.floor - rate_paths) * (self.floor >= rate_paths)
        paths[:, :-1] = 0  # Payment only happens at the final date

        # Path computation
        shape = np.array(paths.shape)
        shape[1] += 1
        paths_amp = np.full(shape, np.nan)
        paths_amp[:, 0] = np.nan
        # For convention considerations (see mc_engines, the line where paths = paths[:, 1:] right after computing the paths)
        paths_amp[:, 1:] = paths

        return paths_amp


# ----------------------------------------------------------------------------------------------
# "Child" rates
# ----------------------------------------------------------------------------------------------


class TARNRate(ProductRate, KnockOutRate, CumulativeRate):
    # Some classes added for the inheritance diagram
    """
    Represents a Target Accrual Redemption Note (TARN) rate process, combining the features of knock-out and cumulative rates to model the accrual and redemption mechanism of TARN products.

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

    Notes
    -----
    Given a rate :math:`R`, the TARN rate :math:`C_j` is defined as:

    .. math::
        C_j = R(T_j) \\times \\prod_{i=0}^j\\text{Ind}\\left( Q_i < B\\right)

    where :math:`Q_j` is the cumulative rate defined as:

    .. math::
        Q_j =\\sum_{i=0}^{j-1} \\tau_i R(T_i)

    with :math:`\\tau_i` representing the set of time intervals between dates.

    This model captures the TARN's accrual mechanism and the knock-out feature, allowing for complex rate path dependencies.

    Other methods from `ProductRate`, `KnockOutRate`, and `CumulativeRate` are inherited and not overwritten.
    """

    def __init__(self, rate, dates, calendar, barrier):
        ko_rate = KnockOutRate(
            CumulativeRate(rate, dates, calendar), barrier, kind="up-and-out"
        )
        ProductRate.__init__(self, rate, ko_rate)


class GFlooredRate(GlobalFloorRate):  # Some classes added for the inheritance diagram
    """
    Represents a rate process with a global floor, extending the functionality of the GlobalFloorRate class to model a rate process where a global floor is applied to the cumulative rate process.

    Class Parameters
    ----------
    rate : Rate
        The underlying rate process to which the global floor will be applied. Mathematically represented by :math:`R`.
    floor : float
        The global floor level. Mathematically represented by :math:`F`.
    dates : List[str] or pd.DatetimeIndex
        The dates for which the rate process is to be computed. Mathematically represented by :math:`(T_j)_j`, the sequence of dates.
    calendar : data.calendars.DayCountCalendar
        The calendar used for computing day count fractions between dates, denoted by :math:`\\text{DC}`, thus :math:`\\tau_j = \\text{DC}(T_{j-1}, T_j)`
        represents the time intervals between consecutive dates :math:`T_{i-1}` and :math:`T_i`.

    Notes
    -----
    Given a rate :math:`R`, the GFlooredRate :math:`C_j` is defined as:

    .. math::
        C_j = R(T_j) + \\Delta_j

    where :math:`\\Delta_j` is the global floor adjustment applied at each point in the rate path, as defined in the GlobalFloorRate class.

    This model captures the behavior of a rate process that is subject to a minimum value (floor) at each point in its path, effectively ensuring that the rate does not fall below a specified level.

    Other methods from `GlobalFloorRate` are inherited and not overwritten.
    """

    def __new__(cls, rate, floor, dates, calendar):  # Different syntax to do the same
        floor_rate = GlobalFloorRate(rate, floor, dates, calendar)
        return rate + floor_rate


# ----------------------------------------------------------------------------------------------
# CDS rates
# ----------------------------------------------------------------------------------------------


class CDSRate:  # TODO: THIS CLASS SHOULD BE EVENTUALLY REMOVED.
    # TODO: Note that the rpvbp and the break even spread can be computed within the class CDSCurve. This is why this class is redundant.
    """
    A class for obtaining the CDS break-even spread (as defined in Section 6.6.2 of [O'Kane], although with (unjustified) approximations).

    Parameters
    ----------
    cds_curve :  pricing.discount_curves.CDSCurve
        Survival probability curve constructed from spreads.
    effective_date : str
        The effective date (as defined in Section 5.2 of [O' Kane]) in 'YYYY-MM-DD' format.
    end_date  : str
        Maturity (as defined in Section 5.2 of [O' Kane]) in 'YYYY-MM-DD' format.
    freq : int
        Premium payment frequency in number of months.
    calendar : data.calendars.DayCountCalendar
        Calendar for computation of time intervals.

    References
    ----------
    [O'Kane] O’Kane, D. (2008). Modelling Single-name and Multi-name Credit Derivatives (first edition).
    """

    def __init__(self, cds_curve, effective_date, end_date, freq, calendar):
        # saving data
        self.curve = cds_curve
        self.effective_date = pd.to_datetime(effective_date)
        self.end_date = pd.to_datetime(end_date)
        self.freq = freq
        self.premium_dates = pd.date_range(
            self.end_date, self.effective_date, freq=-pd.DateOffset(months=self.freq)
        )  # TODO: Not IMM dates, which are ALMOST ALWAYS used.
        self.premium_dates = self.premium_dates.union(
            [self.effective_date]
        ).sort_values()
        self.calendar = calendar
        self.deltas = self.calendar.interval(
            self.premium_dates[:-1], self.premium_dates[1:]
        )

    def get_rpvbp(self, dates, discount_curve):
        """
        Return RPV01(t, T) as described in [O'Kane] assuming t are payment dates (so there is no accrued premium).

        Parameters
        ----------
        dates : pandas.DatetimeIndex
            Valuation dates, :math:`t` in [O'Kane].
        discount_curve : pricing.discount_curves.DiscountCurve
            Gives the discounts, :math:`Z(t, t')` in [O'Kane].

        Returns
        -------
        pandas.Series
            RPV01(t, T). The index is a pandas.DatetimeIndex with the dates :math:`t^0_k`. The values of the pd.Series are RPV01 :math:`(t^0_k, T)`.

        References
        ----------
        [O'Kane] O’Kane, D. (2008). Modelling Single-name and Multi-name Credit Derivatives (first edition).
        """
        dates = afsfun.dates_formatting(dates)

        deltas = self.deltas.reshape(self.deltas.size, 1)

        deltas = np.zeros((self.premium_dates.size - 1, dates.size))
        for i in range(dates.size):
            temp_t = self.calendar.interval(dates[i], self.premium_dates[1:])
            deltas[:, i] = temp_t - np.concatenate(([0], temp_t[:-1]))
        Z = np.zeros((self.premium_dates.size, dates.size))
        Q = np.zeros((self.premium_dates.size, dates.size))
        for i in range(dates.size):
            boolean = dates[i] <= self.premium_dates
            fut_pdates = self.premium_dates[boolean]
            Z[boolean, i] = discount_curve.get_value(
                dates[i], fut_pdates, self.calendar
            )
            previous_premium = self.premium_dates[dates[i] >= self.premium_dates][-1]
            boolean = previous_premium <= self.premium_dates
            Q[boolean, i] = self.curve.get_value(
                dates[i], self.premium_dates[boolean], self.calendar
            )
        rpvbp = (1 / 2) * np.sum(deltas * Z[1:] * (Q[:-1] + Q[1:]), axis=0)
        rpvbp = pd.Series(rpvbp, index=dates)
        return rpvbp

    def get_value(self, dates, discount_curve):
        """
        Calculate par spreads for the given valuation dates.

        Parameters
        ----------
        dates : pandas.DatetimeIndex, list of strings, pandas.Timestamp, or string
            Valuation dates. It can be a single date (as a pandas.Timestamp or its string representation) or an array-like object (as a pandas.DatetimeIndex or a list of its
            string representation) containing dates.
        discount_curve : pricing.discount_curves.DiscountCurve
            Provides the discounts, Z(t, t'), as described in [O'Kane].

        Returns
        -------
        numpy.ndarray
            Array of par spreads calculated for the specified valuation dates.

        References
        ----------
        [O'Kane] O’Kane, D. (2008). Modelling Single-name and Multi-name Credit Derivatives (first edition).
        """
        # remember to keep the pricing consistent with the curve calibration
        dates = afsfun.dates_formatting(dates)
        rpvbp = self.get_rpvbp(dates, discount_curve)

        # deltas = self.deltas.reshape(self.deltas.size, 1)

        integrals = np.zeros(dates.size)
        for index, date in np.ndenumerate(dates):
            init_date = date if date >= self.effective_date else self.effective_date
            integral_steps = pd.date_range(
                start=init_date, end=self.end_date, freq="1D"
            )
            # TODO: We don't need one day (one month is enough), see paragraph before Section 6.6.1 in [O' Kane].
            discounts = discount_curve.get_value(date, integral_steps, self.calendar)
            q = self.curve.get_value(date, integral_steps, self.calendar)
            integrals[index[0]] = np.sum(discounts[1:] * (q[:-1] - q[1:]))

        # # the code below approximates the integral badly
        # Z = np.zeros((self.premium_dates.size, dates.size))
        # Q = np.zeros((self.premium_dates.size, dates.size))
        # for index, premium_date in np.ndenumerate(self.premium_dates):
        #     Z[index[0]] = discount_curve.get_value(dates, premium_date, self.calendar) * (dates <= premium_date)
        #     Q[index[0]] = self.curve.get_value(dates, premium_date, self.calendar) * (dates <= premium_date)
        # # CORRECT THE FORMULA
        # # multiplying by boolean removes difference between the first future premium dates and the last premium date
        # integrals = np.sum(Z[1:]*(Q[:-1] - Q[1:]) * (Q[:-1] > 0), axis=0)

        par_spreads = (1 - self.curve.recovery) * integrals / rpvbp
        return par_spreads


# -----------------------------------------------------------------------
# Compounding curve
# -----------------------------------------------------------------------


class FixedTenorRate:
    """
    Represents a simply-compounded interest rate with a fixed tenor.

    Parameters
    ----------
    spot_curve : pricing.discount_curves.DiscountCurve
        The spot curve from which the interest rates are derived.
    days : int, optional
        Number of days for the fixed tenor. Default is 0.
    months : int, optional
        Number of months for the fixed tenor. Default is 0.
    """

    def __init__(self, spot_curve, days=0, months=0):
        self.spot_curve = spot_curve
        self.tenor = (days, months)  # S-T
        curve_data = np.array(self.spot_curve.interpolation_data.columns)
        if self.tenor == (1, 0):
            boolean = (curve_data > 1 / 361) * (curve_data < 1 / 359)
            if np.max(boolean):
                self.column = curve_data[boolean][0]
        elif self.tenor == (0, 1):
            boolean = (curve_data > 1 / 12) * (curve_data < 1 / 11)
            if np.max(boolean):
                self.column = curve_data[boolean][0]
        else:
            boolean = (
                self.tenor[0] + self.tenor[1] * 30
            ) / 360 in self.spot_curve.interpolation_data.columns
            if boolean:
                self.column = (self.tenor[0] + self.tenor[1] * 30) / 360

    def get_value(self, dates):
        """
        Calculate simply-compounded spot interest rates for the given dates. The simply-compounded spot interest rates are calculated based on the spot curve and the fixed tenor.

        Parameters
        ----------
        dates : pandas.DatetimeIndex, list of strings, pandas.Timestamp, or string
            Dates for which to calculate the spot interest rates.

        Returns
        -------
        pandas.Series
            Series of simply-compounded spot interest rates calculated for the specified dates.

        References
        ----------
        [Brigo and Mercurio, 2006] Brigo & Mercurio (2006). Interest Rate Models–Theory and Practice.

        """

        # Simply-compounded spot interest rate.

        dates = afsfun.dates_formatting(dates)
        values_df = pd.Series(index=dates)
        if hasattr(self, "column"):
            existing_dates = dates[dates.isin(self.spot_curve.interpolation_data.index)]
            values_df.loc[existing_dates] = self.spot_curve.interpolation_data.loc[
                existing_dates, self.column
            ]
        else:
            future_dates = dates + pd.DateOffset(
                days=self.tenor[0], months=self.tenor[1]
            )
            values = self.spot_curve.get_value(
                dates, future_dates, self.spot_curve.calendar
            )
            values_df.loc[dates] = (1 - values) / (
                values * (self.tenor[0] + self.tenor[1] * 30) / 360
            )  # Eq.(1.8) in [Brigo and Mercurio, 2006].
        return values_df

    def get_fvalue(self, dates, future_dates, calendar=None):
        """
        Calculate simply-compounded forward interest rates between dates and future_dates.

        Parameters
        ----------
        dates : pandas.DatetimeIndex, list of strings, pandas.Timestamp, or string
            Start dates for the forward interest rates.
        future_dates : pandas.DatetimeIndex, list of strings, pandas.Timestamp, or string
            End dates for the forward interest rates.
        calendar : data.calendars.DayCountCalendar, optional
            Calendar used to calculate the dates. Default is ``None``.

        Returns
        -------
        numpy.ndarray
            Array of forward interest rates calculated between ``dates`` and ``future_dates``.

        References
        ----------
        [Brigo and Mercurio, 2006] Brigo & Mercurio (2006). Interest Rate Models–Theory and Practice.
        """
        dates = pd.to_datetime(dates)
        future_dates = pd.to_datetime(future_dates)

        fvalue = np.full(
            max(np.asarray(dates).size, np.asarray(future_dates).size), np.nan
        )
        fvalue[future_dates <= dates] = self.get_value(
            future_dates[future_dates <= dates]
        )  # Simply-compounded spot interest rate if future_dates <= dates.
        real_future_dates = future_dates[dates < future_dates]  # T
        spot_dates = (
            dates if np.asarray(dates).size == 1 else dates[dates < future_dates]
        )  # t
        first_bond = self.spot_curve.get_value(spot_dates, real_future_dates)  # P(t,T)
        next_bond = self.spot_curve.get_value(
            spot_dates,
            real_future_dates + pd.DateOffset(days=self.tenor[0], months=self.tenor[1]),
        )  # P(t,S)
        fvalue[dates < future_dates] = (first_bond - next_bond) / (
            next_bond * (self.tenor[0] + self.tenor[1] * 30) / 360
        )  # Eq. (1.20) in [Brigo and Mercurio, 2006] assuming
        # 30/360 day-count convention
        return fvalue


class ShortRate(FixedTenorRate):
    """
    Represents a short-term, annually-compounded interest rate derived from a spot curve.

    Parameters
    ----------
    spot_curve : pricing.discount_curves.DiscountCurve
        The spot curve from which the interest rates are derived.
    """

    def __init__(self, spot_curve):
        FixedTenorRate.__init__(self, spot_curve, days=1)

    def get_fvalue(self, dates, future_dates, calendar=None):
        """
        Forward rates (annually-compounded) between ``dates`` and ``future_dates``.

        Parameters
        ----------
        dates : pandas.DatetimeIndex, list of strings, pandas.Timestamp, or string
            Start dates for the forward interest rates.
        future_dates : pandas.DatetimeIndex, list of strings, pandas.Timestamp, or string
            End dates for the forward interest rates.
        calendar : data.calendars.DayCountCalendar, optional
            Calendar used to calculate the dates. Default is ``None``.

        Returns
        -------
        numpy.ndarray
            Array of forward rates calculated between dates and future_dates (both included).

        References
        ----------
        [Brigo and Mercurio, 2006] Brigo & Mercurio (2006). Interest Rate Models–Theory and Practice.
        """
        dates = pd.to_datetime(dates)
        future_dates = pd.to_datetime(future_dates)

        fvalue = np.full(
            max(np.asarray(dates).size, np.asarray(future_dates).size), np.nan
        )
        fvalue[future_dates <= dates] = self.get_value(
            future_dates[future_dates <= dates]
        )
        real_future_dates = future_dates[dates < future_dates]  # T
        spot_dates = (
            dates if np.asarray(dates).size == 1 else dates[dates < future_dates]
        )  # t
        first_bond = self.spot_curve.get_value(spot_dates, real_future_dates)
        next_bond = self.spot_curve.get_value(
            spot_dates,
            real_future_dates + pd.DateOffset(days=self.tenor[0], months=self.tenor[1]),
        )
        fvalue[dates < future_dates] = (first_bond / next_bond) ** (
            1 / 360
        ) - 1  # See Eq. (1.20) in [Brigo and Mercurio, 2006] (same formula assuming annually-compounding).
        return fvalue


class CompoundingCurve:
    """
    Represents a compounding curve derived from a short-term, annually-compounded interest rate.

    Parameters
    ----------
    shortrate : ShortRate
        The short-term, annually-compounded interest rate curve from which the compounding curve is derived.
    tenor : int
        The compounding tenor, in days, used to calculate the compounded rates.
    calendar : data.calendars.DayCountCalendar
        Calendar used to calculate the compounding intervals.
    """

    def __init__(self, shortrate, tenor, calendar):
        """ "
        tenor: integer
        """
        self.shortrate = shortrate
        self.tenor = tenor
        self.calendar = calendar

    def get_value(self, dates):
        """
        Calculate compounded rates for the specified dates.

        Parameters
        ----------
        dates : pandas.DatetimeIndex, list of strings, pandas.Timestamp, or string
            Dates for which compounded rates are calculated.

        Returns
        -------
        pandas.Series
            Series of compounded rates calculated for the specified dates.
        """
        dates = afsfun.dates_formatting(dates)
        values = pd.Series(index=dates, dtype="float64")
        for date in dates:
            days = self.calendar.days_in_interval(date, self.tenor)
            rates = self.shortrate.get_value(days)
            weights = self.calendar.weights(days)
            compounding = np.prod(1 + weights * rates / 365) - 1
            values.loc[date] = compounding * 365 / self.tenor
        return values


class SRCLookback(CompoundingCurve):
    """
    Represents a compounding curve derived from a short-term, annually-compounded interest rate with a lookback feature.

    Parameters
    ----------
    shortrate :  CompoundingCurve
        The short-term, annually-compounded interest rate curve from which the compounding curve is derived.
    tenor : int
        The compounding tenor, in days, used to calculate the compounded rates.
    lookback_period : int, optional
        The number of days to look back when calculating the compounding rates. Default is 5 days.
    """

    def __init__(self, shortrate, tenor, lookback_period=5):
        CompoundingCurve.__init__(self, shortrate, tenor)
        self.lookback_period = lookback_period

    def get_value(self, dates):
        """
        Calculate compounded rates with a lookback feature for the specified dates.

        Parameters
        ----------
        dates : pandas.DatetimeIndex, list of strings, pandas.Timestamp, or string
            Dates for which compounded rates with a lookback feature are calculated.

        Returns
        -------
        pandas.Series
            Series of compounded rates with a lookback feature calculated for the specified dates.
        """
        dates = afsfun.dates_formatting(dates)
        obsdates = dates - pd.Timedelta("{} days".format(self.lookback_period))
        values = CompoundingCurve.get_value(self, obsdates)
        values = pd.Series(values.values, index=dates)
        return values


class SRCInAdvance(CompoundingCurve):
    """
    Represents a compounding curve derived from a short-term, annually-compounded interest rate with rates calculated in advance.

    Parameters
    ----------
    shortrate : CompoundingCurve
        The short-term, annually-compounded interest rate curve from which the compounding curve is derived.
    tenor : int
        The compounding tenor, in days, used to calculate the compounded rates.
    """

    def __init__(self, shortrate, tenor):
        CompoundingCurve.__init__(self, shortrate, tenor)

    def get_value(self, dates):
        """
        Calculate compounded rates in advance for the specified dates.

        Parameters
        ----------
        dates : pandas.DatetimeIndex, list of strings, pandas.Timestamp, or string
            Dates for which compounded rates in advance are calculated.

        Returns
        -------
        pandas.Series
            Series of compounded rates in advance calculated for the specified dates.
        """
        dates = afsfun.dates_formatting(dates)
        obsdates = dates - pd.Timedelta("{} days".format(self.tenor))
        values = CompoundingCurve.get_value(self, obsdates)
        values = pd.Series(values.values, index=dates)
        return values
