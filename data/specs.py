import pandas as pd

try:
    from .underlyings import *  # (Relative) import needed for the workspace. In this case __package__ is pypricing.data
except (ImportError, ModuleNotFoundError, ValueError):
    from data.underlyings import *  # (Absolute) local import
# TODO: Include here (or in a different script specs_products.py) the specs for PRODUCTS (like structured products).
#  This could be done using the script for automatically reading the necessary attributes (2023_05_16_attributes). We don't need this data in the database.


# # # #  DISCOUNT CURVES SPECS
# # Previously based on specs-curves.xlsx.
# # Curves imported when import_discount_curves is used.
# # Metadata of each curve (like Day Count/calendar) can be found in
# # https://strata.opengamma.io/indices/#:~:text=calendar%20data%20available.-,Ibor%20Indices%3A,-An%20Ibor%20index

# # discount_curves is dict of tuples with the following format for each discount curve:
# Discount curve name : {'kind of curve' (str), 'Day Count' (str), 'Tickers' (tuple)}
# In this case 'Tickers' is a tuple of strings with the tickers from beautifulData used for constructing the curve.
# For swap rates used in constructing the USD LIBOR curve, the floating leg corresponds to the 3-month libor index.
# For swap rates used in constructing the EURIBOR curve, the floating leg corresponds to the 6-month euribor index.

discount_curves = {
    "USD LIBOR": (
        "irsw",
        "Act360",
        (
            "LIBORT1MDX",
            "LIBORT3MDX",
            "LIBORT6MDX",
            "USDAM3LT1YDX",
            "USDAM3LT2YDX",
            "USDAM3LT3YDX",
            "USDAM3LT4YDX",
            "USDAM3LT5YDX",
            "USDAM3LT6YDX",
            "USDAM3LT7YDX",
            "USDAM3LT10YDX",
            "USDAM3LT12YDX",
            "USDAM3LT15YDX",
            "USDAM3LT20YDX",
            "USDAM3LT25YDX",
            "USDAM3LT30YDX",
            "USDAM3LT40YDX",
            "USDAM3LT50YDX",
        ),
    ),
    "ESTR": (
        "irsw",
        "Act360",
        (
            "EESWET3MDX",
            "EESWET1YDX",
            "EESWET2YDX",
            "EESWET3YDX",
            "EESWET4YDX",
            "EESWET5YDX",
            "EESWET7YDX",
            "EESWET10YDX",
            "EESWET20YDX",
            "EESWET30YDX",
        ),
    ),
    "EURIBOR": (
        "irsw",
        "Act360",
        (
            "ERT1MDX",
            "ERT3MDX",
            "ERT6MDX",
            "ERT1YDX",
            "ERT2YDX",
            "ERT3YDX",
            "ERT4YDX",
            "ERT5YDX",
            "ERT6YDX",
            "ERT7YDX",
            "ERT8YDX",
            "ERT9YDX",
            "ERT10YDX",
            "ERT11YDX",
            "ERT12YDX",
            "ERT13YDX",
            "ERT14YDX",
            "ERT15YDX",
            "ERT16YDX",
            "ERT17YDX",
            "ERT18YDX",
            "ERT19YDX",
            "ERT20YDX",
            "ERT21YDX",
            "ERT22YDX",
            "ERT23YDX",
            "ERT24YDX",
            "ERT25YDX",
            "ERT26YDX",
            "ERT27YDX",
            "ERT28YDX",
            "ERT29YDX",
            "ERT30YDX",
            "ERT40YDX",
            "ERT50YDX",
        ),
    ),
}

# # # #  UNDERLYING Specs (from specs-curves.xlsx)
# # # #  Objects imported when import_underlying is used.

# Products available in beautifulData (2023/08/24)
underlying_products = ["SPX", "SX5E", "RTY", "MT"]  # TODO: Update


# # # #  UNDERLYING Dynamics implemented
# # # #  Types of dynamics implemented. These are, essentially, the methods in underlyings.py with a generate_paths method

underlying_dynamics = [
    "NormalAsset",
    "LognormalAsset",
    "MultiAsset",
    "ExposureIndex",
    "Heston",
    "SABR",
    "MultiAssetHeston",
]

underlying_dynamics_classes = [
    globals()[name] for name in underlying_dynamics if name in globals()
]
