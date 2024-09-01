import sys, os, json
import time
import plotly.express as px
import plotly.graph_objs as go
import plotly.subplots as sp
import numpy as np
import pandas as pd
from scipy.stats import norm, gmean, multivariate_normal
import matplotlib.pyplot as plt
import plotly.io as pio

pio.renderers.default = "notebook+pdf"
pypricing_directory = os.path.expanduser("~/ArfimaTools/pypricing")
sys.path.insert(1, pypricing_directory)
import afslibrary as afs

db_directory = os.path.expanduser("~/ArfimaTools/afsdb")
sys.path.insert(1, db_directory)
import db_tools

beautiful_data = db_tools.BeautifulDataAFSStyle()
db = db_tools.BeautifulDataAFSStyleXL()
d = afs.DataFactory(beautiful_data)
factory = afs.DataFactory(db)

equity = factory.import_underlying("SX5E", "SPX", "RTY", "MT")
calendars = d.import_calendar("Act360", "Act365", "Cal30360")
assets = factory.import_underlying("SX5E", "SPX", "RTY", "MT")
discount_curves = factory.import_discount_curves(
    "EURIBOR", "USD LIBOR", "ESTR", start_date="20200101", end_date="20231231"
)
