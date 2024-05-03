from datetime import datetime

import pandas as pd
from blp import Blp
from dateutil.relativedelta import relativedelta


def banner(msg):
    print('*' * 25)
    print(msg)
    print('*' * 25)


d = pd.date_range(start=datetime.today() - relativedelta(days=4), end=datetime.today(), freq='B')
m = pd.date_range(start=datetime.today() - relativedelta(months=2), end=datetime.today(), freq='BM')

blp = Blp()

banner('ReferenceDataRequest: single security, single field, frame response')
resp = blp.get_reference_data('msft us equity', 'px_last')
print(resp.as_dict())
print(resp.as_dataframe())

banner('ReferenceDataRequest: single security, multi-field (with bulk), frame response')
resp = blp.get_reference_data('eurusd curncy', ['px_last', 'fwd_curve'])
print(resp.as_dict())
df = resp.as_dataframe()
print(df.columns)
# show frame within a frame
print(df['fwd_curve'].iloc[0].tail())

banner('ReferenceDataRequest: multi security, multi-field, bad field')
resp = blp.get_reference_data(
    ['eurusd curncy', 'msft us equity'], ['px_last', 'fwd_curve'], raise_field_error=False
)
print(resp.as_dataframe()['fwd_curve']['eurusd curncy'])

banner('HistoricalDataRequest: multi security, multi-field, daily data')
resp = blp.get_historical(['eurusd curncy', 'msft us equity'], ['px_last', 'px_open'], start=d)
print(resp.as_dict())
print(resp.as_dataframe().head(5))

banner('HistoricalDataRequest: multi security, multi-field, weekly data')
resp = blp.get_historical(
    ['eurusd curncy', 'msft us equity'], ['px_last', 'px_open'], start=m, period='WEEKLY'
)
print('--------- AS SINGLE TABLE ----------')
print(resp.as_dataframe().head(5))

#
# HOW TO
#
# - Retrieve an fx vol surface:  BbgReferenceDataRequest('eurusd curncy', 'DFLT_VOL_SURF_MID')
# - Retrieve a fx forward curve:  BbgReferenceDataRequest('eurusd curncy', 'FWD_CURVE')
# - Retrieve dividends:  BbgReferenceDataRequest('csco us equity', 'BDVD_PR_EX_DTS_DVD_AMTS_W_ANN')
