from datetime import datetime

import pandas as pd
from bbg.api import Context
from dateutil.relativedelta import relativedelta

if __name__ == '__main__':
    d = pd.date_range(start=datetime.today() - relativedelta(days=4), end=datetime.today(), freq='B')
    m = pd.date_range(start=datetime.today() - relativedelta(months=2), end=datetime.today(), freq='BM')

    def banner(msg):
        print('*' * 25)
        print(msg)
        print('*' * 25)

    service = Context()

    banner('ReferenceDataRequest: single security, single field, frame response')
    response = service.get_reference_data('msft us equity', 'px_last')
    print(response.as_dictionary())
    print(response.as_dataframe())

    banner('ReferenceDataRequest: single security, multi-field (with bulk), frame response')
    response = service.get_reference_data('eurusd curncy', ['px_last', 'fwd_curve'])
    print(response.as_dictionary())
    rframe = response.as_dataframe()
    print(rframe.columns)
    # show frame within a frame
    print(rframe['fwd_curve'].iloc[0].tail())

    banner('ReferenceDataRequest: multi security, multi-field, bad field')
    response = service.get_reference_data(
        ['eurusd curncy', 'msft us equity'], ['px_last', 'fwd_curve'], ignore_field_error=1
    )
    print(response.as_dataframe()['fwd_curve']['eurusd curncy'])

    banner('HistoricalDataRequest: multi security, multi-field, daily data')
    response = service.get_historical(['eurusd curncy', 'msft us equity'], ['px_last', 'px_open'], start=d)
    print(response.as_dictionary())
    print(response.as_dataframe().head(5))

    banner('HistoricalDataRequest: multi security, multi-field, weekly data')
    response = service.get_historical(
        ['eurusd curncy', 'msft us equity'], ['px_last', 'px_open'], start=m, period='WEEKLY'
    )
    print('--------- AS SINGLE TABLE ----------')
    print(response.as_dataframe().head(5))

    #
    # HOW TO
    #
    # - Retrieve an fx vol surface:  BbgReferenceDataRequest('eurusd curncy', 'DFLT_VOL_SURF_MID')
    # - Retrieve a fx forward curve:  BbgReferenceDataRequest('eurusd curncy', 'FWD_CURVE')
    # - Retrieve dividends:  BbgReferenceDataRequest('csco us equity', 'BDVD_PR_EX_DTS_DVD_AMTS_W_ANN')
