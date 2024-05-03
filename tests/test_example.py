from datetime import datetime

import pandas as pd
import pytest
from blp import Blp
from dateutil.relativedelta import relativedelta


@pytest.fixture(scope='module')
def LocalTerminal(request):
    return Blp()


d = pd.date_range(start=datetime.today() - relativedelta(days=4), end=datetime.today(), freq='B')
m = pd.date_range(start=datetime.today() - relativedelta(months=2), end=datetime.today(), freq='BME')


def test_reference_data_request_single_security_single_field_frame_response(LocalTerminal):
    response = LocalTerminal.get_reference_data('msft us equity', 'px_last')
    print(response.as_dict())
    print(response.as_dataframe())


def test_reference_data_request_single_security_multi_field_frame_response(LocalTerminal):
    response = LocalTerminal.get_reference_data('eurusd curncy', ['px_last', 'fwd_curve'])
    print(response.as_dict())
    rframe = response.as_dataframe()
    print(rframe.columns)
    # show frame within a frame
    print(rframe.iloc[0]['fwd_curve'].tail())


def test_reference_data_request_multi_security_multi_field_bad_field(LocalTerminal):
    response = LocalTerminal.get_reference_data(['eurusd curncy', 'msft us equity'], ['px_last', 'fwd_curve'],
                                                raise_field_error=False)
    print(response.as_dataframe()['fwd_curve']['eurusd curncy'])


def test_historical_data_request_multi_security_multi_field_daily_data(LocalTerminal):
    response = LocalTerminal.get_historical(['eurusd curncy', 'msft us equity'], ['px_last', 'px_open'], start=d)
    print(response.as_dict())
    print(response.as_dataframe().head(5))


def test_historiacal_data_request_multi_security_multi_field_weekly_data(LocalTerminal):
    response = LocalTerminal.get_historical(['eurusd curncy', 'msft us equity'], ['px_last', 'px_open'], start=m,
                                            period='WEEKLY')
    print('--------- AS SINGLE TABLE ----------')
    print(response.as_dataframe().head(5))

#
# HOW TO
#
# - Retrieve an fx vol surface:  BbgReferenceDataRequest('eurusd curncy', 'DFLT_VOL_SURF_MID')
# - Retrieve a fx forward curve:  BbgReferenceDataRequest('eurusd curncy', 'FWD_CURVE')
# - Retrieve dividends:  BbgReferenceDataRequest('csco us equity', 'BDVD_PR_EX_DTS_DVD_AMTS_W_ANN')


if __name__ == '__main__':
    __import__('pytest').main([__file__])
