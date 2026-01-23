import pandas as pd
import pytest

from blp import Blp
from opendate import Date, DateTime


@pytest.fixture(scope='module')
def LocalTerminal(request):
    return Blp()


D = pd.date_range(start=Date.today().subtract(days=4), end=Date.today(), freq='B')
M = pd.date_range(start=Date.today().subtract(months=1), end=Date.today(), freq='BME')


def test_reference_data_request_single_security_single_field_frame_response(LocalTerminal):
    resp = LocalTerminal.get_reference_data(
        'msft us equity', ['px_last', 'last_update', 'time_of_last_news_story']
    )
    print(resp.as_dict())
    print(resp.as_dataframe())


def test_reference_data_request_single_security_single_field_frame_response_invalid(LocalTerminal):
    resp = LocalTerminal.get_reference_data(
        'foobar us equity', ['px_last', 'last_update', 'time_of_last_news_story']
    )
    print(resp.as_dict())
    print(resp.as_dataframe())


def test_reference_data_request_single_security_multi_field_frame_response(LocalTerminal):
    resp = LocalTerminal.get_reference_data('eurusd curncy', ['px_last', 'fwd_curve'])
    print(resp.as_dict())
    rframe = resp.as_dataframe()
    print(rframe.columns)
    # show frame within a frame
    print(rframe.iloc[0]['fwd_curve'].tail())


def test_reference_data_request_multi_security_multi_field_bad_field(LocalTerminal):
    resp = LocalTerminal.get_reference_data(
        ['eurusd curncy', 'msft us equity'],
        ['px_last', 'fwd_curve'],
        raise_field_error=False,
    )
    print(resp.as_dataframe()['fwd_curve']['eurusd curncy'])


def test_historical_data_request_multi_security_multi_field_daily_data(LocalTerminal):
    resp = LocalTerminal.get_historical(
        ['eurusd curncy', 'msft us equity'], ['px_last', 'px_open'],
        start=Date.today().subtract(days=4))
    print(resp.as_dict())
    print(resp.as_dataframe().head(5))


def test_historical_data_request_multi_security_multi_field_daily_data_invalid(LocalTerminal):
    resp = LocalTerminal.get_historical(
        ['foobar uq equity'], ['px_last', 'px_open'],
        start=Date.today().subtract(days=4))
    print(resp.as_dict())
    print(resp.as_dataframe())


def test_historiacal_data_request_multi_security_multi_field_weekly_data(LocalTerminal):
    resp = LocalTerminal.get_historical(
        ['eurusd curncy', 'msft us equity'],
        ['px_last', 'px_open'],
        start=Date.today().subtract(months=1),
        period='WEEKLY',
    )
    print('--------- AS SINGLE TABLE ----------')
    print(resp.as_dataframe().head(5))


def test_yield_curve_request(LocalTerminal):
    """Or hard code curve_members and make a single call
    ['curve_tenor_rates', 'curve_members'],
    """
    index = 'YCSW0045 Index'
    field = 'curve_tenor_rates'
    resp = LocalTerminal.get_reference_data(index, [field])
    tick = {x['Tenor']: x['Tenor Ticker'] for x in resp.as_dict()[index][field].to_dict('records')}



    dates = [DateTime(2024, 12, 3), DateTime(2024, 12, 4), DateTime(2024, 12, 5)]
    resp = LocalTerminal.get_historical(
        list(tick.values()),
        ['px_last'],
        start=dates[0],
        end=dates[-1],
    )
    # we will use this resp to build a list of date to tenor->price
    print('--------- AS SINGLE TABLE ----------')
    print(resp.as_dataframe())

#
# HOW TO
#
# - Retrieve an fx vol surface:  BbgReferenceDataRequest('eurusd curncy', 'DFLT_VOL_SURF_MID')
# - Retrieve a fx forward curve:  BbgReferenceDataRequest('eurusd curncy', 'FWD_CURVE')
# - Retrieve dividends:  BbgReferenceDataRequest('csco us equity', 'BDVD_PR_EX_DTS_DVD_AMTS_W_ANN')


if __name__ == '__main__':
    __import__('pytest').main([__file__])
