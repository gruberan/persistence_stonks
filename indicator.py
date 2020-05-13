import numpy as np
from ripser import ripser
from persim import plot_diagrams
import matplotlib.pyplot as plt
import math
import datetime as dt
from matplotlib import style
import pandas as pd
import pandas_datareader.data as web
import pandas_market_calendars as mcal
from persistence_landscapes.landscapes import Landscape_Reader
from dateutil.parser import isoparse

def log_returns(df):
    """
    Converts a dataframe (or list) of daily close prices to a dataframe of daily log returns, i.e. ln(price today/ price yesterday).
    """
    if type(df) == list:
        return [[math.log(a[i+1][j]/(a[i][j])) for j in range(0,len(a[i]))] for i in range(0,len(a)-1)]
    else:
        a = list(df.index)
        dfdict = df.T.apply(tuple).to_dict()
        return [[math.log(dfdict[a[i+1]][j]/dfdict[a[i]][j]) for j in range(0,len(dfdict[a[i]]))] for i in range(0,len(a)-1)]
def sliding_point_cloud(a, width):
    """
    Returns a sliding window point cloud from a list (or dataframe) of points.
    """
    if type(a) == list:
        return [a[i:i+width] for i in range(0,len(a)-width)]
    else:
        ind = list(df.index)
        dfdict = df.T.apply(tuple).to_dict()
        return [[dfdict[d] for d in ind[i-width:i]] for i in range(width,len(ind))]

def stonk_df(start_dt, end_dt, ticker_symbol_list):
    """
    Reads close prices from each of the tickers in ticker_symbol_list and returns in a dataframe.
    """
    return pd.concat(
        [web.DataReader(sym, 'yahoo', start_dt, end_dt)['Close'] for sym in ticker_symbol_list],
        axis=1,
        keys=ticker_symbol_list
    )
    
def date_by_subtracting_exchange_calendar_days(from_dt, num_days, exchange='NYSE'):
    """
    Returns the date that is num_days before from_date, excluding days when the NYSE is not open.
    
    Arguments:
        from_dt (str or datetime.datetime): date to subtract from
        num_days (int): number of days to subtract
    """
    cal = mcal.get_calendar(exchange)
    from_date = isoparse(from_dt) if type(from_dt) == str else from_dt
    schedule = cal.valid_days(start_date=from_date - dt.timedelta(days=2*num_days), end_date=from_date)
    calendar_days_to_subtract = num_days
    current_date = from_date
    while calendar_days_to_subtract > 0:
        current_date -= dt.timedelta(days=1)
        if current_date in schedule:
            calendar_days_to_subtract -= 1
    return current_date
    
def compute_window_landscapes(start_dt, end_dt, ticker_symbol_list, window_size, maxdim=1):
    """
        
        start_dt (str or datetime.datetime): first date to compute the landscapes for (note that window_size calendar dates must be retrievable before this date)
        start_dt (str or datetime.datetime): last date to compute the landscapes for
        maxdim (int): Maximum homology dimension computed. Will compute all dimensions lower than and equal to this value. For 1, H_0 and H_1 will be computed.
    """
    start_date = date_by_subtracting_exchange_calendar_days(isoparse(start_dt),window_size)
    end_date = isoparse(end_dt) if type(end_dt) == str else end_dt
    
    df = stonk_df(start_dt = start_date, end_dt = end_date, ticker_symbol_list=ticker_symbol_list)
    lfd = log_returns(df)
    sliding_4D = sliding_point_cloud(lfd,window_size)
    
    landscapes = {i : [] for i in range(maxdim+1)}
    integrals = {i : [np.nan] * (window_size + 1) for i in range(maxdim+1)}
    for i in range(len(sliding_4D)):
        dg = ripser(pd.DataFrame(sliding_4D[i]), maxdim=maxdim)['dgms']
        for j in range(maxdim+1):
            land = Landscape_Reader.read_fromlist(dg[j])
            landscapes[j] += [land]
            a = land.integrate()
            integrals[j] += [a]
    for j in range(maxdim+1):
        df['L'+str(j)] = integrals[j]
    return df
  
def plot_window_landscapes(start_dt, end_dt, ticker_symbol_list, window_size, row_syms=["^GSPC","L0","L1"],highlight_dates=[],savefig=None):
    highlight_dates = [isoparse(x) if type(x) == str else x for x in highlight_dates]
    
    df = compute_window_landscapes(start_dt = start_dt, end_dt = end_dt, ticker_symbol_list=ticker_symbol_list, window_size= window_size)

    new_row_syms = [x for x in row_syms if x not in ticker_symbol_list and x not in ['L0','L1']]
    if len(new_row_syms) > 0:
        new_row_syms_df = stonk_df(start_dt=start_dt,end_dt=end_dt,ticker_symbol_list=new_row_syms)
        for row_sym in new_row_syms:
            df[row_sym] = new_row_syms_df[row_sym]
    
    fig, axes = plt.subplots(nrows=len(row_syms))
    
    width=window_size+1
    for ax, row_sym in zip(axes[:-1], row_syms[:-1]):
        ax.set_ylabel(row_sym)
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        for dte in highlight_dates:
            ax.axvline(dte, alpha=0.5)
    axes[-1].set_ylabel(row_syms[-1])
    axes[-1].set_yticklabels([])
    xlabels = []
    for x in df.index[width:]:
        if isoparse(str(x)) in highlight_dates:
            xlabels.append(x)
        else:
            xlabels.append('')
    for dte in highlight_dates:
            if type(highlight_dates) == dict:
                axes[-1].axvline(dte, label=highlight_dates[dte], alpha=0.5)
            else:
                axes[-1].axvline(dte, label=dte, alpha=0.5)
    plt.setp(axes[-1].get_xticklabels(), rotation=30, horizontalalignment='right')
    
    colors = ('k', 'r', 'b')
    colors = [colors[i%3] for i in range(len(row_syms))]
    xs = [df.index[width:]] * len(row_syms)
    ys = []
    for sym in row_syms:
        if sym in ['L0','L1']:
            ys.append(df[sym][width:])
        else:
            ys.append(df[sym].iloc[width:])
    for ax, color, x, y in zip(axes, colors, xs, ys):
        ax.plot(x, y, color=color)
        
    if savefig != None:
        plt.savefig(savefig)
    plt.show()
  