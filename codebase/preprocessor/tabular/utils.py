"""A collection of utility functions for tabular data processing."""
from typing import Callable
import numpy as np
import pandas as pd


def calc_interval(df: pd.DataFrame, name: str,
                  start: str, end: str, unit='m'):
    """
    This function calculates the interval between two dates.
    each date is a column in the dataframe.

    Args:
        df: original dataframe
        start: column name for the start date
        end: column name for the end date
        unit: either h(hour) or d(day) or m(month)

    Returns:
        df with calculation
    """
    deltaT = pd.to_datetime(df[end]) - pd.to_datetime(df[start])

    if unit.lower() == 'd':
        df[name] = (deltaT / pd.Timedelta(days=1)).round()
    elif unit.lower() == 'm':
        df[name] = ((deltaT / pd.Timedelta(days=1)) / 30.4).round()
    elif unit.lower() == 'h':
        df[name] = (deltaT / pd.Timedelta(hours=1)).round()
    else:
        raise ValueError(f'Unit {unit} is not supported.')


def calc_gradient(df: pd.DataFrame, start_pt: tuple[str, str],
                  end_pt: tuple[str, str], name: str,
                  unit='m'):
    """Calculate gradients of a time sequence.
    Args:
        df: original dataframe
        start_pt: tuple[time_pt_name, value_name] for start
        end_pt: tuple[time_pt_name, value_name] for end
        name: name for the newly added columns
    """
    start_time, start_value = start_pt
    end_time, end_value = end_pt
    calc_interval(df, name + '_timediff', start_time, end_time, unit)
    df[name + '_delta'] = (df[end_value] - df[start_value]).astype(float)
    df[name + '_gradient'] = df[name + '_delta'].div(df[name + '_timediff'])


def extract_sequence_statistics(
        df: pd.DataFrame, name: str, x_str: list[str],
        y_str: list[str], func: Callable):
    """Calculates a specific statistic for the sequence.
    Args:
        df: the data
        name: statistics to be named by the user
        x: sequence coordinates
        y: sequence values
        func: statistic function to be called
    """
    df[name] = np.nan
    time_name = 'time_to_' + name
    df[time_name] = np.nan
    for idx in df.index:
        x = df.loc[idx, x_str].to_numpy()
        y = df.loc[idx, y_str].to_numpy()
        value = func(y)
        df.loc[idx, name] = value
        # find the first occurence of the value
        item_index = np.where(y == value)[0][0]
        df.loc[idx, time_name] = x[item_index]
