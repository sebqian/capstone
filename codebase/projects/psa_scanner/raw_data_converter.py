"""This is a module to convert original csv file into the data could be used for training.
    It does the following:
    1) Create sequence data: every n consecutive psa are combined into an example
    2) convert date to time intervel
    3) calculate the psa gradient
    4) convert category data into numerical data or one hot.
"""
from typing import Union
from pathlib import Path
import pandas as pd

_USELESS_COLUMNS = ['Primary Staging Date Format', 'StartDateADT']


def convert_date_to_interval(df: Union[pd.DataFrame, pd.Series],
                             name: str, start: str, end: str, unit='m'):
    """
    This function calculates the interval between two dates.
    each date is a column in the dataframe.

    Args:
        df: original dataframe
        name: new column name for the interval
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
    

def calc_interval_in_row(df: pd.Series, start: str, end: str, unit='m'):
    """
    This function calculates the interval between two dates.
    each date is a column in the series.

    Args:
        df: original series
        start: column name for the start date
        end: column name for the end date
        unit: either h(hour) or d(day) or m(month)

    Returns:
        df with calculation
    """
    deltaT = pd.to_datetime(df[end]) - pd.to_datetime(df[start])
    if unit.lower() == 'd':
        time_diff = (deltaT / pd.Timedelta(days=1))
    elif unit.lower() == 'm':
        time_diff = ((deltaT / pd.Timedelta(days=1)) / 30.4)
    elif unit.lower() == 'h':
        time_diff = (deltaT / pd.Timedelta(hours=1))
    else:
        raise ValueError(f'Unit {unit} is not supported.')
    return round(time_diff)


class RawDataConverter():
    """Common sequence for processing:
    1) break_into_sequence
    2) convert_date_to_interval
    3) clean_tabular_data
    """

    def __init__(self, file_path: Path, max_fu: int, sequence_length: int):
        self.raw_data = pd.read_csv(file_path, header=0)
        self.file_folder = file_path.parent
        self.max_fu = max_fu
        self.sequence_length = sequence_length
        self.data: pd.DataFrame

    def calc_interval(self, name: str,
                      start: str, end: str, unit='m'):
        """This function calculates the interval between two columns of dates.
            each date is a column in the dataframe.

        Args:
            df: original dataframe
            start: column name for the start date
            end: column name for the end date
            unit: either h(hour) or d(day) or m(month)
        """
        deltaT = pd.to_datetime(self.data[end]) - pd.to_datetime(self.data[start])

        if unit.lower() == 'd':
            self.data[name] = (deltaT / pd.Timedelta(days=1)).round()
        elif unit.lower() == 'w':
            self.data[name] = ((deltaT / pd.Timedelta(days=1)) / 7).round()
        elif unit.lower() == 'm':
            self.data[name] = ((deltaT / pd.Timedelta(days=1)) / 30.4).round()
        elif unit.lower() == 'h':
            self.data[name] = (deltaT / pd.Timedelta(hours=1)).round()
        else:
            raise ValueError(f'Unit {unit} is not supported.')

    def calc_gradient(self, start_pt: int,
                      end_pt: int, name: str):
        """Calculate gradients of a time sequence.
        Args:
            start_pt: index for start
            end_pt: index for end
            name: name for the newly added columns
        """
        start_time = 'Interval' + str(start_pt)
        start_value = 'PSA' + str(start_pt)
        end_time = 'Interval' + str(end_pt)
        end_value = 'PSA' + str(end_pt)
        self.data[name + '_delta'] = (self.data[end_value] - self.data[start_value]).astype(float)
        self.data[name + '_gradient'] = self.data[name + '_delta'].div(self.data[end_time] - self.data[start_time])

    def clean_tabular_data(self):
        """Drops invalid columns.
            It should be done just before save.
        """
        invalid_cols = ['MRN', 'RTStart', 'Primary Staging Date Format',
                        'StartDateADT', 'Date1', 'Date2', 'Date3']
        self.data.drop(columns=invalid_cols)

    def break_into_sequence(self) -> pd.DataFrame:
        """ Generates sequence
            Breaks the raw data into multiple rows,
            each of which contains sequence_length PSADate and PSA pairs.
        """
        print(f'Original data has {self.raw_data.shape[0]} patients and {self.raw_data.shape[1]} columns.')
        psa_date_cols = ['PSADate' + str(i) for i in range(1, self.max_fu + 1, 1)]
        psa_cols = ['PSA' + str(i) for i in range(1, self.max_fu + 1, 1)]
        all_cols = self.raw_data.columns
        prior_treat_cols = [x for x in all_cols if x not in psa_date_cols]
        prior_treat_cols = [x for x in prior_treat_cols if x not in psa_cols]

        valid_patients = 0
        new_rows = []
        for _, row in self.raw_data.iterrows():
            print(f"Processing patient: {row['MRN']}")
            keep_cols = row[prior_treat_cols]
            psa_sequence = {}
            valid_case_flag = False
            for i in range(1, self.max_fu - self.sequence_length + 1, 1):
                if pd.isna(row['PSADate' + str(i)]):
                    break  # No more good data in this row
                if pd.isna(row['PSADate' + str(i+self.sequence_length-1)]):
                    break  # too short for a sequence
                time_diff = calc_interval_in_row(row, 'RTStart', 'PSADate' + str(i), 'd')
                if time_diff < 0:
                    continue  # psa is before treatment
                if not valid_case_flag:
                    valid_case_flag = True
                    valid_patients += 1
                for j in range(self.sequence_length):
                    psa_sequence['Date'+str(j+1)] = row['PSADate' + str(i+j)]
                    psa_sequence['PSA'+str(j+1)] = row['PSA' + str(i+j)]

                seq_cols = pd.Series(psa_sequence)
                new_row = pd.concat([keep_cols, seq_cols])
                new_rows.append(new_row)
        print(f'Number of valid patients: {valid_patients}')
        self.data = pd.DataFrame(new_rows)
        return self.data
