"""Class to preprocess tabular format data. """
from copy import deepcopy
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn import model_selection

from projects.psa_scanner import utils


class TabularProcessor:
    """Tabular Data Processor"""

    def __init__(self, input_file: Path, tab: int = 0, anonymize: bool = True):
        if not input_file.exists():
            raise FileExistsError(f'{input_file} cannot be found.')
        suffix = input_file.suffix
        if suffix not in ['.csv', '.xlsx', '.xls']:
            raise ValueError(f'{suffix} type file not supported yet.')

        raw_data = pd.DataFrame()
        if suffix == '.csv':
            raw_data = pd.read_csv(input_file, header=0, index_col=0)
        elif suffix == '.xlsx':
            raw_data = pd.read_excel(input_file, sheet_name=tab, header=0, index_col=0)

        # clean column names
        raw_data.columns = raw_data.columns.str.strip().str.lower().str.replace(' ', '_')
        # clean string values
        df_obj = raw_data.select_dtypes(['object'])
        raw_data[df_obj.columns] = df_obj.apply(
            lambda x: x.str.strip().str.lower().str.replace(' ', '_'))

        self.data = deepcopy(raw_data)
        if anonymize:
            self.data.reset_index(inplace=True, names='AnonymizedID')
            self.data['AnonymizedID'] = 'Anon' + self.data['AnonymizedID'].astype(str)

        self.train_data = None
        self.valid_data = None
        self.test_data = None

    def convert_data_types(self, convert_dict: dict[str, type]):
        """convert column datatype with convert_dict."""
        self.data = self.data.astype(convert_dict)
        # convert all dates to datetime
        date_columns = [col for col in self.data.columns if 'date' in col.lower()]
        self.data[date_columns] = self.data[date_columns].apply(pd.to_datetime)

    def convert_timepoints_to_interval(self, timepoints: list[str],
                                       name: str, unit: str):
        """convert time points in a list to intervals from ref_time.

        Args:
            timepoints: list of the timepoints (dates). index 0 is the reference.
            name: base name for the time interval
            unit: time unit ('m', 'd', 'h')
        Returns:
            None
        """
        ref_time = timepoints[0]
        for i, date_col in enumerate(timepoints[1:]):
            interval_time = name + str(i+1)
            utils.calc_interval(self.data, interval_time, ref_time, date_col, unit)

    def convert_category_to_numerical(self, code_dict: dict[str, dict[str, str]]):
        """Convert categorical columns into numerical columns based on convert_dict.
        Args:
            code_dict: {col_name: {categore: code}}
        """
        self.data.replace(code_dict, inplace=True)

    def calc_sequence_gradient(self, time_points: list[tuple[str, str]],
                               name: str, unit: str = 'm'):
        """Calculates gradients of a time sequence.
        Args:
            time_points: a list of time points of tuple[time_pt_name, value_name].
                They must be in time order from early to late
            name: the basic of the name for gradients and deltas
            unit: time unit
        """
        ntimes = len(time_points)
        for i in range(ntimes - 1):
            name_i = name + str(i+1)
            utils.calc_gradient(self.data, time_points[i],
                                time_points[i+1], name_i, unit)

    def random_split(self, test_ratio: float, valid_ratio: float, random_state: int = 1042):
        """Random split data into train, test and optional valid sets.
        Args:
            test_ratio: between (0, 1)
            valid_ratio: between (0, 1). Note this is a rato for the data excluding test
                if 0, no valid set will be generated.
            random_state
        """
        train, test = model_selection.train_test_split(self.data, test_size=test_ratio)
        if valid_ratio > 0.01:
            train, valid = model_selection.train_test_split(train, test_size=valid_ratio)
            self.valid_data = valid
        self.train_data = train
        self.test_data = test

    def sequence_builder(self, old_base_names: tuple[str, str], reference: tuple[str, str],
                         new_base_names: tuple[str, str], drop_old: bool = True):
        """Builds a time sequence from the ref_time.
            All the date should be already converted to datetime type before calling this function.
        Args:
            old_base_names: (base_date_name, base_value_name) for exisitng columns
            reference: [start time to construct the sequence, start value column name]
            new_base_names: (base_date_name, base_value_name) for new columns of the sequence
            drop_old: if true, drop the old columns from the dataset
        """
        # find old columns
        base_date_name, base_value_name = old_base_names
        date_cols = [str(col) for col in self.data.columns if base_date_name.lower() in col]
        n_cols = len(date_cols)
        print(f'Identified {n_cols} possible time points in the original data.')
        val_cols = [base_value_name + str(i+1) for i in range(n_cols)]
        # create new columns, starting index from 1
        base_date_name, base_value_name = new_base_names
        ref_time, ref_val_col = reference
        self.data[base_date_name + '0'] = self.data[ref_time]
        self.data[base_value_name + '0'] = self.data[ref_val_col]
        new_date_cols = [base_date_name + f'{i+1}' for i in range(n_cols)]
        new_val_cols = [base_value_name + f'{i+1}' for i in range(n_cols)]
        new_df = pd.DataFrame(index=self.data.index, columns=new_date_cols).fillna(
            value=pd.Timestamp('nat'))
        self.data = pd.concat([self.data, new_df], axis=1)
        new_df = pd.DataFrame(index=self.data.index, columns=new_val_cols).fillna(
            value=np.nan)
        self.data = pd.concat([self.data, new_df], axis=1)
        if n_cols != len(val_cols):
            raise ValueError(f'Size does not match: {date_cols} vs {val_cols}')
        print(f'Start to build time sequence for {new_base_names} ...')
        for i in range(len(self.data)):
            new_idx = 0
            for j in range(n_cols):
                time1 = self.data.loc[i, date_cols[j]]
                time0 = self.data.loc[i, ref_time]
                time_diff = time1 - time0  # type: ignore
                if time_diff.days >= 0:
                    self.data.loc[i, new_date_cols[new_idx]] = time1
                    self.data.loc[i, new_val_cols[new_idx]] = self.data.loc[i, val_cols[j]]
                    new_idx += 1
        self.data.dropna(axis=1, how='all', inplace=True)
        if drop_old:
            self.data.drop(date_cols, axis=1, inplace=True)
            self.data.drop(val_cols, axis=1, inplace=True)

    def drop_columns(self, col_names: list[str]):
        """Drop columns from data df."""
        self.data.drop(col_names, axis=1, inplace=True)

    def save_all_to_csv(self, output_path: Path):
        """Save processed data to disk."""
        self.data.to_csv(output_path / 'all_data.csv')

    def save_splits(self, output_path: Path):
        """Save split data to disk."""
        if self.train_data is not None:
            self.train_data.to_csv(output_path / 'train_data.csv')
        if self.test_data is not None:
            self.test_data.to_csv(output_path / 'test_data.csv')
        if self.valid_data is not None:
            self.valid_data.to_csv(output_path / 'valid_data.csv')
