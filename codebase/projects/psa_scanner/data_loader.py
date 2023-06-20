"""Data loader module for PSA Scanner data."""
from pathlib import Path
import numpy as np

from preprocessor.tabular import tabular_processor

# Don't use int type unless you are sure there's no missing values
_DATA_TYPE_DICT = {
    'primary_gleason_score': float,
    'secondary_gleason_score': float,
    'total_gleason_score': float,
    # 'primary_staging_data_format':
}

_CAT_DICT = {
    'primary_cancer_stage': {
        'stage_x': 0,
        'stage_i': 1,
        'stage_ia': 1.2,
        'stage_ib': 1.4,
        'stage_ii': 2,
        'stage_iia': 2.2,
        'stage_iib': 2.4,
        'stage_iic': 2.6,
        'stage_iii': 3,
        'stage_iiia': 3.2,
        'stage_iiib': 3.4,
        'stage_iiic': 3.6,
        'stage_iv': 4,
        'stage_iva': 4.2,
        'stage_ivb': 4.4,
        'stage_ivc': 4.6
    },
    'primary_stage_n': {
        'n0': 0,
        'n1': 1,
        'n2': 2,
        'n3': 3,
        'n4': 4,
    },
    'primary_stage_t': {
        't0': 0,
        't1': 1,
        't1a': 1.2,
        't1b': 1.4,
        't1c': 1.6,
        't2': 2,
        'pt2': 2,
        't2a': 2.2,
        't2b': 2.4,
        't2c': 2.6,
        't3': 3,
        't3a': 3.2,
        'pt3a': 3.2,
        't3b': 3.4,
        'pt3b': 3.4,
        't3c': 3.6,
        't4': 4,
        'pt4': 4,
        'tx': 0,
    },
    'treatmentvolume': {
        'prostate': 1,
        'prostate, pelvic lymph nodes': 2,
        'prostate, pelvic lymph nodes, para--aortic lymph nodes': 3,
    },
    'adtuse': {
        'y': 1,
        'n': 0,
    }
}


class PSAScannerDataLoader:
    """Data loader for PSA scanner project."""

    def __init__(self, data_file: Path) -> None:
        self.preprocessor = tabular_processor.TabularProcessor(data_file)

    def data_preprocessing(self):
        """Preprocessing module for PSA scanner."""
        self.preprocessor.drop_columns(['patient_clinic_number'])
        self.preprocessor.convert_category_to_numerical(_CAT_DICT)
        self.preprocessor.convert_data_types(_DATA_TYPE_DICT)
        self.preprocessor.sequence_builder(
            old_base_names=('psadate', 'psa'), reference=('rtstartdate', 'pretreatmentpsa'),
            new_base_names=('date_after_rt_t', 'psa_t'))
        timepoints = [col for col in self.preprocessor.data.columns if 'date_after_rt_t' in col]
        self.preprocessor.convert_timepoints_to_interval(timepoints, 'time_after_rt_t', 'd')
        timepoints = [('date_after_rt_t' + str(i), 'psa_t' + str(i)) for i in range(len(timepoints))]
        self.preprocessor.calc_sequence_gradient(timepoints, 'psa_t', 'd')
