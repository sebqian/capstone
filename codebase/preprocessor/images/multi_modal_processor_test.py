"""Unit Test for multimodal_processor.py"""
import unittest
import numpy as np

import codebase_settings as cbs
from codebase.preprocessor.images import multi_modal_processor as mmp
from codebase import terminology

_TEST_PATH = cbs.CODEBASE_PATH / 'preprocessor' / 'images' / 'test_data'
_EXPECTED_PATIENT = 'CHUM-024'
_EXPECTED_CT_SHAPE = (1, 512, 512, 237)
_EXPECTED_CT_SPACING = [0.9765625, 0.9765625, 1.5]
_EXPECTED_PET_SHAPE = (1, 144, 144, 87)
_EXPECTED_PET_SPACING = [4.0, 4.0, 4.0]
_EXPECTED_XY = (128, 128)
_EXPECTED_RESAMPLED_SHAPE = (1, 128, 128, 87)
_EXPECTED_RESAMPLED_SPACING = [4.5, 4.5, 4.0]


class TestMultiModalProcessor(unittest.TestCase):

    def setUp(self) -> None:
        super().setUp()
        self.processor = mmp.MultiModalProcessor(
            data_folder=_TEST_PATH,
            phase=terminology.Phase.TRAIN,
            modalities=[terminology.Modality.CT, terminology.Modality.PET],
            reference=terminology.Modality.PET,
            problem_type=terminology.ProblemType.SEGMENTATION
        )

    def test_get_patient_list(self) -> None:
        patients = self.processor.get_patient_lists()
        self.assertEqual(patients[0], _EXPECTED_PATIENT, f'Unexpected patient: {patients[0]}')

    def test_create_subject(self) -> None:
        subject = self.processor.create_subject(_EXPECTED_PATIENT)
        self.assertEqual(subject.ID, _EXPECTED_PATIENT, f'Unexpected subject ID: {subject.ID}')  # type: ignore
        img = subject[terminology.Modality.CT.value]
        self.assertEqual(img.shape, _EXPECTED_CT_SHAPE, f'Unexpected CT shape: {img.shape}')
        self.assertListEqual(list(img.spacing), _EXPECTED_CT_SPACING,
                             f'Unexpected CT spacing {img.spacing}')
        img = subject[terminology.Modality.PET.value]
        self.assertEqual(img.shape, _EXPECTED_PET_SHAPE, f'Unexpected CT shape: {img.shape}')
        self.assertListEqual(list(img.spacing), _EXPECTED_PET_SPACING,
                             f'Unexpected CT spacing {img.spacing}')

    def test_create_subject_list(self) -> None:
        subjects = self.processor.create_subject_list()
        self.assertEqual(len(subjects), 1)
        self.assertEqual(subjects[0].ID, _EXPECTED_PATIENT)  # type: ignore

    def test_resample_to_reference(self) -> None:
        subject = self.processor.create_subject(_EXPECTED_PATIENT)
        subject = self.processor.resample_to_reference(subject, _EXPECTED_XY)
        self.assertEqual(subject.ID, _EXPECTED_PATIENT, f'Unexpected subject ID: {subject.ID}')  # type: ignore
        img = subject[terminology.Modality.CT.value]
        self.assertEqual(img.shape, _EXPECTED_RESAMPLED_SHAPE, f'Unexpected CT shape: {img.shape}')
        self.assertListEqual(list(img.spacing), _EXPECTED_RESAMPLED_SPACING,
                             f'Unexpected CT spacing {img.spacing}')
        img = subject[terminology.Modality.PET.value]
        self.assertEqual(img.shape, _EXPECTED_RESAMPLED_SHAPE, f'Unexpected CT shape: {img.shape}')
        self.assertListEqual(list(img.spacing), _EXPECTED_RESAMPLED_SPACING,
                             f'Unexpected CT spacing {img.spacing}')

    def test_train_histogram_standardization(self) -> None:
        landmarks = self.processor.train_histogram_standardization(terminology.Modality.PET)
        self.assertGreater(len(landmarks), 0)

    # def test_create_transformation(self) -> None:
    #     subject = self.processor.create_subject(_EXPECTED_PATIENT)
    #     subject = self.processor.resample_to_reference(subject, _EXPECTED_XY)
    #     transform_dict = {'flip': {'p': 1.0, 'axes': ('LR', 'AP')}}
    #     transformations = self.processor.create_transformation(transform_dict)
    #     subject = transformations(subject)
    #     transformed_data = subject['CT'].numpy()  # type: ignore
    #     self.assertGreaterEqual(np.min(transformed_data), -1,
    #                             f'Transformed min is too small: {np.min(transformed_data)}')
    #     self.assertLessEqual(np.max(transformed_data), 1,
    #                          f'Transformed max is too large: {np.max(transformed_data)}')


if __name__ == '__main__':
    unittest.main()
