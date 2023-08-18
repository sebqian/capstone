"""Test module for beams_eye_view. """
import unittest
import numpy as np

import codebase.codebase_settings as cbs
from codebase.preprocessor.images import beams_eye_view


_TEST_DATA_FOLDER = cbs.CODEBASE_PATH / 'preprocessor' / 'images' / 'test_data' / 'images'
_TEST_IMAGE_FILE = _TEST_DATA_FOLDER / 'MDA-103__CT.nii.gz'


class TestBeamsEyeView(unittest.TestCase):

    def setUp(self) -> None:
        super().setUp()
        
class BeamsEyeView():

    def align_patient_position(self, img: sitk.Image,
                               position: PositionOrientation = PositionOrientation.hfs
                               ) -> sitk.Image:
        """Align image orientation to gantry view orientation.
        The DICOM coordinate is only relative to the patient, but we also need to know
            how a patient is positioned to find the image orientation to the room and to
            the gantry. Nifti does not have the position information which needs to be
            obtainned from DICOM.
        """
        orient = sitk.DICOMOrientImageFilter_GetOrientationFromDirectionCosines(img.GetDirection())
        print(f'\t Original image orientation: {orient}')

        # TODO: implement other position orientation
        if position == PositionOrientation.hfs:
            img = sitk.DICOMOrient(img, desiredCoordinateOrientation='RPS')
        elif position == PositionOrientation.ffs:
            img = sitk.DICOMOrient(img, desiredCoordinateOrientation='LPI')
        else:
            raise ValueError(f'Patient orientation {position} is not recognized.')
        orient = sitk.DICOMOrientImageFilter_GetOrientationFromDirectionCosines(img.GetDirection())
        print(f'\t New image orientation: {orient}')
        return img

    def get_image_center_coordinates(self, img: sitk.Image) -> Tuple[float, float, float]:
        """Calculates the physical coordinates of the image. """
        width, height, depth = img.GetSize()
        center = img.TransformIndexToPhysicalPoint(
            (int(np.ceil(width/2)), int(np.ceil(height/2)),
             int(np.ceil(depth/2))))
        return center
    
    def resample(self, img: sitk.Image, spacing: List[float, float, float] = _GRID_SPACE):


    # TODO: implement the iso alignment
    def align_isocenter(self, img: sitk.Image, isocenter: Tuple[float, float, float],
                        resample: bool = True) -> sitk.Image:
        """Aligns the image center to beam isocneter.
        Args:
            img:
            isocenter: isocenter location in physical coordinates
            resample: flag to perform resampling
        Returns:
            image with iso at origin, resampled unless flagged.
        """
        # find image center
        old_center = self.get_image_center_coordinates(img)
        # calculate offset to isocenter
        offset = (isocenter[0] - old_center[0],
                  isocenter[1] - old_center[1],
                  isocenter[2] - old_center[2])
        print(offset)
        # define translation
        translation_to_iso = sitk.TranslationTransform(3)  # 3D translation
        translation_to_iso.SetOffset(offset)
        # apply translation
        original = img
        interpolator = sitk.sitkCosineWindowedSinc
        aligned_img = sitk.Resample(img, original, translation_to_iso,
                                    interpolator, _DEFAULT_CT)
        img.SetSpacing(_GRID_SPACE)
        return aligned_img

    def project_nifti_image_to_beam(self, img: sitk.Image) -> sitk.Image:
        """Project a nifti image to beam's eye view.
        This project is only for couch angle 0 and gantry angle 0.
        Args:
            img: sitk image already with orientation aligned.
        Returns:
            image in beam's eye view with couch angle 0 and gantry angle 0.
        """
        # swap axis to match beam's eye view
        pa = sitk.PermuteAxesImageFilter()
        pa.SetOrder([0, 2, 1])
        img = pa.Execute(img)
        return img

    def couch_kick(self, img: sitk.Image, couch_angle: float
                   ) -> sitk.Image:
        """Perform couch kick in beam's eye view.
        Args:
            img: image already in beam's eye view at G0T0.
                Its center is already aligned to isocenter.
            couch_angle: in degrees
        Returns:
            image in BEV with couch kick.
        """
        couch_angle = couch_angle / 180.0 * PI
        center = self.get_image_center_coordinates(img)
        img = geometry.rotate_image(
            img, rotation_centre=center,
            rotation_axis=(0, 0, 1),
            rotation_angle_radians=couch_angle,
            default_value=_DEFAULT_CT
        )
        return img