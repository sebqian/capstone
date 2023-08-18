"""This is a module to convert image in DICOM coordinates into gantry coordinates or vise verse.

Note that the spots of proton spot map are in isocenter plane.
"""
from typing import List, Tuple, Union
from enum import Enum
import numpy as np
from pathlib import Path

import SimpleITK as sitk
import nibabel as nib
from platipy.imaging.utils import geometry

PI = 3.14159
_GRID_SPACE = [2.0, 2.0, 2.0]
_DEFAULT_CT = -1000


class DataType(Enum):
    nifti = 'nifti'
    dicom = 'dicom'


class VirtualSAD(Enum):
    x = 140.0
    y = 190.0


class PositionOrientation(Enum):
    hfs = 'HFS'  # Head first supine
    hfp = 'HFP'  # Head first prone
    ffs = 'FFS'  # Feet first supine
    ffp = 'FFP'  # Feet first prone


class BeamsEyeView():

    # def __int__(self, data_type: DataType):
    #     self.image_io = ''
    #     if data_type == DataType.nifti:
    #         self.image_io = 'NiftiImageIO'

    def read_image(self, filename: Union[str, Path]) -> sitk.Image:
        filename = str(filename)
        return sitk.ReadImage(filename)

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
            img = sitk.DICOMOrient(img, desiredCoordinateOrientation='RSP')
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

    def resample(self, itk_image: sitk.Image, out_spacing: List[float] = _GRID_SPACE,
                 is_label: bool = False):
        """Resample image to desired spacing. Size changes correspondingly."""
        original_spacing = itk_image.GetSpacing()
        original_size = itk_image.GetSize()
        dimension = itk_image.GetDimension()
        out_size = [
            int(np.round(original_size[i] * (original_spacing[i] / out_spacing[0]))
                ) for i in range(dimension)]

        resample = sitk.ResampleImageFilter()
        resample.SetOutputSpacing(out_spacing)
        resample.SetSize(out_size)
        resample.SetOutputDirection(itk_image.GetDirection())
        resample.SetOutputOrigin(itk_image.GetOrigin())
        resample.SetTransform(sitk.Transform())
        resample.SetDefaultPixelValue(itk_image.GetPixelIDValue())

        if is_label:
            resample.SetInterpolator(sitk.sitkNearestNeighbor)
        else:
            resample.SetInterpolator(sitk.sitkBSpline)

        return resample.Execute(itk_image)

    # TODO: implement the iso alignment
    def align_isocenter(self, img: sitk.Image, isocenter: Tuple[float, float, float],
                        ) -> sitk.Image:
        """Aligns the image center to beam isocneter.
        Args:
            img:
            isocenter: isocenter location in physical coordinates
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

    # def project_image_to_beam(self, img: sitk.Image) -> sitk.Image:
    #     """Project a nifti image to beam's eye view.
    #     This project is only for couch angle 0 and gantry angle 0.
    #     Args:
    #         img: sitk image already with orientation aligned.
    #     Returns:
    #         image in beam's eye view with couch angle 0 and gantry angle 0.
    #     """
    #     # swap axis to match beam's eye view
    #     pa = sitk.PermuteAxesImageFilter()
    #     pa.SetOrder([0, 2, 1])
    #     img = pa.Execute(img)
    #     return img

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
        couch_angle = (360 - couch_angle) / 180.0 * PI
        center = self.get_image_center_coordinates(img)
        img = geometry.rotate_image(
            img, rotation_centre=center,
            rotation_axis=(0, -1, 0),
            rotation_angle_radians=couch_angle,
            default_value=_DEFAULT_CT
        )
        return img

    def gantry_rotate(self, img: sitk.Image, gantry_angle: float
                      ) -> sitk.Image:
        """Perform couch kick in beam's eye view.
        Args:
            img: image already in beam's eye view at G0T0.
                Its center is already aligned to isocenter.
            couch_angle: in degrees
        Returns:
            image in BEV with couch kick.
        """
        gantry_angle = gantry_angle / 180.0 * PI
        center = self.get_image_center_coordinates(img)
        img = geometry.rotate_image(
            img, rotation_centre=center,
            rotation_axis=(0, 0, 1),
            rotation_angle_radians=gantry_angle,
            default_value=_DEFAULT_CT
        )
        return img