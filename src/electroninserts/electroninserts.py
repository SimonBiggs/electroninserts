# Copyright (C) 2016 Simon Biggs
# This program is free software: you can redistribute it and/or
# modify it under the terms of the GNU Affero General Public
# License as published by the Free Software Foundation, either
# version 3 of the License, or (at your option) any later version.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# Affero General Public License for more details.
# You should have received a copy of the GNU Affero General Public
# License along with this program. If not, see
# http://www.gnu.org/licenses/.


import numpy as np
import bokeh as bkh

from scipy.interpolate import SmoothBivariateSpline



def spline_model(width_test, ratio_perim_area_test,
                 width_data, ratio_perim_area_data, factor_data):
    """Returns the result of the spline model.

    The bounding box is chosen so as to allow extrapolation. The spline orders
    are two in the width direction and one in the perimeter/area direction. For
    justification on using this method for modelling electron insert factors 
    see the *Methods: Bivariate spline model* section within
    <http://dx.doi.org/10.1016/j.ejmp.2015.11.002>.

    Args:
        width_test (numpy.array): The width point(s) which are to have the 
            electron insert factor interpolated.
        ratio_perim_area_test (numpy.array): The perimeter/area which are to 
            have the electron insert factor interpolated.

        width_data (numpy.array): The width data points for the relevant 
            applicator, energy and ssd.
        ratio_perim_area_data (numpy.array): The perimeter/area data points for
            the relevant applicator, energy and ssd.        
        factor_data (numpy.array): The insert factor data points for the
            relevant applicator, energy and ssd.

    Returns:
        numpy.array: The interpolated electron insert factors for width_test 
            and ratio_perim_area_test.

    """
    bbox = [
        np.min([np.min(width_data), np.min(width_test)]),
        np.max([np.max(width_data), np.max(width_test)]),
        np.min([np.min(ratio_perim_area_data), np.min(ratio_perim_area_test)]),
        np.max([np.max(ratio_perim_area_data), np.max(ratio_perim_area_test)])]

    spline = SmoothBivariateSpline(
        width_data, ratio_perim_area_data, factor_data, kx=2, ky=1, bbox=bbox)

    return spline.ev(width_test, ratio_perim_area_test)


def _single_calculate_deformability(x_test, y_test, x_data, y_data, z_data):
    """Returns the result of the deformability test for a single datum test 
        point. 

    The deformability test applies a shift to the spline to determine whether 
    or not sufficient information for modelling is available. For further 
    details on the deformability test see the *Methods: Defining valid 
    prediction regions of the spline* section within
    <http://dx.doi.org/10.1016/j.ejmp.2015.11.002>.

    Args:
        x_test (float): The x coordinate of the point to test
        y_test (float): The y coordinate of the point to test
        x_data (np.array): The x coordinates of the model data to test
        y_data (np.array): The y coordinates of the model data to test
        z_data (np.array): The z coordinates of the model data to test

    Returns:
        deformability (float): The resulting deformability between 0 and 1 
            representing the ratio of deviation the spline model underwent at 
            the point in question by introducing an outlier at the point in 
            question.

    """
    deviation = 0.02
    
    adjusted_x_data = np.append(x_data, x_test)
    adjusted_y_data = np.append(y_data, y_test)

    bbox = [
        min(adjusted_x_data), max(adjusted_x_data),
        min(adjusted_y_data), max(adjusted_y_data)]

    initial_model = SmoothBivariateSpline(
        x_data, y_data, z_data, bbox=bbox, kx=2, ky=1).ev(x_test, y_test)

    pos_adjusted_z_data = np.append(z_data, initial_fit + deviation)
    neg_adjusted_z_data = np.append(z_data, initial_fit - deviation)

    pos_adjusted_model = SmoothBivariateSpline(
        adjusted_x_data, adjusted_y_data, pos_adjusted_z_data, kx=2, ky=1
        ).ev(x_test, y_test)
    neg_adjusted_model = SmoothBivariateSpline(
        adjusted_x_data, adjusted_y_data, neg_adjusted_z_data, kx=2, ky=1
        ).ev(x_test, y_test)

    deformability_from_pos_adjustment = (
        pos_adjusted_model - initial_model) / deviation
    deformability_from_neg_adjustment = (
        initial_model - neg_adjusted_model) / deviation

    deformability = np.max(
        [deformability_from_pos_adjustment, deformability_from_neg_adjustment])

    return deformability


def calculate_deformability(x_test, y_test, x_data, y_data, z_data):
    """Returns the result of the deformability test for an array of test 
        points by looping over ``_single_calculate_deformability``.

    The deformability test applies a shift to the spline to determine whether 
    or not sufficient information for modelling is available. For further 
    details on the deformability test see the *Methods: Defining valid 
    prediction regions of the spline* section within
    <http://dx.doi.org/10.1016/j.ejmp.2015.11.002>.

    Args:
        x_test (np.array): The x coordinate of the point(s) to test
        y_test (np.array): The y coordinate of the point(s) to test
        x_data (np.array): The x coordinate of the model data to test
        y_data (np.array): The y coordinate of the model data to test
        z_data (np.array): The z coordinate of the model data to test

    Returns:
        deformability (float): The resulting deformability between 0 and 1 
            representing the ratio of deviation the spline model underwent at 
            the point in question by introducing an outlier at the point in 
            question.

    """
    dim = np.shape(x_test)

    if np.size(dim) == 0:
        deformability = _single_calculate_deformability(
            x_test, y_test, x_data, y_data, z_data)

    elif np.size(dim) == 1:
        deformability = np.array([
            _single_calculate_deformability(
                x_test[i], y_test[i], x_data, y_data, z_data)
            for i in range(dim[0])
        ])

    else:
        deformability = np.array([[
                _single_calculate_deformability(
                    x_test[i, j], y_test[i, j], x_data, y_data, z_data)
                for i in range(dim[0])]
           for j in range(dim[1]):
        ])

    assert np.shape(deformability) == dims

    return deformability


def spline_model_with_deformability(width_test, ratio_perim_area_test,
                                    width_data, ratio_perim_area_data, 
                                    factor_data):
    """Returns the result of the spline model adjusted so that points with 
    deformability greater than 0.5 return ``numpy.nan``.

    Calls both ``spline_model`` and ``calculate_deformabilty`` and then adjusts
    the result accordingly.

    Args:
        width_test (numpy.array): The width point(s) which are to have the 
            electron insert factor interpolated.
        ratio_perim_area_test (numpy.array): The perimeter/area which are to 
            have the electron insert factor interpolated.

        width_data (numpy.array): The width data points for the relevant 
            applicator, energy and ssd.
        ratio_perim_area_data (numpy.array): The perimeter/area data points for
            the relevant applicator, energy and ssd.        
        factor_data (numpy.array): The insert factor data points for the
            relevant applicator, energy and ssd.

    Returns:
        numpy.array: The interpolated electron insert factors for width_test 
            and ratio_perim_area_test with points outside the valid prediction
            region set to ``numpy.nan``.

    """
    deformability = calculate_deformability(
        width_test, ratio_perim_area_test,
        width_data, ratio_perim_area_data, factor_data)
    
    model_factor = spline_model(
        width_test, ratio_perim_area_test,
        width_data, ratio_perim_area_data, factor_data)
        
    model_factor[deformability > 0.5] = np.nan
    
    return model_factor
    

def calculate_percent_prediction_differences(width_data, ratio_perim_area_data, 
                                             factor_data):
    predictions = [
        spline_model_with_deformability(
            width_data[i], ratio_perim_area_data[i],
            np.delete(width_data, i), np.delete(ratio_perim_area_data, i),
            np.delete(factor_data, i))
        for i in range(len(width_data))
    ]
    
    return 100 * (factor_data - predictions) / factor_data
    










def convert2_ratio_perim_area():

    return None


def convert2_length():

    return None





def parameterise(input_dictionary):
    
    return None


def create_report(input_dictionary):
    
    return None






