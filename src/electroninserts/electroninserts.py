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

import matplotlib.pyplot
import matplotlib.colors
import matplotlib as mpl

import bokeh.io
import bokeh.plotting
import bokeh.models
import bokeh as bkh

import shapely.geometry
import shapely.affinity
import shapely as shp

import descartes as des

from scipy.interpolate import SmoothBivariateSpline
from scipy.optimize import minimize, basinhopping

viridis = mpl.pyplot.get_cmap('viridis')
default_tools = "hover, box_zoom, reset"


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

    pos_adjusted_z_data = np.append(z_data, initial_model + deviation)
    neg_adjusted_z_data = np.append(z_data, initial_model - deviation)

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
            for j in range(dim[1])]
            for i in range(dim[0])
        ])

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
    """Calculates the model factor for each data point with that point removed
    from the data set. Used to determine an estimated uncertainty for
    prediction.

    Args:
        width_data (numpy.array): The width data points for a specific
            applicator, energy and ssd.
        ratio_perim_area_data (numpy.array): The perimeter/area data points for
            a specific applicator, energy and ssd.
        factor_data (numpy.array): The insert factor data points for a specific
            applicator, energy and ssd.

    Returns:
        numpy.array: The predicted electron insert factors for each data point
            with that given data point removed.

    """
    predictions = [
        spline_model_with_deformability(
            width_data[i], ratio_perim_area_data[i],
            np.delete(width_data, i), np.delete(ratio_perim_area_data, i),
            np.delete(factor_data, i))
        for i in range(len(width_data))
    ]

    return 100 * (factor_data - predictions) / factor_data


def shapely_insert(x, y):
    return shp.geometry.Polygon(np.transpose((x, y)))


def search_for_poi(x, y):
    insert = shapely_insert(x, y)
    boundary = insert.boundary
    centroid = insert.centroid

    furthest_distance = np.hypot(
        np.diff(insert.bounds[::2]),
        np.diff(insert.bounds[1::2]))

    def minimising_function(optimiser_input):
        x, y = optimiser_input
        point = shp.geometry.Point(x, y)

        if insert.contains(point):
            edge_distance = point.distance(boundary)
        else:
            edge_distance = -point.distance(boundary)

        centroid_weighting = (
            point.distance(centroid) / furthest_distance)

        return centroid_weighting - edge_distance

    x0 = np.squeeze(centroid.coords)
    niter = 200
    T = furthest_distance / 3
    stepsize = furthest_distance / 2
    niter_success = 50
    output = basinhopping(
        minimising_function, x0, niter=niter, T=T, stepsize=stepsize,
        niter_success=niter_success)

    return output.x


def calculate_width(x, y, poi):
    insert = shapely_insert(x, y)
    point = shp.geometry.Point(*poi)

    if insert.contains(point):
        distance = point.distance(insert.boundary)
    else:
        raise Exception("poi not within insert")

    return distance * 2


def calculate_length(x, y, width):
    insert = shapely_insert(x, y)
    length = 4 * insert.area / (np.pi * width)

    return length


def parameterise_single_insert(x, y):
    poi = search_for_poi(x, y)
    width = calculate_width(x, y, poi)
    length = calculate_length(x, y, width)

    return width, length, poi


def parameterise_inserts(to_be_parameterised):
    keys = np.sort([key for key in to_be_parameterised])
    for key in keys:
        x, y = to_be_parameterised[key]['x'], to_be_parameterised[key]['y']
        width, length, poi = parameterise_single_insert(x, y)

        to_be_parameterised[key]['width'] = float(round(width, 2))
        to_be_parameterised[key]['length'] = float(round(length, 2))
        to_be_parameterised[key]['poi'] = [
            float(round(item, 2)) for item in poi]

        print("{}:".format(key))
        display_parameterisation(**to_be_parameterised[key])


def fitted_shapely_ellipse(x, y, width, length):
    insert = shapely_insert(x, y)
    unit_circle = shp.geometry.Point(0, 0).buffer(1)
    initial_ellipse = shp.affinity.scale(
        unit_circle, xfact=width/2, yfact=length/2)

    def minimising_function(optimiser_input):
        x, y, angle = optimiser_input
        rotated = shp.affinity.rotate(
            initial_ellipse, angle, use_radians=True)
        translated = shp.affinity.translate(
            rotated, xoff=x, yoff=y)

        disjoint_area = (
            translated.difference(insert).area +
            insert.difference(translated).area)

        return disjoint_area

    x0 = np.append(
        np.squeeze(insert.centroid.coords), np.pi/4)
    niter = 100
    T = insert.area / 4
    stepsize = 3
    niter_success = 3
    output = basinhopping(
        minimising_function, x0, niter=niter, T=T, stepsize=stepsize,
        niter_success=niter_success)

    x, y, angle = output.x
    rotated = shp.affinity.rotate(
        initial_ellipse, angle, use_radians=True)
    ellipse = shp.affinity.translate(
        rotated, xoff=x, yoff=y)

    return ellipse


def display_shapely(ax, shape, alpha=1):
    patch = des.PolygonPatch(
        shape,
        fc=[0, 0, 0, alpha])
    ax.add_patch(patch)


def display_parameterisation(x, y, width, length, poi, **kwargs):
    insert = shapely_insert(x, y)
    circle = shp.geometry.Point(*poi).buffer(width/2)
    ellipse = fitted_shapely_ellipse(x, y, width, length)

    fig = mpl.pyplot.figure()
    ax = fig.add_subplot(111)

    display_shapely(ax, insert, alpha=0.5)
    display_shapely(ax, circle, alpha=0)
    display_shapely(ax, ellipse, alpha=0)

    ax.axis("equal")
    mpl.pyplot.grid(True)
    mpl.pyplot.show()


def convert2_ratio_perim_area(width, length):
    perimeter = (
        np.pi / 2 *
        (3*(width + length) - np.sqrt((3*width + length)*(3*length + width)))
    )
    area = np.pi / 4 * width * length

    return perimeter / area


def convert2_length(width_array, ratio_perim_area_array):
    def to_minimise(length, width, ratio_perim_area):
        return (ratio_perim_area - convert2_ratio_perim_area(width, length))**2

    length_array = np.zeros(len(width_array))

    for i, width in enumerate(width_array):
        ratio_perim_area = ratio_perim_area_array[i]
        length_array[i] = minimize(
            to_minimise, [1], args=(width, ratio_perim_area),
            method='L-BFGS-B', bounds=((width, None),)).x

    return length_array


def create_native_mesh(width_data, ratio_perim_area_data, factor_data):
    x = np.arange(
        np.floor(np.min(width_data)) - 1, np.ceil(np.max(width_data)), 0.1)
    y = np.arange(
        np.floor(np.min(ratio_perim_area_data)*10)/10 - 0.2,
        np.ceil(np.max(ratio_perim_area_data)*10)/10 + 0.1, 0.01)

    xx, yy = np.meshgrid(x, y)

    zz = spline_model(xx, yy, width_data, ratio_perim_area_data, factor_data)
    deformability = calculate_deformability(
        xx, yy, width_data, ratio_perim_area_data, factor_data)

    maximum_ratio_perim_area = convert2_ratio_perim_area(xx, xx)

    mesh_max_area = ((10 * np.sqrt(2) - xx) * xx + (xx/np.sqrt(2))**2)
    mesh_max_length = 4 * mesh_max_area / (np.pi * xx)

    minimum_ratio_perim_area = convert2_ratio_perim_area(xx, mesh_max_length)

    outOfTolerance = (
        (deformability > 0.5) |
        (np.around(yy, decimals=2) > np.around(
            maximum_ratio_perim_area, decimals=2)) |
        (np.around(yy, decimals=2) < np.around(
            minimum_ratio_perim_area, decimals=2)))

    zz[outOfTolerance] = np.nan

    return xx, yy, zz


def create_transformed_mesh(width_data, length_data, factor_data):
    ratio_perim_area_data = convert2_ratio_perim_area(width_data, length_data)

    x = np.arange(
        np.floor(np.min(width_data)) - 1, np.ceil(np.max(width_data)), 0.1)
    y = np.arange(
        np.floor(np.min(length_data)) - 1, np.ceil(np.max(length_data)), 0.1)

    xx, yy = np.meshgrid(x, y)
    mesh_ratio_perim_area = convert2_ratio_perim_area(xx, yy)

    zz = spline_model(
        xx, mesh_ratio_perim_area,
        width_data, ratio_perim_area_data, factor_data)
    deformability = calculate_deformability(
        xx, mesh_ratio_perim_area,
        width_data, ratio_perim_area_data, factor_data)

    mesh_max_area = ((10 * np.sqrt(2) - xx) * xx + (xx/np.sqrt(2))**2)
    mesh_max_length = 4 * mesh_max_area / (np.pi * xx)

    outOfTolerance = (
        (deformability > 0.5) |
        (
            np.around(xx, decimals=1) >
            np.around(yy, decimals=1)) |
        (
            np.around(yy, decimals=1) >
            np.around(mesh_max_length, decimals=1)))

    zz[outOfTolerance] = np.nan

    return xx, yy, zz


def convert2_source(hover_labels, hover_values):
    data = dict()
    for i in range(len(hover_labels)):
        data[hover_labels[i]] = hover_values[i]

    source = bkh.plotting.ColumnDataSource(
        data=data)

    return source


def convert2_tooltips(hover_labels, hover_values):
    tooltips = [(label, " @" + label) for label in hover_labels]
    return tooltips


def find_colour(factor, vmin, vmax):
    colour_reference = (factor - vmin) / (vmax - vmin)
    rgb = viridis(colour_reference)
    rgb = rgb[:, 0:3]
    colour = [mpl.colors.rgb2hex(tuple(item)) for item in rgb]

    return colour


def bokeh_pcolor(xx, yy, zz, hover_labels, hover_values):
    dx = xx[0, 1] - xx[0, 0]
    dy = yy[1, 0] - yy[0, 0]

    xx_flat = np.ravel(xx)
    yy_flat = np.ravel(yy)
    zz_flat = np.ravel(zz)

    reference = ~np.isnan(zz_flat)
    xx_flat = xx_flat[reference]
    yy_flat = yy_flat[reference]
    zz_flat = zz_flat[reference]

    color = find_colour(zz_flat, zz_flat.min(), zz_flat.max())

    fig = bkh.plotting.figure(
        tools=default_tools,
        plot_height=400, plot_width=600)

    source = convert2_source(hover_labels, hover_values)
    tooltips = convert2_tooltips(hover_labels, hover_values)

    fig.rect(xx_flat, yy_flat, dx, dy, color=color, source=source)
    hover = fig.select(dict(type=bkh.models.HoverTool))
    hover.tooltips = tooltips

    return fig


def native_pcolor(width_data, ratio_perim_area_data, factor_data):
    xx, yy, zz = create_native_mesh(
        width_data, ratio_perim_area_data, factor_data)

    xx_flat = np.ravel(xx)
    yy_flat = np.ravel(yy)
    zz_flat = np.ravel(zz)

    reference = ~np.isnan(zz_flat)
    xx_flat = xx_flat[reference]
    yy_flat = yy_flat[reference]
    zz_flat = zz_flat[reference]

    length_calc = convert2_length(xx_flat, yy_flat)

    hover_width = [" %0.1f cm" % (num) for num in xx_flat]
    hover_length = [" %0.1f cm" % (num) for num in length_calc]
    hover_ratio_perim_area = [" %0.2f cm^-1" % (num) for num in yy_flat]
    hover_factor = [" %0.3f" % (num) for num in zz_flat]

    hover_labels = ["Width", "Length", "PonA", "Factor"]
    hover_values = [
        hover_width, hover_length, hover_ratio_perim_area, hover_factor]

    fig = bokeh_pcolor(xx, yy, zz, hover_labels, hover_values)

    fig.title = "Native domain"
    fig.xaxis.axis_label = "Width (cm)"
    fig.yaxis.axis_label = "Perimeter / Area (cm^-1)"

    return fig


def transformed_pcolor(width_data, length_data, factor_data):
    xx, yy, zz = create_transformed_mesh(width_data, length_data, factor_data)

    xx_flat = np.ravel(xx)
    yy_flat = np.ravel(yy)
    zz_flat = np.ravel(zz)

    reference = ~np.isnan(zz_flat)
    xx_flat = xx_flat[reference]
    yy_flat = yy_flat[reference]
    zz_flat = zz_flat[reference]

    ratio_perim_area = convert2_ratio_perim_area(xx_flat, yy_flat)

    hover_width = [" %0.1f cm" % (num) for num in xx_flat]
    hover_length = [" %0.1f cm" % (num) for num in yy_flat]
    hover_ratio_perim_area = [" %0.2f cm^-1" % (num) for num in ratio_perim_area]
    hover_factor = [" %0.3f" % (num) for num in zz_flat]

    hover_labels = ["Width", "Length", "PonA", "Factor"]
    hover_values = [
        hover_width, hover_length, hover_ratio_perim_area, hover_factor]

    fig = bokeh_pcolor(
        xx, yy, zz, hover_labels, hover_values)

    fig.title = "Transformed domain"
    fig.xaxis.axis_label = "Width (cm)"
    fig.yaxis.axis_label = "Length (cm)"

    return fig


def bokeh_scatter(x, y, hover_labels, hover_values):
    fig = bkh.plotting.figure(
        tools=default_tools,
        plot_height=400, plot_width=600)

    source = convert2_source(hover_labels, hover_values)
    tooltips = convert2_tooltips(hover_labels, hover_values)

    fig.scatter(x, y, source=source, size=10)
    hover = fig.select(dict(type=bkh.models.HoverTool))
    hover.tooltips = tooltips

    return fig


def fallback_scatter(width_data, length_data, factor_data, label):
    hover_width = [" %0.1f cm" % (num) for num in width_data]
    hover_length = [" %0.1f cm" % (num) for num in length_data]
    hover_factor = [" %0.3f" % (num) for num in factor_data]

    hover_labels = ["Label", "Width", "Length", "Factor"]
    hover_values = [label, hover_width, hover_length, hover_factor]

    fig = bokeh_scatter(width_data, factor_data, hover_labels, hover_values)

    fig.title = "Fallback scatter plot"
    fig.xaxis.axis_label = "Width (cm)"
    fig.yaxis.axis_label = "Factor"

    return fig


def interactive(width_data, length_data, ratio_perim_area_data, factor_data,
                label):
    model_value = spline_model(
        width_data, ratio_perim_area_data,
        width_data, ratio_perim_area_data, factor_data)
    pred_diff = calculate_percent_prediction_differences(
        width_data, ratio_perim_area_data, factor_data)
    pc_residual = 100 * (factor_data - model_value) / factor_data

    transformed_mesh = dict()
    result = create_transformed_mesh(width_data, length_data, factor_data)
    reference = ~np.isnan(np.ravel(result[2]))
    transformed_mesh['width'] = np.ravel(result[0])[reference]
    transformed_mesh['length'] = np.ravel(result[1])[reference]
    transformed_mesh['factor'] = np.ravel(result[2])[reference]
    transformed_mesh['ratio_perim_area'] = convert2_ratio_perim_area(
        transformed_mesh['width'], transformed_mesh['length'])

    native_mesh = dict()
    result = create_native_mesh(width_data, ratio_perim_area_data, factor_data)
    reference = ~np.isnan(np.ravel(result[2]))
    native_mesh['width'] = np.ravel(result[0])[reference]
    native_mesh['ratio_perim_area'] = np.ravel(result[1])[reference]
    native_mesh['factor'] = np.ravel(result[2])[reference]
    native_mesh['length'] = convert2_length(
        native_mesh['width'], native_mesh['ratio_perim_area'])

    all_factor = np.hstack([
        transformed_mesh['factor'], native_mesh['factor'], factor_data])
    vmin = np.nanmin(all_factor)
    vmax = np.nanmax(all_factor)

    colour = find_colour(factor_data, vmin, vmax)

    measurement_data = dict(
        width=width_data,
        length=length_data,
        ratio_perim_area=np.around(ratio_perim_area_data, decimals=3),
        factor=np.around(factor_data, decimals=3),
        model_value=np.around(model_value, decimals=3),
        pc_residual=np.around(pc_residual, decimals=1),
        pred_diff=np.around(pred_diff, decimals=1),
        colour=colour,
        label=label,
        zeros=np.zeros(len(factor_data))
    )

    measurements_source = bkh.models.ColumnDataSource(measurement_data)

    transformed_hover_width = [
        " %0.1f cm" % (num) for num in transformed_mesh['width']]
    transformed_hover_length = [
        " %0.1f cm" % (num) for num in transformed_mesh['length']]
    transformed_hover_ratio_perim_area = [
        " %0.2f cm^-1" % (num) for num in transformed_mesh['ratio_perim_area']]
    transformed_hover_factor = [
        " %0.3f" % (num) for num in transformed_mesh['factor']]

    transformed_data = dict(
        width=transformed_mesh['width'],
        length=transformed_mesh['length'],
        ratio_perim_area=transformed_mesh['ratio_perim_area'],
        factor=transformed_mesh['factor'],
        zeros=np.zeros(len(transformed_mesh['factor'])),
        colour=find_colour(transformed_mesh['factor'], vmin, vmax),
        hover_width=transformed_hover_width,
        hover_length=transformed_hover_length,
        hover_ratio_perim_area=transformed_hover_ratio_perim_area,
        hover_factor=transformed_hover_factor
    )

    transformed_source = bkh.models.ColumnDataSource(transformed_data)

    native_hover_width = [
        " %0.1f cm" % (num) for num in native_mesh['width']]
    native_hover_length = [
        " %0.1f cm" % (num) for num in native_mesh['length']]
    native_hover_ratio_perim_area = [
        " %0.2f cm^-1" % (num) for num in native_mesh['ratio_perim_area']]
    native_hover_factor = [
        " %0.3f" % (num) for num in native_mesh['factor']]

    native_data = dict(
        width=native_mesh['width'],
        length=native_mesh['length'],
        ratio_perim_area=native_mesh['ratio_perim_area'],
        factor=native_mesh['factor'],
        zeros=np.zeros(len(native_mesh['factor'])),
        colour=find_colour(native_mesh['factor'], vmin, vmax),
        hover_width=native_hover_width,
        hover_length=native_hover_length,
        hover_ratio_perim_area=native_hover_ratio_perim_area,
        hover_factor=native_hover_factor
    )

    native_source = bkh.models.ColumnDataSource(native_data)

    columns = [
        bkh.models.widgets.TableColumn(field="label", title="Label"),
        bkh.models.widgets.TableColumn(field="width", title="Width (cm)"),
        bkh.models.widgets.TableColumn(field="length", title="Length (cm)"),
        bkh.models.widgets.TableColumn(field="ratio_perim_area", title="P/A (cm^-1)"),
        bkh.models.widgets.TableColumn(field="factor", title="Insert factor"),
        bkh.models.widgets.TableColumn(
            field="model_value", title="Model factor"),
        bkh.models.widgets.TableColumn(
            field="pc_residual", title="Residual (%)"),
        bkh.models.widgets.TableColumn(
            field="pred_diff", title="Prediction Diff. (%)"),
    ]

    data_table = bkh.models.widgets.DataTable(
        source=measurements_source, columns=columns, width=900, height=500)

    tools = "box_select, tap, crosshair"
    unselect_rectangle = bkh.models.Rect(line_alpha=0, fill_alpha=0)

    native_y_range = bkh.models.Range1d(
        native_data['ratio_perim_area'].min() - 0.005 -
        native_data['ratio_perim_area'].ptp()*0.03,
        native_data['ratio_perim_area'].max() + 0.005 +
        native_data['ratio_perim_area'].ptp()*0.03)
    native_x_range = bkh.models.Range1d(
        native_data['width'].min() - 0.05 - native_data['width'].ptp()*0.03,
        native_data['width'].max() + 0.05 + native_data['width'].ptp()*0.03)

    native = bkh.plotting.figure(
        tools=tools, width=400, height=350,
        title="Native domain",
        x_axis_label="Width (cm)", y_axis_label="Perimeter / Area cm^-1)",
        x_range=native_x_range, y_range=native_y_range)
    native.rect(
        'width', 'ratio_perim_area', 0.1, 0.01, alpha=0, source=transformed_source,
        name='native_invis')
    native.rect(
        'width', 'ratio_perim_area', 0.1, 0.01, color='colour', source=native_source,
        name='native_visible')
    native.circle(
        'width', 'ratio_perim_area', source=measurements_source,
        size=10, fill_color='colour', line_color='black')
    render_invis = native.select(name='native_invis')
    render_invis.nonselection_glyph = unselect_rectangle
    render_vis = native.select(name='native_visible')
    render_vis.nonselection_glyph = unselect_rectangle

    tooltips = [
        ("Width", "@hover_width"),
        ("Length", "@hover_length"),
        ("P/A", "@hover_ratio_perim_area"),
        ("Factor", "@hover_factor"),
    ]
    native.add_tools(bkh.models.HoverTool(
        tooltips=tooltips,
        renderers=render_vis))

    trans_y_range = bkh.models.Range1d(
        transformed_data['length'].min() - 0.05 -
        transformed_data['length'].ptp()*0.03,
        transformed_data['length'].max() + 0.05)
    trans_x_range = bkh.models.Range1d(
        transformed_data['width'].min() - 0.05 -
        transformed_data['width'].ptp()*0.03,
        transformed_data['width'].max() + 0.05 +
        transformed_data['width'].ptp()*0.03)

    transformed = bkh.plotting.figure(
        tools=tools, width=400, height=350,
        title="Transformed domain",
        x_axis_label="Width (cm)", y_axis_label="Length (cm)",
        x_range=trans_x_range, y_range=trans_y_range)
    transformed.rect(
        'width', 'length', 0.1, 0.1, alpha=0, source=native_source,
        name='trans_invis')
    transformed.rect(
        'width', 'length', 0.1, 0.1, color='colour', source=transformed_source,
        name='trans_visible')
    transformed.circle(
        'width', 'length', source=measurements_source,
        size=10, fill_color='colour', line_color='black')
    render_invis = transformed.select(name='trans_invis')
    render_invis.nonselection_glyph = unselect_rectangle
    render_vis = transformed.select(name='trans_visible')
    render_vis.nonselection_glyph = unselect_rectangle

    transformed.add_tools(bkh.models.HoverTool(
        tooltips=tooltips,
        renderers=render_vis))

    mesh_factors = np.hstack(
        [transformed_mesh['factor'], native_mesh['factor']])
    colour_bar_range = bkh.models.Range1d(
        np.floor(100*mesh_factors.min())/100,
        np.ceil(100*mesh_factors.max())/100)

    colour_bar = bkh.plotting.figure(
        tools=tools, width=135, height=350, title=None,
        y_range=colour_bar_range,
        y_axis_label="Factor")
    colour_bar.rect(
        'zeros', 'factor', 0.5, 0.0005, name='colour_bar_trans',
        source=transformed_source, color='colour')
    render = colour_bar.select(name='colour_bar_trans')
    render.nonselection_glyph = unselect_rectangle
    colour_bar.rect(
        'zeros', 'factor', 0.5, 0.0005, name='colour_bar_native',
        source=native_source, color='colour')
    render = colour_bar.select(name='colour_bar_native')
    render.nonselection_glyph = unselect_rectangle
    colour_bar.rect(
        'zeros', 'factor', 0.5, 0.0005, name='colour_bar_meas',
        source=measurements_source, alpha=0)
    render = colour_bar.select(name='colour_bar_meas')
    render.nonselection_glyph = unselect_rectangle

    crosshair = colour_bar.select(type=bkh.models.CrosshairTool)
    crosshair.dimensions = ['width']

    colour_bar.xgrid.grid_line_color = None
    colour_bar.ygrid.grid_line_color = None
    colour_bar.xaxis.major_label_text_font_size = '0pt'
    colour_bar.xaxis.major_tick_line_color = None
    colour_bar.xaxis.minor_tick_line_color = None

    p = bkh.plotting.gridplot(
        [[colour_bar, native, transformed]], toolbar_location=None)

    return bkh.io.vplot(p, data_table)


def create_report_from_dictionary(input_dictionary):
    label = np.array([key for key in input_dictionary])
    width_data = np.array([input_dictionary[key]['width'] for key in label])
    length_data = np.array([input_dictionary[key]['length'] for key in label])
    factor_data = np.array([input_dictionary[key]['factor'] for key in label])
    ratio_perim_area_data = convert2_ratio_perim_area(width_data, length_data)

    return interactive(
        width_data, length_data, ratio_perim_area_data, factor_data, label)


def create_report_from_pandas(input_dataframe):
    label = np.array(input_dataframe['label']).astype(str)
    width_data = np.array(input_dataframe['width']).astype(float)
    length_data = np.array(input_dataframe['length']).astype(float)
    factor_data = np.array(input_dataframe['factor']).astype(float)
    ratio_perim_area_data = convert2_ratio_perim_area(width_data, length_data)
    
    return interactive(
        width_data, length_data, ratio_perim_area_data, factor_data, label)
