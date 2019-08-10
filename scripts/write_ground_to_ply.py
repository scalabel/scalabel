'''
Script for detecting ground in point clouds and writing it back to the PLY file
'''

import argparse
import os
import re
import sys

import numpy as np


def find_ground_plane(points, normals, iters,
                      min_accepted_size,
                      min_dist, max_dist, min_height, max_height,
                      expected_normal):
    '''
    Find ground plane
    :param points: list of points
    :param normals: normal vectors
    :param iters: number of times to run ransac
    :param min_accepted_size: smallest plane size (number of points)
    :param min_dist: min dist of points from origin to be considered in ransac
    :param max_dist: max dist ""
    :param min_height: min height ""
    :param max_height: max height ""
    :param expected_normal: Expected normal direction of the plane
    :return:
    '''
    depths = np.linalg.norm(points, axis=-1)
    height_filtered = np.logical_and(points[:, 2] >= min_height,
                                     points[:, 2] <= max_height)
    dist_filtered = np.logical_and(depths <= max_dist, depths >= min_dist)
    close_to_expected = np.abs(normals.dot(expected_normal)) >= 0.9
    valid_indices = np.logical_and(
        np.logical_and(height_filtered, dist_filtered),
        close_to_expected)
    valid_points = points[valid_indices]
    valid_normals = normals[valid_indices]

    max_num_inliers = 0
    best_plane = None
    best_plane_inliers = None
    best_plane_outliers = None

    for _ in range(iters):
        sample_index = np.random.randint(valid_points.shape[0])
        sample_point = valid_points[sample_index]
        sample_normal = valid_normals[sample_index]

        normal_cos_angles = valid_normals.dot(sample_normal)

        same_dir_indices = normal_cos_angles >= 0.8
        same_dir_points = valid_points[same_dir_indices]

        diffs = same_dir_points - sample_point
        dists = np.linalg.norm(diffs, axis=-1)
        diffs[:, 0] /= dists
        diffs[:, 1] /= dists
        diffs[:, 2] /= dists

        dists_to_plane = (diffs).dot(sample_normal)

        same_plane_indices = np.abs(dists_to_plane) <= 0.15

        same_plane_points = same_dir_points[same_plane_indices]

        if same_plane_points.shape[0] > max_num_inliers:
            cov = np.cov(same_plane_points, rowvar=False)
            eigvals, eigvecs = np.linalg.eigh(cov)
            normal = eigvecs[:, np.argmin(eigvals)]
            center = np.average(same_plane_points, axis=0)
            offset = -normal.dot(center)

            outliers1 = valid_points[np.logical_not(same_dir_indices)]
            outliers2 = same_dir_points[np.logical_not(same_plane_indices)]
            if same_plane_points.shape[0] > valid_points.shape[0] * \
                    min_accepted_size:
                return normal, offset, same_plane_points, np.vstack(
                    (outliers1, outliers2))
            max_num_inliers = same_plane_points.shape[0]
            best_plane = (normal, offset)
            best_plane_inliers = same_plane_points
            best_plane_outliers = np.vstack((outliers1, outliers2))

    return best_plane[0], best_plane[1], \
        best_plane_inliers, best_plane_outliers


def load_ply_files(directory_name):
    '''
    Load PLY files in directory, must have both points & normals
    :param directory_name: name of directory
    :return: array of all point clouds, array of normals, filenames
    '''
    if os.path.isdir(directory_name):
        filenames = os.listdir(directory_name)

        points_arr = []
        normals_arr = []
        output_file_names = []
        for filename in sorted(filenames):
            if filename.split('.')[-1] != 'ply':
                continue

            output_file_names.append(directory_name + '/' + filename)

            contents = open(output_file_names[-1], 'rb').read()
            end_header_string = 'end_header\n'.encode('utf-8')
            end_header_position = contents.find(end_header_string)
            if end_header_position == -1:
                print('Invalid ply header: {}'.format(filename))
                continue

            data_start_ind = end_header_position + len(end_header_string)

            header = contents[:data_start_ind].decode('utf-8')
            header_lines = header.split('\n')
            if header_lines[0] != 'ply':
                print('Invalid ply header: {}'.format(filename))
                continue

            format_string = header_lines[1].split()[1]

            element_prog = re.compile('element')
            property_prog = re.compile('property')
            properties = []
            num_vertices = -1
            error = False
            for i in range(2, len(header_lines)):
                if element_prog.match(header_lines[i]):
                    if num_vertices != -1:
                        print('Multiple element declarations: {}'.format(
                            filename))
                        error = True
                        break

                    tokens = header_lines[i].split()
                    if len(tokens) != 3:
                        print('Invalid element declaration (Line {}): '
                              '{}'.format(i, filename))
                        error = True
                        break

                    if tokens[1] != 'vertex':
                        print('Only vertex elements accepted: '
                              '{}'.format(filename))
                        error = True
                        break

                    num_vertices = int(tokens[2])
                elif property_prog.match(header_lines[i]):
                    if num_vertices == -1:
                        print('Property declaration without element '
                              'declaration: {}'.format(filename))
                        error = True
                        break

                    tokens = header_lines[i].split()
                    if len(tokens) != 3:
                        print('Invalid property declaration: '
                              '{}'.format(filename))
                        error = True
                        break

                    if tokens[1] != 'float':
                        print('Only floats accepted as property types: '
                              '{}'.format(filename))
                        error = True
                        break

                    if tokens[2] in properties:
                        print('Multiple declarations of same property: '
                              '{}'.format(filename))
                        error = True
                        break

                    properties.append(tokens[2])

            if error:
                continue

            if len(properties) != 6:
                print('Missing vertex properties: {}'.format(filename))
                continue

            raw_data = contents[data_start_ind:]

            if format_string == 'ascii':
                data_tokens = raw_data.split()
                if len(data_tokens) != num_vertices * 6:
                    print('Number of data points is different from what is '
                          'specified: {}'.format(filename))
                    continue
                data = [float(token) for token in data_tokens]
                data = np.array(data).reshape(num_vertices, 6)
            elif format_string == 'binary_little_endian':
                data = np.fromstring(raw_data, dtype='<f4').reshape(
                    (num_vertices, 6))
            elif format_string == 'binary_big_endian':
                data = np.fromstring(raw_data, dtype='>f4').reshape(
                    (num_vertices, 6))

            points = np.zeros((num_vertices, 3))
            normals = np.zeros((num_vertices, 3))

            error = False
            for i, prpty in enumerate(properties):
                if prpty == 'x':
                    points[:, 0] = data[:, i]
                elif prpty == 'y':
                    points[:, 1] = data[:, i]
                elif prpty == 'z':
                    points[:, 2] = data[:, i]
                elif prpty == 'nx':
                    normals[:, 0] = data[:, i]
                elif prpty == 'ny':
                    normals[:, 1] = data[:, i]
                elif prpty == 'nz':
                    normals[:, 2] = data[:, i]
                else:
                    print('Invalid property declaration: {}'.format(filename))
                    error = True
                    break
            if error:
                continue

            points_arr.append(points.astype(np.float32))
            normals_arr.append(normals.astype(np.float32))

        return points_arr, normals_arr, output_file_names
    print("Invalid directory: {}".format(directory_name))
    return None, None, None


def main():
    '''
    Main function
    '''
    parser = argparse.ArgumentParser(
        description='Find ground plane and write it to PLY file')
    parser.add_argument('--pointclouddir',
                        help='Folder containing PLY point clouds',
                        required=True)
    parser.add_argument('--iterations',
                        help='Number of iterations to run for RANSAC.',
                        default=20, type=int)
    parser.add_argument('--min_accepted_fraction',
                        help='Minimum ratio of points in found ground plane '
                             'to total valid points for plane to be '
                             'automatically accepted',
                        default=0.4, type=float)
    parser.add_argument('--min_dist', help='Minimum distance of points from '
                                           'origin to be considered valid',
                        default=3, type=float)
    parser.add_argument('--max_dist', help='Maximum distance of points from '
                                           'origin to be considered valid',
                        default=12, type=float)
    parser.add_argument('--min_height',
                        help='Minimum height of points, value '
                             'on z-axis to be considered valid',
                        default=-2, type=float)
    parser.add_argument('--max_height',
                        help='Maximum height of points, value '
                        'on z-axis to be considered valid',
                        default=-1, type=float)
    parser.add_argument('--expected_normal',
                        help='Expected normal of the ground plane',
                        nargs='+', default=[0, 0, 1], type=float)

    args = parser.parse_args()

    if len(args.expected_normal) != 3:
        print('Expected normal must be a 3 element array.')
        sys.exit(0)

    loaded_points, loaded_normals, file_names = \
        load_ply_files(args.pointclouddir)

    if loaded_points is not None:
        for points, normals, file_name in zip(
                loaded_points, loaded_normals, file_names
        ):
            ground_normal, ground_offset, _, _ = \
                find_ground_plane(points, normals,
                                  args.iterations,
                                  args.min_accepted_fraction,
                                  args.min_dist, args.max_dist,
                                  args.min_height, args.max_height,
                                  np.array(args.expected_normal))
            f = open(file_name, 'w')
            f.write("ply\n")
            f.write("format binary_little_endian 1.0\n")
            f.write("comment [groundCoefficients] {}, {}, {}, {}\n".format(
                ground_normal[0], ground_normal[1], ground_normal[2],
                ground_offset))
            f.write("element vertex {}\n".format(points.shape[0]))
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("property float nx\n")
            f.write("property float ny\n")
            f.write("property float nz\n")
            f.write("end_header\n")

            f.close()
            f = open(file_name, 'ab')

            for i in range(points.shape[0]):
                f.write(points[i].tobytes())
                f.write(normals[i].tobytes())

            f.close()


if __name__ == '__main__':
    main()
