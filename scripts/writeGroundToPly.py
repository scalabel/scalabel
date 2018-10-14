import argparse
import numpy as np
import os
import re
import sys


def find_ground_plane(points, normals, iters,
                      minAcceptedSize,
                      minDist, maxDist, minHeight, maxHeight,
                      expectedNormal):
    depths = np.linalg.norm(points, axis=-1)
    heightFiltered = np.logical_and(points[:, 2] >= minHeight,
                                    points[:, 2] <= maxHeight)
    distFiltered = np.logical_and(depths <= maxDist, depths >= minDist)
    closeToExpected = np.abs(normals.dot(expectedNormal)) >= 0.9
    validIndices = np.logical_and(np.logical_and(heightFiltered, distFiltered),
                                  closeToExpected)
    validPoints = points[validIndices]
    validNormals = normals[validIndices]

    maxNumInliers = 0
    bestPlane = None
    bestPlaneInliers = None
    bestPlaneOutliers = None

    for i in range(iters):
        sampleIndex = np.random.randint(validPoints.shape[0])
        samplePoint = validPoints[sampleIndex]
        sampleNormal = validNormals[sampleIndex]

        normalCosAngles = validNormals.dot(sampleNormal)

        sameDirIndices = normalCosAngles >= 0.8
        sameDirPoints = validPoints[sameDirIndices]

        diffs = sameDirPoints - samplePoint
        dists = np.linalg.norm(diffs, axis=-1)
        diffs[:, 0] /= dists
        diffs[:, 1] /= dists
        diffs[:, 2] /= dists

        distsToPlane = (diffs).dot(sampleNormal)

        samePlaneIndices = np.abs(distsToPlane) <= 0.15

        samePlanePoints = sameDirPoints[samePlaneIndices]

        if samePlanePoints.shape[0] > maxNumInliers:
            cov = np.cov(samePlanePoints, rowvar=False)
            eigvals, eigvecs = np.linalg.eigh(cov)
            normal = eigvecs[:, np.argmin(eigvals)]
            center = np.average(samePlanePoints, axis=0)
            offset = -normal.dot(center)

            outliers1 = validPoints[np.logical_not(sameDirIndices)]
            outliers2 = sameDirPoints[np.logical_not(samePlaneIndices)]
            if samePlanePoints.shape[0] > validPoints.shape[0] * \
               minAcceptedSize:
                return normal, offset, samePlanePoints, np.vstack((outliers1,
                                                                   outliers2))
            maxNumInliers = samePlanePoints.shape[0]
            bestPlane = (normal, offset)
            bestPlaneInliers = samePlanePoints
            bestPlaneOutliers = np.vstack((outliers1, outliers2))

    return bestPlane[0], bestPlane[1], bestPlaneInliers, bestPlaneOutliers


def loadPlyFiles(directoryName):
    if os.path.isdir(directoryName):
        filenames = os.listdir(directoryName)

        pointsArr = []
        normalsArr = []
        outputFileNames = []
        for filename in sorted(filenames):
            if filename.split('.')[-1] != 'ply':
                continue

            outputFileNames.append(directoryName + '/' + filename)

            contents = open(outputFileNames[-1], 'rb').read()
            endHeaderString = 'end_header\n'.encode('utf-8')
            endHeaderPosition = contents.find(endHeaderString)
            if endHeaderPosition == -1:
                print('Invalid ply header: {}'.format(filename))
                continue

            dataStartInd = endHeaderPosition + len(endHeaderString)

            header = contents[:dataStartInd].decode('utf-8')
            headerLines = header.split('\n')
            if headerLines[0] != 'ply':
                print('Invalid ply header: {}'.format(filename))
                continue

            formatString = headerLines[1].split()[1]

            elementProg = re.compile('element')
            propertyProg = re.compile('property')
            properties = []
            numVertices = -1
            error = False
            for i in range(2, len(headerLines)):
                if elementProg.match(headerLines[i]):
                    if numVertices != -1:
                        print('Multiple element declarations: {}'.format(
                            filename))
                        error = True
                        break

                    tokens = headerLines[i].split()
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

                    numVertices = int(tokens[2])
                elif propertyProg.match(headerLines[i]):
                    if numVertices == -1:
                        print('Property declaration without element '
                              'declaration: {}'.format(filename))
                        error = True
                        break

                    tokens = headerLines[i].split()
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

            rawData = contents[dataStartInd:]

            if formatString == 'ascii':
                dataTokens = rawData.split()
                if len(dataTokens) != numVertices * 6:
                    print('Number of data points is different from what is '
                          'specified: {}'.format(filename))
                    continue
                data = [float(token) for token in dataTokens]
                data = np.array(data).reshape(numVertices, 6)
            elif formatString == 'binary_little_endian':
                data = np.fromstring(rawData, dtype='<f4').reshape(
                    (numVertices, 6))
            elif formatString == 'binary_big_endian':
                data = np.fromstring(rawData, dtype='>f4').reshape(
                    (numVertices, 6))

            points = np.zeros((numVertices, 3))
            normals = np.zeros((numVertices, 3))

            error = False
            for i in range(len(properties)):
                property = properties[i]
                if property == 'x':
                    points[:, 0] = data[:, i]
                elif property == 'y':
                    points[:, 1] = data[:, i]
                elif property == 'z':
                    points[:, 2] = data[:, i]
                elif property == 'nx':
                    normals[:, 0] = data[:, i]
                elif property == 'ny':
                    normals[:, 1] = data[:, i]
                elif property == 'nz':
                    normals[:, 2] = data[:, i]
                else:
                    print('Invalid property declaration: {}'.format(filename))
                    error = True
                    break
            if error:
                continue

            pointsArr.append(points.astype(np.float32))
            normalsArr.append(normals.astype(np.float32))

        return pointsArr, normalsArr, outputFileNames
    else:
        print("Invalid directory: {}".format(directoryName))
        return None, None, None


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Find ground plane and write it to PLY file')
    parser.add_argument('--pointclouddir',
                        help='Folder containing PLY point clouds',
                        required=True)
    parser.add_argument('--iterations',
                        help='Number of iterations to run for RANSAC.',
                        default=20, type=int)
    parser.add_argument('--minAcceptedFraction',
                        help='Minimum ratio of points in found ground plane '
                             'to total valid points for plane to be '
                             'automatically accepted',
                        default=0.4, type=float)
    parser.add_argument('--minDist', help='Minimum distance of points from '
                                          'origin to be considered valid',
                        default=3, type=float)
    parser.add_argument('--maxDist', help='Maximum distance of points from '
                                          'origin to be considered valid',
                        default=12, type=float)
    parser.add_argument('--minHeight', help='Minimum height of points, value '
                                            'on z-axis to be considered valid',
                        default=-2, type=float)
    parser.add_argument('--maxHeight', help='Maximum height of points, value '
                                            'on z-axis to be considered valid',
                        default=-1, type=float)
    parser.add_argument('--expectedNormal',
                        help='Expected normal of the ground plane',
                        nargs='+', default=[0, 0, 1], type=float)

    args = parser.parse_args()

    if len(args.expectedNormal) != 3:
        print('Expected normal must be a 3 element array.')
        sys.exit(0)

    pointsArr, normalsArr, filenames = loadPlyFiles(args.pointclouddir)

    if pointsArr is not None:
        for i in range(len(pointsArr)):
            groundNormal, groundOffset, _, _ = \
                find_ground_plane(pointsArr[i], normalsArr[i], args.iterations,
                                  args.minAcceptedFraction,
                                  args.minDist, args.maxDist,
                                  args.minHeight, args.maxHeight,
                                  np.array(args.expectedNormal))
            f = open(filenames[i], 'w')
            f.write("ply\n")
            f.write("format binary_little_endian 1.0\n")
            f.write("comment [groundCoefficients] {}, {}, {}, {}\n".format(
                groundNormal[0], groundNormal[1], groundNormal[2],
                groundOffset))
            f.write("element vertex {}\n".format(pointsArr[i].shape[0]))
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("property float nx\n")
            f.write("property float ny\n")
            f.write("property float nz\n")
            f.write("end_header\n")

            f.close()
            f = open(filenames[i], 'ab')

            for j in range(pointsArr[i].shape[0]):
                f.write(pointsArr[i][j].tobytes())
                f.write(normalsArr[i][j].tobytes())

            f.close()
