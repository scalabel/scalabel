/**
 * Constant type definition
 */

export enum LabelTypes {
  EMPTY = 'empty',
  TAG = 'tag',
  BOX_2D = 'box2d',
  POLYGON_2D = 'polygon2d',
  POLYLINE_2D = 'polyline2d',
  BOX_3D = 'box3d',
  PLANE_3D = 'plane3d'
}

export enum ShapeTypes {
  UNKNOWN = 'unknown',
  RECT = 'rect',
  CUBE = 'cube',
  POINT_2D = 'point2d',
  PATH_POINT_2D = 'path_point2d',
  POLYGON_2D = 'polygon2d'
}

export enum PathPointTypes {
  LINE = 'line',
  CURVE = 'bezier' // cubic Bezier curve path points
}
