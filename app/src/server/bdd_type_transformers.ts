import { ShapeTypeName } from "../const/common"
import * as bdd from "../types/export"
import * as types from "../types/state"

/**
 * Transform internal rect to export 2d box
 *
 * @param shape
 */
export function transformBox2D(shape: types.ShapeType): bdd.Box2DType {
  if (shape.shapeType !== ShapeTypeName.RECT) {
    throw TypeError(
      `Received wrong shape type ${shape.shapeType} for shape ${shape.id}. ` +
        `Expecting ${ShapeTypeName.RECT}`
    )
  }
  const box2d = shape as types.RectType
  return {
    x1: box2d.x1,
    x2: box2d.x2,
    y1: box2d.y1,
    y2: box2d.y2
  }
}

/**
 * Transform internal cube to export 3d box
 *
 * @param box3d
 * @param shape
 */
export function transformBox3D(shape: types.ShapeType): bdd.Box3DType {
  if (shape.shapeType !== ShapeTypeName.CUBE) {
    throw TypeError(
      `Received wrong shape type ${shape.shapeType} for shape ${shape.id}. ` +
        `Expecting ${ShapeTypeName.CUBE}`
    )
  }
  const box3d = shape as types.CubeType
  return {
    center: box3d.center,
    size: box3d.size,
    orientation: box3d.orientation
  }
}

/**
 * Transform internal plane to export 3d plane
 *
 * @param plane
 * @param shape
 */
export function transformPlane3D(shape: types.ShapeType): bdd.Plane3DType {
  if (shape.shapeType !== ShapeTypeName.GRID) {
    throw TypeError(
      `Received wrong shape type ${shape.shapeType} for shape ${shape.id}. ` +
        `Expecting ${ShapeTypeName.GRID}`
    )
  }
  const plane = shape as types.Plane3DType
  return {
    center: plane.center,
    orientation: plane.orientation
  }
}
