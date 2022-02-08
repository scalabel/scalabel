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
    location: [box3d.center.x, box3d.center.y, box3d.center.z],
    dimension: [box3d.size.x, box3d.size.y, box3d.size.z],
    orientation: [box3d.orientation.x, box3d.orientation.y, box3d.orientation.z]
  }
}

/**
 * Transform exported 3d box to internal cube
 *
 * @param box3d
 */
export function box3dToCube(box3d: bdd.Box3DType): types.SimpleCube {
  return {
    size: {
      x: box3d.dimension[0],
      y: box3d.dimension[1],
      z: box3d.dimension[2]
    },
    orientation: {
      x: box3d.orientation[0],
      y: box3d.orientation[1],
      z: box3d.orientation[2]
    },
    center: {
      x: box3d.location[0],
      y: box3d.location[1],
      z: box3d.location[2]
    },
    anchorIndex: 0
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

/**
 * Transform internal intrinsic parameters to export type
 *
 * @param intrinsics
 */
export function intrinsicsToExport(
  intrinsics: types.IntrinsicsType
): bdd.IntrinsicsExportType {
  return {
    focal: [intrinsics.focalLength.x, intrinsics.focalLength.y],
    center: [intrinsics.focalCenter.x, intrinsics.focalCenter.y]
  }
}

/**
 * Transform exported intrinsics to internal type
 *
 * @param intrinsicsExport
 */
export function intrinsicsFromExport(
  intrinsicsExport: bdd.IntrinsicsExportType
): types.IntrinsicsType {
  return {
    focalLength: {
      x: intrinsicsExport.focal[0],
      y: intrinsicsExport.focal[1]
    },
    focalCenter: {
      x: intrinsicsExport.center[0],
      y: intrinsicsExport.center[1]
    }
  }
}

/**
 * Transform internal extrinsic parameters to export type
 *
 * @param extrinsics
 */
export function extrinsicsToExport(
  extrinsics: types.ExtrinsicsType
): bdd.ExtrinsicsExportType {
  return {
    location: [
      extrinsics.translation.x,
      extrinsics.translation.y,
      extrinsics.translation.z
    ],
    rotation: [
      extrinsics.rotation.x,
      extrinsics.rotation.y,
      extrinsics.rotation.z
    ]
  }
}

/**
 * Transform exported extrinsics to internal type
 *
 * @param extrinsicsExport
 */
export function extrinsicsFromExport(
  extrinsicsExport: bdd.ExtrinsicsExportType
): types.ExtrinsicsType {
  return {
    rotation: {
      x: extrinsicsExport.rotation[0],
      y: extrinsicsExport.rotation[1],
      z: extrinsicsExport.rotation[2],
      w: 0.0
    },
    translation: {
      x: extrinsicsExport.location[0],
      y: extrinsicsExport.location[1],
      z: extrinsicsExport.location[2]
    }
  }
}

/**
 * Transform exported sensors to internal type
 *
 * @param sensors
 */
export function sensorsFromExport(sensorExportMap: {
  [id: number]: bdd.SensorExportType
}): types.SensorMapType {
  const sensors: types.SensorMapType = {}
  for (const key of Object.keys(sensorExportMap)) {
    const sensorExport = sensorExportMap[Number(key)]
    sensors[Number(key)] = {
      ...sensorExport,
      intrinsics:
        sensorExport.intrinsics === undefined ||
        sensorExport.intrinsics === null
          ? undefined
          : intrinsicsFromExport(sensorExport.intrinsics),
      extrinsics:
        sensorExport.extrinsics === undefined ||
        sensorExport.extrinsics === null
          ? undefined
          : extrinsicsFromExport(sensorExport.extrinsics)
    }
  }
  return sensors
}
