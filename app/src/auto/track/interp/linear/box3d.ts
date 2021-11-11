import _ from "lodash"

import {
  LabelType,
  Vector3Type,
  SimpleCube,
  CubeType,
  ShapeType
} from "../../../../types/state"
import { assignShapesInRange, getAutoLabelRange, TrackInterp } from "../interp"
import * as THREE from "three"

/**
 * Linearly interpolate the cubes from the first to the last in
 * the shape array.
 *
 * @param labels
 * @param shapes
 */
function linearInterp3DBoxes(
  labels: LabelType[],
  shapes: CubeType[]
): CubeType[] {
  const newShapes = _.cloneDeep(shapes)
  if (newShapes.length <= 2) {
    return newShapes
  }
  const first = shapes[0]
  const last = shapes[shapes.length - 1]
  const num = labels[labels.length - 1].item - labels[0].item

  // calculate the offset for position and orientation
  const diffCentre: Vector3Type = {
    x: (last.center.x - first.center.x) / num,
    y: (last.center.y - first.center.y) / num,
    z: (last.center.z - first.center.z) / num
  }
  const diffSize = {
    x: (last.size.x - first.size.x) / num,
    y: (last.size.y - first.size.y) / num,
    z: (last.size.z - first.size.z) / num
  }
  const eulerFirst = new THREE.Euler().set(
    first.orientation.x,
    first.orientation.y,
    first.orientation.z
  )
  const quaternionFirst = new THREE.Quaternion().setFromEuler(eulerFirst)
  const eulerLast = new THREE.Euler().set(
    last.orientation.x,
    last.orientation.y,
    last.orientation.z
  )
  const quaternionLast = new THREE.Quaternion().setFromEuler(eulerLast)
  const diff: SimpleCube = {
    center: diffCentre,
    size: diffSize,
    orientation: first.orientation,
    anchorIndex: first.anchorIndex
  }

  // calculate final position and orientation
  for (let i = 1; i < newShapes.length - 1; i += 1) {
    const shape = newShapes[i]
    const dist = labels[i].item - labels[0].item
    shape.center.x = diff.center.x * dist + first.center.x
    shape.center.y = diff.center.y * dist + first.center.y
    shape.center.z = diff.center.z * dist + first.center.z
    shape.size.x = diff.size.x * dist + first.size.x
    shape.size.y = diff.size.y * dist + first.size.y
    shape.size.z = diff.size.z * dist + first.size.z
    const newQuaternion = new THREE.Quaternion().slerpQuaternions(
      quaternionFirst,
      quaternionLast,
      dist / num
    )
    const newEuler = new THREE.Euler().setFromQuaternion(newQuaternion)
    shape.orientation.x = newEuler.x
    shape.orientation.y = newEuler.y
    shape.orientation.z = newEuler.z
  }
  return newShapes
}

/**
 * Linearly interpolate the bounding boxes in [start, end]
 * The results will be put in [start, end] of allShapes
 *
 * @param start
 * @param end
 * @param allLabels
 * @param allShapes
 */
function linearInterp3DBoxesInRange(
  start: number,
  end: number,
  allLabels: LabelType[],
  allShapes: ShapeType[][]
): ShapeType[][] {
  allShapes = [...allShapes]
  let boxes = allShapes.slice(start, end + 1).map((s) => s[0]) as CubeType[]
  const labels = allLabels.slice(start, end + 1)
  boxes = linearInterp3DBoxes(labels, boxes)
  for (let i = start + 1; i < end; i += 1) {
    allShapes[i] = [boxes[i - start]]
  }
  return allShapes
}

/**
 * Interpolating 3D bounding boxes linearly
 */
export class Box3DLinearInterp extends TrackInterp {
  /**
   * Main method for interpolation. It assumes allLabels is sorted by itemIndex
   *
   * @param newLabel
   * @param newShape
   * @param labels
   * @param shapes
   * @param allLabels
   * @param allShapes
   */
  public interp(
    newLabel: LabelType,
    newShape: ShapeType[],
    allLabels: LabelType[],
    allShapes: ShapeType[][]
  ): ShapeType[][] {
    const [labelIndex, manual0, manual1] = getAutoLabelRange(
      newLabel,
      allLabels
    )
    // Copy the array
    let newShapes = [...allShapes]
    newShapes[labelIndex] = _.cloneDeep(newShape)
    if (manual0 === -1) {
      newShapes = assignShapesInRange(0, labelIndex, newShape, newShapes)
    } else if (manual0 >= 0) {
      newShapes = linearInterp3DBoxesInRange(
        manual0,
        labelIndex,
        allLabels,
        newShapes
      )
    }
    if (manual1 === -1) {
      newShapes = assignShapesInRange(
        labelIndex + 1,
        newShapes.length,
        newShape,
        newShapes
      )
    } else if (manual1 >= 0) {
      newShapes = linearInterp3DBoxesInRange(
        labelIndex,
        manual1,
        allLabels,
        newShapes
      )
    }
    return newShapes
  }
}
