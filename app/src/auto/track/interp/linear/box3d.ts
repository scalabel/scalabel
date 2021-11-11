import _ from "lodash"

import {
  LabelType,
  Vector3Type,
  CubeType,
  ShapeType
} from "../../../../types/state"
import { assignShapesInRange, getAutoLabelRange, TrackInterp } from "../interp"

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
  const diffCenter: Vector3Type = {
    x: (last.center.x - first.center.x) / num,
    y: (last.center.y - first.center.y) / num,
    z: (last.center.z - first.center.z) / num
  }
  const diffSize = {
    x: (last.size.x - first.size.x) / num,
    y: (last.size.y - first.size.y) / num,
    z: (last.size.z - first.size.z) / num
  }
  const diffOrientation = {
    x: (last.orientation.x - first.orientation.x) / num,
    y: (last.orientation.y - first.orientation.y) / num,
    z: (last.orientation.z - first.orientation.z) / num
  }

  // calculate final position and orientation
  for (let i = 1; i < newShapes.length - 1; i += 1) {
    const shape = newShapes[i]
    const dist = labels[i].item - labels[0].item
    shape.center.x = diffCenter.x * dist + first.center.x
    shape.center.y = diffCenter.y * dist + first.center.y
    shape.center.z = diffCenter.z * dist + first.center.z
    shape.size.x = diffSize.x * dist + first.size.x
    shape.size.y = diffSize.y * dist + first.size.y
    shape.size.z = diffSize.z * dist + first.size.z
    shape.orientation.x = diffOrientation.x * dist + first.orientation.x
    shape.orientation.y = diffOrientation.y * dist + first.orientation.y
    shape.orientation.z = diffOrientation.z * dist + first.orientation.z
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
