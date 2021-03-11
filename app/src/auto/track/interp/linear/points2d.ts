import _ from "lodash"

import { makePathPoint2D } from "../../../../functional/states"
import {
  LabelType,
  PathPoint2DType,
  PathPointType,
  ShapeType,
  SimplePathPoint2DType
} from "../../../../types/state"
import { assignShapesInRange, getAutoLabelRange, TrackInterp } from "../interp"

/**
 * Linearly interpolate the rectangles from the first to the last in
 * the shape array.
 *
 * @param labels
 * @param shapes
 */
function linearInterpPoints(
  labels: LabelType[],
  shapes: PathPoint2DType[][]
): PathPoint2DType[][] {
  const newShapes = [...shapes]
  if (newShapes.length <= 2) {
    return newShapes
  }
  const first = shapes[0]
  const last = shapes[shapes.length - 1]
  if (first.length !== last.length) {
    // The start and end of shape lengths don't match
    return newShapes
  }
  const num = labels[labels.length - 1].item - labels[0].item
  const diffs: SimplePathPoint2DType[] = []
  for (let i = 0; i < first.length; i += 1) {
    diffs.push({
      x: (last[i].x - first[i].x) / num,
      y: (last[i].y - first[i].y) / num,
      pointType: PathPointType.UNKNOWN
    })
  }
  for (let i = 1; i < newShapes.length - 1; i += 1) {
    let shape = newShapes[i]
    if (shape.length !== first.length) {
      shape = first.map(() => makePathPoint2D())
    } else {
      shape = _.cloneDeep(shape)
    }
    const dist = labels[i].item - labels[0].item
    for (let j = 0; j < first.length; j += 1) {
      shape[j].x = diffs[j].x * dist + first[j].x
      shape[j].y = diffs[j].y * dist + first[j].y
    }
    newShapes[i] = shape
    // Console.log(shape, diffs, first)
  }
  return newShapes
}

/**
 * Linearly interpolate the path points in [start, end]
 * The results will be put in [start, end] of allShapes
 *
 * @param start
 * @param end
 * @param allLabels
 * @param allShapes
 */
function linearInterpPointsInRange(
  start: number,
  end: number,
  allLabels: LabelType[],
  allShapes: ShapeType[][]
): ShapeType[][] {
  allShapes = [...allShapes]
  let points = allShapes.slice(start, end + 1) as PathPoint2DType[][]
  const labels = allLabels.slice(start, end + 1)
  points = linearInterpPoints(labels, points)
  for (let i = start + 1; i < end; i += 1) {
    allShapes[i] = points[i - start]
  }
  return allShapes
}

/**
 * Interpolating path points linearly
 */
export class Points2DLinearInterp extends TrackInterp {
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
    // Copy the input array
    let newShapes = [...allShapes]
    newShapes[labelIndex] = newShape
    if (manual0 === -1) {
      newShapes = assignShapesInRange(0, labelIndex, newShape, newShapes)
    } else {
      newShapes = linearInterpPointsInRange(
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
    } else {
      newShapes = linearInterpPointsInRange(
        labelIndex,
        manual1,
        allLabels,
        newShapes
      )
    }
    return newShapes
  }
}
