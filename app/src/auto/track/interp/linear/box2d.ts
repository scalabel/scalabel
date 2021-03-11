import _ from "lodash"

import {
  LabelType,
  RectType,
  ShapeType,
  SimpleRect
} from "../../../../types/state"
import { assignShapesInRange, getAutoLabelRange, TrackInterp } from "../interp"

/**
 * Linearly interpolate the rectangles from the first to the last in
 * the shape array.
 *
 * @param labels
 * @param shapes
 */
function linearInterpBoxes(
  labels: LabelType[],
  shapes: RectType[]
): RectType[] {
  const newShapes = _.cloneDeep(shapes)
  if (newShapes.length <= 2) {
    return newShapes
  }
  const first = shapes[0]
  const last = shapes[shapes.length - 1]
  const num = labels[labels.length - 1].item - labels[0].item
  const diff: SimpleRect = {
    x1: (last.x1 - first.x1) / num,
    y1: (last.y1 - first.y1) / num,
    x2: (last.x2 - first.x2) / num,
    y2: (last.y2 - first.y2) / num
  }
  for (let i = 1; i < newShapes.length - 1; i += 1) {
    const shape = newShapes[i]
    const dist = labels[i].item - labels[0].item
    shape.x1 = diff.x1 * dist + first.x1
    shape.y1 = diff.y1 * dist + first.y1
    shape.x2 = diff.x2 * dist + first.x2
    shape.y2 = diff.y2 * dist + first.y2
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
function linearInterpBoxesInRange(
  start: number,
  end: number,
  allLabels: LabelType[],
  allShapes: ShapeType[][]
): ShapeType[][] {
  allShapes = [...allShapes]
  let boxes = allShapes.slice(start, end + 1).map((s) => s[0]) as RectType[]
  const labels = allLabels.slice(start, end + 1)
  boxes = linearInterpBoxes(labels, boxes)
  for (let i = start + 1; i < end; i += 1) {
    allShapes[i] = [boxes[i - start]]
  }
  return allShapes
}

/**
 * Interpolating 2D bounding boxes linearly
 */
export class Box2DLinearInterp extends TrackInterp {
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
    // Let newShapes = allShapes.map((shapes) => [...shapes]
    newShapes[labelIndex] = _.cloneDeep(newShape)
    if (manual0 === -1) {
      newShapes = assignShapesInRange(0, labelIndex, newShape, newShapes)
    } else {
      newShapes = linearInterpBoxesInRange(
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
      newShapes = linearInterpBoxesInRange(
        labelIndex,
        manual1,
        allLabels,
        newShapes
      )
    }
    return newShapes
  }
}
