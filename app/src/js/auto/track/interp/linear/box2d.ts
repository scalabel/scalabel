import _ from 'lodash'
import { LabelType, RectType, ShapeType, SimpleRect } from '../../../../functional/types'
import { assignShapesInRange, getAutoLabelRange, TrackInterp } from '../interp'

/**
 * Linearly interpolate the rectangles from the first to the last in
 * the shape array.
 * @param shapes
 */
function linearInterpBoxes (
    labels: LabelType[], shapes: RectType[]): RectType[] {
  const newShapes = shapes.map((s) => _.cloneDeep(s))
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
    shape.x1 = diff.x1 * i + first.x1
    shape.y1 = diff.y1 * i + first.y1
    shape.x2 = diff.x2 * i + first.x2
    shape.y2 = diff.y2 * i + first.y2
  }
  return newShapes
}

/**
 * Linearly interpolate the bounding boxes in [start, end]
 * The results will be put in [start, end] of allShapes
 * @param start
 * @param end
 * @param allLabels
 * @param allShapes
 */
function linearInterpBoxesInRange (
    start: number, end: number,
    allLabels: LabelType[], allShapes: ShapeType[][]): void {
  let boxes = allShapes.slice(
    start, end + 1).map((s) => s[0]) as RectType[]
  const labels = allLabels.slice(start, end + 1)
  boxes = linearInterpBoxes(labels, boxes)
  for (let i = start + 1; i < end; i += 1) {
    allShapes[i][0] = boxes[i - start]
  }
}

/**
 * Interpolating 2D bounding boxes linearly
 */
export class Box2DLinearInterp extends TrackInterp {
  /**
   * Main method for interpolation. It assumes allLabels is sorted by itemIndex
   * @param newLabel
   * @param newShape
   * @param labels
   * @param shapes
   */
  public interp (
    newLabel: LabelType, newShape: ShapeType[],
    allLabels: LabelType[], allShapes: ShapeType[][]): ShapeType[][] {
    const [labelIndex, manual0, manual1] = getAutoLabelRange(
      newLabel, allLabels)
    // Copy the double array
    const newShapes = allShapes.map((shapes) => shapes.map((s) => s))
    newShapes[labelIndex] = newShape
    if (manual0 === -1) {
      assignShapesInRange(0, labelIndex, newShape, newShapes)
    } else {
      linearInterpBoxesInRange(manual0, labelIndex, allLabels, newShapes)
    }
    if (manual1 === -1) {
      assignShapesInRange(labelIndex + 1, newShapes.length, newShape, newShapes)
    } else {
      linearInterpBoxesInRange(labelIndex, manual1, allLabels, newShapes)
    }
    return newShapes
  }
}
