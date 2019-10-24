import { addDuplicatedTrack } from '../../action/track'
import { CHANGE_SHAPES } from '../../action/types'
import { Box2D } from '../../drawable/2d/box2d'
import { Rect2D } from '../../drawable/2d/rect2d'
import { makeLabel } from '../../functional/states'
import { ItemType, RectType, ShapeType } from '../../functional/types'
import { Vector2D } from '../../math/vector2d'
import Session from '../session'
import { Label, Track } from '../track'
import * as types from '../types'
import { TrackPolicy } from './track_policy'

/**
 * interpolation
 * @param state
 * @param labels
 * @param firstItemIndex
 * @param lastItemIndex
 * @param updatedIndices
 * @param updatedShapeIds
 * @param updatedShapes
 */
function interpolateRects (
  items: ItemType[],
  labels: {[index: number]: number},
  firstItemIndex: number,
  lastItemIndex: number,
  firstRect: RectType,
  lastRect: RectType,
  updatedIndices: number[],
  updatedShapeIds: number[][],
  updatedShapes: Array<Array<Partial<RectType>>>
) {
  const firstCenter = new Vector2D(
    (firstRect.x1 + firstRect.x2) / 2.,
    (firstRect.y1 + firstRect.y2) / 2.
  )
  const firstDimension = new Vector2D(
    firstRect.x1 - firstRect.x2,
    firstRect.y1 - firstRect.y2
  )
  const lastCenter = new Vector2D(
    (lastRect.x1 + lastRect.x2) / 2.,
    (lastRect.y1 + lastRect.y2) / 2.
  )
  const lastDimension = new Vector2D(
    lastRect.x1 - lastRect.x2,
    lastRect.y1 - lastRect.y2
  )

  const numItems = lastItemIndex - firstItemIndex

  const centerDelta = lastCenter.clone()
  centerDelta.subtract(firstCenter)
  centerDelta.scale(1. / numItems)

  const dimensionDelta = lastDimension.clone()
  dimensionDelta.subtract(firstDimension)
  dimensionDelta.scale(1. / numItems)

  for (let i = firstItemIndex + 1; i < lastItemIndex; i += 1) {
    if (i in labels) {
      const indexDelta = i - firstItemIndex
      const labelId = labels[i]
      const label = items[i].labels[labelId]

      const newCenter = centerDelta.clone()
      newCenter.scale(indexDelta)
      newCenter.add(firstCenter)

      const newDimension = dimensionDelta.clone()
      newDimension.scale(indexDelta)
      newDimension.add(firstDimension)

      updatedIndices.push(i)
      updatedShapeIds.push([label.shapes[0]])
      updatedShapes.push([{
        x1: newCenter.x - newDimension.x / 2.,
        x2: newCenter.x + newDimension.x / 2.,
        y1: newCenter.y - newDimension.y / 2.,
        y2: newCenter.y + newDimension.y / 2.
      }])
    }
  }
}

/**
 * Box 3d linear interpolation
 * @param trackId
 * @param updatedItemIndex
 * @param newShapes
 */
function linearInterpolateBox2D (
  trackId: number,
  updatedItemIndex: number,
  newShapes: ShapeType[]
) {
  const state = Session.getState()
  const track = state.task.tracks[trackId]
  const items = state.task.items

  const updatedIndices: number[] = []
  const updatedShapeIds: number[][] = []
  const updatedShapes: Array<Array<Partial<RectType>>> = []

  // Go backward
  let lastManualIndex = -1
  let lastLabel
  for (let i = updatedItemIndex - 1; i >= 0; i -= 1) {
    if (i in track.labels) {
      const labelId = track.labels[i]
      const label = items[i].labels[labelId]
      if (label.manual) {
        lastManualIndex = i
        lastLabel = label
        break
      }
    }
  }

  // Go forward
  let nextManualIndex = -1
  let nextLabel
  for (let i = updatedItemIndex + 1; i < items.length; i += 1) {
    if (i in track.labels) {
      const labelId = track.labels[i]
      const label = items[i].labels[labelId]
      if (label.manual) {
        nextManualIndex = i
        nextLabel = label
        break
      }
    }
  }

  const newRect = newShapes[0] as RectType

  if (lastManualIndex >= 0 && lastLabel) {
    interpolateRects(
      items,
      track.labels,
      lastManualIndex,
      updatedItemIndex,
      items[lastManualIndex].shapes[lastLabel.shapes[0]].shape as RectType,
      newRect,
      updatedIndices,
      updatedShapeIds,
      updatedShapes
    )
  }

  if (nextManualIndex >= 0 && nextLabel) {
    interpolateRects(
      items,
      track.labels,
      updatedItemIndex,
      nextManualIndex,
      newRect,
      items[nextManualIndex].shapes[nextLabel.shapes[0]].shape as RectType,
      updatedIndices,
      updatedShapeIds,
      updatedShapes
    )
  }

  return {
    type: CHANGE_SHAPES,
    sessionId: Session.id,
    itemIndices: updatedIndices,
    shapeIds: updatedShapeIds,
    shapes: updatedShapes
  }
}

/**
 * Class for linear interpolating box 3d's
 */
export class LinearInterpolationBox2DPolicy extends TrackPolicy {
  constructor (track: Track) {
    super(track)
    this._policyType = types.TrackPolicyType.LINEAR_INTERPOLATION_BOX_2D
  }

  /**
   * Callback for when a label in the track is updated
   * @param itemIndex
   * @param labelId
   * @param newShapes
   */
  public onLabelUpdated (
    itemIndex: number, newShapes: ShapeType[]
  ) {
    Session.dispatch(linearInterpolateBox2D(
      this._track.id, itemIndex, newShapes
    ))
  }

  /**
   * Callback for label creation
   * @param itemIndex
   * @param label
   * @param shapes
   * @param shapeTypes
   */
  public onLabelCreated (
    itemIndex: number,
    label: Label
  ) {
    const rect = ((label as Box2D).shapes[0] as Rect2D).toRect()
    const labelObject = makeLabel({
      type: types.LabelTypeName.BOX_2D,
      category: label.category
    })

    const state = Session.getState()
    if (state.task.config.tracking) {
      Session.dispatch(addDuplicatedTrack(
        labelObject,
        [types.ShapeTypeName.RECT],
        [rect],
        itemIndex
      ))
    }
  }
}
