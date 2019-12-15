
import { addDuplicatedTrack } from '../../action/track'
import { CHANGE_SHAPES } from '../../action/types'
import { Plane3D } from '../../drawable/3d/plane3d'
import { makeLabel } from '../../functional/states'
import { ItemType, Plane3DType, ShapeType } from '../../functional/types'
import { Vector3D } from '../../math/vector3d'
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
function interpolatePlanes (
  items: ItemType[],
  labels: {[index: number]: number},
  firstItemIndex: number,
  lastItemIndex: number,
  firstPlane: Plane3DType,
  lastPlane: Plane3DType,
  updatedIndices: number[],
  updatedShapeIds: number[][],
  updatedShapes: Array<Array<Partial<Plane3DType>>>
) {
  const firstCenter = (new Vector3D()).fromObject(firstPlane.center)
  const firstOrientation =
      (new Vector3D()).fromObject(firstPlane.orientation)
  // const firstSize = (new Vector3D()).fromObject(firstPlane.size)

  const lastCenter = (new Vector3D()).fromObject(lastPlane.center)
  const lastOrientation =
      (new Vector3D()).fromObject(lastPlane.orientation)
  // const lastSize = (new Vector3D()).fromObject(lastPlane.size)

  const numItems = lastItemIndex - firstItemIndex

  const positionDelta = new Vector3D()
  positionDelta.fromObject(lastCenter)
  positionDelta.subtract(firstCenter)
  positionDelta.scale(1. / numItems)

  const rotationDelta = new Vector3D()
  rotationDelta.fromObject(lastOrientation)
  rotationDelta.subtract(firstOrientation)
  rotationDelta.scale(1. / numItems)

  // const scaleDelta = new Vector3D()
  // scaleDelta.fromObject(lastSize)
  // scaleDelta.subtract(firstSize)
  // scaleDelta.scale(1. / numItems)

  for (let i = firstItemIndex + 1; i < lastItemIndex; i += 1) {
    if (i in labels) {
      const indexDelta = i - firstItemIndex
      const labelId = labels[i]
      const label = items[i].labels[labelId]

      const newCenter = (new Vector3D()).fromObject(positionDelta)
      newCenter.multiplyScalar(indexDelta)
      newCenter.add((new Vector3D()).fromObject(firstCenter))

      const newOrientation = (new Vector3D()).fromObject(rotationDelta)
      newOrientation.multiplyScalar(indexDelta)
      newOrientation.add((new Vector3D().fromObject(firstOrientation)))

      // const newSize = (new Vector3D()).fromObject(scaleDelta)
      // newSize.multiplyScalar(indexDelta)
      // newSize.add((new Vector3D().fromObject(firstSize)))

      updatedIndices.push(i)
      updatedShapeIds.push([label.shapes[0]])
      updatedShapes.push([{
        center: newCenter.toObject(),
        orientation: newOrientation.toObject()
        // size: newSize.toObject()
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
function linearInterpolateBox3D (
  trackId: number,
  updatedItemIndex: number,
  newShapes: ShapeType[]
) {
  const state = Session.getState()
  const track = state.task.tracks[trackId]
  const items = state.task.items

  const updatedIndices: number[] = []
  const updatedShapeIds: number[][] = []
  const updatedShapes: Array<Array<Partial<Plane3DType>>> = []

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

  const newPlane = newShapes[0] as Plane3DType

  if (lastManualIndex >= 0 && lastLabel) {
    interpolatePlanes(
      items,
      track.labels,
      lastManualIndex,
      updatedItemIndex,
      items[lastManualIndex].shapes[lastLabel.shapes[0]].shape as Plane3DType,
      newPlane,
      updatedIndices,
      updatedShapeIds,
      updatedShapes
    )
  }

  if (nextManualIndex >= 0 && nextLabel) {
    interpolatePlanes(
      items,
      track.labels,
      updatedItemIndex,
      nextManualIndex,
      newPlane,
      items[nextManualIndex].shapes[nextLabel.shapes[0]].shape as Plane3DType,
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
export class LinearInterpolationPlane3DPolicy extends TrackPolicy {
  constructor (track: Track) {
    super(track)
    this._policyType = types.TrackPolicyType.LINEAR_INTERPOLATION_PLANE_3D
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
    Session.dispatch(linearInterpolateBox3D(
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
    label: Label,
    sensors: number[]
  ) {
    label = label as Plane3D
    const plane = label.shapes()[0].toObject()
    const labelObject = makeLabel({
      type: types.LabelTypeName.PLANE_3D,
      category: label.category,
      sensors
    })

    if (Session.tracking) {
      Session.dispatch(addDuplicatedTrack(
        labelObject,
        [types.ShapeTypeName.GRID],
        [plane],
        itemIndex
      ))
    }
  }
}
