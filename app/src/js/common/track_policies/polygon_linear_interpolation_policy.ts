import { addDuplicatedTrack } from '../../action/track'
import { Polygon2D } from '../../drawable/2d/polygon2d'
import { makeLabel } from '../../functional/states'
import Session from '../session'
import { Label, Track } from '../track'
import * as types from '../types'
import { TrackPolicy } from './track_policy'
/**
 * Class for linear interpolating polygon's
 */
export class LinearInterpolationPolygonPolicy extends TrackPolicy {
  constructor (track: Track) {
    super(track)
    this._policyType = types.TrackPolicyType.LINEAR_INTERPOLATION_POLYGON
  }

  /**
   * Callback for when a label in the track is updated
   */
  public onLabelUpdated () {
    return
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
    const polygon = (label as Polygon2D).shapeObjects()[2]
    const labelObject = makeLabel({
      type: types.LabelTypeName.POLYGON_2D,
      category: label.category,
      sensors
    })

    const state = Session.getState()
    if (state.task.config.tracking) {
      Session.dispatch(addDuplicatedTrack(
        labelObject,
        [types.ShapeTypeName.POLYGON_2D],
        polygon,
        itemIndex
      ))
    }
  }
}
