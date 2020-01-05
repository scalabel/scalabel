import { addDuplicatedTrack } from '../../action/track'
import { CustomLabel2D } from '../../drawable/2d/custom_label'
import { makeLabel } from '../../functional/states'
import Session from '../session'
import { Label, Track } from '../track'
import * as types from '../types'
import { TrackPolicy } from './track_policy'
/**
 * Class for linear interpolating polygon's
 */
export class LinearInterpolationCustomLabel2DPolicy extends TrackPolicy {
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
    const [,shapeTypes, shapeStates] = (label as CustomLabel2D).shapeObjects()
    const labelObject = makeLabel(
      { ...label.label, sensors }
    )

    const state = Session.getState()
    if (state.task.config.tracking) {
      Session.dispatch(addDuplicatedTrack(
        labelObject,
        shapeTypes,
        shapeStates,
        itemIndex
      ))
    }
  }
}
