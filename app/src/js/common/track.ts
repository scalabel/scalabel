import Label2D from '../drawable/2d/label2d'
import Label3D from '../drawable/3d/label3d'
import { ShapeType, TrackType } from '../functional/types'
import { LinearInterpolationBox2DPolicy } from './track_policies/box2d_linear_interpolation_policy'
import { LinearInterpolationBox3DPolicy } from './track_policies/box3d_linear_interpolation_policy'
import { LinearInterpolationCustomLabel2DPolicy } from './track_policies/custom_label_linear_interpolation_policy'
import { LinearInterpolationPlane3DPolicy } from './track_policies/plane3d_linear_interpolation_policy'
import { LinearInterpolationPolygonPolicy } from './track_policies/polygon_linear_interpolation_policy'
import { TrackPolicy } from './track_policies/track_policy'
import { TrackPolicyType } from './types'

export type Label = Label2D | Label3D

/**
 * Make track policy
 */
export function makeTrackPolicy (track: Track, policyType: string) {
  switch (policyType) {
    case TrackPolicyType.LINEAR_INTERPOLATION_BOX_3D:
      return new LinearInterpolationBox3DPolicy(track)
    case TrackPolicyType.LINEAR_INTERPOLATION_BOX_2D:
      return new LinearInterpolationBox2DPolicy(track)
    case TrackPolicyType.LINEAR_INTERPOLATION_POLYGON:
      return new LinearInterpolationPolygonPolicy(track)
    case TrackPolicyType.LINEAR_INTERPOLATION_PLANE_3D:
      return new LinearInterpolationPlane3DPolicy(track)
    case TrackPolicyType.LINEAR_INTERPOLATION_CUSTOM_2D:
      return new LinearInterpolationCustomLabel2DPolicy(track)
  }
}

/**
 * Object representation of track
 */
export class Track {
  /** map between item indices and label objects */
  // protected _labels: {[itemIndex: number]: Label2D | Label3D}
  /** policy */
  protected _policy?: TrackPolicy
  /** track state */
  protected _track: TrackType | null

  constructor () {
    this._track = null
  }

  /**
   * Run when state is updated
   * @param state
   */
  public updateState (track: TrackType, policy?: TrackPolicy) {
    this._track = track
    if (policy) {
      this._policy = policy
    }
  }

  /**
   * Get track policy
   */
  public get policyType () {
    if (this._policy) {
      return this._policy.policyType
    }
    return undefined
  }

  /**
   * Get policy object
   */
  public get trackPolicy () {
    return this._policy
  }

  /**
   * Get track id
   */
  public get id () {
    if (this._track) {
      return this._track.id
    }
    return -1
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
    if (this._policy) {
      this._policy.onLabelUpdated(itemIndex, newShapes)
    }
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
    if (this._policy) {
      this._policy.onLabelCreated(itemIndex, label, sensors)
    }
  }
}
