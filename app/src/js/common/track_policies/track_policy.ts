import { ShapeType } from '../../functional/types'
import { Label, Track } from '../track'
import { TrackPolicyType } from '../types'

/** Convert policy type name to enum */
export function policyFromString (
  typeName: string
): TrackPolicyType {
  switch (typeName) {
    case TrackPolicyType.LINEAR_INTERPOLATION_BOX_2D:
      return TrackPolicyType.LINEAR_INTERPOLATION_BOX_2D
    case TrackPolicyType.LINEAR_INTERPOLATION_BOX_3D:
      return TrackPolicyType.LINEAR_INTERPOLATION_BOX_3D
    case TrackPolicyType.LINEAR_INTERPOLATION_POLYGON:
      return TrackPolicyType.LINEAR_INTERPOLATION_POLYGON
  }

  throw new Error(`Unrecognized policy type: ${typeName}`)
}

/**
 * Track policy for updating track
 */
export abstract class TrackPolicy {
  /** policy type */
  protected _policyType?: string
  /** track object */
  protected _track: Track

  constructor (track: Track) {
    this._track = track
  }

  /**
   * Callback for updating track
   * @param itemIndex
   * @param newShapes
   */
  public abstract onLabelUpdated (
    itemIndex: number, newShapes: ShapeType[]
  ): void

  /**
   * Callback for label creation
   * @param itemIndex
   * @param label
   * @param shapes
   * @param shapeTypes
   */
  public abstract onLabelCreated (
    itemIndex: number,
    label: Label,
    sensors: number[]
  ): void

  /**
   * Get track policy
   */
  public get policyType () {
    return this._policyType
  }
}
