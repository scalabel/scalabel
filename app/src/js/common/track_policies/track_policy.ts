import { ShapeType } from '../../functional/types'
import { Label, Track } from '../track'

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
    label: Label
  ): void

  /**
   * Get track policy
   */
  public get policyType () {
    return this._policyType
  }
}
