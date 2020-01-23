import { RectType } from '../../../../functional/types'
import { Vector2D } from '../../../../math/vector2d'
import { Track } from '../../track'
import { LinearInterpolationPolicy } from './linear_interpolation_policy'

/**
 * Class for linear interpolating box 3d's
 */
export class Box2DLinearInterpolationPolicy extends LinearInterpolationPolicy {
  /** Initial center for adding deltas */
  protected _initialCenter: Vector2D
  /** Initial dimension for adding deltas */
  protected _initialDimension: Vector2D
  /** Delta between centers per index */
  protected _centerDelta: Vector2D
  /** Delta between dimensions per index */
  protected _dimensionDelta: Vector2D

  constructor (track: Track) {
    super(track)
    this._initialCenter = new Vector2D()
    this._initialDimension = new Vector2D()
    this._centerDelta = new Vector2D()
    this._dimensionDelta = new Vector2D()
  }

  /** Calculate deltas */
  protected calculateDeltas (start: number, end: number) {
    const numItems = end - start
    const firstRect = this._track.getShapes(start)[0].shape as RectType
    const lastRect = this._track.getShapes(end)[0].shape as RectType
    this._initialCenter = new Vector2D(
      (firstRect.x1 + firstRect.x2) / 2.,
      (firstRect.y1 + firstRect.y2) / 2.
    )
    this._initialDimension = new Vector2D(
      Math.abs(firstRect.x1 - firstRect.x2),
      Math.abs(firstRect.y1 - firstRect.y2)
    )
    const lastCenter = new Vector2D(
      (lastRect.x1 + lastRect.x2) / 2.,
      (lastRect.y1 + lastRect.y2) / 2.
    )
    const lastDimension = new Vector2D(
      Math.abs(lastRect.x1 - lastRect.x2),
      Math.abs(lastRect.y1 - lastRect.y2)
    )

    this._centerDelta = lastCenter
    this._centerDelta.subtract(this._initialCenter)
    this._centerDelta.scale(1. / numItems)

    this._dimensionDelta = lastDimension
    this._dimensionDelta.subtract(this._initialDimension)
    this._dimensionDelta.scale(1. / numItems)
  }

  /** Linear interpolate shapes at index */
  protected interpolateIndex (start: number, index: number): void {
    const offset = index - start
    const newCenter = this._centerDelta.clone()
    newCenter.scale(offset)
    newCenter.add(this._initialCenter)

    const newDimension = this._dimensionDelta.clone()
    newDimension.scale(offset)
    newDimension.add(this._initialDimension)

    const shapes = this._track.getShapes(index)
    const newShapes = [
      {
        ...shapes[0],
        shape: {
          x1: newCenter.x - newDimension.x / 2.,
          x2: newCenter.x + newDimension.x / 2.,
          y1: newCenter.y - newDimension.y / 2.,
          y2: newCenter.y + newDimension.y / 2.
        }
      }
    ]

    this._track.setShapes(index, newShapes)
  }
}
