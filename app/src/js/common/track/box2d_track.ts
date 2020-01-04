import { RectType } from '../../functional/types'
import { Vector2D } from '../../math/vector2d'
import { Track } from './track'

/**
 * Class for linear interpolating box 3d's
 */
export class Box2DTrack extends Track {
  constructor () {
    super()
  }

  /** Linear interpolate shapes at index */
  protected linearInterpolateIndex (
    itemIndex: number,
    previousIndex: number,
    nextIndex: number
  ): void {
    const firstRect = this._shapes[previousIndex][0].shape as RectType
    const lastRect = this._shapes[nextIndex][0].shape as RectType
    const firstCenter = new Vector2D(
      (firstRect.x1 + firstRect.x2) / 2.,
      (firstRect.y1 + firstRect.y2) / 2.
    )
    const firstDimension = new Vector2D(
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

    const numItems = nextIndex - previousIndex

    const centerDelta = lastCenter.clone()
    centerDelta.subtract(firstCenter)
    centerDelta.scale(1. / numItems)

    const dimensionDelta = lastDimension.clone()
    dimensionDelta.subtract(firstDimension)
    dimensionDelta.scale(1. / numItems)

    const indexDelta = itemIndex - previousIndex

    const newCenter = centerDelta.clone()
    newCenter.scale(indexDelta)
    newCenter.add(firstCenter)

    const newDimension = dimensionDelta.clone()
    newDimension.scale(indexDelta)
    newDimension.add(firstDimension)

    this._shapes[itemIndex][0].shape = {
      x1: newCenter.x - newDimension.x / 2.,
      x2: newCenter.x + newDimension.x / 2.,
      y1: newCenter.y - newDimension.y / 2.,
      y2: newCenter.y + newDimension.y / 2.
    }
  }
}
