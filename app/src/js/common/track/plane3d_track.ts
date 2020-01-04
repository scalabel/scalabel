
import { Plane3DType } from '../../functional/types'
import { Vector3D } from '../../math/vector3d'
import { Track } from './track'

/**
 * Class for linear interpolating box 3d's
 */
export class Plane3DTrack extends Track {
  constructor () {
    super()
  }

  /** Linear interpolate shapes at index */
  protected linearInterpolateIndex (
    itemIndex: number,
    previousIndex: number,
    nextIndex: number
  ): void {
    const firstCube = this._shapes[previousIndex][0].shape as Plane3DType
    const lastCube = this._shapes[nextIndex][0].shape as Plane3DType
    const firstCenter = (new Vector3D()).fromState(firstCube.center)
    const firstOrientation =
        (new Vector3D()).fromState(firstCube.orientation)

    const lastCenter = (new Vector3D()).fromState(lastCube.center)
    const lastOrientation =
        (new Vector3D()).fromState(lastCube.orientation)

    const numItems = nextIndex - previousIndex

    const positionDelta = new Vector3D()
    positionDelta.fromState(lastCenter)
    positionDelta.subtract(firstCenter)
    positionDelta.scale(1. / numItems)

    const rotationDelta = new Vector3D()
    rotationDelta.fromState(lastOrientation)
    rotationDelta.subtract(firstOrientation)
    rotationDelta.scale(1. / numItems)

    const indexDelta = itemIndex - previousIndex

    const newCenter = (new Vector3D()).fromState(positionDelta)
    newCenter.multiplyScalar(indexDelta)
    newCenter.add((new Vector3D()).fromState(firstCenter))

    const newOrientation = (new Vector3D()).fromState(rotationDelta)
    newOrientation.multiplyScalar(indexDelta)
    newOrientation.add((new Vector3D().fromState(firstOrientation)))

    const updatedCube = this._shapes[itemIndex][0].shape as Plane3DType
    updatedCube.center = newCenter.toState()
    updatedCube.orientation = newOrientation.toState()
  }
}
