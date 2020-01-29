import { CubeType } from '../../../../functional/types'
import { Vector3D } from '../../../../math/vector3d'
import { Track } from '../../track'
import { LinearInterpolationPolicy } from './linear_interpolation_policy'

/**
 * Class for Box 3d track
 */
export class Box3DLinearInterpolationPolicy extends LinearInterpolationPolicy {
  /** Initial center for adding deltas */
  protected _initialCenter: Vector3D
  /** Initial orientation for adding deltas */
  protected _initialOrientation: Vector3D
  /** Initial size for adding deltas */
  protected _initialSize: Vector3D
  /** Delta between centers per index */
  protected _centerDelta: Vector3D
  /** Delta between orientations per index */
  protected _orientationDelta: Vector3D
  /** Delta between dimensions per index */
  protected _sizeDelta: Vector3D

  constructor (track: Track) {
    super(track)
    this._initialCenter = new Vector3D()
    this._initialOrientation = new Vector3D()
    this._initialSize = new Vector3D()
    this._centerDelta = new Vector3D()
    this._orientationDelta = new Vector3D()
    this._sizeDelta = new Vector3D()
  }

  /** Calculate deltas */
  protected calculateDeltas (start: number, end: number) {
    const numItems = end - start
    const firstCube = this._track.getShapes(start)[0].shape as CubeType
    const lastCube = this._track.getShapes(end)[0].shape as CubeType
    this._initialCenter = (new Vector3D()).fromState(firstCube.center)
    this._initialOrientation =
        (new Vector3D()).fromState(firstCube.orientation)
    this._initialSize = (new Vector3D()).fromState(firstCube.size)

    this._centerDelta.fromState(lastCube.center)
    this._centerDelta.subtract(this._initialCenter)
    this._centerDelta.scale(1. / numItems)

    this._orientationDelta.fromState(lastCube.orientation)
    this._orientationDelta.subtract(this._initialOrientation)
    this._orientationDelta.scale(1. / numItems)

    this._sizeDelta.fromState(lastCube.size)
    this._sizeDelta.subtract(this._initialSize)
    this._sizeDelta.scale(1. / numItems)
  }

  /** Linear interpolate shapes at index */
  protected interpolateIndex (start: number, index: number): void {
    const indexDelta = index - start

    const newCenter = (new Vector3D()).fromState(this._centerDelta)
    newCenter.multiplyScalar(indexDelta)
    newCenter.add((new Vector3D()).fromState(this._initialCenter))

    const newOrientation = (new Vector3D()).fromState(this._orientationDelta)
    newOrientation.multiplyScalar(indexDelta)
    newOrientation.add((new Vector3D().fromState(this._initialOrientation)))

    const newSize = (new Vector3D()).fromState(this._sizeDelta)
    newSize.multiplyScalar(indexDelta)
    newSize.add((new Vector3D().fromState(this._initialSize)))

    const shapes = this._track.getShapes(index)
    const newShapes = [
      {
        ...shapes[0],
        shape: {
          center: newCenter.toState(),
          orientation: newOrientation.toState(),
          size: newSize.toState()
        }
      }
    ]

    this._track.setShapes(index, newShapes)
  }
}
