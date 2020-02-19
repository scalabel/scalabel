import * as THREE from 'three'
import { BLUE, GREEN, RED } from '../common'
import Label3D from '../label3d'
import { Controller } from './controller'
import { ScaleAxis } from './scale_axis'

const SCALE_AMOUNT = 0.05

/**
 * perform scaling ops
 */
export class ScaleControl extends Controller {
  constructor (labels: Label3D[], bounds: THREE.Box3) {
    super(labels, bounds)
    this._controlUnits.push(
      new ScaleAxis(
        'x',
        false,
        RED
      )
    )
    this._controlUnits.push(
      new ScaleAxis(
        'y',
        false,
        GREEN
      )
    )
    this._controlUnits.push(
      new ScaleAxis(
        'z',
        false,
        BLUE
      )
    )
    this._controlUnits.push(
      new ScaleAxis(
        'x',
        true,
        RED
      )
    )
    this._controlUnits.push(
      new ScaleAxis(
        'y',
        true,
        GREEN
      )
    )
    this._controlUnits.push(
      new ScaleAxis(
        'z',
        true,
        BLUE
      )
    )
    for (const unit of this._controlUnits) {
      this.add(unit)
    }
    this._local = true
  }

  /** Apply pre-determined transformation amount based on camera direction */
  public transformDiscrete (direction: THREE.Vector3): void {
    const center = new THREE.Vector3()
    this._bounds.getCenter(center)
    const dimensions = new THREE.Vector3()
    dimensions.copy(this._bounds.max)
    dimensions.sub(this._bounds.min)
    for (const label of this._labels) {
      const inverseQuaternion = (new THREE.Quaternion()).copy(label.orientation)
      const localDirection = (new THREE.Vector3()).copy(direction)
        .applyQuaternion(inverseQuaternion).toArray()

      let maxAxis = 0
      for (let i = 0; i < localDirection.length; i++) {
        if (localDirection[i] > localDirection[maxAxis]) {
          maxAxis = i
        }
      }

      const scaleArr = [1, 1, 1]
      scaleArr[maxAxis] += SCALE_AMOUNT

      const anchor = new THREE.Vector3()
      anchor.copy(direction)
      anchor.multiply(dimensions)
      anchor.divideScalar(-2.0)
      anchor.add(center)
      label.scale((new THREE.Vector3()).fromArray(scaleArr), anchor, true)
    }
  }
}
