import * as THREE from 'three'
import { BLUE, GREEN, RED } from '../common'
import { Controller } from './controller'
import { RotationRing } from './rotation_ring'

/**
 * perform rotation ops
 */
export class RotationControl extends Controller {
  constructor () {
    super()
    this._controlUnits.push(
      new RotationRing(
        new THREE.Vector3(1, 0, 0),
        RED
      )
    )
    this._controlUnits.push(
      new RotationRing(
        new THREE.Vector3(0, 1, 0),
        GREEN
      )
    )
    this._controlUnits.push(
      new RotationRing(
        new THREE.Vector3(0, 0, 1),
        BLUE
      )
    )
    for (const unit of this._controlUnits) {
      this.add(unit)
    }
  }
}
