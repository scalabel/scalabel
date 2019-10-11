import * as THREE from 'three'
import { Controller } from './controller'
import { RotationRing } from './rotation_ring'

/**
 * Groups TranslationAxis's and TranslationPlanes to perform translation ops
 */
export class RotationControl extends Controller {
  constructor () {
    super()
    this._controlUnits.push(
      new RotationRing(
        new THREE.Vector3(1, 0, 0),
        0xff0000
      )
    )
    this._controlUnits.push(
      new RotationRing(
        new THREE.Vector3(0, 1, 0),
        0x00ff00
      )
    )
    this._controlUnits.push(
      new RotationRing(
        new THREE.Vector3(0, 0, 1),
        0x0000ff
      )
    )
    for (const unit of this._controlUnits) {
      this.add(unit)
    }
  }
}
