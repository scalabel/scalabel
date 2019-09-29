import * as THREE from 'three'
import { Controller } from './controller'
import { TranslationAxis } from './translation_axis'
import { TranslationPlane } from './translation_plane'

/**
 * Groups TranslationAxis's and TranslationPlanes to perform translation ops
 */
export class TranslationControl extends Controller {
  constructor (camera: THREE.Camera) {
    super(camera)
    this._controlUnits = []
    this._controlUnits.push(
      new TranslationAxis(new THREE.Vector3(1, 0, 0), 0xff0000)
    )
    this._controlUnits.push(
      new TranslationAxis(new THREE.Vector3(0, 1, 0), 0x00ff00)
    )
    this._controlUnits.push(
      new TranslationAxis(new THREE.Vector3(0, 0, 1), 0x0000ff)
    )
    this._controlUnits.push(
      new TranslationAxis(new THREE.Vector3(-1, 0, 0), 0xff0000)
    )
    this._controlUnits.push(
      new TranslationAxis(new THREE.Vector3(0, -1, 0), 0x00ff00)
    )
    this._controlUnits.push(
      new TranslationAxis(new THREE.Vector3(0, 0, -1), 0x0000ff)
    )
    this._controlUnits.push(
      new TranslationPlane(
        new THREE.Vector3(1, 0, 0),
        0xff0000
      )
    )
    this._controlUnits.push(
      new TranslationPlane(
        new THREE.Vector3(0, 1, 0),
        0x00ff00
      )
    )
    this._controlUnits.push(
      new TranslationPlane(
        new THREE.Vector3(0, 0, 1),
        0x0000ff
      )
    )
    for (const unit of this._controlUnits) {
      this.add(unit)
    }
  }
}
