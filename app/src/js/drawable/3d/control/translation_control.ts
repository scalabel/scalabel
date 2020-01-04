import * as THREE from 'three'
import { BLUE, GREEN, RED } from '../common'
import Label3D from '../label3d'
import { Controller } from './controller'
import { TranslationAxis } from './translation_axis'
import { TranslationPlane } from './translation_plane'

/**
 * Groups TranslationAxis's and TranslationPlanes to perform translation ops
 */
export class TranslationControl extends Controller {
  constructor (labels: Label3D[], bounds: THREE.Box3) {
    super(labels, bounds)
    this._controlUnits = []
    this._controlUnits.push(
      new TranslationAxis(new THREE.Vector3(1, 0, 0), RED)
    )
    this._controlUnits.push(
      new TranslationAxis(new THREE.Vector3(0, 1, 0), GREEN)
    )
    this._controlUnits.push(
      new TranslationAxis(new THREE.Vector3(0, 0, 1), BLUE)
    )
    this._controlUnits.push(
      new TranslationAxis(new THREE.Vector3(-1, 0, 0), RED)
    )
    this._controlUnits.push(
      new TranslationAxis(new THREE.Vector3(0, -1, 0), GREEN)
    )
    this._controlUnits.push(
      new TranslationAxis(new THREE.Vector3(0, 0, -1), BLUE)
    )
    this._controlUnits.push(
      new TranslationPlane(
        new THREE.Vector3(1, 0, 0),
        RED
      )
    )
    this._controlUnits.push(
      new TranslationPlane(
        new THREE.Vector3(0, 1, 0),
        GREEN
      )
    )
    this._controlUnits.push(
      new TranslationPlane(
        new THREE.Vector3(0, 0, 1),
        BLUE
      )
    )
    for (const unit of this._controlUnits) {
      this.add(unit)
    }
  }
}
