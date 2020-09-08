import * as THREE from "three"

import { Key } from "../../../const/common"
import { BLUE, GREEN, RED } from "../common"
import Label3D from "../label3d"
import { Controller } from "./controller"
import { TranslationAxis } from "./translation_axis"
import { TranslationPlane } from "./translation_plane"

const MOVE_AMOUNT = 0.03

/**
 * Groups TranslationAxis's and TranslationPlanes to perform translation ops
 */
export class TranslationControl extends Controller {
  /**
   * Constructor
   *
   * @param labels
   * @param bounds
   */
  constructor(labels: Label3D[], bounds: THREE.Box3) {
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
      new TranslationPlane(new THREE.Vector3(1, 0, 0), RED)
    )
    this._controlUnits.push(
      new TranslationPlane(new THREE.Vector3(0, 1, 0), GREEN)
    )
    this._controlUnits.push(
      new TranslationPlane(new THREE.Vector3(0, 0, 1), BLUE)
    )
    for (const unit of this._controlUnits) {
      this.add(unit)
    }
  }

  /**
   * Apply pre-determined transformation amount based on camera direction
   *
   * @param key
   * @param camera
   */
  public keyDown(key: string, camera: THREE.Camera): void {
    super.keyDown(key, camera)
    const direction = new THREE.Vector3()
    const up = new THREE.Vector3(0, 1, 0)
    up.applyQuaternion(camera.quaternion)
    const forward = camera.getWorldDirection(new THREE.Vector3())
    const left = new THREE.Vector3().crossVectors(up, forward).normalize()
    switch (key) {
      case Key.I_UP:
      case Key.I_LOW:
        direction.copy(up)
        break
      case Key.K_UP:
      case Key.K_LOW:
        direction.copy(up)
        direction.negate()
        break
      case Key.J_UP:
      case Key.J_LOW:
        direction.copy(left)
        break
      case Key.L_UP:
      case Key.L_LOW:
        direction.copy(left)
        direction.negate()
        break
    }
    const delta = new THREE.Vector3().copy(direction).normalize()
    delta.multiplyScalar(MOVE_AMOUNT)
    for (const label of this._labels) {
      label.translate(delta)
    }
  }
}
