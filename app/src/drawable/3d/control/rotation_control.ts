import * as THREE from "three"

import { Key } from "../../../const/common"
import { BLUE, GREEN, RED } from "../common"
import Label3D from "../label3d"
import { Controller } from "./controller"
import { RotationRing } from "./rotation_ring"

const ROTATION_AMOUNT = 0.025

/**
 * perform rotation ops
 */
export class RotationControl extends Controller {
  /**
   * Constructor
   *
   * @param labels
   * @param bounds
   */
  constructor(labels: Label3D[], bounds: THREE.Box3) {
    super(labels, bounds)
    this._controlUnits.push(new RotationRing(new THREE.Vector3(1, 0, 0), RED))
    this._controlUnits.push(new RotationRing(new THREE.Vector3(0, 1, 0), GREEN))
    this._controlUnits.push(new RotationRing(new THREE.Vector3(0, 0, 1), BLUE))
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
    let rotationAmount = ROTATION_AMOUNT
    if (key === Key.J_LOW || key === Key.J_UP) {
      rotationAmount *= -1
    }
    switch (key) {
      case Key.J_LOW:
      case Key.J_UP:
      case Key.L_LOW:
      case Key.L_UP: {
        const cameraDirection = camera.getWorldDirection(new THREE.Vector3())
        const quaternion = new THREE.Quaternion()
        quaternion.setFromAxisAngle(cameraDirection, rotationAmount)
        for (const label of this._labels) {
          label.rotate(quaternion)
        }
        break
      }
    }
  }
}
