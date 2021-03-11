import * as THREE from "three"

import { Key } from "../../../const/common"
import { BLUE, GREEN, RED } from "../common"
import Label3D from "../label3d"
import { Controller } from "./controller"
import { ScaleAxis } from "./scale_axis"

const SCALE_AMOUNT = 0.01

/**
 * perform scaling ops
 */
export class ScaleControl extends Controller {
  /**
   * Constructor
   *
   * @param labels
   * @param bounds
   */
  constructor(labels: Label3D[], bounds: THREE.Box3) {
    super(labels, bounds)
    this._controlUnits.push(new ScaleAxis("x", false, RED))
    this._controlUnits.push(new ScaleAxis("y", false, GREEN))
    this._controlUnits.push(new ScaleAxis("z", false, BLUE))
    this._controlUnits.push(new ScaleAxis("x", true, RED))
    this._controlUnits.push(new ScaleAxis("y", true, GREEN))
    this._controlUnits.push(new ScaleAxis("z", true, BLUE))
    for (const unit of this._controlUnits) {
      this.add(unit)
    }
    this._local = true
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

    let scaleDelta = SCALE_AMOUNT
    if (this._keyDownMap[Key.SHIFT]) {
      scaleDelta *= -1
    }

    const center = new THREE.Vector3()
    this._bounds.getCenter(center)
    const dimensions = new THREE.Vector3()
    dimensions.copy(this._bounds.max)
    dimensions.sub(this._bounds.min)
    for (const label of this._labels) {
      const inverseQuaternion = new THREE.Quaternion().copy(label.orientation)
      const localDirection = new THREE.Vector3()
        .copy(direction)
        .applyQuaternion(inverseQuaternion)
        .toArray()

      let maxAxis = 0
      for (let i = 0; i < localDirection.length; i++) {
        if (Math.abs(localDirection[i]) > Math.abs(localDirection[maxAxis])) {
          maxAxis = i
        }
      }

      const scaleArr = [1, 1, 1]
      scaleArr[maxAxis] += scaleDelta

      const scaleFactor = new THREE.Vector3().fromArray(scaleArr)

      const anchorDirection = [0, 0, 0]
      anchorDirection[maxAxis] = Math.sign(localDirection[maxAxis])
      if (anchorDirection[maxAxis] === 0) {
        anchorDirection[maxAxis] = 1
      }

      const anchor = new THREE.Vector3().fromArray(anchorDirection)
      anchor.multiply(dimensions)
      anchor.divideScalar(-2.0)
      anchor.add(center)

      anchor.sub(label.center)
      anchor.applyQuaternion(label.orientation)
      anchor.add(label.center)

      label.scale(scaleFactor, anchor, true)
    }
  }
}
