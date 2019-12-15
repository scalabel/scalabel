import * as THREE from 'three'
import { CameraIntrinsicsType } from '../../functional/types'

/** Camera for rendering with calibrated intrinsics */
export class IntrinsicCamera extends THREE.Camera {
  /** inverse of projection matrix */
  public projectionMatrixInverse: THREE.Matrix4
  /** Set to true for raycaster to treat as perspective camera for projecting */
  public isPerspectiveCamera: boolean

  constructor (
    intrinsics: CameraIntrinsicsType,
    width: number,
    height: number,
    near: number = 0.1,
    far: number = 1000
  ) {
    super()
    this.projectionMatrix = new THREE.Matrix4()
    this.projectionMatrix.set(
      2 * intrinsics.focalLength.x / width, 0,
        -(2 * intrinsics.focalCenter.x / width) + 1, 0,
      0, 2 * intrinsics.focalLength.y / height,
        (2 * intrinsics.focalCenter.y / height) - 1, 0,
      0, 0, (near + far) / (near - far), 2 * far * near / (near - far),
      0, 0, -1, 0
    )
    this.projectionMatrixInverse = new THREE.Matrix4()
    this.projectionMatrixInverse.getInverse(this.projectionMatrix)
    this.isPerspectiveCamera = true
  }
}
