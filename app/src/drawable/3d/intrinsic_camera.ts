import * as THREE from "three"

import { CameraIntrinsicsType } from "../../types/state"

/** Camera for rendering with calibrated intrinsics */
export class IntrinsicCamera extends THREE.Camera {
  /** inverse of projection matrix */
  public projectionMatrixInverse: THREE.Matrix4
  /** Set to true for raycaster to treat as perspective camera for projecting */
  public isPerspectiveCamera: boolean
  /** width of the image */
  public width: number
  /** height of the image */
  public height: number
  /** Minimum distance */
  public near: number
  /** Maximum distance */
  public far: number
  /** intrinsics */
  public intrinsics?: CameraIntrinsicsType

  /**
   * Constructor
   *
   * @param width
   * @param height
   * @param near
   * @param far
   * @param intrinsics
   */
  constructor(
    width: number = 0,
    height: number = 0,
    near: number = 0.1,
    far: number = 1000,
    intrinsics?: CameraIntrinsicsType
  ) {
    super()
    this.width = width
    this.height = height
    this.near = near
    this.far = far
    this.intrinsics = intrinsics
    this.isPerspectiveCamera = true
    this.projectionMatrix = new THREE.Matrix4()
    this.projectionMatrixInverse = new THREE.Matrix4()
    this.calculateProjectionMatrix()
  }

  /** Use parameters to calculate internal projection matrix */
  public calculateProjectionMatrix(): void {
    if (this.intrinsics !== undefined) {
      this.projectionMatrix.set(
        (2 * this.intrinsics.focalLength.x) / this.width,
        0,
        -((2 * this.intrinsics.focalCenter.x) / this.width) + 1,
        0,
        0,
        (2 * this.intrinsics.focalLength.y) / this.height,
        (2 * this.intrinsics.focalCenter.y) / this.height - 1,
        0,
        0,
        0,
        (this.near + this.far) / (this.near - this.far),
        (2 * this.far * this.near) / (this.near - this.far),
        0,
        0,
        -1,
        0
      )
      this.projectionMatrixInverse.getInverse(this.projectionMatrix)
    }
  }
}
