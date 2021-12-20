import * as THREE from "three"

/**
 * ThreeJS class for rendering 3D ground plane
 */
export class GroundPlane3D {
  private readonly _plane: THREE.Plane
  private readonly _shape: THREE.PlaneHelper

  /** Constructor
   *
   * @param points
   */
  constructor(points: THREE.Vector3[]) {
    this._plane = new THREE.Plane().setFromCoplanarPoints(
      points[0],
      points[1],
      points[2]
    )
    this._shape = new THREE.PlaneHelper(this._plane, 50)
  }

  /**
   * Add to scene for rendering
   *
   * @param scene
   */
  public render(scene: THREE.Scene): void {
    scene.add(this._shape)
  }

  /** Get plane */
  public get plane(): THREE.Plane {
    return this._plane
  }
}
