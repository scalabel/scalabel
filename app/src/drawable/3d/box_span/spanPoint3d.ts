import * as THREE from "three"

import { Vector3D } from "../../../math/vector3d"

/**
 * ThreeJS class for rendering 3D point
 */
export class SpanPoint3D {
  private readonly _point: THREE.Mesh
  /** x coordinate in 3D space */
  private _x: number
  /** y coordinate in 3D space */
  private _y: number
  /** z coordinate in 3D space */
  private _z: number

  /**
   * Constructor
   *
   * @param point
   */
  constructor(point: Vector3D) {
    this._x = point.x
    this._y = point.y
    this._z = point.z
    const geometry = new THREE.SphereGeometry(0.05)
    const material = new THREE.MeshBasicMaterial({ color: "#ff0000" })
    this._point = new THREE.Mesh(geometry, material)
  }

  /**
   * Update point coordinates
   *
   * @param point
   */
  public update(point: Vector3D): void {
    this._x = point.x
    this._y = point.y
    this._z = point.z
  }

  /**
   * Add to scene for rendering
   *
   * @param scene
   */
  public render(scene: THREE.Scene): void {
    this._point.position.set(this.x, this.y, this.z)
    if (!scene.children.includes(this._point)) {
      scene.add(this._point)
    }
  }

  /** get x */
  public get x(): number {
    return this._x
  }

  /** get y */
  public get y(): number {
    return this._y
  }

  /** get z */
  public get z(): number {
    return this._z
  }

  /** return point coordinates as Vector3D */
  public toVector3D(): Vector3D {
    return new Vector3D(this.x, this.y, this.z)
  }
}
