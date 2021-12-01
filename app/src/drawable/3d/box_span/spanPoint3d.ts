import * as THREE from "three"

import { Vector3D } from "../../../math/vector3d"

/**
 * ThreeJS class for rendering 3D point
 */
export class SpanPoint3D {
  private _x: number
  private _y: number
  private _z: number
  private readonly _color: string
  private readonly _radius: number

  /**
   * Constructor
   *
   * @param point
   */
  constructor(point: Vector3D) {
    this._x = point.x
    this._y = point.y
    this._z = point.z
    this._color = "#ff0000"
    this._radius = 0.1
  }

  /**
   * Add to scene for rendering
   *
   * @param scene
   * @param camera
   * @param pointCloud
   */
  public render(scene: THREE.Scene): void {
    // (radius, widthSegments, heightSegments)
    const geometry = new THREE.SphereGeometry(this._radius, 3, 2)
    const material = new THREE.MeshBasicMaterial({ color: this._color })
    const point = new THREE.Mesh(geometry, material)
    // console.log(this.x, this.y, this.z)
    point.position.set(this.x, this.y, this.z)
    scene.add(point)
  }

  /** get x */
  public get x(): number {
    return this._x
  }

  /** set x */
  public set x(x: number) {
    this._x = x
  }

  /** get y */
  public get y(): number {
    return this._y
  }

  /** set y */
  public set y(y: number) {
    this._y = y
  }

  /** get z */
  public get z(): number {
    return this._z
  }

  /** set z */
  public set z(z: number) {
    this._z = z
  }
}
