import * as THREE from "three"

import { SpanPoint3D } from "./spanPoint3d"
import { Vector3D } from "../../../math/vector3d"

/**
 * ThreeJS class for rendering 3D line
 */
export class SpanLine3D {
  private readonly _p1: SpanPoint3D
  private readonly _p2: SpanPoint3D
  private readonly _color: number
  private readonly _lineWidth: number

  /**
   * Constructor
   *
   * @param p1
   * @param p2
   */
  constructor(p1: SpanPoint3D, p2: SpanPoint3D) {
    this._p1 = p1
    this._p2 = p2
    this._color = 0x00ff00
    this._lineWidth = 0.1
  }

  /**
   * Add to scene for rendering
   *
   * @param scene
   */
  public render(scene: THREE.Scene): void {
    // generate line formed by two points and add to scene
    const points = []
    points.push(new THREE.Vector3(this._p1.x, this._p1.y, this._p1.z))
    points.push(new THREE.Vector3(this._p2.x, this._p2.y, this._p2.z))
    const geometry = new THREE.BufferGeometry().setFromPoints(points)
    const material = new THREE.LineBasicMaterial({
      color: this._color,
      linewidth: this._lineWidth
    })
    const line = new THREE.Line(geometry, material)
    scene.add(line)
  }

  /**
   * calculate unit normal of line
   *
   * @param point
   */
  public calculateNormal(point: Vector3D): Vector3D {
    const v1 = new Vector3D(this._p1.x, this._p1.y, this._p1.z)
    const v2 = new Vector3D(this._p2.x, this._p2.y, this._p2.z)
    const v12 = v2.clone().subtract(v1)
    const vDir = point.clone().subtract(v2)
    if (vDir.x + vDir.y >= 0) {
      return new Vector3D(-v12.y, v12.x, 0).unitVector()
    } else {
      return new Vector3D(v12.y, -v12.x, 0).unitVector()
    }
  }

  /** align point to unit normal
   *
   * @param point
   */
  public alignPointToNormal(point: Vector3D): Vector3D {
    const v2 = new Vector3D(this._p2.x, this._p2.y, this._p2.z)
    const normal = this.calculateNormal(point)
    const dist = point.clone().distanceTo(v2)
    normal.multiplyScalar(dist)
    const v3 = v2.clone().add(normal)
    return v3
  }
}
