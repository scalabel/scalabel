import * as THREE from "three"

import { SpanPoint3D } from "./spanPoint3d"

/**
 * ThreeJS class for rendering 3D line
 */
export class SpanLine3D {
  private _p1: SpanPoint3D
  private _p2: SpanPoint3D
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
    this._lineWidth = 0.05
  }

  /**
   * Add to scene for rendering
   *
   * @param scene
   */
  public render(scene: THREE.Scene): void {
    // TODO: generate line formed by two points and add to scene
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
   * Update line vertices
   *
   * @param p1
   * @param p2
   */
  public updateState(p1: SpanPoint3D, p2: SpanPoint3D): void {
    this._p1 = p1
    this._p2 = p2
  }
}
