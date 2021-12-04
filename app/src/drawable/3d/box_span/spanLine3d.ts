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
   * align point to unit normal
   *
   * @param point
   */
  public alignPointToNormal(point: Vector3D): Vector3D {
    const unitNormal = this.calculateUnitNormal()
    const footPerpendicular = this.footPerpendicular(point, unitNormal)
    return footPerpendicular
  }

  /**
   * calculate unit normal of line
   *
   * @param point
   */
  private calculateUnitNormal(): Vector3D {
    const v1 = new Vector3D(this._p1.x, this._p1.y, this._p1.z)
    const v2 = new Vector3D(this._p2.x, this._p2.y, this._p2.z)
    const v12 = v2.clone().subtract(v1)
    return new Vector3D(-v12.y, v12.x, 0).normalize()
  }

  /**
   * calculate perpendicular distance between point and normal of line
   *
   * @param point
   * @param unitNormal
   */
  private perpendicularDist(point: Vector3D, unitNormal: Vector3D): number {
    const v2 = new Vector3D(this._p2.x, this._p2.y, this._p2.z)
    const vTmp = point.clone().subtract(v2)
    const perpendicularDist = vTmp.clone().cross(unitNormal).magnitude()
    return perpendicularDist
  }

  /**
   * calculate foot of perpendicular between point and normal of line
   *
   * @param point
   * @param unitNormal
   */
  private footPerpendicular(point: Vector3D, unitNormal: Vector3D): Vector3D {
    const v2 = new Vector3D(this._p2.x, this._p2.y, this._p2.z)
    const perpendicularDist = this.perpendicularDist(point, unitNormal)
    const dTmp = point.clone().distanceTo(v2)
    const dist = Math.sqrt(dTmp * dTmp - perpendicularDist * perpendicularDist)
    const normalDist = unitNormal.clone()
    normalDist.multiplyScalar(dist)
    let footPerpendicular = v2.clone().add(normalDist)
    const dPoint = point.distanceTo(footPerpendicular)
    const errorDeg = 0.01
    if (
      Math.abs(dPoint - this.perpendicularDist(point, unitNormal)) > errorDeg
    ) {
      normalDist.multiplyScalar(-1)
      footPerpendicular = v2.clone().add(normalDist)
    }
    return footPerpendicular
  }
}
