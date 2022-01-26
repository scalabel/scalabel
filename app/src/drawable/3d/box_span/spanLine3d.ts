import * as THREE from "three"

import { SpanPoint3D } from "./spanPoint3d"
import { Vector3D } from "../../../math/vector3d"

/**
 * ThreeJS class for rendering 3D line
 */
export class SpanLine3D {
  /** first point */
  private readonly _p1: SpanPoint3D
  /** second point */
  private readonly _p2: SpanPoint3D
  /** line color */
  private readonly _color: number
  /** line width */
  private readonly _lineWidth: number
  /** line object */
  private _line: THREE.Object3D | null

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
    this._line = null
  }

  /**
   * Remove line from Three.js scene
   *
   * @param scene
   */
  public removeFromScene(scene: THREE.Scene): void {
    if (this._line !== null) {
      scene.remove(this._line)
    }
  }

  /**
   * Add to scene for rendering
   *
   * @param scene
   */
  public render(scene: THREE.Scene): void {
    // generate line formed by two points and add to scene
    const points: THREE.Vector3[] = []
    points.push.apply(
      points,
      [this._p1, this._p2].map((p) => new THREE.Vector3(p.x, p.y, p.z))
    )
    const geometry = new THREE.BufferGeometry().setFromPoints(points)
    const material = new THREE.LineBasicMaterial({
      color: this._color,
      linewidth: this._lineWidth
    })
    this._line = new THREE.Line(geometry, material)
    scene.add(this._line)
  }

  /**
   * align point to unit normal
   *
   * @param point
   * @param plane
   */
  public alignPointToNormal(point: Vector3D, plane: THREE.Plane): Vector3D {
    const [first, second] = [this._p1, this._p2].map((p) =>
      p.toVector3D().toThree()
    )
    const firstToSecond = second.clone().sub(first)
    const normal = plane.normal
    const cross = new THREE.Vector3()
    cross.crossVectors(normal, firstToSecond)
    cross.setLength(1)
    const perpendicularPlane = new THREE.Plane()
    perpendicularPlane.setFromNormalAndCoplanarPoint(cross, first)
    const distance = perpendicularPlane.distanceToPoint(point.toThree())
    const p3 = second.clone().add(cross.clone().setLength(distance))
    return new Vector3D(p3.x, p3.y, p3.z)
  }
}
