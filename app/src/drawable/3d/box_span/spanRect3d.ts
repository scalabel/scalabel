import * as THREE from "three"

import { SpanPoint3D } from "./spanPoint3d"
import { SpanLine3D } from "./spanLine3d"
import { Vector3D } from "../../../math/vector3d"

/**
 * ThreeJS class for rendering 3D plane
 */
export class SpanRect3D {
  private _p1: SpanPoint3D
  private _p2: SpanPoint3D
  private _p3: SpanPoint3D

  constructor(p1: SpanPoint3D, p2: SpanPoint3D, p3: SpanPoint3D) {
    this._p1 = p1
    this._p2 = p2
    this._p3 = p3
  }

  /**
   * Add to scene for rendering
   *
   * @param scene
   */
  public render(scene: THREE.Scene): void {
    /**
     * TODO: generate rectangle formed by three points and add to scene
     * TODO: display p1, p2, p3
     * TODO: (p3 is along the line orthogonal to the line p1-p2 and intersects p2)
     * TODO: display lines p1-p2, p2-p3
     * TODO: calculate p4 (cross of p1-p2 and p2-p3)
     * TODO: complete the parallelogram
     */
    this._p1.render(scene)
    this._p2.render(scene)
    this._p3.render(scene)
    const l12 = new SpanLine3D(this._p1, this._p2)
    const l23 = new SpanLine3D(this._p2, this._p3)
    const v4 = this.completeParallelogram()
    const p4 = new SpanPoint3D(0, 0)
    p4.setCoords(v4.x, v4.y, v4.z)
    const l34 = new SpanLine3D(this._p3, p4)
    const l41 = new SpanLine3D(p4, this._p1)

    l12.render(scene)
    l23.render(scene)
    l34.render(scene)
    l41.render(scene)
  }

  /** Update plane vertices */
  public updateState(p1: SpanPoint3D, p2: SpanPoint3D): void {
    this._p1 = p1
    this._p2 = p2
  }

  /**
   * TODO: make this function a common math function
   */
  /** Complete parallelogram from 3 points */
  private completeParallelogram(): Vector3D {
    const v1 = new Vector3D(this._p1.x, this._p1.y, this._p1.z)
    const v2 = new Vector3D(this._p2.x, this._p2.y, this._p2.z)
    const v3 = new Vector3D(this._p3.x, this._p3.y, this._p3.z)

    const v12 = v2.clone().subtract(v1)
    const v13 = v3.clone().subtract(v1)

    const vCross = v12.clone().cross(v13)
    return vCross
  }
}
