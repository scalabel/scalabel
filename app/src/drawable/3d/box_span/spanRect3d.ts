import * as THREE from "three"

import { SpanPoint3D } from "./spanPoint3d"
import { SpanLine3D } from "./spanLine3d"
import { Vector3D } from "../../../math/vector3d"

/**
 * ThreeJS class for rendering 3D plane
 */
export class SpanRect3D {
  private readonly _p1: SpanPoint3D
  private readonly _p2: SpanPoint3D
  private readonly _p3: SpanPoint3D
  private readonly _p4: SpanPoint3D

  /**
   * Constructor
   *
   * @param p1
   * @param p2
   * @param p3
   */
  constructor(p1: SpanPoint3D, p2: SpanPoint3D, p3: SpanPoint3D) {
    this._p1 = p1
    this._p2 = p2
    this._p3 = p3

    const v4 = this.completeParallelogram()
    this._p4 = new SpanPoint3D(v4)
  }

  /**
   * Add to scene for rendering
   *
   * @param scene
   * @param camera
   */
  public render(scene: THREE.Scene): void {
    // generate rectangle formed by three points and add to scene
    // display lines p1-p2, p2-p3, p3-p4, p4-p1
    const l12 = new SpanLine3D(this._p1, this._p2)
    const l23 = new SpanLine3D(this._p2, this._p3)
    const l34 = new SpanLine3D(this._p3, this._p4)
    const l41 = new SpanLine3D(this._p4, this._p1)

    l12.render(scene)
    l23.render(scene)
    l34.render(scene)
    l41.render(scene)
  }

  /** Get points data */
  public get points(): SpanPoint3D[] {
    return [this._p1, this._p2, this._p3, this._p4]
  }

  /** Complete parallelogram from 3 points */
  private completeParallelogram(): Vector3D {
    const v1 = new Vector3D(this._p1.x, this._p1.y, this._p1.z)
    const v2 = new Vector3D(this._p2.x, this._p2.y, this._p2.z)
    const v3 = new Vector3D(this._p3.x, this._p3.y, this._p3.z)

    const v4 = v1.clone().add(v3).subtract(v2)
    return v4
  }
}
