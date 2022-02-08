import * as THREE from "three"

import { SpanPoint3D } from "./spanPoint3d"
import { SpanLine3D } from "./spanLine3d"
import { Vector3D } from "../../../math/vector3d"

/**
 * ThreeJS class for rendering 3D plane
 */
export class SpanRect3D {
  /** first point */
  private readonly _p1: SpanPoint3D
  /** second point */
  private readonly _p2: SpanPoint3D
  /** third point */
  private readonly _p3: SpanPoint3D
  /** fourth point */
  private readonly _p4: SpanPoint3D
  /** lines in the rect */
  private _lines: SpanLine3D[] | null

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
    this._lines = null
  }

  /**
   * Remove rect from Three.js scene
   *
   * @param scene
   */
  public removeFromScene(scene: THREE.Scene): void {
    if (this._lines !== null) {
      this._lines.forEach((line) => line.removeFromScene(scene))
    }
  }

  /**
   * Add to scene for rendering
   *
   * @param scene
   */
  public render(scene: THREE.Scene): void {
    // generate rectangle formed by three points and add to scene
    // display lines p1-p2, p2-p3, p3-p4, p4-p1
    const points = this.points
    const lines = points.map((p, i) => {
      const next = (i + 1) % points.length
      return new SpanLine3D(p, points[next])
    })
    this._lines = lines
    lines.map((l) => l.render(scene))
  }

  /** Get points data */
  public get points(): SpanPoint3D[] {
    return [this._p1, this._p2, this._p3, this._p4]
  }

  /** Complete parallelogram from 3 points */
  private completeParallelogram(): Vector3D {
    const [v1, v2, v3] = [this._p1, this._p2, this._p3].map((p) =>
      p.toVector3D()
    )
    return v1.clone().add(v3).subtract(v2)
  }
}
