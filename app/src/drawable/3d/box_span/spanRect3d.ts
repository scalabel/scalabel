import * as THREE from "three"

import { SpanPoint3D } from "./spanPoint3d"

/**
 * ThreeJS class for rendering 3D plane
 */
export class SpanRect3D {
  private _p1: SpanPoint3D
  private _p2: SpanPoint3D
  private _p3: SpanPoint3D
  private readonly _color: number

  constructor(p1: SpanPoint3D, p2: SpanPoint3D, p3: SpanPoint3D) {
    this._p1 = p1
    this._p2 = p2
    this._p3 = p3
    this._color = 0xffff00
  }

  /**
   * Add to scene for rendering
   *
   * @param scene
   */
  public render(scene: THREE.Scene): void {
    /**
     * TODO: generate rectangle formed by three points and add to scene
     */
  }

  /** Update plane vertices */
  public updateState(p1: SpanPoint3D, p2: SpanPoint3D): void {
    this._p1 = p1
    this._p2 = p2
  }
}
