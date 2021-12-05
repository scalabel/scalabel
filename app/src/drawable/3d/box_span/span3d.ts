import * as THREE from "three"

import { SpanPoint3D } from "./spanPoint3d"
import { SpanLine3D } from "./spanLine3d"
import { SpanRect3D } from "./spanRect3d"
import { SpanCuboid3D } from "./spanCuboid3d"
import { Vector3D } from "../../../math/vector3d"

/**
 * ThreeJS class for rendering 3D span object
 */
export class Span3D {
  /** First point */
  private _p1: SpanPoint3D | null
  /** Second point */
  private _p2: SpanPoint3D | null
  /** Third point */
  private _p3: SpanPoint3D | null
  /** Fourth point */
  private _p4: SpanPoint3D | null
  /** Temporary point that follows mouse cursor */
  private _pTmp: SpanPoint3D
  /** Line between first and second point */
  private _line: SpanLine3D | null
  /** Cuboid formed by the four points */
  private _cuboid: SpanCuboid3D | null
  /** Whether span box is complete */
  private _complete: boolean

  /** Constructor */
  constructor() {
    this._p1 = null
    this._p2 = null
    this._p3 = null
    this._p4 = null
    this._pTmp = new SpanPoint3D(new Vector3D(0, 0, 0))
    this._line = null
    this._cuboid = null
    this._complete = false
  }

  /**
   * Modify ThreeJS objects to draw labels
   *
   * @param scene
   * @param camera
   * @param canvas
   */
  public render(scene: THREE.Scene): void {
    // use points data to render temporary geometries
    // render lines/planes/cuboids natively
    if (!this._complete) {
      this._pTmp.render(scene)
    }
    if (this._p1 !== null) {
      if (this._p2 === null) {
        // render first point
        // render line between first point and temp point
        this._p1.render(scene)
        const line = new SpanLine3D(this._p1, this._pTmp)
        this._line = line
        line.render(scene)
      } else if (this._p3 === null) {
        // render first, second point
        // render plane formed by first, second point and temp point
        if (this._p1 !== null && this._p2 !== null) {
          this._p1.render(scene)
          this._p2.render(scene)
          const plane = new SpanRect3D(this._p1, this._p2, this._pTmp)
          plane.render(scene)
        }
      } else if (this._p4 === null) {
        // render first, second, third point
        // render cuboid formed by first, second, third point and temp point
        this._p1.render(scene)
        this._p2.render(scene)
        this._p3.render(scene)
        const cuboid = new SpanCuboid3D(
          this._p1,
          this._p2,
          this._p3,
          this._pTmp
        )
        this._cuboid = cuboid
        this._cuboid.render(scene)
      } else if (this._cuboid !== null) {
        this._cuboid.render(scene)
      } else {
        throw new Error("Span3D: rendering error")
      }
    }
  }

  /**
   * Register new temporary point given current mouse position
   *
   * @param point
   */
  public updatePointTmp(point: Vector3D): this {
    if (this._p2 !== null && this._p3 === null) {
      // make second point orthogonal to line
      if (this._line !== null) {
        const newCoords = this._line.alignPointToNormal(point)
        this._pTmp = new SpanPoint3D(newCoords)
      }
    } else if (this._p3 !== null && this._p4 === null) {
      // make third point orthogonal to plane
      this._pTmp = new SpanPoint3D(
        new Vector3D(this._p3.x, this._p3.y, point.y)
      )
    } else {
      this._pTmp = new SpanPoint3D(point)
    }
    return this
  }

  /** Register new point */
  public registerPoint(): this {
    if (this._p1 === null) {
      this._p1 = this._pTmp
    } else if (this._p2 === null && this._p1 !== this._pTmp) {
      this._p2 = this._pTmp
    } else if (this._p3 === null && this._p2 !== this._pTmp) {
      this._p3 = this._pTmp
    } else if (this._p4 === null && this._p3 !== this._pTmp) {
      this._p4 = this._pTmp
      this._complete = true
    } else {
      throw new Error("Span3D: error registering new point")
    }
    return this
  }

  /** Remove last registered point */
  public removeLastPoint(): this {
    if (this._p4 !== null) {
      this._p4 = null
      this._complete = false
    } else if (this._p3 !== null) {
      this._p3 = null
    } else if (this._p2 !== null) {
      this._p2 = null
    } else if (this._p1 !== null) {
      this._p1 = null
    }
    return this
  }

  /** Return whether span box is complete */
  public get complete(): boolean {
    return this._complete
  }

  /** Return cuboid center */
  public get center(): THREE.Vector3 {
    if (this._cuboid !== null) {
      return this._cuboid.center
    }

    throw new Error("Span3D: cannot get cuboid center")
  }

  /** Return cuboid dimensions */
  public get dimensions(): THREE.Vector3 {
    if (this._cuboid !== null) {
      return this._cuboid.dimensions
    }

    throw new Error("Span3D: cannot get cuboid dimensions")
  }

  /** Return cuboid rotation */
  public get rotation(): THREE.Quaternion {
    if (this._cuboid !== null) {
      return this._cuboid.rotation
    }

    throw new Error("Span3D: cannot get cuboid rotation")
  }
}
