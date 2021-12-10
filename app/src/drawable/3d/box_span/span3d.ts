import * as THREE from "three"

import { SpanPoint3D } from "./spanPoint3d"
import { SpanLine3D } from "./spanLine3d"
import { SpanRect3D } from "./spanRect3d"
import { SpanCuboid3D } from "./spanCuboid3d"
import { Vector2D } from "../../../math/vector2d"
import { Vector3D } from "../../../math/vector3d"

/**
 * ThreeJS class for rendering 3D span object
 */
export class Span3D {
  /** Points list */
  private readonly _points: SpanPoint3D[]
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
    this._points = []
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
    this._points.map((l) => l.render(scene))
    switch (this._points.length) {
      case 1: {
        // render line between first point and temp point
        const line = new SpanLine3D(this._points[0], this._pTmp)
        this._line = line
        line.render(scene)
        break
      }
      case 2: {
        // render plane formed by first, second point and temp point
        const plane = new SpanRect3D(
          this._points[0],
          this._points[1],
          this._pTmp
        )
        plane.render(scene)
        break
      }
      case 3: {
        // render cuboid formed by first, second, third point and temp point
        const cuboid = new SpanCuboid3D(
          this._points[0],
          this._points[1],
          this._points[2],
          this._pTmp
        )
        this._cuboid = cuboid
        this._cuboid.render(scene)
        break
      }
      default:
        if (this._cuboid !== null) {
          this._cuboid.render(scene)
        }
    }
  }

  /**
   * Register new temporary point given current mouse position
   *
   * @param point
   * @param mousePos
   */
  public updatePointTmp(point: Vector3D, mousePos: Vector2D): this {
    switch (this._points.length) {
      case 2:
        // make second point orthogonal to line
        if (this._line !== null) {
          const newCoords = this._line.alignPointToNormal(point)
          this._pTmp = new SpanPoint3D(newCoords)
        }
        break
      case 3: {
        // make third point orthogonal to plane
        const scaleFactor = 5
        this._pTmp = new SpanPoint3D(
          new Vector3D(
            this._points[2].x,
            this._points[2].y,
            mousePos.y * scaleFactor
          )
        )
        break
      }
      default:
        this._pTmp = new SpanPoint3D(point)
    }
    return this
  }

  /** Register new point */
  public registerPoint(): this {
    switch (this._points.length) {
      case 1:
        if (this._points[0] === this._pTmp) {
          return this
        }
        break
      case 2:
        if (this._points[1] === this._pTmp) {
          return this
        }
        break
      case 3:
        if (this._points[2] !== this._pTmp) {
          this._complete = true
        } else {
          return this
        }
    }
    this._points.push(this._pTmp)
    return this
  }

  /** Remove last registered point */
  public removeLastPoint(): this {
    if (this._points.length > 0) {
      this._points.pop()
      this._complete = false
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
