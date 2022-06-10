import * as THREE from "three"

import { SpanPoint3D } from "./spanPoint3d"
import { SpanLine3D } from "./spanLine3d"
import { SpanRect3D } from "./spanRect3d"
import { SpanCuboid3D } from "./spanCuboid3d"
import { Vector3D } from "../../../math/vector3d"
import { projectionFromNDC } from "../../../view_config/point_cloud"
import { Vector2D } from "../../../math/vector2d"

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
  /** Rectangle formed by first three points */
  private _rect: SpanRect3D | null
  /** Cuboid formed by the four points */
  private _cuboid: SpanCuboid3D | null
  /** Whether span box is complete */
  private _complete: boolean

  /** Constructor */
  constructor() {
    this._points = []
    this._pTmp = new SpanPoint3D(new Vector3D(0, 0, 0))
    this._line = null
    this._rect = null
    this._cuboid = null
    this._complete = false
  }

  /**
   * Modify ThreeJS objects to draw labels
   *
   * @param scene
   */
  public render(scene: THREE.Scene): void {
    // use points data to render temporary geometries
    // render lines/planes/cuboids natively
    if (!this._complete) {
      this._pTmp.render(scene)
    }
    this._points.map((l) => l.render(scene))
    switch (this._points.length) {
      case 0:
        break
      case 1: {
        // render line between first point and temp point
        this._line?.removeFromScene(scene)
        const line = new SpanLine3D(this._points[0], this._pTmp)
        this._line = line
        line.render(scene)
        break
      }
      case 2: {
        // render plane formed by first, second point and temp point
        this._rect?.removeFromScene(scene)
        const rect = new SpanRect3D(
          this._points[0],
          this._points[1],
          this._pTmp
        )
        this._rect = rect
        rect.render(scene)
        break
      }
      case 3: {
        // render cuboid formed by first, second, third point and temp point
        this._cuboid?.removeFromScene(scene)
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
   * @param coords
   * @param plane
   * @param camera
   */
  public updatePointTmp(
    coords: Vector2D,
    plane: THREE.Plane,
    camera: THREE.Camera
  ): this {
    if (this._pTmp === null) {
      this._pTmp = new SpanPoint3D(new Vector3D())
    }
    const projection = projectionFromNDC(coords.x, coords.y, camera)
    const point3d = new THREE.Vector3()
    projection.intersectPlane(plane, point3d)
    const point = new Vector3D(point3d.x, point3d.y, point3d.z)
    switch (this._points.length) {
      case 2:
        // make second point orthogonal to line
        if (this._line !== null) {
          const newCoords = this._line.alignPointToNormal(point, plane)
          this._pTmp.update(newCoords)
        }
        break
      case 3: {
        // make third point orthogonal to plane
        const newPlane = new THREE.Plane()
        const p3 = this._points[2].toVector3D().toThree()
        const newNormal = p3.clone().normalize()
        newPlane.setFromNormalAndCoplanarPoint(newNormal, p3)

        projection.intersectPlane(newPlane, point3d)

        const distance = plane.distanceToPoint(point3d)
        const normal = plane.normal.clone()
        normal.setLength(distance)
        const newPoint = normal.add(this._points[2].toVector3D().toThree())
        this._pTmp.update(new Vector3D(newPoint.x, newPoint.y, newPoint.z))
        break
      }
      default:
        this._pTmp.update(point)
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
    this._pTmp = new SpanPoint3D(new Vector3D())
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

  /**
   * Return cuboid dimensions
   *
   * @param up
   * @param forward
   * @param left
   * */
  public dimensions(
    up: Vector3D,
    forward: Vector3D,
    left: Vector3D
  ): THREE.Vector3 {
    if (this._cuboid !== null) {
      return this._cuboid.dimensions(up, forward, left)
    }

    throw new Error("Span3D: cannot get cuboid dimensions")
  }

  /**
   * Get side line
   */
  public get sideEdge(): THREE.Vector3 {
    if (this._points.length < 3) {
      throw new Error("Point 3 has not been set yet")
    }
    return this._points[2]
      .toVector3D()
      .toThree()
      .sub(this._points[1].toVector3D().toThree())
  }

  /**
   * Number of points set
   */
  public get numPoints(): number {
    return this._points.length
  }
}
