import * as THREE from "three"

import { SpanPoint3D } from "./spanPoint3d"
import { SpanLine3D } from "./spanLine3d"
import { SpanRect3D } from "./spanRect3d"
import { SpanCuboid3D } from "./spanCuboid3d"
import { Vector3D } from "../../../math/vector3d"

import { convertMouseToNDC } from "../../../view_config/point_cloud"

/**
 * ThreeJS class for rendering 3D span object
 */
export class Span3D {
  private _camera: THREE.Camera | null
  private _canvas: HTMLCanvasElement | null
  private _p1: SpanPoint3D | null
  private _p2: SpanPoint3D | null
  private _p3: SpanPoint3D | null
  private _p4: SpanPoint3D | null
  private _pTmp: SpanPoint3D | null
  private _cuboid: SpanCuboid3D | null
  private _complete: boolean

  /** Constructor */
  constructor() {
    this._camera = null
    this._canvas = null
    this._p1 = null
    this._p2 = null
    this._p3 = null
    this._p4 = null
    this._pTmp = new SpanPoint3D(0, 0, new THREE.Camera())
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
  public render(
    scene: THREE.Scene,
    camera: THREE.Camera,
    canvas: HTMLCanvasElement
  ): void {
    // TODO: use points data to render temporary geometries
    // TODO: figure out whether to render lines/planes/cuboids natively
    // TODO: with ThreeJS or encapsulate in a custom class
    if (this._camera === null) {
      this._camera = camera
    }
    if (this._canvas === null) {
      this._canvas = canvas
    }

    // const plane = new THREE.Plane(new THREE.Vector3(0, 0, 1), 0)

    // // Create a basic rectangle geometry
    // const planeGeometry = new THREE.PlaneGeometry(10, 10)

    // // Align the geometry to the plane
    // const coplanarPoint = plane.coplanarPoint(new THREE.Vector3(0, 0, 1))
    // const focalPoint = new THREE.Vector3().copy(coplanarPoint).add(plane.normal)
    // planeGeometry.lookAt(focalPoint)
    // planeGeometry.translate(coplanarPoint.x, coplanarPoint.y, coplanarPoint.z)

    // // Create mesh with the geometry
    // const planeMaterial = new THREE.MeshBasicMaterial({
    //   color: 0xffff00,
    //   side: THREE.DoubleSide
    // })
    // const dispPlane = new THREE.Mesh(planeGeometry, planeMaterial)
    // scene.add(dispPlane)

    if (this._pTmp !== null) {
      this._pTmp.render(scene)

      if (this._p1 !== null) {
        if (this._p2 === null) {
          // TODO: render first point
          // TODO: render line between first point and temp point
          this._p1.render(scene)
          const line = new SpanLine3D(this._p1, this._pTmp)
          line.render(scene)
        } else if (this._p3 === null) {
          // TODO: render first, second point
          // TODO: render plane formed by first, second point and temp point
          if (this._p1 !== null && this._p2 !== null) {
            this._p1.render(scene)
            this._p2.render(scene)
            const plane = new SpanRect3D(this._p1, this._p2, this._pTmp)
            plane.render(scene)
          }
        } else if (this._p4 === null) {
          // TODO: render first, second, third point
          // TODO: render cuboid formed by first, second, third point and temp point
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
  }

  /**
   * Register new temp point given current mouse position
   *
   * @param x
   * @param y
   */
  public updatePointTmp(x: number, y: number): void {
    const normalized = this.normalizeCoordinatesToCanvas(
      x,
      y,
      this._canvas as HTMLCanvasElement
    )
    console.log(normalized)
    const NDC = convertMouseToNDC(
      normalized[0],
      normalized[1],
      this._canvas as HTMLCanvasElement
    )
    console.log(NDC)
    x = NDC[0]
    y = NDC[1]

    this._pTmp = new SpanPoint3D(x, y, this._camera as THREE.Camera)
  }

  /**
   * Register new point given current mouse position
   *
   * @param x
   * @param y
   */
  public registerPoint(): void {
    if (this._p1 === null) {
      this._p1 = this._pTmp
    } else if (this._p2 === null) {
      this._p2 = this._pTmp
    } else if (this._p3 === null) {
      this._p3 = this._pTmp
    } else if (this._p4 === null) {
      this._p4 = this._pTmp
      this._complete = true
    } else {
      throw new Error("Span3D: error registering new point")
    }
  }

  /** Return whether span box is complete */
  public get complete(): boolean {
    return this._complete
  }

  /** Return cuboid center */
  public get center(): THREE.Vector3 {
    if (
      this._p1 !== null &&
      this._p2 !== null &&
      this._p3 !== null &&
      this._p4 !== null
    ) {
      const v1 = new Vector3D(this._p1.x, this._p1.y, this._p1.z)
      const v2 = new Vector3D(this._p2.x, this._p2.y, this._p2.z)
      const v3 = new Vector3D(this._p3.x, this._p3.y, this._p3.z)
      const v4 = new Vector3D(this._p4.x, this._p4.y, this._p4.z)

      const center = v1.clone().add(v2).add(v3).add(v4)
      center.x /= 4
      center.y /= 4
      center.z /= 4

      return new THREE.Vector3(center.x, center.y, center.z)
    }

    return new THREE.Vector3(0, 0, 0)
  }

  /** Return cuboid dimensions */
  public get dimensions(): THREE.Vector3 {
    if (
      this._p1 !== null &&
      this._p2 !== null &&
      this._p3 !== null &&
      this._p4 !== null
    ) {
      const v1 = new Vector3D(this._p1.x, this._p1.y, this._p1.z)
      const v2 = new Vector3D(this._p2.x, this._p2.y, this._p2.z)
      const v3 = new Vector3D(this._p3.x, this._p3.y, this._p3.z)
      const v4 = new Vector3D(this._p4.x, this._p4.y, this._p4.z)

      const width = v2.distanceTo(v3)
      const depth = v1.distanceTo(v2)
      const height = v3.distanceTo(v4)

      return new THREE.Vector3(width, depth, height)
    }

    return new THREE.Vector3(0, 0, 0)
  }

  /**
   * Normalize mouse coordinates to make canvas left top origin
   *
   * @param x
   * @param y
   * @param canvas
   */
  private normalizeCoordinatesToCanvas(
    x: number,
    y: number,
    canvas: HTMLCanvasElement
  ): number[] {
    return [
      x - canvas.getBoundingClientRect().left,
      y - canvas.getBoundingClientRect().top
    ]
  }
}
