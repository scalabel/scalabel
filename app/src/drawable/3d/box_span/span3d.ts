import * as THREE from "three"

// import { Box3D } from "../box3d"
import { SpanPoint3D } from "./spanPoint3d"
import { SpanLine3D } from "./spanLine3d"
import { SpanRect3D } from "./spanRect3d"
// import { SpanCuboid3D } from "./spanCuboid3d"

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
  private _pTmp: SpanPoint3D
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

    const plane = new THREE.Plane(new THREE.Vector3(0, 0, 1), 0)

    // Create a basic rectangle geometry
    const planeGeometry = new THREE.PlaneGeometry(10, 10)

    // Align the geometry to the plane
    const coplanarPoint = plane.coplanarPoint(new THREE.Vector3(0, 0, 1))
    const focalPoint = new THREE.Vector3().copy(coplanarPoint).add(plane.normal)
    planeGeometry.lookAt(focalPoint)
    planeGeometry.translate(coplanarPoint.x, coplanarPoint.y, coplanarPoint.z)

    // Create mesh with the geometry
    const planeMaterial = new THREE.MeshBasicMaterial({
      color: 0xffff00,
      side: THREE.DoubleSide
    })
    const dispPlane = new THREE.Mesh(planeGeometry, planeMaterial)
    scene.add(dispPlane)

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
          plane.render(scene, this._camera)
        }
      } else if (this._p4 === null) {
        // TODO: render first, second, third point
        // TODO: render cuboid formed by first, second, third point and temp point
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
    const NDC = convertMouseToNDC(
      normalized[0],
      normalized[1],
      this._canvas as HTMLCanvasElement
    )
    x = NDC[0]
    y = NDC[1]
    console.log(x, y)

    this._pTmp = new SpanPoint3D(x, y, this._camera as THREE.Camera)
  }

  /**
   * Register new point given current mouse position
   *
   * @param x
   * @param y
   */
  public registerPoint(x: number, y: number): void {
    const normalized = this.normalizeCoordinatesToCanvas(
      x,
      y,
      this._canvas as HTMLCanvasElement
    )
    const NDC = convertMouseToNDC(
      normalized[0],
      normalized[1],
      this._canvas as HTMLCanvasElement
    )
    x = NDC[0]
    y = NDC[1]
    console.log(x, y)

    if (this._p1 === null) {
      this._p1 = new SpanPoint3D(x, y, this._camera as THREE.Camera)
    } else if (this._p2 === null) {
      this._p2 = new SpanPoint3D(x, y, this._camera as THREE.Camera)
    } else if (this._p3 === null) {
      this._p3 = new SpanPoint3D(x, y, this._camera as THREE.Camera)
    } else if (this._p4 === null) {
      this._p4 = new SpanPoint3D(x, y, this._camera as THREE.Camera)
      this._complete = true
    } else {
      throw new Error("Span3D: error registering new point")
    }
  }

  // /**
  //  * Handle mouse up
  //  *
  //  * @param x
  //  * @param y
  //  * @param camera
  //  */
  // public onMouseUp(x: number, y: number): void {
  //   /**
  //    * TODO: set next point as current mouse position
  //    * TODO: render new point that follows mouse along an axis
  //    * TODO: orthogonal to vectors generated by previous points
  //    */
  //   if (this._p1 === null) {
  //     this._p1 = new SpanPoint3D(x, y)
  //   } else if (this._p2 === null) {
  //     this._p2 = new SpanPoint3D(x, y)
  //   } else if (this._p3 === null) {
  //     this._p3 = new SpanPoint3D(x, y)
  //   } else if (this._p4 === null) {
  //     this._p4 = new SpanPoint3D(x, y)
  //     this._complete = true
  //   } else {
  //     throw new Error("Span3D: error registering new point")
  //   }
  // }

  // /**
  //  * Handle mouse move
  //  *
  //  * @param x
  //  * @param y
  //  * @param camera
  //  */
  // public onMouseMove(x: number, y: number): void {
  //   /**
  //    * TODO: update temp point to current mouse position
  //    */
  //   this._pTmp.set(x, y)
  // }

  /** whether span box is complete */
  public get complete(): boolean {
    return this._complete
  }

  /** convert span box to Box3D */
  public spanToBox3D(): void {
    // TODO: convert point data to box coordinates
    // TODO: add an appropriate Label3D class based on
    // TODO: currently selected category or default
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
