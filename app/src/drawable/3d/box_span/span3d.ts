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
  private _pTmp: SpanPoint3D
  private _line: SpanLine3D | null
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
  public render(
    scene: THREE.Scene,
    camera: THREE.Camera,
    canvas: HTMLCanvasElement
  ): void {
    if (this._camera === null) {
      this._camera = camera
    }
    if (this._canvas === null) {
      this._canvas = canvas
    }

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
   * Register new temp point given current mouse position
   *
   * @param x
   * @param y
   */
  public updatePointTmp(x: number, y: number): void {
    // TODO: figure out why this offset is necessary
    const offset = [160, 92]
    x += offset[0]
    y += offset[1]
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

    if (this._p2 !== null && this._p3 === null) {
      // make second point orthogonal to line
      if (this._camera !== null) {
        const worldCoords = this.raycast(x, y, this._camera)
        if (this._line !== null) {
          const newCoords = this._line.alignPointToNormal(worldCoords)
          this._pTmp = new SpanPoint3D(newCoords)
        }
      }
    } else if (this._p3 !== null && this._p4 === null) {
      // make third point orthogonal to plane
      const scaleFactor = 5
      this._pTmp = new SpanPoint3D(
        new Vector3D(this._p3.x, this._p3.y, y * scaleFactor)
      )
    } else {
      if (this._camera !== null) {
        const worldCoords = this.raycast(x, y, this._camera)
        this._pTmp = new SpanPoint3D(worldCoords)
      }
    }
  }

  /** Register new point */
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

  /** Remove last registered point */
  public removeLastPoint(): void {
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

    return new THREE.Vector3(0, 0, 0)
  }

  /** Return cuboid dimensions */
  public get dimensions(): THREE.Vector3 {
    if (this._cuboid !== null) {
      return this._cuboid.dimensions
    }

    return new THREE.Vector3(0, 0, 0)
  }

  /** Return cuboid rotation */
  public get rotation(): THREE.Quaternion {
    if (this._cuboid !== null) {
      return this._cuboid.rotation
    }

    return new THREE.Quaternion()
  }

  /**
   * Convert mouse pos to 3D world coordinates
   *
   * @param x - mouse x
   * @param y - mouse y
   * @param camera - camera
   */
  private raycast(x: number, y: number, camera: THREE.Camera): Vector3D {
    // TODO: figure out why this offset is necessary
    const offset = new THREE.Vector3(0, 0, -1.5)
    const plane = new THREE.Plane(new THREE.Vector3(0, 0, 1), 0).translate(
      offset
    )
    const raycaster = new THREE.Raycaster()
    const mousePos = new THREE.Vector2(x, y)
    raycaster.setFromCamera(mousePos, camera)
    if (raycaster.ray.intersectsPlane(plane)) {
      const intersects = new THREE.Vector3()
      raycaster.ray.intersectPlane(plane, intersects)
      return new Vector3D().fromThree(intersects)
    } else {
      return new Vector3D(0, 0, 0)
    }
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
