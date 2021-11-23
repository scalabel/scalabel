import * as THREE from "three"

/**
 * ThreeJS class for rendering 3D point
 */
export class SpanPoint3D {
  private _x: number
  private _y: number
  private _z: number
  private readonly _color: string
  private readonly _radius: number

  /**
   * Constructor
   *
   * @param x - mouse x
   * @param y - mouse y
   */
  constructor(x: number, y: number) {
    /**
     * TODO: convert mouse pos to world coords
     */
    this._x = 0
    this._y = 0
    this._z = 0
    this._color = "#ff0000"
    this._radius = 0.05
  }

  /**
   * Add to scene for rendering
   *
   * @param scene
   */
  public render(scene: THREE.Scene): void {
    // (radius, widthSegments, heightSegments)
    const geometry = new THREE.SphereGeometry(this._radius, 3, 2)
    const material = new THREE.MeshBasicMaterial({ color: this._color })
    const point = new THREE.Mesh(geometry, material)
    scene.add(point)
  }

  /** x coords */
  public get x() {
    return this._x
  }

  /** y coords */
  public get y() {
    return this._y
  }

  /** z coords */
  public get z() {
    return this._z
  }

  /**
   * Set x,y,z coords based on mouse pos
   *
   * @param x - mouse x
   * @param y - mouse y
   */
  public updateState(x: number, y: number) {
    /**
     * TODO: convert mouse pos to world coords
     */
    this._x = 0
    this._y = 0
    this._z = 0
  }
}
