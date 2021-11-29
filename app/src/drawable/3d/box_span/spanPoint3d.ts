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
   * @param camera - camera
   * @param pointCloud - point cloud
   * @param temp - temporary
   */
  constructor(x: number, y: number, camera: THREE.Camera, temp: boolean) {
    if (temp) {
      this._x = 0
      this._y = 0
      this._z = 0
    } else {
      const point = this.raycast(x, y, camera)
      this._x = point.x
      this._y = point.y
      this._z = point.z
    }
    this._color = "#ff0000"
    this._radius = 0.05
  }

  /**
   * Add to scene for rendering
   *
   * @param scene
   * @param camera
   * @param pointCloud
   */
  public render(scene: THREE.Scene): void {
    // (radius, widthSegments, heightSegments)
    const geometry = new THREE.SphereGeometry(this._radius, 3, 2)
    const material = new THREE.MeshBasicMaterial({ color: this._color })
    const point = new THREE.Mesh(geometry, material)
    scene.add(point)
  }

  /** x coords */
  public get x(): number {
    return this._x
  }

  /** y coords */
  public get y(): number {
    return this._y
  }

  /** z coords */
  public get z(): number {
    return this._z
  }

  /**
   * Set raw x,y,z coords
   *
   * @param x
   * @param y
   * @param z
   */
  public setCoords(x: number, y: number, z: number): void {
    this._x = x
    this._y = y
    this._z = z
  }

  /**
   * Convert mouse pos to 3D world coordinates
   *
   * @param x - mouse x
   * @param y - mouse y
   * @param camera - camera
   */
  public raycast(x: number, y: number, camera: THREE.Camera): THREE.Vector3 {
    const plane = new THREE.Plane(new THREE.Vector3(0, 0, 1), 0)
    const raycaster = new THREE.Raycaster()
    const mouse = new THREE.Vector2(x, y)
    raycaster.setFromCamera(mouse, camera)
    if (raycaster.ray.intersectsPlane(plane)) {
      console.log("intersects")
      const intersects = new THREE.Vector3()
      raycaster.ray.intersectPlane(plane, intersects)
      console.log(intersects)
      return intersects
    } else {
      return new THREE.Vector3(0, 0, 0)
    }
  }
}
