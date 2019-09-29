import * as THREE from 'three'
import { Plane3DType } from '../../functional/types'
import { Vector3D } from '../../math/vector3d'
import { TransformationControl } from './control/transformation_control'

/**
 * ThreeJS class for rendering grid
 */
export class Grid3D extends THREE.Group {
  /** grid lines */
  private _lines: THREE.GridHelper
  /** label id */
  private _id: number
  /** control */
  private _control: TransformationControl | null

  constructor (id: number) {
    super()
    this._id = id
    this._lines = new THREE.GridHelper(6, 6, 0xffffff, 0xffffff)
    this._lines.rotation.x = Math.PI / 2
    this.add(this._lines)
    this._control = null
  }

  /**
   * Add to scene for rendering
   * @param scene
   */
  public render (scene: THREE.Scene): void {
    scene.add(this)
  }

  /**
   * Get id
   */
  public get labelId (): number {
    return this._id
  }

  /**
   * Set id
   * @param {number} id
   */
  public set labelId (id: number) {
    this._id = id
  }

  /**
   * Object representation
   */
  public toPlane (): Plane3DType {
    return {
      center: (new Vector3D()).fromThree(this.position).toObject(),
      orientation:
        (new Vector3D()).fromThree(this.rotation.toVector3()).toObject()
    }
  }

  /**
   * Override ThreeJS raycast
   * @param raycaster
   * @param intersects
   */
  public raycast (
    raycaster: THREE.Raycaster,
    intersects: THREE.Intersection[]
  ) {
    if (this._control) {
      this._control.raycast(raycaster, intersects)
    }
  }

  /**
   * Add/remove controls
   * @param control
   * @param show
   */
  public setControl (control: TransformationControl, show: boolean) {
    if (show) {
      this.add(control)
      this._control = control
      this._control.attach(this)
    } else if (this._control) {
      this._control.detach()
      this.remove(control)
      this._control = null
    }
  }
}
