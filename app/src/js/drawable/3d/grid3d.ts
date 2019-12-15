import * as THREE from 'three'
import { ShapeTypeName } from '../../common/types'
import { Plane3DType, ShapeType } from '../../functional/types'
import { Vector3D } from '../../math/vector3d'
import { TransformationControl } from './control/transformation_control'
import Label3D from './label3d'
import { Shape3D } from './shape3d'

/**
 * ThreeJS class for rendering grid
 */
export class Grid3D extends Shape3D {
  /** grid lines */
  private _lines: THREE.GridHelper

  constructor (label: Label3D) {
    super(label)
    this._lines = new THREE.GridHelper(6, 6, 0xffffff, 0xffffff)
    this._lines.rotation.x = Math.PI / 2
    this.add(this._lines)
  }

  /** Get shape type name */
  public get typeName () {
    return ShapeTypeName.GRID
  }

  /** Set center */
  public set center (center: Vector3D) {
    this.position.copy(center.toThree())
  }

  /**
   * Add to scene for rendering
   * @param scene
   */
  public render (scene: THREE.Scene): void {
    scene.add(this)
  }

  /** Do not highlight plane for now */
  public setHighlighted (_intersection: THREE.Intersection) {
    return
  }

  /**
   * Object representation
   */
  public toObject (): ShapeType {
    return{
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
    this._lines.raycast(raycaster, intersects)
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
      this._control.attachShape(this)
    } else if (this._control) {
      this._control.detachShape()
      this.remove(control)
      this._control = null
    }
  }

  /** update parameters */
  public updateState (
    shape: ShapeType, id: number, _activeCamera?: THREE.Camera
  ) {
    super.updateState(shape, id)
    const newShape = shape as Plane3DType
    this.position.copy(
      (new Vector3D()).fromObject(newShape.center).toThree()
    )
    this.rotation.setFromVector3(
      (new Vector3D()).fromObject(newShape.orientation).toThree()
    )
  }
}
