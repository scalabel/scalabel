import { LabelTypeName, ShapeTypeName } from '../../common/types'
import { makeLabel } from '../../functional/states'
import { ShapeType, State } from '../../functional/types'
import { Vector3D } from '../../math/vector3d'
import { TransformationControl } from './control/transformation_control'
import { Grid3D } from './grid3d'
import Label3D from './label3d'
import { Label3DList } from './label3d_list'
import { Shape3D } from './shape3d'

/**
 * Class for managing plane for holding 3d labels
 */
export class Plane3D extends Label3D {
  /** ThreeJS object for rendering shape */
  private _shape: Grid3D

  constructor (labelList: Label3DList) {
    super(labelList)
    this._shape = new Grid3D(this)
  }

  /**
   * Modify ThreeJS objects to draw label
   * @param {THREE.Scene} scene: ThreeJS Scene Object
   */
  public render (scene: THREE.Scene, _camera: THREE.Camera): void {
    this._shape.render(scene)
  }

  /** Attach control */
  public attachControl (control: TransformationControl): void {
    this._shape.attachControl(control)
  }

  /** Detach control */
  public detachControl (): void {
    this._shape.detachControl()
  }

  /**
   * Handle mouse move
   * @param projection
   */
  public onMouseDown (_x: number, _y: number, _camera: THREE.Camera) {
    return false
  }

  /**
   * Handle mouse up
   * @param projection
   */
  public onMouseUp () {
    return
  }

  /**
   * Handle mouse move
   * @param projection
   */
  public onMouseMove (
    _x: number, _y: number, _camera: THREE.Camera
  ): boolean {
    return false
  }

  /**
   * Handle keyboard events
   * @param {KeyboardEvent} e
   * @returns true if consumed, false otherwise
   */
  public onKeyDown (_e: KeyboardEvent): boolean {
    return false
  }

  /**
   * Handle keyboard events
   * @returns true if consumed, false otherwise
   */
  public onKeyUp (_e: KeyboardEvent): boolean {
    return false
  }

  /** Rotate */
  public rotate (quaternion: THREE.Quaternion) {
    this._shape.applyQuaternion(quaternion)
  }

  /** Translate */
  public translate (delta: THREE.Vector3) {
    this._shape.position.add(delta)
  }

  /** Scale */
  public scale (scale: THREE.Vector3, anchor: THREE.Vector3) {
    this._shape.scale.x *= scale.x
    this._shape.scale.y *= scale.y
    this._shape.position.sub(anchor)
    this._shape.position.multiply(scale)
    this._shape.position.add(anchor)
  }

  /** Move */
  public move (position: THREE.Vector3): void {
    this._shape.position.copy(position)
    this._labelList.addUpdatedLabel(this)
  }

  /** center of box */
  public get center (): THREE.Vector3 {
    return this._shape.position
  }

  /** orientation of box */
  public get orientation (): THREE.Quaternion {
    return this._shape.quaternion
  }

  /**
   * Expand the primitive shapes to drawable shapes
   * @param {ShapeType[]} shapes
   */
  public updateState (
    state: State,
    itemIndex: number,
    labelId: number,
    activeCamera?: THREE.Camera
  ): void {
    super.updateState(state, itemIndex, labelId)
    const label = state.task.items[itemIndex].labels[labelId]
    this._shape.updateState(
      state.task.items[itemIndex].shapes[label.shapes[0]].shape,
      label.shapes[0],
      activeCamera
    )
  }

  /**
   * Initialize label
   * @param {State} state
   */
  public init (
    itemIndex: number,
    category: number,
    center?: Vector3D
  ): void {
    this._label = makeLabel({
      type: LabelTypeName.PLANE_3D, id: -1, item: itemIndex,
      category: [category]
    })
    this._labelId = -1
    if (center) {
      this._shape.center = center
    }
  }

  /**
   * Return a list of the shape for inspection and testing
   */
  public shapes (): Shape3D[] {
    return [this._shape]
  }

  /** State representation of shape */
  public shapeStates (): [number[], ShapeTypeName[], ShapeType[]] {
    if (!this._label) {
      throw new Error('Uninitialized label')
    }
    return [
      [this._label.shapes[0]],
      [ShapeTypeName.GRID],
      [this._shape.toObject()]
    ]
  }
}
