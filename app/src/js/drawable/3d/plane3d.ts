import { changeLabelShape } from '../../action/common'
import { addPlaneLabel } from '../../action/plane3d'
import Session from '../../common/session'
import { LabelTypes } from '../../common/types'
import { makeLabel } from '../../functional/states'
import { Plane3DType, ShapeType, State } from '../../functional/types'
import { Vector3D } from '../../math/vector3d'
import { TransformationControl } from './control/transformation_control'
import { Grid3D } from './grid3d'
import Label3D from './label3d'

/**
 * Class for managing plane for holding 3d labels
 */
export class Plane3D extends Label3D {
  /** ThreeJS object for rendering shape */
  private _shape: Grid3D

  constructor () {
    super()
    this._shape = new Grid3D(this._index)
  }

  /** select the label */
  public setSelected (s: boolean) {
    super.setSelected(s)
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
    this._shape.setControl(control, true)
  }

  /** Detach control */
  public detachControl (control: TransformationControl): void {
    this._shape.setControl(control, false)
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

  /**
   * Expand the primitive shapes to drawable shapes
   * @param {ShapeType[]} shapes
   */
  public updateState (
    state: State, itemIndex: number, labelId: number): void {
    super.updateState(state, itemIndex, labelId)
    this._shape.labelId = labelId
  }

  /** Update the shapes of the label to the state */
  public commitLabel (): void {
    if (this._label !== null) {
      const shape = this._shape.toPlane()
      Session.dispatch(changeLabelShape(
        this._label.item, this._label.shapes[0], shape
      ))
    }
  }

  /**
   * Initialize label
   * @param {State} state
   */
  public init (state: State, _surfaceId?: number, _temporary?: boolean): void {
    const itemIndex = state.user.select.item
    this._order = state.task.status.maxOrder + 1
    this._label = makeLabel({
      type: LabelTypes.PLANE_3D, id: -1, item: itemIndex,
      category: [state.user.select.category],
      order: this._order
    })
    this._labelId = -1
    const plane = this._shape.toPlane()
    Session.dispatch(addPlaneLabel(
      this._label.item, plane.center, plane.orientation
    ))
  }

  /**
   * Return a list of the shape for inspection and testing
   */
  public shapes (): Array<Readonly<Grid3D>> {
    return [this._shape]
  }

  /**
   * Expand the primitive shapes to drawable shapes
   * @param {ShapeType[]} shapes
   */
  public updateShapes (shapes: ShapeType[]): void {
    const newShape = shapes[0] as Plane3DType
    this._shape.position.copy(
      (new Vector3D()).fromObject(newShape.center).toThree()
    )
    this._shape.rotation.setFromVector3(
      (new Vector3D()).fromObject(newShape.orientation).toThree()
    )
  }
}
