import { LabelTypeName, ShapeTypeName } from '../../common/types'
import { makeLabel } from '../../functional/states'
import { Plane3DType, ShapeType, State } from '../../functional/types'
import { Vector3D } from '../../math/vector3d'
import { TransformationControl } from './control/transformation_control'
import { Grid3D } from './grid3d'
import Label3D from './label3d'
import { Shape3D } from './shape3d'

/**
 * Class for managing plane for holding 3d labels
 */
export class Plane3D extends Label3D {
  /** ThreeJS object for rendering shape */
  private _shape: Grid3D

  constructor () {
    super()
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

  /**
   * Expand the primitive shapes to drawable shapes
   * @param {ShapeType[]} shapes
   */
  public updateState (
    state: State, itemIndex: number, labelId: number): void {
    super.updateState(state, itemIndex, labelId)
  }

  /**
   * Initialize label
   * @param {State} state
   */
  public init (
    itemIndex: number,
    category: number
  ): void {
    this._label = makeLabel({
      type: LabelTypeName.PLANE_3D, id: -1, item: itemIndex,
      category: [category]
    })
    this._labelId = -1
  }

  /**
   * Return a list of the shape for inspection and testing
   */
  public shapes (): Shape3D[] {
    return [this._shape]
  }

  /** State representation of shape */
  public shapeObjects (): [number[], ShapeTypeName[], ShapeType[]] {
    if (!this._label) {
      throw new Error('Uninitialized label')
    }
    return [
      [this._label.shapes[0]],
      [ShapeTypeName.GRID],
      [this._shape.toObject()]
    ]
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
