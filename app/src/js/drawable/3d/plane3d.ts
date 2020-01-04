import * as THREE from 'three'
import { LabelTypeName, ShapeTypeName } from '../../common/types'
import { makeLabel } from '../../functional/states'
import { ShapeType, State } from '../../functional/types'
import { Vector3D } from '../../math/vector3d'
import { Box3D } from './box3d'
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
  /** temporary shape */
  private _temporaryLabel: Label3D | null

  constructor (labelList: Label3DList) {
    super(labelList)
    this._shape = new Grid3D(this)
    this._temporaryLabel = null
  }

  /** Override set selected method */
  public set selected (s: boolean) {
    super.selected = s
  }

  /** Override get selected */
  public get selected (): boolean {
    return super.selected
  }

  /**
   * Modify ThreeJS objects to draw label
   * @param {THREE.Scene} scene: ThreeJS Scene Object
   */
  public render (scene: THREE.Scene, _camera: THREE.Camera): void {
    this._shape.render(scene)
  }

  /**
   * Highlight box
   * @param intersection
   */
  public setHighlighted (intersection?: THREE.Intersection) {
    super.setHighlighted(intersection)
    this._shape.setHighlighted(intersection)
  }

  /**
   * Handle mouse move
   * @param projection
   */
  public onMouseDown (x: number, y: number, camera: THREE.Camera) {
    if (
      this._label &&
      (this.selected || this.anyChildSelected()) &&
      this._labelList.currentLabelType === LabelTypeName.BOX_3D
    ) {
      this._temporaryLabel = new Box3D(this._labelList)
      this._temporaryLabel.init(
        this._label.item,
        0,
        undefined,
        this._label.sensors,
        true
      )
      this.addChild(this._temporaryLabel)
      for (const shape of this._temporaryLabel.shapes()) {
        this._shape.attach(shape)
      }
      return this._temporaryLabel.onMouseDown(x, y, camera)
    }
    return false
  }

  /**
   * Handle mouse up
   * @param projection
   */
  public onMouseUp () {
    if (this._temporaryLabel) {
      this._temporaryLabel.onMouseUp()
      this._temporaryLabel = null
    }
  }

  /**
   * Handle mouse move
   * @param projection
   */
  public onMouseMove (
    x: number, y: number, camera: THREE.Camera
  ): boolean {
    if (this._temporaryLabel) {
      return this._temporaryLabel.onMouseMove(x, y, camera)
    }
    return false
  }

  /** Rotate */
  public rotate (quaternion: THREE.Quaternion) {
    this._labelList.addUpdatedLabel(this)
    this._shape.applyQuaternion(quaternion)
    for (const child of this.children) {
      child.rotate(quaternion, this._shape.position)
    }
  }

  /** Translate */
  public translate (delta: THREE.Vector3) {
    this._labelList.addUpdatedLabel(this)
    this._shape.position.add(delta)
    for (const child of this.children) {
      child.translate(delta)
    }
  }

  /** Scale */
  public scale (scale: THREE.Vector3, anchor: THREE.Vector3) {
    this._labelList.addUpdatedLabel(this)
    this._shape.scale.x *= scale.x
    this._shape.scale.y *= scale.y
    this._shape.position.sub(anchor)
    this._shape.position.multiply(scale)
    this._shape.position.add(anchor)
  }

  /** center of plane */
  public get center (): THREE.Vector3 {
    return this._shape.position
  }

  /** orientation of plane */
  public get orientation (): THREE.Quaternion {
    return this._shape.quaternion
  }

  /** scale of plane */
  public get size (): THREE.Vector3 {
    return this._shape.scale
  }

  /** bounds of plane */
  public bounds (local?: boolean): THREE.Box3 {
    const box = new THREE.Box3()
    if (!local) {
      box.copy(this._shape.lines.geometry.boundingBox)
      this._shape.updateMatrixWorld(true)
      box.applyMatrix4(this._shape.matrixWorld)
    } else {
      box.setFromCenterAndSize(this.center, this.size)
    }
    return box
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

    if (this._label) {
      this._shape.updateState(
        state.task.items[itemIndex].shapes[this._label.shapes[0]].shape,
        this._label.shapes[0],
        activeCamera
      )

      const currentChildren = [...this._children]
      for (const child of currentChildren) {
        if (!this._label.children.includes(child.labelId)) {
          this.removeChild(child)
          for (const shape of child.shapes()) {
            this._shape.remove(shape)
          }
        }
      }
    }
  }

  /**
   * Initialize label
   * @param {State} state
   */
  public init (
    itemIndex: number,
    category: number,
    center?: Vector3D,
    sensors?: number[]
  ): void {
    this._label = makeLabel({
      type: LabelTypeName.PLANE_3D, id: -1, item: itemIndex,
      category: [category], sensors
    })
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
      [this._shape.toState()]
    ]
  }
}
