import * as THREE from "three"

import { LabelTypeName } from "../../const/common"
import { makeLabel } from "../../functional/states"
import { Vector3D } from "../../math/vector3d"
import { IdType, INVALID_ID, ShapeType, State } from "../../types/state"
import { Box3D } from "./box3d"
import { Grid3D } from "./grid3d"
import Label3D from "./label3d"
import { Label3DList } from "./label3d_list"
import { Shape3D } from "./shape3d"

/**
 * Class for managing plane for holding 3d labels
 */
export class Plane3D extends Label3D {
  /** ThreeJS object for rendering shape */
  private _shape: Grid3D
  /** temporary shape */
  private _temporaryLabel: Label3D | null

  /**
   * Constructor
   *
   * @param labelList
   */
  constructor(labelList: Label3DList) {
    super(labelList)
    this._shape = new Grid3D(this)
    this._temporaryLabel = null
  }

  /** Override set selected method */
  public set selected(s: boolean) {
    super.selected = s
  }

  /** Override get selected */
  public get selected(): boolean {
    return super.selected
  }

  /**
   * Modify ThreeJS objects to draw label
   *
   * @param {THREE.Scene} scene: ThreeJS Scene Object
   * @param scene
   * @param _camera
   */
  public render(scene: THREE.Scene /* _camera: THREE.Camera */): void {
    this._shape.render(scene)
  }

  /**
   * Highlight box
   *
   * @param intersection
   */
  public setHighlighted(intersection?: THREE.Intersection): void {
    super.setHighlighted(intersection)
    this._shape.setHighlighted(intersection)
  }

  /**
   * Handle mouse move
   *
   * @param projection
   * @param x
   * @param y
   * @param camera
   */
  public onMouseDown(x: number, y: number, camera: THREE.Camera): boolean {
    if (
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
      for (const shape of this._temporaryLabel.internalShapes()) {
        this._labelList.scene.add(shape)
      }
      return this._temporaryLabel.onMouseDown(x, y, camera)
    }
    return false
  }

  /**
   * Handle mouse up
   *
   * @param projection
   */
  public onMouseUp(): void {
    if (this._temporaryLabel !== null) {
      this._temporaryLabel.onMouseUp()
      this._temporaryLabel = null
    }
  }

  /**
   * Handle mouse move
   *
   * @param projection
   * @param x
   * @param y
   * @param camera
   */
  public onMouseMove(x: number, y: number, camera: THREE.Camera): boolean {
    if (this._temporaryLabel !== null) {
      return this._temporaryLabel.onMouseMove(x, y, camera)
    }
    return false
  }

  /**
   * Rotate
   *
   * @param quaternion
   */
  public rotate(quaternion: THREE.Quaternion): void {
    this._labelList.addUpdatedLabel(this)
    this._shape.applyQuaternion(quaternion)
    for (const child of this.children) {
      child.rotate(quaternion, this._shape.position)
    }
  }

  /**
   * Translate
   *
   * @param delta
   */
  public translate(delta: THREE.Vector3): void {
    this._labelList.addUpdatedLabel(this)
    this._shape.position.add(delta)
    for (const child of this.children) {
      child.translate(delta)
    }
  }

  /**
   * Scale
   *
   * @param scale
   * @param anchor
   */
  public scale(scale: THREE.Vector3, anchor: THREE.Vector3): void {
    this._labelList.addUpdatedLabel(this)
    this._shape.scale.x *= scale.x
    this._shape.scale.y *= scale.y
    this._shape.position.sub(anchor)
    this._shape.position.multiply(scale)
    this._shape.position.add(anchor)
  }

  /**
   * Move
   *
   * @param position
   */
  public move(position: THREE.Vector3): void {
    this._shape.position.copy(position)
    this._labelList.addUpdatedLabel(this)
  }

  /** center of plane */
  public get center(): THREE.Vector3 {
    return this._shape.position
  }

  /** orientation of plane */
  public get orientation(): THREE.Quaternion {
    return this._shape.quaternion
  }

  /** scale of plane */
  public get size(): THREE.Vector3 {
    return this._shape.scale
  }

  /**
   * bounds of plane
   *
   * @param local
   */
  public bounds(local?: boolean): THREE.Box3 {
    const box = new THREE.Box3()
    if (local === undefined || !local) {
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
   *
   * @param {ShapeType[]} shapes
   * @param state
   * @param itemIndex
   * @param labelId
   * @param activeCamera
   */
  public updateState(
    state: State,
    itemIndex: number,
    labelId: IdType,
    activeCamera?: THREE.Camera
  ): void {
    super.updateState(state, itemIndex, labelId)

    this._shape.updateState(
      state.task.items[itemIndex].shapes[this._label.shapes[0]],
      this._label.shapes[0],
      activeCamera
    )

    const currentChildren = [...this._children]
    for (const child of currentChildren) {
      if (!this._label.children.includes(child.labelId)) {
        this.removeChild(child)
        for (const shape of child.internalShapes()) {
          this._shape.remove(shape)
        }
      }
    }
  }

  /**
   * Initialize label
   *
   * @param {State} state
   * @param itemIndex
   * @param category
   * @param center
   * @param sensors
   */
  public init(
    itemIndex: number,
    category: number,
    center?: Vector3D,
    sensors?: number[]
  ): void {
    this._label = makeLabel({
      type: LabelTypeName.PLANE_3D,
      id: INVALID_ID,
      item: itemIndex,
      category: [category],
      sensors
    })
    if (center !== undefined) {
      this._shape.center = center
    }
  }

  /**
   * Return a list of the shape for inspection and testing
   */
  public internalShapes(): Shape3D[] {
    return [this._shape]
  }

  /** State representation of shape */
  public shapes(): ShapeType[] {
    /**
     * This is a temporary solution for assigning the correct ID to the shapes
     * We should initialize the shape when the temporary label is created.
     * Also store the shape id properly so that the generated shape state has
     * the right id directly.
     */
    const shape = this._shape.toState()
    if (!this._temporary) {
      shape.id = this._label.shapes[0]
    }
    return [shape]
  }
}
