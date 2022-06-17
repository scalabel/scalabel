import * as THREE from "three"

import { LabelTypeName } from "../../const/common"
import { makeLabel } from "../../functional/states"
import { Vector3D } from "../../math/vector3d"
import { IdType, INVALID_ID, ShapeType, State } from "../../types/state"
import { Grid3D } from "./grid3d"
import Label3D from "./label3d"
import { Label3DList } from "./label3d_list"
import { Shape3D } from "./shape3d"

/**
 * Class for managing plane for holding 3d labels
 */
export class Plane3D extends Label3D {
  /** ThreeJS object for rendering shape */
  private readonly _shape: Grid3D

  /**
   * Constructor
   *
   * @param labelList
   */
  constructor(labelList: Label3DList) {
    super(labelList)
    this._shape = new Grid3D(this)
  }

  /**
   * Initialize label
   *
   * @param {State} state
   * @param itemIndex
   * @param _category
   * @param center
   * @param orientation
   * @param sensors
   */
  public init(
    itemIndex: number,
    _category: number,
    center?: Vector3D,
    orientation?: Vector3D,
    sensors?: number[]
  ): void {
    if (sensors === null || sensors === undefined || sensors.length === 0) {
      sensors = [-1]
    }

    this._label = makeLabel({
      type: LabelTypeName.PLANE_3D,
      id: INVALID_ID,
      item: itemIndex,
      category: [],
      sensors,
      shapes: [this._shape.shapeId]
    })
    if (center !== undefined && center !== null) {
      // this._shape.center = center
      this._shape.position.copy(center.toThree())
    }

    this._shape.rotation.copy(
      (orientation ?? new Vector3D(Math.PI / 2, 0, 0)).toThreeEuler()
    )
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
   */
  public onMouseDown(): boolean {
    return false
  }

  /**
   * Handle mouse up
   *
   * @param projection
   */
  public onMouseUp(): void {}

  /**
   * Handle mouse move
   */
  public onMouseMove(): boolean {
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

  /** rotation of plane */
  public get rotation(): THREE.Euler {
    return this._shape.rotation
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
      if (this._shape.lines.geometry.boundingBox !== null) {
        box.copy(this._shape.lines.geometry.boundingBox)
      }
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

    // If label is being edited, don't overwrite state
    if (!this.editing) {
      const label = state.task.items[itemIndex].labels[labelId]
      this._shape.updateState(
        state.task.items[itemIndex].shapes[label.shapes[0]],
        label.shapes[0],
        activeCamera
      )
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
    if (this._label === null || this._label === undefined) {
      throw new Error("Uninitialized label")
    }
    const shape = this._shape.toState()
    if (!this._temporary) {
      shape.id = this._label.shapes[0]
    }
    return [shape]
  }

  /**
   * copy the label
   *
   * @param shape
   */
  public setShape(shape: ShapeType): void {
    if (shape.id !== this._shape.shapeId) {
      this.labelList.clearHistShapes()
      return
    }
    super.setShape(shape)
    this._shape.updateState(shape, this._shape.shapeId)
  }
}
