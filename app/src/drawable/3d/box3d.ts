import * as THREE from "three"

import { LabelTypeName } from "../../const/common"
import { makeLabel } from "../../functional/states"
import { rotateScale } from "../../math/3d"
import { Vector3D } from "../../math/vector3d"
import { IdType, INVALID_ID, ShapeType, State } from "../../types/state"
import { Cube3D } from "./cube3d"
import { Label3D } from "./label3d"
import { Label3DList } from "./label3d_list"
import { Plane3D } from "./plane3d"
import { Shape3D } from "./shape3d"

/**
 * Box3d Label
 */
export class Box3D extends Label3D {
  /** ThreeJS object for rendering shape */
  private _shape: Cube3D

  /**
   * Constructor
   *
   * @param labelList
   */
  constructor(labelList: Label3DList) {
    super(labelList)
    this._shape = new Cube3D(this)
  }

  /**
   * Initialize label
   *
   * @param {State} state
   * @param itemIndex
   * @param category
   * @param center
   * @param sensors
   * @param temporary
   */
  public init(
    itemIndex: number,
    category: number,
    center?: Vector3D,
    sensors?: number[],
    temporary?: boolean
  ): void {
    if (sensors === null || sensors === undefined || sensors.length === 0) {
      sensors = [-1]
    }

    this._label = makeLabel({
      type: LabelTypeName.BOX_3D,
      id: INVALID_ID,
      item: itemIndex,
      category: [category],
      sensors,
      shapes: [this._shape.shapeId]
    })

    if (center !== null && center !== undefined) {
      this._shape.position.copy(center.toThree())
    }

    if (temporary !== null && temporary !== undefined && temporary) {
      this._temporary = true
    }
  }

  /** Set active camera
   */
  public set activeCamera(camera: THREE.Camera) {
    this._shape.setControlSpheres(camera)
  }

  /**
   * Get active camera
   */
  public get activeCamera(): THREE.Camera {
    throw new Error("Method not implemented.")
  }

  /** Indexed shapes */
  public shapes(): ShapeType[] {
    if (this._label === null || this._label === undefined) {
      throw new Error("Uninitialized label")
    }
    const box = this._shape.toState()
    if (!this._temporary) {
      box.id = this._label.shapes[0]
    }
    return [box]
  }

  /** Override set parent */
  public set parent(parent: Label3D | null) {
    this._parent = parent
    if (parent !== null && this._label !== null && this._label !== undefined) {
      this._label.parent = parent.labelId
    } else if (this._label !== null && this._label !== undefined) {
      this._label.parent = INVALID_ID
    }
    if (parent !== null && parent.label.type === LabelTypeName.PLANE_3D) {
      this._shape.attachToPlane(parent as Plane3D)
    } else {
      this._shape.detachFromPlane()
    }
  }

  /** Get parent label */
  public get parent(): Label3D | null {
    return this._parent
  }

  /**
   * Return a list of the shape for inspection and testing
   */
  public internalShapes(): Shape3D[] {
    return [this._shape]
  }

  /**
   * Modify ThreeJS objects to draw label
   *
   * @param {THREE.Scene} scene: ThreeJS Scene Object
   * @param scene
   * @param camera
   */
  public render(scene: THREE.Scene, camera: THREE.Camera): void {
    this._shape.render(scene, camera)
  }

  /**
   * move anchor to next corner
   */
  public incrementAnchorIndex(): void {
    this._shape.incrementAnchorIndex()
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
    if (this._temporary) {
      this._shape.clickInit(x, y, camera)
    }
    return this._shape.shouldDrag()
  }

  /**
   * Handle mouse up
   *
   * @param projection
   */
  public onMouseUp(): void {}

  /**
   * Handle mouse move
   *
   * @param x
   * @param y
   * @param camera
   */
  public onMouseMove(x: number, y: number, camera: THREE.Camera): boolean {
    const success = this._shape.drag(x, y, camera)
    if (success) {
      this._temporary = false
    }
    this._labelList.addUpdatedLabel(this)
    return success
  }

  /**
   * Rotate
   *
   * @param quaternion
   * @param anchor
   */
  public rotate(quaternion: THREE.Quaternion, anchor?: THREE.Vector3): void {
    this._labelList.addUpdatedLabel(this)
    this._shape.applyQuaternion(quaternion)
    if (anchor !== null && anchor !== undefined) {
      const newPosition = new THREE.Vector3()
      newPosition.copy(this._shape.position)
      newPosition.sub(anchor)
      newPosition.applyQuaternion(quaternion)
      newPosition.add(anchor)
      this._shape.position.copy(newPosition)
    }
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

  /**
   * Translate
   *
   * @param delta
   */
  public translate(delta: THREE.Vector3): void {
    this._labelList.addUpdatedLabel(this)
    this._shape.position.add(delta)
  }

  /**
   * Scale
   *
   * @param scale
   * @param anchor
   * @param local
   */
  public scale(
    scale: THREE.Vector3,
    anchor: THREE.Vector3,
    local: boolean
  ): void {
    this._labelList.addUpdatedLabel(this)
    const inverseRotation = new THREE.Quaternion()
    inverseRotation.copy(this.orientation)
    inverseRotation.inverse()

    if (!local) {
      scale = rotateScale(scale, this.orientation)
    }
    this._shape.scale.multiply(scale)

    this._shape.position.sub(anchor)
    this._shape.position.applyQuaternion(inverseRotation)
    this._shape.position.multiply(scale)
    this._shape.position.applyQuaternion(this.orientation)
    this._shape.position.add(anchor)
  }

  /** center of box */
  public get center(): THREE.Vector3 {
    const position = new THREE.Vector3()
    this._shape.getWorldPosition(position)
    return position
  }

  /** orientation of box */
  public get orientation(): THREE.Quaternion {
    const quaternion = new THREE.Quaternion()
    this._shape.getWorldQuaternion(quaternion)
    return quaternion
  }

  /** scale of box */
  public get size(): THREE.Vector3 {
    const scale = new THREE.Vector3()
    this._shape.getWorldScale(scale)
    return scale
  }

  /**
   * bounds of box
   *
   * @param local
   */
  public bounds(local?: boolean): THREE.Box3 {
    const box = new THREE.Box3()
    if (local === null || local === undefined) {
      box.copy(this._shape.box.geometry.boundingBox)
      this._shape.updateMatrixWorld(true)
      box.applyMatrix4(this._shape.matrixWorld)
    } else {
      box.setFromCenterAndSize(this.center, this.size)
    }
    return box
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
   * Update params when state is updated
   *
   * @param state
   * @param itemIndex
   * @param labelId
   */
  public updateState(state: State, itemIndex: number, labelId: IdType): void {
    super.updateState(state, itemIndex, labelId)
    this._shape.color = this._color
    const label = state.task.items[itemIndex].labels[labelId]
    this._shape.updateState(
      state.task.items[itemIndex].shapes[label.shapes[0]],
      label.shapes[0]
    )
  }
}
