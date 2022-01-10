import _ from "lodash"
import * as THREE from "three"

import { policyFromString } from "../../common/track"
import { LabelTypeName, TrackPolicyType } from "../../const/common"
import { makeState } from "../../functional/states"
import { Vector3D } from "../../math/vector3d"
import { IdType, ShapeType, State } from "../../types/state"
import { commitLabels } from "../states"
import { Box3D } from "./box3d"
import { TransformationControl } from "./control/transformation_control"
import { Label3D, labelTypeFromString } from "./label3d"
import { Plane3D } from "./plane3d"

/**
 * Make a new drawable label based on the label type
 *
 * @param {string} labelType: type of the new label
 * @param labelList
 * @param labelType
 */
export function makeDrawableLabel3D(
  labelList: Label3DList,
  labelType: string
): Label3D | null {
  switch (labelType) {
    case LabelTypeName.BOX_3D:
      return new Box3D(labelList)
    case LabelTypeName.PLANE_3D:
      return new Plane3D(labelList)
  }
  return null
}
/**
 * Commit new plane 3D label to state
 *
 * @param labelList
 * @param itemIndex
 * @param category
 * @param sensors
 * @param center
 * @param orientation
 */
export function createPlaneLabel(
  labelList: Label3DList,
  itemIndex: number,
  category: number,
  center?: Vector3D,
  orientation?: Vector3D,
  sensors?: number[]
): void {
  const plane = new Plane3D(labelList)
  plane.init(itemIndex, category, center, orientation, sensors)
  commitLabels([plane], false)
}
/**
 * Commit new Box3D label to state
 *
 * @param labelList
 * @param itemIndex
 * @param sensors
 * @param category
 * @param center
 * @param dimension
 * @param orientation
 * @param tracking
 */
export function createBox3dLabel(
  labelList: Label3DList,
  itemIndex: number,
  sensors: number[],
  category: number,
  center: Vector3D,
  dimension: Vector3D,
  orientation: Vector3D,
  tracking: boolean
): void {
  const box = new Box3D(labelList)
  box.init(itemIndex, category, center, orientation, dimension, sensors)
  commitLabels([box], tracking)
}

/**
 * List of drawable labels
 */
export class Label3DList {
  /** transformation control */
  public control: TransformationControl

  /** Scalabel id to labels */
  private _labels: { [labelId: string]: Label3D }
  /** ThreeJS Object id to labels */
  private _raycastMap: { [id: number]: Label3D }
  /** Recorded state of last update */
  private _state: State
  /** Scene for rendering */
  private _scene: THREE.Scene
  /** selected label */
  private _selectedLabel: Label3D | null
  /** List of ThreeJS objects for raycasting */
  private _raycastableShapes: Readonly<Array<Readonly<THREE.Object3D>>>
  /** callbacks */
  private readonly _callbacks: Array<() => void>
  /** New labels to be committed */
  private readonly _updatedLabels: Set<Label3D>
  /** History shapes */
  private readonly _histShapes: ShapeType[]

  /**
   * Constructor
   */
  constructor() {
    this.control = new TransformationControl()
    this.control.layers.enableAll()
    this._labels = {}
    this._raycastMap = {}
    this._selectedLabel = null
    this._scene = new THREE.Scene()
    this._scene.add(this.control)
    this._scene.layers.enableAll()
    this._raycastableShapes = []
    this._state = makeState()
    this._callbacks = []
    this._updatedLabels = new Set()
    this._histShapes = []
  }

  /**
   * Return scene object
   */
  public get scene(): THREE.Scene {
    return this._scene
  }

  /**
   * Subscribe callback for drawable update
   *
   * @param callback
   */
  public subscribe(callback: () => void): void {
    this._callbacks.push(callback)
  }

  /**
   * Unsubscribe callback for drawable update
   *
   * @param callback
   */
  public unsubscribe(callback: () => void): void {
    const index = this._callbacks.indexOf(callback)
    if (index >= 0) {
      this._callbacks.splice(index, 1)
    }
  }

  /**
   * Get label by id
   *
   * @param id
   */
  public get(id: string): Label3D | null {
    if (id in this._labels) {
      return this._labels[id]
    }
    return null
  }

  /** Get all labels */
  public labels(): Readonly<Array<Readonly<Label3D>>> {
    return Object.values(this._labels)
  }

  /** Call when any drawable has been updated */
  public onDrawableUpdate(): void {
    for (const callback of this._callbacks) {
      callback()
    }
  }

  /**
   * Get selected label
   */
  public get selectedLabel(): Label3D | null {
    return this._selectedLabel
  }

  /**
   * Get id's of selected labels
   */
  public get selectedLabelIds(): { [index: number]: IdType[] } {
    return this._state.user.select.labels
  }

  /** Get all policy types in config */
  public get policyTypes(): TrackPolicyType[] {
    return this._state.task.config.policyTypes.map(policyFromString)
  }

  /** Get all label types in config */
  public get labelTypes(): LabelTypeName[] {
    return this._state.task.config.labelTypes.map(labelTypeFromString)
  }

  /**
   * Get current policy type
   */
  public get currentPolicyType(): TrackPolicyType {
    return policyFromString(
      this._state.task.config.policyTypes[this._state.user.select.policyType]
    )
  }

  /**
   * Get current label type
   */
  public get currentLabelType(): LabelTypeName {
    return labelTypeFromString(
      this._state.task.config.labelTypes[this._state.user.select.labelType]
    )
  }

  /**
   * Get index of current category
   */
  public get currentCategory(): number {
    return this._state.user.select.category
  }

  /**
   * update labels from the state
   *
   * @param state
   */
  public updateState(state: State): void {
    this._state = state

    const newLabels: { [labelId: string]: Label3D } = {}
    const newRaycastableShapes: Array<Readonly<THREE.Object3D>> = [this.control]
    const newRaycastMap: { [id: string]: Label3D } = {}
    const item = state.task.items[state.user.select.item]

    if (this._selectedLabel !== null) {
      this._selectedLabel.selected = false
    }
    this._selectedLabel = null

    // Reset control & scene
    this._scene.children = [this.control]

    // Update & create labels
    for (const key of Object.keys(item.labels)) {
      const id = key
      if (id in this._labels) {
        newLabels[id] = this._labels[id]
      } else {
        const newLabel = makeDrawableLabel3D(this, item.labels[id].type)
        if (newLabel !== null) {
          newLabels[id] = newLabel
        }
      }
      if (newLabels[id] !== undefined) {
        newLabels[id].updateState(state, state.user.select.item, id)
        for (const shape of Object.values(newLabels[id].internalShapes())) {
          newRaycastableShapes.push(shape)
          newRaycastMap[shape.id] = newLabels[id]
          this._scene.add(shape)
        }

        newLabels[id].selected = false
      }
    }

    // Assign parents
    for (const key of Object.keys(newLabels)) {
      const id = key
      if (item.labels[id].parent in newLabels) {
        newLabels[item.labels[id].parent].addChild(newLabels[id])
      }
    }

    this._raycastableShapes = newRaycastableShapes
    this._labels = newLabels
    this._raycastMap = newRaycastMap

    this.control.clearLabels()
    const select = state.user.select
    if (select.item in select.labels) {
      const selectedLabelIds = select.labels[select.item]
      if (
        selectedLabelIds.length === 1 &&
        selectedLabelIds[0] in this._labels
      ) {
        this._selectedLabel = this._labels[select.labels[select.item][0]]
        this._selectedLabel.selected = true
        this.control.addLabel(this._selectedLabel)
      }
    }

    if (this.selectedLabel !== null) {
      this.control.visible = true
    } else {
      this.control.visible = false
    }
  }

  /**
   * Get raycastable list
   */
  public get raycastableShapes(): Readonly<Array<Readonly<THREE.Object3D>>> {
    return this._raycastableShapes
  }

  /**
   * Get the label associated with the raycasted object 3d
   *
   * @param obj
   */
  public getLabelFromRaycastedObject3D(obj: THREE.Object3D): Label3D | null {
    while (obj.parent !== null && !(obj.id in this._raycastMap)) {
      obj = obj.parent
    }
    if (obj.id in this._raycastMap) {
      return this._raycastMap[obj.id]
    }
    return null
  }

  /**
   * Set active camera
   *
   * @param camera
   */
  public setActiveCamera(camera: THREE.Camera): void {
    for (const label of Object.values(this._labels)) {
      label.activeCamera = camera
    }
    this.onDrawableUpdate()
  }

  /** Get uncommitted labels */
  public get updatedLabels(): Readonly<Set<Readonly<Label3D>>> {
    return this._updatedLabels
  }

  /**
   * Push updated label to array
   *
   * @param label
   */
  public addUpdatedLabel(label: Label3D): void {
    this._updatedLabels.add(label)
  }

  /** Clear uncommitted label list */
  public clearUpdatedLabels(): void {
    this._updatedLabels.clear()
  }

  /**
   * push new shape to the history shape list
   *
   * @param shape
   */
  public addShapeToHistShapes(shape: ShapeType): void {
    if (
      this._histShapes.length > 0 &&
      shape.id !== this._histShapes[this._histShapes.length - 1].id
    ) {
      this.clearHistShapes()
    }
    this._histShapes.push(shape)
  }

  /**
   * reset label to previous status
   */
  public getLastShape(): ShapeType | null {
    if (this._histShapes.length > 0) {
      const shape = this._histShapes[this._histShapes.length - 1]
      this._histShapes.splice(this._histShapes.length - 1, 1)
      return shape
    }
    return null
  }

  /**
   * clear the history label list
   */
  public clearHistShapes(): void {
    this._histShapes.splice(0, this._histShapes.length)
  }

  /**
   * return current selected object's shape
   */
  public getCurrentShape(): ShapeType {
    const state = this._state
    const item = state.task.items[state.user.select.item]
    const label =
      item.labels[state.user.select.labels[state.user.select.item][0]]
    return _.cloneDeep(item.shapes[label.shapes[0]])
  }

  /**
   * Get ground plane for item
   *
   * @param itemIndex
   */
  public getItemGroundPlane(itemIndex: number): Plane3D | null {
    const labels = this.labels()
    const itemPlanes = labels.filter(
      (l) => l.item === itemIndex && l.label.type === LabelTypeName.PLANE_3D
    )
    return (itemPlanes[0] as Plane3D) ?? null
  }
}
