import * as THREE from "three"

import { policyFromString } from "../../common/track"
import { LabelTypeName, TrackPolicyType } from "../../const/common"
import { makeState } from "../../functional/states"
import { IdType, State } from "../../types/state"
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
}
