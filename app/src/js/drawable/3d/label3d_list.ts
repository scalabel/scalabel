import _ from 'lodash'
import * as THREE from 'three'
import { policyFromString } from '../../common/track/track'
import { LabelTypeName, TrackPolicyType } from '../../common/types'
import { makeState } from '../../functional/states'
import { State } from '../../functional/types'
import { Box3D } from './box3d'
import { TransformationControl } from './control/transformation_control'
import { Label3D, labelTypeFromString } from './label3d'
import { Plane3D } from './plane3d'

/**
 * Make a new drawable label based on the label type
 * @param {string} labelType: type of the new label
 */
export function makeDrawableLabel3D (
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
  private _labels: {[labelId: number]: Label3D}
  /** ThreeJS Object id to labels */
  private _raycastMap: {[id: number]: Label3D}
  /** Recorded state of last update */
  private _state: State
  /** Scene for rendering */
  private _scene: THREE.Scene
  /** selected label */
  private _selectedLabel: Label3D | null
  /** List of ThreeJS objects for raycasting */
  private _raycastableShapes: Readonly<Array<Readonly<THREE.Object3D>>>
  /** callbacks */
  private _callbacks: Array<() => void>
  /** New labels to be committed */
  private _updatedLabels: Set<Label3D>

  constructor () {
    this.control = new TransformationControl()
    this.control.layers.enableAll()
    this._labels = {}
    this._raycastMap = {}
    this._selectedLabel = null
    this._scene = new THREE.Scene()
    this._scene.add(this.control)
    this._raycastableShapes = []
    this._state = makeState()
    this._callbacks = []
    this._updatedLabels = new Set()
  }

  /**
   * Return scene object
   */
  public get scene (): THREE.Scene {
    return this._scene
  }

  /** Subscribe callback for drawable update */
  public subscribe (callback: () => void) {
    this._callbacks.push(callback)
  }

  /** Unsubscribe callback for drawable update */
  public unsubscribe (callback: () => void) {
    const index = this._callbacks.indexOf(callback)
    if (index >= 0) {
      this._callbacks.splice(index, 1)
    }
  }

  /** Get label by id */
  public get (id: number): Label3D | null {
    if (id in this._labels) {
      return this._labels[id]
    }
    return null
  }

  /** Call when any drawable has been updated */
  public onDrawableUpdate (): void {
    for (const callback of this._callbacks) {
      callback()
    }
  }

  /**
   * Get selected label
   */
  public get selectedLabel (): Label3D | null {
    return this._selectedLabel
  }

  /**
   * Get id's of selected labels
   */
  public get selectedLabelIds (): {[index: number]: number[]} {
    return this._state.user.select.labels
  }

  /** Get all policy types in config */
  public get policyTypes (): TrackPolicyType[] {
    return this._state.task.config.policyTypes.map(policyFromString)
  }

  /** Get all label types in config */
  public get labelTypes (): LabelTypeName[] {
    return this._state.task.config.labelTypes.map(labelTypeFromString)
  }

  /**
   * Get current policy type
   */
  public get currentPolicyType (): TrackPolicyType {
    return policyFromString(
      this._state.task.config.policyTypes[this._state.user.select.policyType]
    )
  }

  /**
   * Get current label type
   */
  public get currentLabelType (): LabelTypeName {
    return labelTypeFromString(
      this._state.task.config.labelTypes[this._state.user.select.labelType]
    )
  }

  /**
   * Get index of current category
   */
  public get currentCategory (): number {
    return this._state.user.select.category
  }

  /**
   * update labels from the state
   */
  public updateState (state: State): void {
    this._state = state

    const newLabels: {[labelId: number]: Label3D} = {}
    const newRaycastableShapes: Array<Readonly<THREE.Object3D>> = [this.control]
    const newRaycastMap: {[id: number]: Label3D} = {}
    const item = state.task.items[state.user.select.item]

    if (this._selectedLabel) {
      this._selectedLabel.selected = false
    }
    this._selectedLabel = null

    // Reset control & scene
    for (const key of Object.keys(this._labels)) {
      const id = Number(key)
      if (!(id in item.labels)) {
        for (const shape of Object.values(this._labels[id].shapes())) {
          this._scene.remove(shape)
        }
      }
    }

    // Update & create labels
    for (const key of Object.keys(item.labels)) {
      const id = Number(key)
      if (id in this._labels) {
        newLabels[id] = this._labels[id]
      } else {
        const newLabel = makeDrawableLabel3D(this, item.labels[id].type)
        if (newLabel) {
          newLabels[id] = newLabel
        }
      }
      if (newLabels[id]) {
        newLabels[id].updateState(
          state, state.user.select.item, id
        )
        for (const shape of Object.values(newLabels[id].shapes())) {
          newRaycastableShapes.push(shape)
          newRaycastMap[shape.id] = newLabels[id]
          this._scene.add(shape)
        }

        newLabels[id].selected = false

        // Disable all layers. Viewers will re-enable
        // for (const shape of newLabels[id].shapes()) {
        //   shape.layers.disableAll()
        // }
      }
    }

    // Assign parents
    for (const key of Object.keys(newLabels)) {
      const id = Number(key)
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
      if (selectedLabelIds.length === 1 &&
          selectedLabelIds[0] in this._labels) {
        this._selectedLabel = this._labels[select.labels[select.item][0]]
        this._selectedLabel.selected = true
        this.control.addLabel(this._selectedLabel)
      }
    }

    if (this.selectedLabel) {
      this.control.visible = true
    } else {
      this.control.visible = false
    }
  }

  /**
   * Get raycastable list
   */
  public get raycastableShapes (): Readonly<Array<Readonly<THREE.Object3D>>> {
    return this._raycastableShapes
  }

  /**
   * Get the label associated with the raycasted object 3d
   * @param obj
   */
  public getLabelFromRaycastedObject3D (
    obj: THREE.Object3D
  ): Label3D | null {
    while (obj.parent && !(obj.id in this._raycastMap)) {
      obj = obj.parent
    }

    if (obj.id in this._raycastMap) {
      return this._raycastMap[obj.id]
    }
    return null
  }

  /** Set active camera */
  public setActiveCamera (camera: THREE.Camera) {
    for (const label of Object.values(this._labels)) {
      label.activeCamera = camera
    }
    this.onDrawableUpdate()
  }

  /** Get uncommitted labels */
  public get updatedLabels (): Readonly<Set<Readonly<Label3D>>> {
    return this._updatedLabels
  }

  /** Push updated label to array */
  public addUpdatedLabel (label: Label3D) {
    this._updatedLabels.add(label)
  }

  /** Clear uncommitted label list */
  public clearUpdatedLabels () {
    this._updatedLabels.clear()
  }
}
