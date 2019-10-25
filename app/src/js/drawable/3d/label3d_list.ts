import _ from 'lodash'
import * as THREE from 'three'
import { selectLabel } from '../../action/select'
import Session from '../../common/session'
import { makeTrackPolicy, Track } from '../../common/track'
import { Key, LabelTypeName } from '../../common/types'
import { makeTrack } from '../../functional/states'
import { CubeType, State } from '../../functional/types'
import { Box3D } from './box3d'
import { TransformationControl } from './control/transformation_control'
import { Cube3D } from './cube3d'
import { Label3D } from './label3d'
import { Plane3D } from './plane3d'
/**
 * Make a new drawable label based on the label type
 * @param {string} labelType: type of the new label
 */
function makeDrawableLabel (
  labelType: string
): Label3D {
  switch (labelType) {
    case LabelTypeName.BOX_3D:
      return new Box3D()
    case LabelTypeName.PLANE_3D:
      return new Plane3D()
  }
  return new Box3D()
}

type Shape = Cube3D

/**
 * List of drawable labels
 * ViewController for the labels
 */
export class Label3DList {
  /** Scalabel id to labels */
  private _labels: {[labelId: number]: Label3D}
  /** ThreeJS Object id to labels */
  private _raycastMap: {[id: number]: Label3D}
  /** Recorded state of last update */
  private _state: State
  /** selected label */
  private _selectedLabel: Label3D | null
  /** highlighted label */
  private _highlightedLabel: Label3D | null
  /** whether mouse is down on the selected box */
  private _mouseDownOnSelection: boolean
  /** whether the selected label is changed */
  private _labelChanged: boolean
  /** List of ThreeJS objects for raycasting */
  private _raycastableShapes: Readonly<Array<Readonly<Shape>>>
  /** Plane visualization */
  private _plane?: Plane3D
  /** transformation control */
  private _control: TransformationControl
  /** The hashed list of keys currently down */
  private _keyDownMap: { [key: string]: boolean }

  constructor () {
    this._labels = {}
    this._raycastMap = {}
    this._selectedLabel = null
    this._highlightedLabel = null
    this._mouseDownOnSelection = false
    this._labelChanged = false
    this._raycastableShapes = []
    this._control = new TransformationControl()
    if (Session.itemType === 'image') {
      let planeExists = false
      const state = Session.getState()
      const itemIndex = state.user.select.item
      const item = state.task.items[itemIndex]
      for (const key of Object.keys(item.labels)) {
        if (item.labels[Number(key)].type === LabelTypeName.PLANE_3D) {
          planeExists = true
          break
        }
      }
      if (!planeExists) {
        this._plane = new Plane3D()
        this._plane.init(state)
      }
    }
    this._keyDownMap = {}
    this._state = Session.getState()
    this.updateState(this._state, this._state.user.select.item)
  }

  /**
   * Add labels to scene object
   * @param {THREE.Scene} scene: ThreeJS Scene Object
   */
  public render (scene: THREE.Scene, camera: THREE.Camera): void {
    for (const id of Object.keys(this._labels)) {
      this._labels[Number(id)].render(scene, camera)
    }
  }

  /**
   * update labels from the state
   */
  public updateState (state: State, itemIndex: number): void {
    this._state = state

    const newLabels: {[labelId: number]: Label3D} = {}
    const newRaycastableShapes: Array<Readonly<Shape>> = []
    const newRaycastMap: {[id: number]: Label3D} = {}
    const item = state.task.items[itemIndex]

    for (const key of Object.keys(this._labels)) {
      const id = Number(key)
      if (!(id in item.labels)) {
        this._labels[id].detachFromPlane()
        this._labels[id].detachControl(this._control)
      }
    }
    for (const key of Object.keys(item.labels)) {
      const id = Number(key)
      if (id in this._labels) {
        newLabels[id] = this._labels[id]
      } else {
        newLabels[id] =
          makeDrawableLabel(item.labels[id].type)
      }
      if (item.labels[id].type === LabelTypeName.PLANE_3D) {
        this._plane = newLabels[id] as Plane3D
      }
      newLabels[id].updateState(state, itemIndex, id)
      for (const shape of newLabels[id].shapes()) {
        newRaycastableShapes.push(shape as Shape)
        newRaycastMap[shape.id] = newLabels[id]
      }

      newLabels[id].setSelected(false)
    }

    // Attach shapes to plane
    for (const key of Object.keys(item.labels)) {
      const id = Number(key)
      if (item.labels[id].type === LabelTypeName.BOX_3D) {
        const shape = item.shapes[item.labels[id].shapes[0]].shape as CubeType
        if (shape.surfaceId >= 0) {
          newLabels[id].attachToPlane(newLabels[shape.surfaceId] as Plane3D)
        }
      }
    }

    this._raycastableShapes = newRaycastableShapes
    this._labels = newLabels
    this._raycastMap = newRaycastMap

    if (this._selectedLabel) {
      this._selectedLabel.setSelected(false)
      this._selectedLabel.detachControl(this._control)
    }
    this._selectedLabel = null

    const select = state.user.select
    if (select.item in select.labels) {
      const selectedLabelIds = select.labels[select.item]
      if (selectedLabelIds.length === 1 &&
          selectedLabelIds[0] in this._labels) {
        this._selectedLabel = this._labels[select.labels[select.item][0]]
        this._selectedLabel.attachControl(this._control)
      }
    }
  }

  /**
   * Handle double click, select label for editing
   * @returns true if consumed, false otherwise
   */
  public onDoubleClick (): boolean {
    this.selectHighlighted()
    return false
  }

  /**
   * Process mouse down action
   */
  public onMouseDown (x: number, y: number, camera: THREE.Camera): boolean {
    if (this._highlightedLabel === this._selectedLabel && this._selectedLabel) {
      this._mouseDownOnSelection = true
      if (this._control.attached()) {
        const consumed = this._control.onMouseDown(camera)
        if (consumed) {
          return false
        }
      }
    }

    if (this._highlightedLabel) {
      const consumed = this._highlightedLabel.onMouseDown(x, y, camera)
      if (consumed) {
        this._mouseDownOnSelection = true
        // Set current label as selected label
        this.selectHighlighted()
        return false
      }
    }

    if (this._plane) {
      const state = this._state
      const currentPolicyType =
        state.task.config.policyTypes[state.user.select.policyType]
      const newTrack = new Track()
      newTrack.updateState(
        makeTrack(-1), makeTrackPolicy(newTrack, currentPolicyType)
      )
      Session.tracks[-1] = newTrack

      const newLabel = new Box3D()
      newLabel.init(this._state, this._plane.labelId, true)
      this._labels[-1] = newLabel
      newLabel.attachToPlane(this._plane)
      if (this._highlightedLabel) {
        this._highlightedLabel.setHighlighted()
      }
      if (this._selectedLabel) {
        this._selectedLabel.setSelected(false)
        this._selectedLabel.detachControl(this._control)
      }
      this._highlightedLabel = newLabel
      this._selectedLabel = newLabel
      this._mouseDownOnSelection = true

      this._highlightedLabel.onMouseDown(x, y, camera)
    }

    return false
  }

  /**
   * Process mouse up action
   */
  public onMouseUp (): boolean {
    this._mouseDownOnSelection = false
    let consumed = false
    if (this._control.attached()) {
      consumed = this._control.onMouseUp()
    }
    if (!consumed && this._selectedLabel) {
      this._selectedLabel.onMouseUp()
    }
    if (this._labelChanged && this._selectedLabel !== null) {
      this._selectedLabel.commitLabel()
    }
    this._labelChanged = false
    return false
  }

  /**
   * Process mouse move action
   * @param x NDC
   * @param y NDC
   * @param camera
   */
  public onMouseMove (
    x: number,
    y: number,
    camera: THREE.Camera,
    raycastIntersection?: THREE.Intersection
  ): boolean {
    if (this._mouseDownOnSelection && this._selectedLabel) {
      this._labelChanged = true
      if (this._control.attached()) {
        const consumed = this._control.onMouseMove(x, y, camera)
        if (consumed) {
          return true
        }
      }
      this._selectedLabel.onMouseMove(x, y, camera)
      return true
    } else {
      this.highlight(raycastIntersection)
    }
    return false
  }

  /**
   * Handle keyboard events
   * @param {KeyboardEvent} e
   * @returns true if consumed, false otherwise
   */
  public onKeyDown (e: KeyboardEvent): boolean {
    const state = this._state
    switch (e.key) {
      case Key.SPACE:
        const currentPolicyType =
          state.task.config.policyTypes[state.user.select.policyType]
        const newTrack = new Track()
        newTrack.updateState(
          makeTrack(-1), makeTrackPolicy(newTrack, currentPolicyType)
        )
        Session.tracks[-1] = newTrack

        const label = new Box3D()
        const planeId = (this._plane) ? this._plane.labelId : -1
        label.init(state, planeId)
        return true
      case Key.ESCAPE:
      case Key.ENTER:
        Session.dispatch(selectLabel(state, -1, -1))
        return true
      case Key.P_UP:
      case Key.P_LOW:
        if (this._plane) {
          if (this._selectedLabel === this._plane) {
            Session.dispatch(selectLabel(state, -1, -1))
          } else {
            Session.dispatch(selectLabel(
              state,
              state.user.select.item,
              this._plane.labelId
            ))
          }
          return true
        }
        return false
      default:
        this._keyDownMap[e.key] = true
    }
    if (this._selectedLabel !== null) {
      return this._control.onKeyDown(e)
    }
    return false
  }

  /**
   * Handle key up
   */
  public onKeyUp (e: KeyboardEvent) {
    delete this._keyDownMap[e.key]
    return false
  }

  /**
   * Get raycastable list
   */
  public getRaycastableShapes (): Readonly<Array<Readonly<Shape>>> {
    return this._raycastableShapes
  }

  /**
   * Highlight label if ray from mouse is intersecting a label
   * @param object
   * @param point
   */
  private highlight (intersection?: THREE.Intersection) {
    if (this._highlightedLabel) {
      this._highlightedLabel.setHighlighted()
      this._control.setHighlighted()
    }
    this._highlightedLabel = null

    if (intersection) {
      let object = intersection.object
      while (object.parent && !(object.id in this._raycastMap)) {
        object = object.parent
      }

      if (object.id in this._raycastMap) {
        this._highlightedLabel = this._raycastMap[object.id]
        this._highlightedLabel.setHighlighted(intersection)
        if (this._highlightedLabel === this._selectedLabel) {
          this._control.setHighlighted(intersection)
        }
        return
      }
    }
  }

  /**
   * Whether a specific key is pressed down
   * @param {string} key - the key to check
   * @return {boolean}
   */
  private isKeyDown (key: string): boolean {
    return this._keyDownMap[key]
  }

  /**
   * Select highlighted label
   */
  private selectHighlighted () {
    if (this._highlightedLabel !== null) {
      if ((this.isKeyDown(Key.CONTROL) || this.isKeyDown(Key.META)) &&
          this._highlightedLabel !== this._selectedLabel) {
        Session.dispatch(selectLabel(
          this._state,
          this._state.user.select.item,
          this._highlightedLabel.labelId,
          this._highlightedLabel.category[0],
          this._highlightedLabel.attributes,
          true
        ))
      } else {
        Session.dispatch(selectLabel(
          this._state,
          this._state.user.select.item,
          this._highlightedLabel.labelId,
          this._highlightedLabel.category[0],
          this._highlightedLabel.attributes
        ))
      }
      return true
    }
  }
}
