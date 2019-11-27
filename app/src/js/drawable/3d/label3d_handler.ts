import _ from 'lodash'
import * as THREE from 'three'
import { addLabel, changeShapes } from '../../action/common'
import { selectLabel } from '../../action/select'
import Session from '../../common/session'
import { makeTrackPolicy, Track } from '../../common/track'
import { Key } from '../../common/types'
import { getCurrentViewerConfig } from '../../functional/state_util'
import { makePointCloudViewerConfig, makeTrack } from '../../functional/states'
import { PointCloudViewerConfigType, State } from '../../functional/types'
import { Box3D } from './box3d'
import { Label3D } from './label3d'
import { Plane3D } from './plane3d'

/**
 * Handles user interactions with labels
 */
export class Label3DHandler {
  /** highlighted label */
  private _highlightedLabel: Label3D | null
  /** whether mouse is down on the selected box */
  private _mouseDownOnSelection: boolean
  /** whether the selected label is changed */
  private _labelChanged: boolean
  /** Plane visualization */
  private _plane?: Plane3D
  /** The hashed list of keys currently down */
  private _keyDownMap: { [key: string]: boolean }
  /** viewer config */
  private _viewerConfig: PointCloudViewerConfigType
  /** index of selected item */
  private _selectedItemIndex: number

  constructor () {
    this._highlightedLabel = null
    this._mouseDownOnSelection = false
    this._labelChanged = false
    // if (Session.itemType === 'image') {
    //   let planeExists = false
    //   const state = Session.getState()
    //   const itemIndex = state.user.select.item
    //   const item = state.task.items[itemIndex]
    //   for (const key of Object.keys(item.labels)) {
    //     if (item.labels[Number(key)].type === LabelTypeName.PLANE_3D) {
    //       planeExists = true
    //       break
    //     }
    //   }
    //   if (!planeExists) {
    //     this._plane = new Plane3D()
    //     this._plane.init(itemIndex, -1)
    //   }
    // }
    this._keyDownMap = {}
    this._viewerConfig = makePointCloudViewerConfig(-1)
    this._selectedItemIndex = -1
  }

  /**
   * Update handler params when state updated
   * @param itemIndex
   * @param viewerId
   */
  public updateState (state: State, itemIndex: number, viewerId: number) {
    this._selectedItemIndex = itemIndex
    this._viewerConfig =
      getCurrentViewerConfig(state, viewerId) as PointCloudViewerConfigType
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
    if (this._highlightedLabel &&
        this._highlightedLabel === Session.label3dList.selectedLabel) {
      this._mouseDownOnSelection = true
      if (Session.label3dList.control.attached()) {
        const consumed = Session.label3dList.control.onMouseDown(camera)
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

    return false
  }

  /**
   * Process mouse up action
   */
  public onMouseUp (): boolean {
    this._mouseDownOnSelection = false
    let consumed = false
    if (Session.label3dList.control.attached()) {
      consumed = Session.label3dList.control.onMouseUp()
    }
    if (!consumed && Session.label3dList.selectedLabel) {
      Session.label3dList.selectedLabel.onMouseUp()
    }
    if (this._labelChanged && Session.label3dList.selectedLabel) {
      const [ids,,shapes] = Session.label3dList.selectedLabel.shapeObjects()
      Session.dispatch(changeShapes(
        this._selectedItemIndex,
        ids,
        shapes
      ))
      const label = Session.label3dList.selectedLabel.label
      if (Session.tracking && label.track in Session.tracks) {
        Session.tracks[label.track].onLabelUpdated(
          this._selectedItemIndex,
          shapes
        )
      }
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
    if (this._mouseDownOnSelection && Session.label3dList.selectedLabel) {
      this._labelChanged = true
      if (Session.label3dList.control.attached()) {
        const consumed = Session.label3dList.control.onMouseMove(x, y, camera)
        if (consumed) {
          return true
        }
      }
      Session.label3dList.selectedLabel.onMouseMove(x, y, camera)
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
    switch (e.key) {
      case Key.SPACE:
        const box = new Box3D()
        const planeId = (this._plane) ? this._plane.labelId : -1
        box.init(
          this._selectedItemIndex,
          Session.label3dList.currentCategory,
          this._viewerConfig,
          planeId
        )
        if (box.label) {
          const [, types, shapes] = box.shapeObjects()
          if (Session.tracking) {
            const newTrack = new Track()
            newTrack.updateState(
              makeTrack(-1),
              makeTrackPolicy(newTrack, Session.label3dList.policyType)
            )
            newTrack.onLabelCreated(this._selectedItemIndex, box, [-1])
          } else {
            Session.dispatch(addLabel(
              this._selectedItemIndex,
              box.label,
              types,
              shapes
            ))
          }
        }
        return true
      case Key.ESCAPE:
      case Key.ENTER:
        Session.dispatch(selectLabel(
          Session.label3dList.selectedLabelIds, -1, -1
        ))
        return true
      case Key.P_UP:
      case Key.P_LOW:
        if (this._plane) {
          if (Session.label3dList.selectedLabel === this._plane) {
            Session.dispatch(selectLabel(
              Session.label3dList.selectedLabelIds, -1, -1
            ))
          } else {
            Session.dispatch(selectLabel(
              Session.label3dList.selectedLabelIds,
              this._selectedItemIndex,
              this._plane.labelId
            ))
          }
          return true
        }
        return false
      default:
        this._keyDownMap[e.key] = true
    }
    if (Session.label3dList.selectedLabel !== null) {
      return Session.label3dList.control.onKeyDown(e)
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
   * Highlight label if ray from mouse is intersecting a label
   * @param object
   * @param point
   */
  private highlight (intersection?: THREE.Intersection) {
    if (this._highlightedLabel) {
      this._highlightedLabel.setHighlighted()
      Session.label3dList.control.setHighlighted()
    }
    this._highlightedLabel = null

    if (intersection) {
      const object = intersection.object
      const label = Session.label3dList.getLabelFromRaycastedObject3D(object)

      if (label) {
        label.setHighlighted(intersection)
        this._highlightedLabel = label
        if (this._highlightedLabel === Session.label3dList.selectedLabel) {
          Session.label3dList.control.setHighlighted(intersection)
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
          this._highlightedLabel !== Session.label3dList.selectedLabel) {
        Session.dispatch(selectLabel(
          Session.label3dList.selectedLabelIds,
          this._selectedItemIndex,
          this._highlightedLabel.labelId,
          this._highlightedLabel.category[0],
          this._highlightedLabel.attributes,
          true
        ))
      } else {
        Session.dispatch(selectLabel(
          Session.label3dList.selectedLabelIds,
          this._selectedItemIndex,
          this._highlightedLabel.labelId,
          this._highlightedLabel.category[0],
          this._highlightedLabel.attributes
        ))
      }
    }
  }
}
