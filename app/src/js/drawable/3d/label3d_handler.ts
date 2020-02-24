import _ from 'lodash'
import * as THREE from 'three'
import { selectLabel, selectLabel3dType } from '../../action/select'
import Session from '../../common/session'
import { DataType, Key, LabelTypeName, ViewerConfigTypeName } from '../../common/types'
import { getCurrentViewerConfig } from '../../functional/state_util'
import { makePointCloudViewerConfig, makeSensor } from '../../functional/states'
import { PointCloudViewerConfigType, SensorType, State, ViewerConfigType } from '../../functional/types'
import { Vector3D } from '../../math/vector3d'
import { commitLabels } from '../states'
import { Label3D } from './label3d'
import { makeDrawableLabel3D } from './label3d_list'

/**
 * Handles user interactions with labels
 */
export class Label3DHandler {
  /** highlighted label */
  private _highlightedLabel: Label3D | null
  /** whether mouse is down on the selected box */
  private _mouseDownOnSelection: boolean
  /** The hashed list of keys currently down */
  private _keyDownMap: { [key: string]: boolean }
  /** viewer config */
  private _viewerConfig: ViewerConfigType
  /** index of selected item */
  private _selectedItemIndex: number
  /** Sensors that are currently in use */
  private _sensorIds: number[]
  /** Current sensor */
  private _sensor: SensorType
  /** camera */
  private _camera: THREE.Camera
  /** timer for throttling committing effects of key presses to state */
  private _keyThrottleTimer: ReturnType<typeof setTimeout> | null

  constructor (camera: THREE.Camera) {
    this._highlightedLabel = null
    this._mouseDownOnSelection = false
    this._keyDownMap = {}
    this._viewerConfig = makePointCloudViewerConfig(-1)
    this._selectedItemIndex = -1
    this._sensorIds = []
    this._sensor = makeSensor(-1, '', DataType.POINT_CLOUD)
    this._camera = camera
    this._keyThrottleTimer = null
  }

  /** Set camera */
  public set camera (camera: THREE.Camera) {
    this._camera = camera
  }

  /**
   * Update handler params when state updated
   * @param itemIndex
   * @param viewerId
   */
  public updateState (state: State, itemIndex: number, viewerId: number) {
    this._selectedItemIndex = itemIndex
    this._viewerConfig =
      getCurrentViewerConfig(state, viewerId)
    this._sensorIds = Object.keys(state.task.sensors).map((key) => Number(key))
    if (this._viewerConfig.sensor in state.task.sensors) {
      this._sensor = state.task.sensors[this._viewerConfig.sensor]
    }
  }

  /**
   * Handle double click, select label for editing
   * @returns true if consumed, false otherwise
   */
  public onDoubleClick (): boolean {
    this.selectHighlighted()
    return this._highlightedLabel !== null
  }

  /**
   * Process mouse down action
   */
  public onMouseDown (x: number, y: number): boolean {
    if (Session.label3dList.control.highlighted) {
      this._mouseDownOnSelection = true
      Session.label3dList.control.onMouseDown(this._camera)
      return false
    }

    if (this._highlightedLabel) {
      const consumed = this._highlightedLabel.onMouseDown(x, y, this._camera)
      if (consumed) {
        this._mouseDownOnSelection = true
        return false
      }
    }

    return false
  }

  /**
   * Process mouse up action
   */
  public onMouseUp (): boolean {
    let consumed = false
    if (Session.label3dList.control.visible) {
      consumed = Session.label3dList.control.onMouseUp()
    }
    if (!consumed && Session.label3dList.selectedLabel) {
      Session.label3dList.selectedLabel.onMouseUp()
    }
    commitLabels([...Session.label3dList.updatedLabels.values()])
    Session.label3dList.clearUpdatedLabels()
    // Set current label as selected label
    if (
      this._mouseDownOnSelection &&
      this._highlightedLabel !== Session.label3dList.selectedLabel
    ) {
      this.selectHighlighted()
    }
    this._mouseDownOnSelection = false
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
    raycastIntersection?: THREE.Intersection
  ): boolean {
    if (
      this._mouseDownOnSelection &&
      Session.label3dList.selectedLabel &&
      Session.label3dList.control.visible &&
      Session.label3dList.control.highlighted
    ) {
      const consumed = Session.label3dList.control.onMouseMove(
        x, y, this._camera
      )
      if (consumed) {
        return true
      }
    }
    if (this._mouseDownOnSelection && this._highlightedLabel) {
      this._highlightedLabel.onMouseMove(x, y, this._camera)
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
        const label = makeDrawableLabel3D(
          Session.label3dList, Session.label3dList.currentLabelType
        )
        if (label) {
          const center = new Vector3D()
          switch (this._viewerConfig.type) {
            case ViewerConfigTypeName.POINT_CLOUD:
              center.fromState(
                (this._viewerConfig as PointCloudViewerConfigType).target
              )
              break
            case ViewerConfigTypeName.IMAGE_3D:
              if (this._sensor.extrinsics) {
                const worldDirection = new THREE.Vector3()
                this._camera.getWorldDirection(worldDirection)
                worldDirection.normalize()
                worldDirection.multiplyScalar(5)
                center.fromState(this._sensor.extrinsics.translation)
                center.add((new Vector3D()).fromThree(worldDirection))
              }
          }
          label.init(
            this._selectedItemIndex,
            Session.label3dList.currentCategory,
            center,
            this._sensorIds
          )
          Session.label3dList.addUpdatedLabel(label)
          commitLabels([...Session.label3dList.updatedLabels.values()])
          Session.label3dList.clearUpdatedLabels()
          return true
        }
        return false
      case Key.ESCAPE:
      case Key.ENTER:
        Session.dispatch(selectLabel(
          Session.label3dList.selectedLabelIds, -1, -1
        ))
        return true
      case Key.P_UP:
      case Key.P_LOW:
        Session.dispatch(selectLabel3dType(
          LabelTypeName.PLANE_3D
        ))
        return true
      case Key.B_UP:
      case Key.B_LOW:
        Session.dispatch(selectLabel3dType(
          LabelTypeName.BOX_3D
        ))
        return true
      case Key.T_UP:
      case Key.T_LOW:
        if (this.isKeyDown(Key.SHIFT)) {
          if (Session.label3dList.selectedLabel) {
            const target =
              (this._viewerConfig as PointCloudViewerConfigType).target
            Session.label3dList.selectedLabel.move(
              (new Vector3D()).fromState(target).toThree()
            )
            commitLabels([...Session.label3dList.updatedLabels.values()])
            Session.label3dList.clearUpdatedLabels()
          }
        }
        break
    }
    if (Session.label3dList.selectedLabel !== null && !this.isKeyDown(e.key)) {
      const consumed = Session.label3dList.control.onKeyDown(e, this._camera)
      if (consumed) {
        this._keyDownMap[e.key] = true
        if (this._keyThrottleTimer) {
          window.clearTimeout(this._keyThrottleTimer)
        }
        this.timedRepeat(
          () => {
            Session.label3dList.control.onKeyDown(e, this._camera)
            Session.label3dList.onDrawableUpdate()
          },
          e.key
        )
        return true
      }
    }
    this._keyDownMap[e.key] = true
    return false
  }

  /**
   * Handle key up
   */
  public onKeyUp (e: KeyboardEvent) {
    delete this._keyDownMap[e.key]
    if (Session.label3dList.selectedLabel !== null) {
      Session.label3dList.control.onKeyUp(e)
    }
    this._keyThrottleTimer = setTimeout(() => {
      if (Session.label3dList.updatedLabels.size > 0) {
        commitLabels([...Session.label3dList.updatedLabels.values()])
        Session.label3dList.clearUpdatedLabels()
      }
    }, 200)
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
    }
    this._highlightedLabel = null

    if (intersection) {
      const object = intersection.object
      const label = Session.label3dList.getLabelFromRaycastedObject3D(object)

      if (label) {
        label.setHighlighted(intersection)
        this._highlightedLabel = label
      }
    }
    Session.label3dList.control.setHighlighted(intersection)
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

  /** Repeat function as long as key is held down */
  private timedRepeat (fn: () => void, key: string, timeout: number = 30) {
    if (this.isKeyDown(key)) {
      fn()
      setTimeout(() => this.timedRepeat(fn, key, timeout), timeout)
    }
  }
}
