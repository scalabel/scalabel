import _ from 'lodash'
import * as THREE from 'three'
import { selectLabel, selectLabel3dType } from '../../action/select'
import Session from '../../common/session'
import { DataType, Key, LabelTypeName, ViewerConfigTypeName } from '../../common/types'
import { getCurrentViewerConfig } from '../../functional/state_util'
import { makePointCloudViewerConfig, makeSensor } from '../../functional/states'
import { PointCloudViewerConfigType, SensorType, State, ViewerConfigType } from '../../functional/types'
import { Vector2D } from '../../math/vector2d'
import { Vector3D } from '../../math/vector3d'
import { commitLabels } from '../states'
import { makeDrawableLabel3D } from './label3d_list'

const MOUSE_MOVE_THRESHOLD = .01

/**
 * Handles user interactions with labels
 */
export class Label3DHandler {
  /** whether mouse is down on the selected box */
  private _mouseDown: boolean
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
  /** Current mouse position */
  private _mousePosition: Vector2D
  /** Whether mouse moved */
  // private _mouseMoved: boolean

  constructor (camera: THREE.Camera) {
    this._mouseDown = false
    this._keyDownMap = {}
    this._viewerConfig = makePointCloudViewerConfig(-1)
    this._selectedItemIndex = -1
    this._sensorIds = []
    this._sensor = makeSensor(-1, '', DataType.POINT_CLOUD)
    this._camera = camera
    this._mousePosition = new Vector2D()
    // this._mouseMoved = false
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
    return Session.label3dList.highlightedLabel !== null
  }

  /**
   * Process mouse down action
   */
  public onMouseDown (): boolean {
    this._mouseDown = true

    return false
  }

  /**
   * Process mouse up action
   */
  public onMouseUp (_x: number, _y: number): boolean {
    // if (
    //   !this._mouseMoved &&
    //   Session.label3dList.highlightedLabel
    // ) {
    //   Session.label3dList.highlightedLabel.click(x, y)
    // }
    commitLabels(
        [...Session.label3dList.updatedLabels.values()],
        [...Session.label3dList.updatedShapes.values()]
      )
    Session.label3dList.clearUpdated()
    // Set current label as selected label
    if (
      this._mouseDown &&
      Session.label3dList.highlightedLabel !== Session.label3dList.selectedLabel
    ) {
      this.selectHighlighted()
    }
    this._mouseDown = false
    // this._mouseMoved = false
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
    if (this._mouseDown) {
      const dx = x - this._mousePosition.x
      const dy = y - this._mousePosition.y
      if ((dx * dx + dy * dy) > MOUSE_MOVE_THRESHOLD * MOUSE_MOVE_THRESHOLD) {
        this._mousePosition.set(x, y)
        // this._mouseMoved = true
        if (
          Session.label3dList.control.visible &&
          Session.label3dList.control.highlighted
        ) {
          const consumed = Session.label3dList.control.drag(
            dx, dy, this._camera
          )
          if (consumed) {
            return true
          }
        }
        if (Session.label3dList.highlightedLabel) {
          const consumed = Session.label3dList.highlightedLabel.drag(
            dx, dy, this._camera
          )
          if (consumed) {
            Session.label3dList.selectedLabel =
              Session.label3dList.highlightedLabel
            Session.label3dList.highlightedLabel.editing = true
            this.selectHighlighted()
          }
        }
      }
      return (
        (Session.label3dList.control.highlighted &&
          Session.label3dList.control.visible) ||
        (Session.label3dList.selectedLabel !== null &&
        Session.label3dList.selectedLabel ===
          Session.label3dList.highlightedLabel
        )
      )
    } else {
      this.highlight(raycastIntersection)
      this._mousePosition.set(x, y)
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
          commitLabels(
            [...Session.label3dList.updatedLabels.values()],
            [...Session.label3dList.updatedShapes.values()]
          )
          Session.label3dList.clearUpdated()
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
            commitLabels(
              [...Session.label3dList.updatedLabels.values()],
              [...Session.label3dList.updatedShapes.values()]
            )
            Session.label3dList.clearUpdated()
          }
        }
        break
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
    if (Session.label3dList.highlightedLabel) {
      Session.label3dList.highlightedLabel.setHighlighted()
    }
    Session.label3dList.highlightedLabel = null

    if (intersection) {
      const object = intersection.object
      const label = Session.label3dList.getLabelFromRaycastedObject3D(object)

      if (label) {
        label.setHighlighted(intersection)
        Session.label3dList.highlightedLabel = label
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
    if (Session.label3dList.highlightedLabel !== null) {
      if ((this.isKeyDown(Key.CONTROL) || this.isKeyDown(Key.META)) &&
          Session.label3dList.highlightedLabel !==
          Session.label3dList.selectedLabel) {
        Session.dispatch(selectLabel(
          Session.label3dList.selectedLabelIds,
          this._selectedItemIndex,
          Session.label3dList.highlightedLabel.labelId,
          Session.label3dList.highlightedLabel.category[0],
          Session.label3dList.highlightedLabel.attributes,
          true
        ))
      } else if (
        Session.label3dList.highlightedLabel !==
          Session.label3dList.selectedLabel
      ) {
        Session.dispatch(selectLabel(
          Session.label3dList.selectedLabelIds,
          this._selectedItemIndex,
          Session.label3dList.highlightedLabel.labelId,
          Session.label3dList.highlightedLabel.category[0],
          Session.label3dList.highlightedLabel.attributes
        ))
      }
    }
  }
}
