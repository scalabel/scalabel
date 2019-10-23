import _ from 'lodash'
import * as THREE from 'three'

import { addBox3dLabel } from '../../action/box3d'
import { changeLabelProps, changeLabelShape } from '../../action/common'
import Session from '../../common/session'

import { getCurrentPointCloudViewerConfig } from '../../functional/state_util'
import { makeLabel } from '../../functional/states'
import {
  CubeType, PointCloudViewerConfigType, ShapeType, State
} from '../../functional/types'

import { Vector3D } from '../../math/vector3d'

import { LabelTypeName } from '../../common/types'
import { TransformationControl } from './control/transformation_control'
import { Cube3D } from './cube3d'
import { Label3D } from './label3d'
import { Plane3D } from './plane3d'

/**
 * Box3d Label
 */
export class Box3D extends Label3D {
  /** ThreeJS object for rendering shape */
  private _shape: Cube3D
  /** Whether this is temporary */
  private _temporary: boolean

  constructor () {
    super()
    this._shape = new Cube3D(this._index)
    this._temporary = false
  }

  /**
   * Initialize label
   * @param {State} state
   */
  public init (state: State, surfaceId?: number, temporary?: boolean): void {
    const itemIndex = state.user.select.item
    this._order = state.task.status.maxOrder + 1
    this._label = makeLabel({
      type: LabelTypeName.BOX_3D, id: -1, item: itemIndex,
      category: [state.user.select.category],
      order: this._order
    })
    this._labelId = -1
    const viewerConfig: PointCloudViewerConfigType =
      getCurrentPointCloudViewerConfig(state)
    const center = (new Vector3D()).fromObject(viewerConfig.target)
    if (surfaceId && surfaceId >= 0) {
      center.z = 0.5
      this._shape.setSurfaceId(surfaceId)
    }
    this._shape.setCenter(center)

    if (temporary && surfaceId && surfaceId >= 0) {
      this._temporary = temporary
    } else if (!temporary) {
      this.addToState()
    }
  }

  /**
   * Override set selected
   * @param s
   */
  public setSelected (s: boolean) {
    super.setSelected(s)
    this._shape.setSelected(s)
  }

  /** Attach label to plane */
  public attachToPlane (plane: Plane3D) {
    super.attachToPlane(plane)
    this._shape.attachToPlane(plane)
  }

  /** Attach label to plane */
  public detachFromPlane () {
    this._shape.detachFromPlane()
    super.detachFromPlane()
  }

  /**
   * Attach control
   */
  public attachControl (control: TransformationControl) {
    this._shape.setControl(control, true)
  }

  /**
   * Attach control
   */
  public detachControl (control: TransformationControl) {
    this._shape.setControl(control, false)
  }

  /**
   * Return a list of the shape for inspection and testing
   */
  public shapes (): Array<Readonly<Cube3D>> {
    return [this._shape]
  }

  /**
   * Modify ThreeJS objects to draw label
   * @param {THREE.Scene} scene: ThreeJS Scene Object
   */
  public render (scene: THREE.Scene, camera: THREE.Camera): void {
    this._shape.render(scene, camera)
  }

  /**
   * Update Box3D internal parameters based on new state
   * @param state
   * @param itemIndex
   * @param labelId
   */
  public updateState (
    state: State, itemIndex: number, labelId: number): void {
    super.updateState(state, itemIndex, labelId)
    this._shape.setId(labelId)
    this._shape.setColor(this._color)
  }

  /**
   * Expand the primitive shapes to drawable shapes
   * @param {ShapeType[]} shapes
   */
  public updateShapes (shapes: ShapeType[]): void {
    const newShape = shapes[0] as CubeType
    this._shape.setCenter((new Vector3D()).fromObject(newShape.center))
    this._shape.setSize((new Vector3D()).fromObject(newShape.size))
    this._shape.setOrientation(
      (new Vector3D()).fromObject(newShape.orientation)
    )
    this._shape.setSurfaceId(newShape.surfaceId)
    if (newShape.surfaceId < 0) {
      this.detachFromPlane()
    }
  }

  /**
   * move anchor to next corner
   */
  public incrementAnchorIndex (): void {
    this._shape.incrementAnchorIndex()
  }

  /** Update the shapes of the label to the state */
  public commitLabel (): void {
    if (this._label !== null) {
      const cube = this._shape.toCube()
      if (this._plane) {
        cube.center.z = 0.5 * cube.size.z
        cube.orientation.x = 0
        cube.orientation.y = 0
      }

      if (this.labelId < 0 && !this._temporary) {
        this.addToState()
      } else {
        Session.dispatch(changeLabelShape(
          this._label.item, this._label.shapes[0], cube
        ))
        Session.dispatch(changeLabelProps(
          this._label.item, this._labelId, { manual: true }
        ))
        if (Session.tracking && this._trackId in Session.tracks) {
          Session.tracks[this._trackId].onLabelUpdated(
            this._label.item, [cube]
          )
        }
      }
    }
  }

  /**
   * Handle mouse move
   * @param projection
   */
  public onMouseDown (x: number, y: number, camera: THREE.Camera) {
    if (this._temporary) {
      this._shape.clickInit(x, y, camera)
    }
    return this._shape.shouldDrag()
  }

  /**
   * Handle mouse up
   * @param projection
   */
  public onMouseUp () {
    return
  }

  /**
   * Handle mouse move
   * @param projection
   */
  public onMouseMove (x: number, y: number, camera: THREE.Camera) {
    const success = this._shape.drag(x, y, camera)
    if (this._temporary && success) {
      this._temporary = false
    }
    return success
  }

  /**
   * Highlight box
   * @param h
   * @param raycaster
   */
  public setHighlighted (intersection?: THREE.Intersection) {
    super.setHighlighted(intersection)
    this._shape.setHighlighted(intersection)
  }

  /**
   * Add this label to state when newly created
   */
  private addToState () {
    if (this._label) {
      const cube = this._shape.toCube()
      if (Session.tracking && this._trackId in Session.tracks) {
        Session.tracks[this._trackId].onLabelCreated(
          this._label.item, this
        )
      } else {
        Session.dispatch(addBox3dLabel(
          this._label.item,
          this._label.category,
          cube.center,
          cube.size,
          cube.orientation,
          cube.surfaceId
        ))
      }
    }
  }
}
