import _ from 'lodash'
import * as THREE from 'three'

import { makeLabel } from '../../functional/states'
import { ShapeType, State } from '../../functional/types'

import { Vector3D } from '../../math/vector3d'

import { LabelTypeName, ShapeTypeName } from '../../common/types'
import { TransformationControl } from './control/transformation_control'
import { Cube3D } from './cube3d'
import { Label3D } from './label3d'
import { Label3DList } from './label3d_list'
import { Plane3D } from './plane3d'
import { Shape3D } from './shape3d'

/**
 * Box3d Label
 */
export class Box3D extends Label3D {
  /** ThreeJS object for rendering shape */
  private _shape: Cube3D
  /** Whether this is temporary */
  private _temporary: boolean

  constructor (labelList: Label3DList) {
    super(labelList)
    this._shape = new Cube3D(this)
    this._temporary = false
  }

  /**
   * Initialize label
   * @param {State} state
   */
  public init (
    itemIndex: number,
    category: number,
    center?: Vector3D,
    sensors?: number[]
  ): void {
    this._label = makeLabel({
      type: LabelTypeName.BOX_3D, id: -1, item: itemIndex,
      category: [category], sensors
    })
    this._labelId = -1

    if (center) {
      this._shape.center = center
    }
  }

  /** Set active camera */
  public set activeCamera (camera: THREE.Camera) {
    this._shape.setControlSpheres(camera)
  }

  /** Indexed shapes */
  public shapeStates (): [number[], ShapeTypeName[], ShapeType[]] {
    if (!this._label) {
      throw new Error('Uninitialized label')
    }
    return [
      [this._label.shapes[0]],
      [ShapeTypeName.CUBE],
      [this._shape.toObject()]
    ]
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
    this._shape.attachControl(control)
  }

  /**
   * Attach control
   */
  public detachControl () {
    this._shape.detachControl()
  }

  /**
   * Return a list of the shape for inspection and testing
   */
  public shapes (): Shape3D[] {
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
   * move anchor to next corner
   */
  public incrementAnchorIndex (): void {
    this._shape.incrementAnchorIndex()
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
    this._labelList.addUpdatedLabel(this)
    return success
  }

  /** Rotate */
  public rotate (quaternion: THREE.Quaternion) {
    this._shape.applyQuaternion(quaternion)
    this._labelList.addUpdatedLabel(this)
  }

  /** Translate */
  public translate (delta: THREE.Vector3) {
    this._shape.position.add(delta)
    this._labelList.addUpdatedLabel(this)
  }

  /** Scale */
  public scale (scale: THREE.Vector3, anchor: THREE.Vector3) {
    this._shape.scale.multiply(scale)
    this._shape.position.sub(anchor)
    this._shape.position.multiply(scale)
    this._shape.position.add(anchor)
    this._labelList.addUpdatedLabel(this)
  }

  /** center of box */
  public get center (): THREE.Vector3 {
    return this._shape.position
  }

  /** orientation of box */
  public get orientation (): THREE.Quaternion {
    return this._shape.quaternion
  }

  /** size of box */
  public get size (): THREE.Vector3 {
    return this._shape.scale
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
   * Update params when state is updated
   * @param state
   * @param itemIndex
   * @param labelId
   */
  public updateState (
    state: State,
    itemIndex: number,
    labelId: number
  ): void {
    super.updateState(state, itemIndex, labelId)
    this._shape.color = this._color
    const label = state.task.items[itemIndex].labels[labelId]
    this._shape.updateState(
      state.task.items[itemIndex].shapes[label.shapes[0]].shape,
      label.shapes[0]
    )
  }

  // /**
  //  * Add this label to state when newly created
  //  */
  // private addToState () {
  //   if (this._label) {
  //     const cube = this._shape.toCube()
  //     if (Session.tracking && this._trackId in Session.tracks) {
  //       Session.tracks[this._trackId].onLabelCreated(
  //         this._label.item, this
  //       )
  //     } else {
  //       Session.dispatch(addBox3dLabel(
  //         this._label.item,
  //         this._label.category,
  //         cube.center,
  //         cube.size,
  //         cube.orientation,
  //         cube.surfaceId
  //       ))
  //     }
  //   }
  // }
}
