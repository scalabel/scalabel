import _ from 'lodash'
import * as THREE from 'three'

import { makeLabel } from '../../functional/states'
import { ShapeType, State } from '../../functional/types'

import { Vector3D } from '../../math/vector3d'

import { LabelTypeName, ShapeTypeName } from '../../common/types'
import { rotateScale } from '../../math/3d'
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

  constructor (labelList: Label3DList) {
    super(labelList)
    this._shape = new Cube3D(this)
  }

  /**
   * Initialize label
   * @param {State} state
   */
  public init (
    itemIndex: number,
    category: number,
    center?: Vector3D,
    sensors?: number[],
    temporary?: boolean
  ): void {
    if (!sensors || sensors.length === 0) {
      sensors = [-1]
    }

    this._label = makeLabel({
      type: LabelTypeName.BOX_3D, id: -1, item: itemIndex,
      category: [category], sensors
    })

    if (center) {
      this._shape.position.copy(center.toThree())
    }

    if (temporary) {
      this._temporary = true
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
      [this._shape.toState()]
    ]
  }

  /** Override set parent */
  public set parent (parent: Label3D | null) {
    this._parent = parent
    if (parent && this._label) {
      this._label.parent = parent.labelId
    } else if (this._label) {
      this._label.parent = -1
    }
    if (parent && parent.label.type === LabelTypeName.PLANE_3D) {
      this._shape.attachToPlane(parent as Plane3D)
    } else {
      this._shape.detachFromPlane()
    }
  }

  /** Get parent label */
  public get parent (): Label3D | null {
    return this._parent
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

  /** Handle mouse move */
  public onMouseMove (x: number, y: number, camera: THREE.Camera) {
    const success = this._shape.drag(x, y, camera)
    if (success) {
      this._temporary = false
    }
    this._labelList.addUpdatedLabel(this)
    return success
  }

  /** Rotate */
  public rotate (quaternion: THREE.Quaternion, anchor?: THREE.Vector3) {
    this._labelList.addUpdatedLabel(this)
    this._shape.applyQuaternion(quaternion)
    if (anchor) {
      const newPosition = new THREE.Vector3()
      newPosition.copy(this._shape.position)
      newPosition.sub(anchor)
      newPosition.applyQuaternion(quaternion)
      newPosition.add(anchor)
      this._shape.position.copy(newPosition)
    }
  }

  /** Translate */
  public translate (delta: THREE.Vector3) {
    this._labelList.addUpdatedLabel(this)
    this._shape.position.add(delta)
  }

  /** Scale */
  public scale (scale: THREE.Vector3, anchor: THREE.Vector3, local: boolean) {
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
  public get center (): THREE.Vector3 {
    const position = new THREE.Vector3()
    this._shape.getWorldPosition(position)
    return position
  }

  /** orientation of box */
  public get orientation (): THREE.Quaternion {
    const quaternion = new THREE.Quaternion()
    this._shape.getWorldQuaternion(quaternion)
    return quaternion
  }

  /** scale of box */
  public get size (): THREE.Vector3 {
    const scale = new THREE.Vector3()
    this._shape.getWorldScale(scale)
    return scale
  }

  /** bounds of box */
  public bounds (local?: boolean): THREE.Box3 {
    const box = new THREE.Box3()
    if (!local) {
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
   * @param intersection
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
}
