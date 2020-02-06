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
  private _cube: Cube3D

  constructor (labelList: Label3DList) {
    super(labelList)
    this._cube = new Cube3D()
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
    sensors?: number[],
    temporary?: boolean
  ): void {
    if (!sensors || sensors.length === 0) {
      sensors = [-1]
    }

    if (center) {
      this._cube.position.copy(center.toThree())
    }

    this._shapes = [this._cube]
    this._cube.associateLabel(this)

    this._labelState = makeLabel({
      type: LabelTypeName.BOX_3D,
      id: -1,
      item: itemIndex,
      category: [category],
      sensors,
      shapes: [this._cube.shapeId]
    })

    if (temporary) {
      this._temporary = true
    } else {
      this._labelList.addUpdatedLabel(this)
      this._labelList.addTemporaryShape(this._cube)
    }
  }

  /** Set active camera */
  public set activeCamera (camera: THREE.Camera) {
    this._cube.setControlSpheres(camera)
  }

  /** Indexed shapes */
  public shapeStates (): [number[], ShapeTypeName[], ShapeType[]] {
    if (!this._labelState) {
      throw new Error('Uninitialized label')
    }
    return [
      [this._labelState.shapes[0]],
      [ShapeTypeName.CUBE],
      [this._cube.toState().shape]
    ]
  }

  /** Override set parent */
  public set parent (parent: Label3D | null) {
    this._parent = parent
    if (parent && this._labelState) {
      this._labelState.parent = parent.labelId
    } else if (this._labelState) {
      this._labelState.parent = -1
    }
    if (parent && parent.type === LabelTypeName.PLANE_3D) {
      this._cube.attachToPlane(parent as Plane3D)
    } else {
      this._cube.detachFromPlane()
    }
  }

  /** Get parent label */
  public get parent (): Label3D | null {
    return this._parent
  }

  /** select the label */
  public set selected (s: boolean) {
    this._selected = s
    this._cube.selected = s
  }

  /** return whether label selected */
  public get selected (): boolean {
    return this._selected
  }

  /**
   * Return a list of the shape for inspection and testing
   */
  public shapes (): Shape3D[] {
    return [this._cube]
  }

  /**
   * move anchor to next corner
   */
  public incrementAnchorIndex (): void {
    this._cube.incrementAnchorIndex()
  }

  /**
   * Handle click
   */
  public click () {
    return false
  }

  /** Handle mouse move */
  public drag (dx: number, dy: number, camera: THREE.Camera) {
    const success = this._cube.drag(dx, dy, camera)
    if (success) {
      this._temporary = false
      this._labelList.addUpdatedShape(this._cube)
      this._labelList.addUpdatedLabel(this)
    }
    return success
  }

  /** Rotate */
  public rotate (quaternion: THREE.Quaternion, anchor?: THREE.Vector3) {
    this._labelList.addUpdatedShape(this._cube)
    this._labelList.addUpdatedLabel(this)
    this._cube.applyQuaternion(quaternion)
    if (anchor) {
      const newPosition = new THREE.Vector3()
      newPosition.copy(this._cube.position)
      newPosition.sub(anchor)
      newPosition.applyQuaternion(quaternion)
      newPosition.add(anchor)
      this._cube.position.copy(newPosition)
    }
  }

  /** Move */
  public move (position: THREE.Vector3): void {
    this._cube.position.copy(position)
    this._labelList.addUpdatedShape(this._cube)
    this._labelList.addUpdatedLabel(this)
  }

  /** Translate */
  public translate (delta: THREE.Vector3) {
    this._labelList.addUpdatedShape(this._cube)
    this._labelList.addUpdatedLabel(this)
    this._cube.position.add(delta)
  }

  /** Scale */
  public scale (scale: THREE.Vector3, anchor: THREE.Vector3, local: boolean) {
    this._labelList.addUpdatedShape(this._cube)
    this._labelList.addUpdatedLabel(this)
    const inverseRotation = new THREE.Quaternion()
    inverseRotation.copy(this.orientation)
    inverseRotation.inverse()

    if (!local) {
      scale = rotateScale(scale, this.orientation)
    }
    this._cube.scale.multiply(scale)

    this._cube.position.sub(anchor)
    this._cube.position.applyQuaternion(inverseRotation)
    this._cube.position.multiply(scale)
    this._cube.position.applyQuaternion(this.orientation)
    this._cube.position.add(anchor)
  }

  /** center of box */
  public get center (): THREE.Vector3 {
    const position = new THREE.Vector3()
    this._cube.getWorldPosition(position)
    return position
  }

  /** orientation of box */
  public get orientation (): THREE.Quaternion {
    const quaternion = new THREE.Quaternion()
    this._cube.getWorldQuaternion(quaternion)
    return quaternion
  }

  /** scale of box */
  public get size (): THREE.Vector3 {
    const scale = new THREE.Vector3()
    this._cube.getWorldScale(scale)
    return scale
  }

  /** bounds of box */
  public bounds (local?: boolean): THREE.Box3 {
    const box = new THREE.Box3()
    if (!local) {
      box.copy(this._cube.box.geometry.boundingBox)
      this._cube.updateMatrixWorld(true)
      box.applyMatrix4(this._cube.matrixWorld)
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
    this._cube.setHighlighted(intersection)
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
    this._cube = this._shapes[0] as Cube3D
    this._cube.color = this._color
  }
}
