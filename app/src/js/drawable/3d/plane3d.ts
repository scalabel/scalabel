import * as THREE from 'three'
import { LabelTypeName, ShapeTypeName } from '../../common/types'
import { makeLabel } from '../../functional/states'
import { ShapeType, State } from '../../functional/types'
import { Vector3D } from '../../math/vector3d'
import { Box3D } from './box3d'
import { Grid3D } from './grid3d'
import Label3D from './label3d'
import { Label3DList } from './label3d_list'
import { Shape3D } from './shape3d'

/**
 * Class for managing plane for holding 3d labels
 */
export class Plane3D extends Label3D {
  /** ThreeJS object for rendering shape */
  private _grid: Grid3D
  /** temporary shape */
  private _temporaryLabel: Label3D | null

  constructor (labelList: Label3DList) {
    super(labelList)
    this._grid = new Grid3D()
    this._temporaryLabel = null
  }

  /** Override set selected method */
  public set selected (s: boolean) {
    super.selected = s
  }

  /** Override get selected */
  public get selected (): boolean {
    return super.selected
  }

  /**
   * Modify ThreeJS objects to draw label
   * @param {THREE.Scene} scene: ThreeJS Scene Object
   */
  public render (scene: THREE.Scene, _camera: THREE.Camera): void {
    this._grid.render(scene)
  }

  /**
   * Highlight box
   * @param intersection
   */
  public setHighlighted (intersection?: THREE.Intersection) {
    super.setHighlighted(intersection)
    this._grid.setHighlighted(intersection)
  }

  /**
   * Handle click
   */
  public click () {
    return false
  }

  /**
   * Handle drag
   * @param projection
   */
  public drag (
    dx: number, dy: number, camera: THREE.Camera
  ): boolean {
    if (this._temporaryLabel) {
      return this._temporaryLabel.drag(dx, dy, camera)
    } else {
      if (
        this._labelState &&
        (this.selected || this.anyChildSelected()) &&
        this._labelList.currentLabelType === LabelTypeName.BOX_3D
      ) {
        this._temporaryLabel = new Box3D(this._labelList)
        this._temporaryLabel.init(
          this._labelState.item,
          0,
          undefined,
          this._labelState.sensors,
          true
        )
        this.addChild(this._temporaryLabel)
        for (const shape of this._temporaryLabel.shapes()) {
          this._grid.attach(shape)
        }
        return this._temporaryLabel.drag(dx, dy, camera)
      }
    }
    return false
  }

  /** Rotate */
  public rotate (quaternion: THREE.Quaternion) {
    this._labelList.addUpdatedShape(this._grid)
    this._grid.applyQuaternion(quaternion)
    for (const child of this.children) {
      child.rotate(quaternion, this._grid.position)
    }
  }

  /** Translate */
  public translate (delta: THREE.Vector3) {
    this._labelList.addUpdatedShape(this._grid)
    this._grid.position.add(delta)
    for (const child of this.children) {
      child.translate(delta)
    }
  }

  /** Scale */
  public scale (scale: THREE.Vector3, anchor: THREE.Vector3) {
    this._labelList.addUpdatedShape(this._grid)
    this._grid.scale.x *= scale.x
    this._grid.scale.y *= scale.y
    this._grid.position.sub(anchor)
    this._grid.position.multiply(scale)
    this._grid.position.add(anchor)
  }

  /** Move */
  public move (position: THREE.Vector3): void {
    this._grid.position.copy(position)
    this._labelList.addUpdatedShape(this._grid)
  }

  /** center of plane */
  public get center (): THREE.Vector3 {
    return this._grid.position
  }

  /** orientation of plane */
  public get orientation (): THREE.Quaternion {
    return this._grid.quaternion
  }

  /** scale of plane */
  public get size (): THREE.Vector3 {
    return this._grid.scale
  }

  /** bounds of plane */
  public bounds (local?: boolean): THREE.Box3 {
    const box = new THREE.Box3()
    if (!local) {
      box.copy(this._grid.lines.geometry.boundingBox)
      this._grid.updateMatrixWorld(true)
      box.applyMatrix4(this._grid.matrixWorld)
    } else {
      box.setFromCenterAndSize(this.center, this.size)
    }
    return box
  }

  /**
   * Expand the primitive shapes to drawable shapes
   * @param {ShapeType[]} shapes
   */
  public updateState (
    state: State,
    itemIndex: number,
    labelId: number
  ): void {
    super.updateState(state, itemIndex, labelId)
    this._grid = this._shapes[0] as Grid3D
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
    this._labelState = makeLabel({
      type: LabelTypeName.PLANE_3D, id: -1, item: itemIndex,
      category: [category], sensors
    })
    if (center) {
      this._grid.center = center
    }
  }

  /**
   * Return a list of the shape for inspection and testing
   */
  public shapes (): Shape3D[] {
    return [this._grid]
  }

  /** State representation of shape */
  public shapeStates (): [number[], ShapeTypeName[], ShapeType[]] {
    if (!this._labelState) {
      throw new Error('Uninitialized label')
    }
    return [
      [this._labelState.shapes[0]],
      [ShapeTypeName.GRID],
      [this._grid.toState().shape]
    ]
  }
}
