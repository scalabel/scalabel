import * as THREE from 'three'
import { LabelType, ShapeType, State } from '../functional/types'
import { Cube3D } from './cube3d'
import { getColorById } from './util'

type Shape = Cube3D

/**
 * Abstract class for 3D drawable labels
 */
export abstract class Label3D {
  /* The members are public for testing purpose */
  /** label id in state */
  protected _labelId: number
  /** index of the label */
  protected _index: number
  /** drawing order of the label */
  protected _order: number
  /** the corresponding label in the state */
  protected _label: LabelType | null
  /** whether the label is selected */
  protected _selected: boolean
  /** whether the label is highlighted */
  protected _highlighted: boolean
  /** rgba color decided by labelId */
  protected _color: number[]

  constructor () {
    this._index = -1
    this._labelId = -1
    this._order = -1
    this._selected = false
    this._highlighted = false
    this._label = null
    this._color = [0, 0, 0, 1]
  }

  /**
   * Set index of this label
   */
  public set index (i: number) {
    this._index = i
  }

  /** get index */
  public get index (): number {
    return this._index
  }

  /** get labelId */
  public get labelId (): number {
    return this._labelId
  }

  /** select the label */
  public setSelected (s: boolean) {
    this._selected = s
  }

  /** highlight the label */
  public setHighlighted (h: boolean) {
    this._highlighted = h
  }

  /**
   * Modify ThreeJS objects to draw label
   * @param {THREE.Scene} scene: ThreeJS Scene Object
   */
  public abstract render (scene: THREE.Scene): void

  /**
   * Set up for drag action
   * @param viewPlaneNormal
   * @param cameraPosition
   * @param intersectionPoint
   */
  public abstract startDrag (
    viewPlaneNormal: THREE.Vector3,
    cameraPosition: THREE.Vector3,
    intersectionPoint: THREE.Vector3
  ): void

  /**
   * Mouse movement while mouse down on box (from raycast)
   * @param projection
   */
  public abstract drag (projection: THREE.Vector3): void

  /**
   * Clean up for drag action
   * @param viewPlaneNormal
   * @param cameraPosition
   * @param intersectionPoint
   * @param editMode
   */
  public abstract stopDrag (): void

  /**
   * Handle keyboard events
   * @param {KeyboardEvent} e
   * @returns true if consumed, false otherwise
   */
  public abstract onKeyDown (e: KeyboardEvent): boolean

  /**
   * Handle keyboard events
   * @returns true if consumed, false otherwise
   */
  public abstract onKeyUp (e: KeyboardEvent): boolean

  /**
   * Expand the primitive shapes to drawable shapes
   * @param {ShapeType[]} shapes
   */
  public abstract updateShapes (shapes: ShapeType[]): void

  /** Update the shapes of the label to the state */
  public abstract commitLabel (): void

  /**
   * Initialize label
   * @param {State} state
   */
  public abstract init (state: State): void

  /**
   * Return a list of the shape for inspection and testing
   */
  public abstract shapes (): Array<Readonly<Shape>>

  /** Convert label state to drawable */
  public updateState (
    state: State, itemIndex: number, labelId: number): void {
    const item = state.task.items[itemIndex]
    this._label = item.labels[labelId]
    this._labelId = this._label.id
    this._color = getColorById(this._labelId)
    this.updateShapes(this._label.shapes.map((i) => item.shapes[i].shape))
  }
}

export default Label3D
