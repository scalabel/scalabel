import * as THREE from 'three'
import { LabelType, ShapeType, State } from '../../functional/types'
import { getColorById } from '../util'
import { TransformationControl } from './control/transformation_control'
import { Cube3D } from './cube3d'
import { Grid3D } from './grid3d'
import { Plane3D } from './plane3d'

type Shape = Cube3D | Grid3D

/**
 * Abstract class for 3D drawable labels
 */
export abstract class Label3D {
  /* The members are public for testing purpose */
  /** label id in state */
  protected _labelId: number
  /** track id in state */
  protected _trackId: number
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
  /** plane if attached */
  protected _plane: Plane3D | null

  constructor () {
    this._index = -1
    this._labelId = -1
    this._trackId = -1
    this._order = -1
    this._selected = false
    this._highlighted = false
    this._label = null
    this._color = [0, 0, 0, 1]
    this._plane = null
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
  public setHighlighted (intersection?: THREE.Intersection) {
    if (intersection) {
      this._highlighted = true
    } else {
      this._highlighted = false
    }
  }

  /** Attach label to plane */
  public attachToPlane (plane: Plane3D) {
    if (plane === this._plane) {
      return
    }
    this._plane = plane
  }

  /** Attach label to plane */
  public detachFromPlane () {
    if (this._plane) {
      this._plane = null
    }
  }

  /** get category */
  public get category (): number[] {
    if (this._label && this._label.category) {
      return this._label.category
    }
    return []
  }

  /** get attributes */
  public get attributes (): {[key: number]: number[]} {
    if (this._label && this._label.attributes) {
      return this._label.attributes
    }
    return {}
  }

  /** Attach control */
  public abstract attachControl (control: TransformationControl): void

  /** Attach control */
  public abstract detachControl (control: TransformationControl): void

  /**
   * Modify ThreeJS objects to draw label
   * @param {THREE.Scene} scene: ThreeJS Scene Object
   */
  public abstract render (scene: THREE.Scene, camera: THREE.Camera): void

  /**
   * Handle mouse move
   * @param projection
   */
  public abstract onMouseDown (
    x: number, y: number, camera: THREE.Camera
  ): boolean

  /**
   * Handle mouse up
   * @param projection
   */
  public abstract onMouseUp (): void

  /**
   * Handle mouse move
   * @param projection
   */
  public abstract onMouseMove (
    x: number, y: number, camera: THREE.Camera
  ): boolean

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
  public abstract init (
    state: State, surfaceId?: number, temporary?: boolean
  ): void

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
    this._trackId = this._label.track
    this._color = getColorById(this._labelId, this._trackId)
    this.updateShapes(this._label.shapes.map((i) => item.shapes[i].shape))
  }
}

export default Label3D
