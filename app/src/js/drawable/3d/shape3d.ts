import * as THREE from 'three'
import { ShapeType } from '../../functional/types'

/**
 * Base shape class
 */
export abstract class Shape3D extends THREE.Object3D {
  /** id */
  protected _id: number
  /** shape state */
  protected _shape: ShapeType | null
  /** whether highlighted */
  protected _highlighted: boolean
  /** whether selected */
  protected _selected: boolean

  constructor () {
    super()
    this._id = -1
    this._shape = null
    this._highlighted = false
    this._selected = false
  }

  /** Get shape id */
  public get id (): number {
    return this._id
  }

  /** Get selected */
  public get selected (): boolean {
    return this._selected
  }

  /** Set selected */
  public set selected (s: boolean) {
    this._selected = s
  }

  /** return shape type */
  public abstract get typeName (): string

  /** update parameters */
  public updateState (
    shape: ShapeType, id: number, _activeCamera?: THREE.Camera
  ) {
    this._shape = shape
    this._id = id
  }

  /** Convert shape to state representation */
  public abstract toState (): ShapeType

  /** function for setting highlight status */
  public abstract setHighlighted (intersection?: THREE.Intersection): void
}
