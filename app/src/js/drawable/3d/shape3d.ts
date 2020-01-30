import * as THREE from 'three'
import { IndexedShapeType } from '../../functional/types'

/**
 * Base shape class
 */
export abstract class Shape3D extends THREE.Object3D {
  /** id */
  protected _id: number
  /** shape state */
  protected _indexedShape: IndexedShapeType | null
  /** whether highlighted */
  protected _highlighted: boolean
  /** whether selected */
  protected _selected: boolean

  constructor () {
    super()
    this._id = -1
    this._indexedShape = null
    this._highlighted = false
    this._selected = false
  }

  /** Get shape id */
  public get id (): number {
    if (this._indexedShape) {
      return this._indexedShape.id
    }
    return -1
  }

  /** Set shape id */
  public set id (id: number) {
    this._id = id
  }

  /** Get item */
  public get item (): number {
    if (this._indexedShape) {
      return this._indexedShape.item
    }
    return -1
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
    indexedShape: IndexedShapeType
  ) {
    this._indexedShape = indexedShape
  }

  /** Convert shape to state representation */
  public abstract toState (): IndexedShapeType

  /** function for setting highlight status */
  public abstract setHighlighted (intersection?: THREE.Intersection): void
}
