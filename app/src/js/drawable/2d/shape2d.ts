
import { ShapeTypeName } from '../../common/types'
import { IndexedShapeType } from '../../functional/types'
import Label2D from './label2d'

/**
 * Base shape class
 */
export abstract class Shape2D {
  /** id */
  protected _id: number
  /** shape state */
  protected _indexedShape: IndexedShapeType | null
  /** corresponding label objects */
  protected _labels: { [id: number]: Label2D }
  /** whether highlighted */
  protected _highlighted: boolean

  constructor () {
    this._id = -1
    this._labels = []
    this._indexedShape = null
    this._highlighted = false
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
    if (this._indexedShape) {
      this._indexedShape.id = id
    }
  }

  /** clear associated labels */
  public clearLabels () {
    this._labels = {}
  }

  /** return shape type */
  public get typeName (): string {
    if (this._indexedShape) {
      return this._indexedShape.type
    }
    return ShapeTypeName.UNKNOWN
  }

  /** update parameters */
  public updateState (indexedShape: IndexedShapeType) {
    this._indexedShape = indexedShape
  }

  /** Convert shape to state representation */
  public toState (): IndexedShapeType {
    if (this._indexedShape) {
      return this._indexedShape
    }
    throw new Error('Uninitialized shape')
  }

  /** function for setting highlight status */
  public setHighlighted (h: boolean): void {
    this._highlighted = h
  }

  /** copy */
  public abstract copy (shape: Shape2D): void
}
