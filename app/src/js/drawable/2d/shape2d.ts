
import { ShapeType } from '../../functional/types'
import { Vector2D } from '../../math/vector2d'
import Label2D from './label2d'

/**
 * Base shape class
 */
export abstract class Shape2D extends Vector2D {
  /** id */
  protected _id: number
  /** shape state */
  protected _shape: ShapeType | null
  /** corresponding label objects */
  protected _labels: Label2D[]
  /** whether highlighted */
  protected _highlighted: boolean

  constructor () {
    super()
    this._id = -1
    this._labels = []
    this._shape = null
    this._highlighted = false
  }

  /** Get shape id */
  public get id (): number {
    return this._id
  }

  /** Get associated label */
  public get labels (): Label2D[] {
    return this._labels
  }

  /** Set associated labels */
  public set labels (labels: Label2D[]) {
    this._labels = labels
  }

  /** return shape type */
  public abstract get typeName (): string

  /** update parameters */
  public updateState (shape: ShapeType, id: number) {
    this._shape = shape
    this._id = id
  }

  /** Convert shape to state representation */
  public abstract toState (): ShapeType

  /** function for setting highlight status */
  public setHighlighted (h: boolean): void {
    this._highlighted = h
  }
}
