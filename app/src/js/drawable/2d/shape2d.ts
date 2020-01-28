
import { ShapeType } from '../../functional/types'
import { Vector } from '../../math/vector'
import Label2D from './label2d'

/**
 * Base shape class
 */
export abstract class Shape2D extends Vector {
  /** id */
  protected _id: number
  /** shape state */
  protected _shape: ShapeType | null
  /** corresponding label objects */
  protected _labels: { [id: number]: Label2D }
  /** whether highlighted */
  protected _highlighted: boolean

  constructor (dim: number) {
    super(dim)
    this._id = -1
    this._labels = []
    this._shape = null
    this._highlighted = false
  }

  /** Get shape id */
  public get id (): number {
    return this._id
  }

  /** clear associated labels */
  public clearLabels () {
    this._labels = {}
  }

  /** Add associated label */
  public associateLabel (label: Label2D) {
    this._labels[label.labelId] = label
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
