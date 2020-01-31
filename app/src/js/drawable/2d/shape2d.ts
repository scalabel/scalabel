import _ from 'lodash'
import { ShapeTypeName } from '../../common/types'
import { makeIndexedShape } from '../../functional/states'
import { IndexedShapeType } from '../../functional/types'
import Label2D from './label2d'

/**
 * Base shape class
 */
export abstract class Shape2D {
  /** shape state */
  protected _indexedShape: IndexedShapeType
  /** corresponding label objects */
  protected _labels: { [id: number]: Label2D }
  /** whether highlighted */
  protected _highlighted: boolean

  constructor () {
    this._labels = []
    this._highlighted = false
    this._indexedShape = makeIndexedShape(-1, -1, [], ShapeTypeName.UNKNOWN, {})
  }

  /** Get shape id */
  public get id (): number {
    return this._indexedShape.id
  }

  /** Set shape id */
  public set id (id: number) {
    this._indexedShape.id = id
  }

  /** Get item */
  public get item (): number {
    return this._indexedShape.item
  }

  /** Set item */
  public set item (item: number) {
    this._indexedShape.item = item
  }

  /** clear associated labels */
  public clearLabels () {
    this._labels = {}
  }

  /** return shape type */
  public get typeName (): string {
    return this._indexedShape.type
  }

  /** update parameters */
  public updateState (indexedShape: IndexedShapeType) {
    this._indexedShape = _.cloneDeep(indexedShape)
  }

  /** Convert shape to state representation */
  public toState (): IndexedShapeType {
    return this._indexedShape
  }

  /** function for setting highlight status */
  public setHighlighted (h: boolean): void {
    this._highlighted = h
  }

  /** Associate another label with this shape */
  public associateLabel (label: Label2D) {
    this._indexedShape.labels.push(label.labelId)
  }

  /** copy */
  public abstract copy (shape: Shape2D): void
}
