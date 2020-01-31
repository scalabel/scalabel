import * as THREE from 'three'
import { ShapeTypeName } from '../../common/types'
import { makeIndexedShape } from '../../functional/states'
import { IndexedShapeType } from '../../functional/types'
import Label3D from './label3d'

/**
 * Base shape class
 */
export abstract class Shape3D extends THREE.Object3D {
  /** shape state */
  protected _indexedShape: IndexedShapeType
  /** whether highlighted */
  protected _highlighted: boolean
  /** whether selected */
  protected _selected: boolean

  constructor () {
    super()
    this._indexedShape = makeIndexedShape(-1, -1, [], ShapeTypeName.UNKNOWN, {})
    this._highlighted = false
    this._selected = false
  }

  /** Get shape id */
  public get shapeId (): number {
    return this._indexedShape.id
  }

  /** Set shape id */
  public set shapeId (id: number) {
    this._indexedShape.id = id
  }

  /** Get item */
  public get item (): number {
    return this._indexedShape.item
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

  /** Associate another label with this shape */
  public associateLabel (label: Label3D) {
    this._indexedShape.labels.push(label.labelId)
  }

  /** Unasssociate another label with this shape */
  public unassociateLabel (label: Label3D) {
    const idIndex = this._indexedShape.labels.indexOf(label.labelId)
    if (idIndex >= 0) {
      this._indexedShape.labels.splice(idIndex, 1)
    }
  }

  /** Convert shape to state representation */
  public abstract toState (): IndexedShapeType

  /** function for setting highlight status */
  public abstract setHighlighted (intersection?: THREE.Intersection): void
}
