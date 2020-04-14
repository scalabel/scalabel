import * as THREE from 'three'
import { makeDefaultId } from '../../functional/states'
import { IdType, ShapeType } from '../../functional/types'
import Label3D from './label3d'
import { Object3D } from './object3d'

/**
 * Base shape class
 */
export abstract class Shape3D extends Object3D {
  /** id */
  protected _shapeId: IdType
  /** shape state */
  protected _shape: ShapeType | null
  /** corresponding label object */
  protected _label: Label3D
  /** whether highlighted */
  protected _highlighted: boolean

  constructor (label: Label3D) {
    super()
    this._shapeId = makeDefaultId()
    this._label = label
    this._shape = null
    this._highlighted = false
  }

  /** Get shape id */
  public get shapeId (): IdType {
    return this._shapeId
  }

  /** Get associated label */
  public get label (): Label3D {
    return this._label
  }

  /** return shape type */
  public abstract get typeName (): string

  /** update parameters */
  public updateState (
    shape: ShapeType, id: IdType, _activeCamera?: THREE.Camera
  ) {
    this._shape = shape
    this._shapeId = id
  }

  /** Set visibility for viewer */
  public setVisible (viewerId: number, v: boolean = true) {
    if (v) {
      this.layers.enable(viewerId)
    } else {
      this.layers.disable(viewerId)
    }
  }

  /** Convert shape to state representation */
  public abstract toState (): ShapeType

  /** function for setting highlight status */
  public abstract setHighlighted (intersection?: THREE.Intersection): void
}
