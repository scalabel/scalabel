import * as THREE from "three"

import { IdType, INVALID_ID, ShapeType } from "../../types/state"
import Label3D from "./label3d"
import { Object3D } from "./object3d"

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

  /**
   * Constructor
   *
   * @param label
   */
  constructor(label: Label3D) {
    super()
    this._shapeId = INVALID_ID
    this._label = label
    this._shape = null
    this._highlighted = false
  }

  /** Get shape id */
  public get shapeId(): IdType {
    return this._shapeId
  }

  /** Get associated label */
  public get label(): Label3D {
    return this._label
  }

  /** return shape type */
  public abstract get typeName(): string

  /**
   * update parameters
   *
   * @param shape
   * @param id
   * @param _activeCamera
   */
  public updateState(
    shape: ShapeType,
    id: IdType
    // _activeCamera?: THREE.Camera
  ): void {
    this._shape = shape
    this._shapeId = id
  }

  /**
   * Set visibility for viewer
   *
   * @param viewerId
   * @param v
   */
  public setVisible(viewerId: number, v: boolean = true): void {
    if (v) {
      this.layers.enable(viewerId)
    } else {
      this.layers.disable(viewerId)
    }
  }

  /** Convert shape to state representation */
  public abstract toState(): ShapeType

  /** function for setting highlight status */
  public abstract setHighlighted(intersection?: THREE.Intersection): void
}
