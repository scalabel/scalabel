import * as THREE from 'three'
import { ShapeType } from '../../functional/types'
import { TransformationControl } from './control/transformation_control'
import Label3D from './label3d'

/**
 * Base shape class
 */
export abstract class Shape3D extends THREE.Object3D {
  /** id */
  protected _id: number
  /** shape state */
  protected _shape: ShapeType | null
  /** corresponding label object */
  protected _label: Label3D
  /** whether highlighted */
  protected _highlighted: boolean
  /** controls */
  protected _control: TransformationControl | null

  constructor (label: Label3D) {
    super()
    this._id = -1
    this._label = label
    this._shape = null
    this._highlighted = false
    this._control = null
  }

  /** Get shape id */
  public get id (): number {
    return this._id
  }

  /** Get associated label */
  public get label (): Label3D {
    return this._label
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
  public abstract toObject (): ShapeType

  /** function for setting highlight status */
  public abstract setHighlighted (intersection?: THREE.Intersection): void

  /** attach control */
  public attachControl (control: TransformationControl) {
    this.detachControl()
    this.add(control)
    this._control = control
    this._control.attachShape(this)
  }

  /** detach control */
  public detachControl () {
    if (this._control) {
      this._control.detachShape()
      this.remove(this._control)
      this._control = null
    }
  }
}
