import { ShapeTypeName } from '../../common/types'
import { IndexedShapeType, Point2DType } from '../../functional/types'
import { Vector2D } from '../../math/vector2d'
import { Context2D, toCssColor } from '../util'
import { Shape2D } from './shape2d'

export interface Point2DStyle {
  /** radius of the point on drawing */
  radius: number
  /** color of the rect */
  color: number[]
}

/**
 * Generate Point2D style with default parameters
 * @param {Partial<Point2DStyle>} style
 */
export function makePoint2DStyle (
    style: Partial<Point2DStyle> = {}): Point2DStyle {
  return {
    radius: 1,
    color: [0, 0, 0],
    ...style
  }
}

/**
 * Drawable 2D point
 */
export class Point2D extends Shape2D {
  /** point state */
  private _pointState: Point2DType
  constructor (x: number = 0, y: number = 0) {
    super()
    this._pointState = { x: 0, y: 0 }
    this.x = x
    this.y = y
  }

  /** get x */
  public get x (): number {
    return this._pointState.x
  }

  /** set x */
  public set x (v: number) {
    this._pointState.x = v
  }

  /** get y */
  public get y (): number {
    return this._pointState.y
  }

  /** set x */
  public set y (v: number) {
    this._pointState.y = v
  }

  /** Get type name */
  public get typeName () {
    return ShapeTypeName.POINT_2D
  }

  /** Update State */
  public updateState (indexedShape: IndexedShapeType) {
    super.updateState(indexedShape)
    if (this._indexedShape) {
      this._pointState = this._indexedShape.shape as Point2DType
    }
  }

  /**
   * Draw the point on a 2D context
   * @param {Context2D} context
   * @param {number} ratio: display to image ratio
   * @param {RectStyle} style
   */
  public draw (
    context: Context2D, ratio: number, style: Point2DStyle): void {
    context.save()
    // convert to display resolution
    context.beginPath()
    context.fillStyle = toCssColor(style.color)
    context.arc(
      this.x * ratio,
      this.y * ratio,
      style.radius,
      0,
      2 * Math.PI,
      false
    )
    context.closePath()
    context.fill()
    context.restore()
  }

  /** Copy point */
  public copy (shape: Shape2D) {
    const point = shape as Point2D
    this.x = point.x
    this.y = point.y
  }

  /** Set x and y */
  public set (x: number, y: number) {
    this.x = x
    this.y = y
  }

  /** Make vector with same coordinates */
  public toVector () {
    return new Vector2D(this.x, this.y)
  }
}
