import { ShapeTypeName } from '../../common/types'
import { Point2DType } from '../../functional/types'
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
  constructor (x: number = 0, y: number = 0) {
    super(2)
    this.x = x
    this.y = y
  }

  /** get x */
  public get x (): number {
    return this[0]
  }

  /** set x */
  public set x (v: number) {
    this[0] = v
  }

  /** get y */
  public get y (): number {
    return this[1]
  }

  /** set x */
  public set y (v: number) {
    this[1] = v
  }

  /** Get type name */
  public get typeName () {
    return ShapeTypeName.POINT_2D
  }

  /** Convert to state */
  public toState (): Point2DType {
    return {
      x: this.x,
      y: this.y
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
    const real = this.clone().scale(ratio)
    context.beginPath()
    context.fillStyle = toCssColor(style.color)
    context.arc(real.x, real.y, style.radius, 0, 2 * Math.PI, false)
    context.closePath()
    context.fill()
    context.restore()
  }
}
