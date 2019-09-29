import { Vector2D } from '../../math/vector2d'
import { Context2D, toCssColor } from '../util'

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
export class Point2D extends Vector2D {
  constructor (x: number = 0, y: number = 0) {
    super()
    this.x = x
    this.y = y
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
