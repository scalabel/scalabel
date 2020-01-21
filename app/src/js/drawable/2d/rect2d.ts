import { ShapeTypeName } from '../../common/types'
import { makeRect } from '../../functional/states'
import { RectType, ShapeType } from '../../functional/types'
import { Context2D, toCssColor } from '../util'
import { Shape2D } from './shape2d'

export interface Rect2DStyle {
  /** line width of the rect sides */
  lineWidth: number
  /** color of the rect */
  color: number[]
  /** whether dash or not */
  dashed: boolean
}

/**
 * Generate Rect2D style with default parameters
 * @param {Partial<Rect2DStyle>} style
 */
export function makeRect2DStyle (
    style: Partial<Rect2DStyle> = {}): Rect2DStyle {
  return {
    lineWidth: 1,
    color: [0, 0, 0, 1],
    dashed: false,
    ...style
  }
}

/** Drawable rectangle */
export class Rect2D extends Shape2D {
  constructor (x: number = 0, y: number = 0, w: number = 0, h: number = 0) {
    super(4)
    this.x = x
    this.y = y
    this.w = w
    this.h = h
    this._shape = makeRect({ x1: x, y1: y, x2: x + w, y2: y + h })
  }

  /** type */
  public get typeName () {
    return ShapeTypeName.RECT
  }

  /** Convert to state */
  public toState (): RectType {
    return {
      x1: this.x,
      y1: this.y,
      x2: this.x + this.w,
      y2: this.y + this.h
    }
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

  public get w (): number {
    return this[2]
  }

  /** set x */
  public set w (v: number) {
    this[2] = v
  }

  /** get y */
  public get h (): number {
    return this[3]
  }

  /** set x */
  public set h (v: number) {
    this[3] = v
  }

  /** convert this drawable rect to a rect state */
  public toRect (): RectType {
    return makeRect({
      x1: this.x, y1: this.y, x2: this.x + this.w, y2: this.y + this.h})
  }

  /**
   * Draw the rect on a 2D context
   * @param {Context2D} context
   * @param {number} ratio: display to image ratio
   * @param {RectStyle} style
   */
  public draw (
    context: Context2D, ratio: number, style: Rect2DStyle): void {
    context.save()
    // convert to display resolution
    const real = this.clone().scale(ratio)
    context.strokeStyle = toCssColor(style.color)
    if (style.dashed) {
      context.setLineDash([6, 2])
    }
    context.lineWidth = style.lineWidth
    context.strokeRect(real.x, real.y, real.w, real.h)
    context.restore()
  }

  /**
   * Update the values of the drawable shapes
   * @param {RectType} rect
   */
  public updateState (shape: ShapeType, id: number) {
    super.updateState(shape, id)
    if (this._shape) {
      const newShape = this._shape as RectType
      this.x = newShape.x1
      this.y = newShape.y1
      this.w = newShape.x2 - newShape.x1
      this.h = newShape.y2 - newShape.y1
    }
  }
}
