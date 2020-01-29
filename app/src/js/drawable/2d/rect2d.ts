import { ShapeTypeName } from '../../common/types'
import { makeRect } from '../../functional/states'
import { IndexedShapeType, RectType } from '../../functional/types'
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
  /** Shape state */
  private _rectState: RectType
  constructor (x: number = 0, y: number = 0, w: number = 0, h: number = 0) {
    super()
    this.x = x
    this.y = y
    this.w = w
    this.h = h
    this._rectState = makeRect()
  }

  /** type */
  public get typeName () {
    return ShapeTypeName.RECT
  }

  /** get x */
  public get x (): number {
    return this._rectState.x1
  }

  /** set x */
  public set x (v: number) {
    this._rectState.x2 -= this._rectState.x1 - v
    this._rectState.x1 = v
  }

  /** get y */
  public get y (): number {
    return this._rectState.y1
  }

  /** set x */
  public set y (v: number) {
    this._rectState.y2 -= this._rectState.y1 - v
    this._rectState.y1 = v
  }

  public get w (): number {
    return this._rectState.x2 - this._rectState.x1
  }

  /** set x */
  public set w (v: number) {
    this._rectState.x2 = this._rectState.x1 + v
  }

  /** get y */
  public get h (): number {
    return this._rectState.y2 - this._rectState.y1
  }

  /** set x */
  public set h (v: number) {
    this._rectState.y2 = this._rectState.y1 + v
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
    context.strokeStyle = toCssColor(style.color)
    if (style.dashed) {
      context.setLineDash([6, 2])
    }
    context.lineWidth = style.lineWidth
    context.strokeRect(
      this.x * ratio,
      this.y * ratio,
      this.w * ratio,
      this.h * ratio
    )
    context.restore()
  }

  /**
   * Update the values of the drawable shapes
   * @param {RectType} rect
   */
  public updateState (indexedShape: IndexedShapeType) {
    super.updateState(indexedShape)
    if (this._indexedShape) {
      this._rectState = this._indexedShape.shape as RectType
    }
  }

  /** Copy other rect */
  public copy (shape: Shape2D) {
    const rect = shape as Rect2D
    this.x = rect.x
    this.y = rect.y
    this.w = rect.w
    this.h = rect.h
  }
}
