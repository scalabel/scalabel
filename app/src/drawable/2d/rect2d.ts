import _ from "lodash"

import { makeRect } from "../../functional/states"
import { Vector } from "../../math/vector"
import { RectType } from "../../types/state"
import { Context2D, toCssColor } from "../util"

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
 *
 * @param {Partial<Rect2DStyle>} style
 */
export function makeRect2DStyle(style: Partial<Rect2DStyle> = {}): Rect2DStyle {
  return {
    lineWidth: 1,
    color: [0, 0, 0, 1],
    dashed: false,
    ...style
  }
}

/** Drawable rectangle */
export class Rect2D {
  /** The shape of the rect */
  private _rect: RectType

  /**
   * Constructor
   *
   * @param rect
   */
  constructor(rect: RectType | null = null) {
    if (rect === null) {
      this._rect = makeRect()
    } else {
      this._rect = _.cloneDeep(rect)
    }
  }

  /**
   * Get x coordinate of upper left corner
   */
  public get x1(): number {
    return this._rect.x1
  }

  /** set x */
  public set x1(v: number) {
    this._rect.x1 = v
  }

  /** get y */
  public get y1(): number {
    return this._rect.y1
  }

  /** set y */
  public set y1(v: number) {
    this._rect.y1 = v
  }

  /**
   * Get x coordinate of lower right corner
   */
  public get x2(): number {
    return this._rect.x2
  }

  /** set w */
  public set x2(v: number) {
    this._rect.x2 = v
  }

  /** get h */
  public get y2(): number {
    return this._rect.y2
  }

  /** set h */
  public set y2(v: number) {
    this._rect.y2 = v
  }

  /**
   * Width of the rect
   */
  public width(): number {
    return this.x2 - this.x1
  }

  /**
   * Height of the rect
   */
  public height(): number {
    return this.y2 - this.y1
  }

  /**
   * Set the shape to the new rect
   *
   * @param rect
   */
  public set(rect: RectType): void {
    this._rect = _.cloneDeep(rect)
  }

  /**
   * Make a copy of the object
   */
  public clone(): Rect2D {
    return new Rect2D(this._rect)
  }

  /**
   * Return a copy of the shape
   */
  public shape(): RectType {
    return _.cloneDeep(this._rect)
  }

  /**
   * Draw the rect on a 2D context
   *
   * @param {Context2D} context
   * @param {number} ratio: display to image ratio
   * @param ratio
   * @param {RectStyle} style
   */
  public draw(context: Context2D, ratio: number, style: Rect2DStyle): void {
    context.save()
    // Convert to display resolution
    const real = this.vector().scale(ratio)
    context.strokeStyle = toCssColor(style.color)
    if (style.dashed) {
      context.setLineDash([6, 2])
    }
    context.lineWidth = style.lineWidth
    context.strokeRect(real[0], real[1], real[2], real[3])
    context.restore()
  }

  /**
   * Convert to vector for drawing
   */
  private vector(): Vector {
    const v = new Vector(4)
    v.set(this.x1, this.y1, this.width(), this.height())
    return v
  }
}
