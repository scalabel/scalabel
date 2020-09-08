import _ from "lodash"

import { isValidId, makePathPoint2D } from "../../functional/states"
import { Vector2D } from "../../math/vector2d"
import { IdType, PathPoint2DType, PathPointType } from "../../types/state"
import { Context2D, toCssColor } from "../util"

export interface PathPoint2DStyle {
  /** radius of the point on drawing */
  radius: number
  /** color of the point */
  color: number[]
}

export interface Edge2DStyle {
  /** width of the line on drawing */
  lineWidth: number
  /** color of the line */
  color: number[]
  /** whether dashed */
  dashed: boolean
}

/**
 * Generate Edge2D style with default parameters
 *
 * @param style
 */
export function makeEdge2DStyle(style: Partial<Edge2DStyle> = {}): Edge2DStyle {
  return {
    lineWidth: 1,
    color: [0, 0, 0, 1],
    dashed: false,
    ...style
  }
}

/**
 * Generate PathPoint2D style with default parameters
 *
 * @param style
 */
export function makePathPoint2DStyle(
  style: Partial<PathPoint2DStyle> = {}
): PathPoint2DStyle {
  return {
    radius: 1,
    color: [0, 0, 0],
    ...style
  }
}

/**
 * Utility function to make new drawable path point
 *
 * @param x
 * @param y
 * @param pointType
 * @param labelId
 */
export function makeDrawablePathPoint2D(
  x: number,
  y: number,
  pointType: PathPointType,
  labelId: IdType | undefined
): PathPoint2D {
  const label: IdType[] = []
  if (labelId !== null && labelId !== undefined && isValidId(labelId)) {
    label.push(labelId)
  }
  return new PathPoint2D(makePathPoint2D({ x, y, pointType, label }))
}

/**
 * Drawable 2D path point
 */
export class PathPoint2D {
  /**
   * The actual path point data type
   * This separate the drawing and actual content. If we add new fields to the
   * shape, we don't have to change the drawable PathPoint2D
   */
  private _point: PathPoint2DType

  /**
   * Constructor
   *
   * @param point
   */
  constructor(point: PathPoint2DType | null = null) {
    if (point === null) {
      this._point = makePathPoint2D()
    } else {
      this._point = _.cloneDeep(point)
    }
  }

  /**
   * Access the path point type
   */
  public get type(): PathPointType {
    return this._point.pointType
  }

  /**
   * Set type
   */
  public set type(t: PathPointType) {
    this._point.pointType = t
  }

  /**
   * Access x
   */
  public get x(): number {
    return this._point.x
  }

  /** set x */
  public set x(v: number) {
    this._point.x = v
  }

  /**
   * Access y
   */
  public get y(): number {
    return this._point.y
  }

  /** set y */
  public set y(v: number) {
    this._point.y = v
  }

  /**
   * Get point id
   */
  public get id(): IdType {
    return this._point.id
  }

  /**
   * Return a copy of the shape
   */
  public shape(): PathPoint2DType {
    return _.cloneDeep(this._point)
  }

  /**
   * Convert the point to a vector for easy numeric manipulation
   */
  public vector(): Vector2D {
    return new Vector2D(this._point.x, this._point.y)
  }

  /**
   * Make a copy of this
   */
  public clone(): PathPoint2D {
    return new PathPoint2D(this._point)
  }

  /**
   * Copy from another point
   *
   * @param p
   */
  public copy(p: PathPoint2D): void {
    this._point = p.shape()
  }

  /**
   * Draw the point on a 2D context
   *
   * @param context
   * @param ratio
   * @param style
   */
  public draw(
    context: Context2D,
    ratio: number,
    style: PathPoint2DStyle
  ): void {
    context.save()
    // Convert to display resolution
    const real = this.vector().scale(ratio)
    context.beginPath()
    context.fillStyle = toCssColor(style.color)
    context.arc(real.x, real.y, style.radius, 0, 2 * Math.PI, false)
    context.closePath()
    context.fill()
    context.restore()
  }
}
