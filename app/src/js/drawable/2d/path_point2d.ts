import { makePathPoint } from '../../functional/states'
import { PathPoint2DType } from '../../functional/types'
import { Context2D, toCssColor } from '../util'
import { Point2D } from './point2d'

export enum PointType {
  VERTEX = 'vertex',
  MID = 'mid',
  CURVE = 'bezier'
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
 * @param style
 */
export function makeEdge2DStyle (
  style: Partial<Edge2DStyle> = {}): Edge2DStyle {
  return {
    lineWidth: 1,
    color: [0, 0, 0, 1],
    dashed: false,
    ...style
  }
}

export interface PathPoint2DStyle {
  /** radius of the point on drawing */
  radius: number
  /** color of the point */
  color: number[]
}

/**
 * Generate PathPoint2D style with default parameters
 * @param style
 */
export function makePathPoint2DStyle (
    style: Partial<PathPoint2DStyle> = {}): PathPoint2DStyle {
  return {
    radius: 1,
    color: [0, 0, 0],
    ...style
  }
}

/** points2D for polygon */
export class PathPoint2D extends Point2D {

  /** point type */
  private _type: PointType

  constructor (
    x: number = 0, y: number = 0, type: PointType = PointType.VERTEX) {
    super(x, y)
    this._type = type
  }

  /** get and set type */
  public get type (): PointType {
    return this._type
  }

  public set type (t: PointType) {
    this._type = t
  }

  /**
   * convert this drawable pathPoint to a pathPoint state
   */
  public toPathPoint (): PathPoint2DType {
    return makePathPoint({
      x: this.x, y: this.y, type: this.type
    })
  }

  /**
   * pass the value to the current point
   * @param target
   */
  public copy (target: PathPoint2D): void {
    this.x = target.x
    this.y = target.y
    this.type = target.type
  }

  /**
   * Draw the point on a 2D context
   * @param context
   * @param ratio
   * @param style
   */
  public draw (
    context: Context2D, ratio: number, style: PathPoint2DStyle): void {
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
