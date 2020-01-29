import { ShapeTypeName } from '../../common/types'
import { IndexedShapeType, PathPoint2DType } from '../../functional/types'
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

  /** path point state */
  private _type: PointType

  constructor (
    x: number = 0,
    y: number = 0,
    type: PointType = PointType.VERTEX
  ) {
    super(x, y)
    this._type = type
  }

  /** Get type name */
  public get typeName () {
    return ShapeTypeName.PATH_POINT_2D
  }

  /** get and set type */
  public get type (): PointType {
    return this._type
  }

  /** Set type */
  public set type (t: PointType) {
    this._type = t
    if (this._indexedShape) {
      (this._indexedShape.shape as PathPoint2DType).type = t
    }
  }

  /** Update State */
  public updateState (indexedShape: IndexedShapeType) {
    super.updateState(indexedShape)
    this._type = (indexedShape.shape as PathPoint2DType).type as PointType
  }

  /**
   * pass the value to the current point
   * @param target
   */
  public copy (target: PathPoint2D): void {
    this.x = target.x
    this.y = target.y
    this._type = target.type
  }
}
