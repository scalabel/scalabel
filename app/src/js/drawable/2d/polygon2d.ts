import _ from 'lodash'
import { sprintf } from 'sprintf-js'
import { changeLabelShape } from '../../action/common'
import { addPolygon2dLabel } from '../../action/polygon2d'
import Session from '../../common/session'
import { LabelTypes } from '../../common/types'
import { makeLabel, makePolygon } from '../../functional/states'
import { PathPoint2DType, PolygonType, ShapeType, State } from '../../functional/types'
import { Size2D } from '../../math/size2d'
import { Vector2D } from '../../math/vector2d'
import { Context2D, encodeControlColor, getColorById, toCssColor } from '../util'
import { DrawMode, Label2D } from './label2d'
import { makeEdge2DStyle, makePathPoint2DStyle, PathPoint2D, PointType } from './path_point2d'

const DEFAULT_VIEW_EDGE_STYLE = makeEdge2DStyle({ lineWidth: 4 })
const DEFAULT_VIEW_POINT_STYLE = makePathPoint2DStyle({ radius: 8 })
const DEFAULT_VIEW_HIGH_POINT_STYLE = makePathPoint2DStyle({ radius: 12 })
const DEFAULT_CONTROL_EDGE_STYLE = makeEdge2DStyle({ lineWidth: 10 })
const DEFAULT_CONTROL_POINT_STYLE = makePathPoint2DStyle({ radius: 12 })
const DEFAULT_CONTROL_HIGH_POINT_STYLE = makePathPoint2DStyle({ radius: 14 })

/** list all states */
enum Polygon2DState {
  FREE,
  DRAW,
  CLOSED,
  RESHAPE,
  MOVE,
  LINK
}

/** list all orientation types */
enum OrientationType {
  COLLINEAR,
  CLOCKWISE,
  COUNTERCLOCKWISE
}

/**
 * polygon 2d label
 */
export class Polygon2D extends Label2D {
  /** array for vertices */
  private _points: PathPoint2D[]
  /** polygon label state */
  private _state: Polygon2DState
  /** mouse position */
  private _mouseCoord: Vector2D
  /** cache shape points for dragging, both move and reshape */
  private _startingPoints: PathPoint2D[]
  /** key for drawing bezier curve */
  private _keyDownC: boolean
  constructor () {
    super()
    this._points = []
    this._state = Polygon2DState.FREE
    this._mouseCoord = new Vector2D()
    this._startingPoints = []
    this._keyDownC = false
  }

  /**
   * Draw the label on viewing or control canvas
   * @param _context
   * @param _ratio
   * @param _mode
   */
  public draw (context: Context2D, ratio: number, mode: DrawMode): void {
    const self = this

    if (self._points.length === 0) return
    let pointStyle = makePathPoint2DStyle()
    let highPointStyle = makePathPoint2DStyle()
    let edgeStyle = makeEdge2DStyle()
    let assignColor: (i: number) => number[] = () => [0]
    switch (mode) {
      case DrawMode.VIEW:
        pointStyle = _.assign(pointStyle, DEFAULT_VIEW_POINT_STYLE)
        highPointStyle = _.assign(highPointStyle,
          DEFAULT_VIEW_HIGH_POINT_STYLE)
        edgeStyle = _.assign(edgeStyle, DEFAULT_VIEW_EDGE_STYLE)
        assignColor = (_i: number): number[] => {
          return self._color
        }
        break
      case DrawMode.CONTROL:
        pointStyle = _.assign(pointStyle, DEFAULT_CONTROL_POINT_STYLE)
        highPointStyle = _.assign(
          highPointStyle, DEFAULT_CONTROL_HIGH_POINT_STYLE)
        edgeStyle = _.assign(edgeStyle, DEFAULT_CONTROL_EDGE_STYLE)
        assignColor = (i: number): number[] => {
          return encodeControlColor(self._index, i)
        }
        break
    }

    // draw line first
    edgeStyle.color = assignColor(0)
    context.save()
    context.strokeStyle = toCssColor(edgeStyle.color)
    context.lineWidth = edgeStyle.lineWidth
    context.beginPath()
    const begin = self._points[0].clone().scale(ratio)
    context.moveTo(begin.x, begin.y)
    for (let i = 1; i < self._points.length; ++i) {
      const point = self._points[i].clone().scale(ratio)
      if (point.type === PointType.MID) continue
      else if (point.type === PointType.CURVE) {
        const nextPoint = self._points[i + 1].clone().scale(ratio)
        let nextVertex
        if (self._points.length === i + 2) {
          nextVertex = self._points[0].clone().scale(ratio)
        } else {
          nextVertex = self._points[i + 2].clone().scale(ratio)
        }
        context.bezierCurveTo(point.x, point.y,
          nextPoint.x, nextPoint.y, nextVertex.x, nextVertex.y)
        i = i + 2
        continue
      } else {
        context.lineTo(point.x, point.y)
      }
    }

    if (self._state === Polygon2DState.DRAW) {
      const tmp = self._mouseCoord.clone().scale(ratio)
      context.lineTo(tmp.x, tmp.y)
      context.lineTo(begin.x, begin.y)
    } else {
      context.lineTo(begin.x, begin.y)
    }
    context.closePath()
    context.stroke()

    if (mode === DrawMode.VIEW) {
      const fillStyle = self._color.concat([0.3])
      context.fillStyle = toCssColor(fillStyle)
      context.fill()
      context.restore()
    }

    // next draw points
    if (mode === DrawMode.CONTROL || self._selected || self._highlighted) {
      if (self._state === Polygon2DState.DRAW) {
        const tmpPoint = new PathPoint2D(self._mouseCoord.x, self._mouseCoord.y)
        const tmpStyle = pointStyle
        tmpStyle.color = assignColor(self._points.length + 1)
        tmpPoint.draw(context, ratio, tmpStyle)
        let vertexNum = 1
        for (const point of self._points) {
          if (point.type === PointType.VERTEX) {
            let style = pointStyle
            if (vertexNum === self._highlightedHandle) {
              style = highPointStyle
            }
            style.color = assignColor(vertexNum)
            point.draw(context, ratio, style)
            vertexNum++
          }
        }
      } else if (self._state === Polygon2DState.CLOSED) {
        for (let i = 0; i < self._points.length; ++i) {
          const point = self._points[i]
          let style = pointStyle
          if (point.type === PointType.VERTEX) {
            if (i + 1 === self._highlightedHandle) {
              style = highPointStyle
            }
          }
          if (point.type === PointType.MID) {
            if (i + 1 === self._highlightedHandle) {
              style = highPointStyle
            }
          }
          if (point.type === PointType.CURVE) {
            if (i + 1 === self._highlightedHandle) {
              style = highPointStyle
            }
          }
          style.color = assignColor(i + 1)
          point.draw(context, ratio, style)
        }
      }
    }
  }

  /**
   * reshape the polygon: drag vertex or control points
   * @param _end
   * @param _limit
   */
  public reshape (_end: Vector2D, _limit: Size2D): void {
    if (this._selectedHandle <= 0) {
      throw new Error(sprintf('not operation reshape'))
    }
    return
  }

  /**
   * Move the polygon
   * @param _end
   * @param _limit
   */
  public move (_end: Vector2D, _limit: Size2D): void {
    if (this._selectedHandle !== 0) {
      throw new Error(sprintf('not operation move'))
    }
    return
  }

  /**
   * add a new vertex to polygon label
   * it will return whether it is closed
   * @param _coord
   * @param _limit
   */
  public addVertex (coord: Vector2D): boolean {
    if (this._points.length === 0) {
      const newPoint = new PathPoint2D(coord.x, coord.y, PointType.VERTEX)
      this._points.push(newPoint)
    } else if (
      this._highlightedHandle === 1) {
      const firstPoint = this._points[0]
      const lastPoint = this._points[this._points.length - 1]
      const midPoint = this.getMidpoint(firstPoint, lastPoint)
      this._points.push(midPoint)
      return true
    } else {
      const lastPoint = this._points[this._points.length - 1]
      const midPoint = this.getMidpoint(lastPoint, coord)
      this._points.push(midPoint)
      this._points.push(new PathPoint2D(coord.x, coord.y, PointType.VERTEX))
    }
    return false
  }

  /**
   * delete latest vertex in polygon
   */
  public deleteVertex (): boolean {
    return true
  }

  /**
   * return the midpoint of the line
   * @param prev the previous vertex
   * @param next the next vertex
   */
  public getMidpoint (prev: Vector2D, next: Vector2D): PathPoint2D {
    const mid = prev.clone().add(next).scale(0.5)
    return new PathPoint2D(mid.x, mid.y, PointType.MID)
  }

  /**
   * convert a midpoint to a vertex
   */
  public midToVertex (): void {
    return
  }

  /**
   * convert a line to a curve and vice-versa
   */
  public lineToCurve (): void {
    return
  }

  /**
   * Handle mouse down
   * @param coord
   */
  public onMouseDown (coord: Vector2D): boolean {
    this._mouseDown = true
    this._mouseCoord = coord.clone()
    if (this._selected) {
      this._mouseDownCoord = coord.clone()
      if (this._state === Polygon2DState.CLOSED && this._selectedHandle < 0) {
        // not click edge or point
        return true
      } else if (this._state === Polygon2DState.CLOSED &&
        this._selectedHandle > 0) {
          // click point
        if (this._keyDownC) {
          // convert line to bezier curve
          this.lineToCurve()
        } else {
          // drag vertex or midpoint
          this._state = Polygon2DState.RESHAPE
          this.editing = true
          this._startingPoints = []
          for (const point of this._points) {
            this._startingPoints.push(
              new PathPoint2D(point.x, point.y, point.type))
          }
          if (this._points[this._selectedHandle - 1].type === PointType.MID) {
            // drag midpoint: convert midpoint to vertex first
            this.midToVertex()
          }
        }
        return true
      } else if (this._state === Polygon2DState.CLOSED &&
        this._selectedHandle === 0) {
        // drag edges
        this._state = Polygon2DState.MOVE
        this.editing = true
        this._startingPoints = []
        for (const point of this._points) {
          this._startingPoints.push(point.clone())
        }
        return true
      }
      return true
    }
    return false
  }

  /**
   * Handle mouse move
   * @param coord
   * @param _limit
   */
  public onMouseMove (coord: Vector2D, _limit: Size2D,
                      labelIndex: number, handleIndex: number): boolean {
    if (this._state === Polygon2DState.DRAW) {
      // move to add vertex
      this._mouseCoord = coord.clone()
      if (labelIndex === this._index) {
        this._highlightedHandle = handleIndex
      }
    } else if (this._mouseDown === true &&
      this._state === Polygon2DState.RESHAPE) {
      // dragging point
      this.reshape(coord, _limit)
    } else if (this._mouseDown === true &&
      this._state === Polygon2DState.MOVE) {
      // dragging edges
      this.move(coord, _limit)
    }
    return true
  }

  /**
   * Handle mouse up
   * @param coord
   */
  public onMouseUp (coord: Vector2D): boolean {
    this._mouseCoord = coord.clone()
    if (this.editing === true &&
      this._state === Polygon2DState.DRAW) {
      // add vertex
      const isClosed = this.addVertex(coord)
      if (isClosed) {
        // finish adding when it is closed
        this._state = Polygon2DState.CLOSED
        this.editing = false
      }
    } else if (this.editing === true &&
      this._state === Polygon2DState.RESHAPE) {
      // finish dragging point
      this._state = Polygon2DState.CLOSED
      this.editing = false
    } else if (this.editing === true &&
      this._state === Polygon2DState.MOVE) {
      // finish dragging edges
      this._state = Polygon2DState.CLOSED
      this.editing = false
    }
    this._mouseDown = false
    return this.commitLabel()
  }

  /**
   * handle keyboard down event
   * @param e pressed key
   */
  public onKeyDown (e: string): boolean {
    switch (e) {
      case 'D':
      case 'd':
        return this.deleteVertex()
      case 'C':
      case 'c':
        this._keyDownC = true
        break
    }
    return true
  }

  /**
   * handle keyboard up event
   * @param e pressed key
   */
  public onKeyUp (e: string): void {
    switch (e) {
      case 'C':
      case 'c':
        this._keyDownC = false
        break
    }
  }

  /**
   * convert this drawable polygon to a polygon state
   */
  public toPolygon (): PolygonType {
    const pathPoints: PathPoint2DType[] = new Array()
    for (const point of this._points) {
      if (point.type === PointType.MID) continue
      pathPoints.push(point.toPathPoint())
    }
    return makePolygon({ points: pathPoints })
  }

  /**
   * finish one operation and whether add new label, save changes
   */
  public commitLabel (): boolean {
    const valid = this.isValid()
    if (!this._label) {
      return false
    }
    if (!valid && !this.editing) {
      if (this.labelId === -1) { // create invalid
        return false
      } else {                   // drag invalid
        this._points = []
        for (const point of this._startingPoints) {
          this._points.push(point.clone())
        }
        return true
      }
    }
    if (!this.editing) {
      if (this._labelId < 0) {
        const p = this.toPolygon()
        Session.dispatch(addPolygon2dLabel(
          this._label.item, this._label.category, p.points))
      } else {
        const p = this.toPolygon()
        Session.dispatch(changeLabelShape(
          this._label.item, this._label.shapes[0], p))
      }
      return true
    }
    return true
  }

  /**
   * create new polygon label
   * @param _state
   * @param _start
   */
  public initTemp (state: State, _start: Vector2D): void {
    this.editing = true
    this._state = Polygon2DState.DRAW
    const itemIndex = state.user.select.item
    this._order = state.task.status.maxOrder + 1
    this._label = makeLabel({
      type: LabelTypes.POLYGON_2D, id: -1, item: itemIndex,
      category: [state.user.select.category],
      order: this._order
    })
    this._labelId = -1
    this._color = getColorById(state.task.status.maxLabelId + 1)
    this.setSelected(true, 1)
  }

  /**
   * to update the shape of polygon
   * @param _shapes
   */
  public updateShapes (shapes: ShapeType[]): void {
    if (this._label) {
      const polygon = shapes[0] as PolygonType
      if (!_.isEqual(this.toPolygon, polygon)) {
        this._points = new Array()
        for (const point of polygon.points) {
          switch (point.type) {
            case PointType.VERTEX: {
              const currPoint =
                new PathPoint2D(point.x, point.y, PointType.VERTEX)
              if (this._points.length !== 0) {
                const prevPoint = this._points[this._points.length - 1]
                if (prevPoint.type === PointType.VERTEX) {
                  this._points.push(this.getMidpoint(prevPoint, currPoint))
                }
              }
              this._points.push(currPoint)
              break
            }
            case PointType.CURVE: {
              this._points.push(
                new PathPoint2D(point.x, point.y, PointType.CURVE))
              break
            }
          }
        }
        const tmp = this._points[this._points.length - 1]
        if (tmp.type === PointType.VERTEX) {
          this._points.push(this.getMidpoint(tmp, this._points[0]))
        }
        this._state = Polygon2DState.CLOSED
      }
    }
  }

  /**
   * Given three collinear points p, q, r, the function checks if q lies
   * on line segment pr
   */
  public onSegment (p: PathPoint2D, q: PathPoint2D, r: PathPoint2D): boolean {
    if (q.x <= Math.max(p.x, r.x) && q.x >= Math.min(p.x, r.x) &&
    q.y <= Math.max(p.y, r.y) && q.y >= Math.min(p.y, r.y)) {
      return true
    }
    return false
  }

  /**
   * To find orientation of ordered triplet
   * The function returns following values
   * 0 -> p, q and r are collinear
   * 1 -> Clockwise
   * 2 -> Counterclockwise
   */
  public orientation (p: PathPoint2D, q: PathPoint2D, r: PathPoint2D):
  OrientationType {
    const val = (q.y - p.y) * (r.x - q.x) - (q.x - p.x) * (r.y - q.y)
    switch (true) {
      case val === 0:
        return OrientationType.COLLINEAR
      case val > 0:
        return OrientationType.CLOCKWISE
      default:
        return OrientationType.COUNTERCLOCKWISE
    }
  }

  /**
   * to check whether two line segments intersect
   */
  public intersect (a: PathPoint2D[], b: PathPoint2D[]): boolean {
    const p1 = a[0]
    const q1 = a[1]
    const p2 = b[0]
    const q2 = b[1]
    const o1 = this.orientation(p1, q1, p2)
    const o2 = this.orientation(p1, q1, q2)
    const o3 = this.orientation(p2, q2, p1)
    const o4 = this.orientation(p2, q2, q1)
    if (o1 !== o2 && o3 !== o4) {
      return true
    }
    if (o1 === OrientationType.COLLINEAR
      && this.onSegment(p1, p2, q1)) return true
    if (o2 === OrientationType.COLLINEAR
      && this.onSegment(p1, q2, q1)) return true
    if (o3 === OrientationType.COLLINEAR
      && this.onSegment(p2, p1, q2)) return true
    if (o4 === OrientationType.COLLINEAR
      && this.onSegment(p2, q1, q2)) return true
    return false
  }

  /**
   * to check whether the label is valid
   */
  public isValid (): boolean {
    const lines: PathPoint2D[][] = []
    let l = 0
    let r = 1
    while (r < this._points.length) {
      if (this._points[r].type === PointType.VERTEX) {
        lines.push([this._points[l], this._points[r]])
        l = r
      }
      r++
    }
    if (this._state === Polygon2DState.CLOSED) {
      if (this._points[l].type === PointType.VERTEX) {
        lines.push([this._points[l], this._points[0]])
      }
    }
    for (let i = 0; i < lines.length; i++) {
      for (let j = i + 1; j < lines.length; j++) {
        if (lines[i][0].x === lines[j][0].x &&
          lines[i][0].y === lines[j][0].y) {
          continue
        }
        if (lines[i][0].x === lines[j][1].x &&
          lines[i][0].y === lines[j][1].y) {
          continue
        }
        if (lines[i][1].x === lines[j][0].x &&
          lines[i][1].y === lines[j][0].y) {
          continue
        }
        if (lines[i][1].x === lines[j][1].x &&
          lines[i][1].y === lines[j][1].y) {
          continue
        }
        if (this.intersect(lines[i], lines[j])) {
          return false
        }
      }
    }
    return true
  }
}
