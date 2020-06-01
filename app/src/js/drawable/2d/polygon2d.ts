import _ from 'lodash'
import { sprintf } from 'sprintf-js'
import { Cursor, Key, LabelTypeName } from '../../common/types'
import { makeLabel, makePolygon } from '../../functional/states'
import { LabelType, PolygonType, PolyPathPoint2DType, ShapeType, State } from '../../functional/types'
import { Size2D } from '../../math/size2d'
import { Vector2D } from '../../math/vector2d'
import { blendColor, Context2D, encodeControlColor, toCssColor } from '../util'
import { DASH_LINE, MIN_SIZE, OPACITY } from './common'
import { DrawMode, Label2D } from './label2d'
import { Label2DList } from './label2d_list'
import { makeEdge2DStyle, makePathPoint2DStyle, PathPoint2D, PointType } from './path_point2d'

const DEFAULT_VIEW_EDGE_STYLE = makeEdge2DStyle({ lineWidth: 4 })
const DEFAULT_VIEW_POINT_STYLE = makePathPoint2DStyle({ radius: 8 })
const DEFAULT_VIEW_HIGH_POINT_STYLE = makePathPoint2DStyle({ radius: 12 })
const DEFAULT_CONTROL_EDGE_STYLE = makeEdge2DStyle({ lineWidth: 10 })
const DEFAULT_CONTROL_POINT_STYLE = makePathPoint2DStyle({ radius: 12 })
const DEFAULT_CONTROL_HIGH_POINT_STYLE = makePathPoint2DStyle({ radius: 14 })

/** list all states */
export enum Polygon2DState {
  FREE,
  DRAW,
  FINISHED,
  RESHAPE,
  MOVE
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
  /** The hashed list of keys currently down */
  private _keyDownMap: { [key: string]: boolean }
  /** open or closed */
  private _closed: boolean

  constructor (labelList: Label2DList, closed: boolean) {
    super(labelList)
    this._points = []
    this._state = Polygon2DState.FREE
    this._mouseCoord = new Vector2D()
    this._startingPoints = []
    this._keyDownMap = {}
    this._closed = closed
  }

  /** Get cursor for highlighting */
  public get highlightCursor () {
    if (this._state === Polygon2DState.DRAW) {
      return Cursor.CROSSHAIR
    } else if (
      this._highlightedHandle > 0 &&
      this._highlightedHandle <= this._points.length
    ) {
      return Cursor.DEFAULT
    } else {
      return Cursor.MOVE
    }
  }

  /**
   * Draw the label on viewing or control canvas
   * @param _context
   * @param _ratio
   * @param _mode
   */
  public draw (context: Context2D, ratio: number, mode: DrawMode): void {
    const self = this
    const numPoints = self._points.length

    if (numPoints === 0) return
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
        assignColor = (i: number): number[] => {
          if (i > 0 && i <= this._points.length &&
            this._points[i - 1].type !== PointType.VERTEX) {
            return blendColor(self._color, [255, 255, 255], 0.7)
          } else {
            return self._color
          }
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
    for (let i = 1; i < numPoints; ++i) {
      const point = self._points[i].clone().scale(ratio)
      if (point.type === PointType.CURVE) {
        const nextPoint = self._points[(i + 1) % numPoints].clone().scale(ratio)
        const nextVertex =
          self._points[(i + 2) % numPoints].clone().scale(ratio)
        context.bezierCurveTo(point.x, point.y,
          nextPoint.x, nextPoint.y, nextVertex.x, nextVertex.y)
        i = i + 2
      } else if (point.type === PointType.VERTEX) {
        context.lineTo(point.x, point.y)
      }
    }

    if (self._state === Polygon2DState.DRAW) {
      const tmp = self._mouseCoord.clone().scale(ratio)
      context.lineTo(tmp.x, tmp.y)
    }

    if (this._closed) {
      context.lineTo(begin.x, begin.y)
      context.closePath()
      if (mode === DrawMode.VIEW) {
        const fillStyle = self._color.concat(OPACITY)
        context.fillStyle = toCssColor(fillStyle)
        context.fill()
      }
    }
    context.stroke()
    context.restore()

    if (mode === DrawMode.CONTROL || self._selected || self._highlighted) {
      // for bezier curve
      context.save()
      context.setLineDash(DASH_LINE)
      context.beginPath()
      for (let i = 0; i < numPoints; ++i) {
        const point = self._points[i].clone().scale(ratio)
        const nextPoint = self._points[(i + 1) % numPoints].clone().scale(ratio)
        if ((point.type === PointType.VERTEX &&
          nextPoint.type === PointType.CURVE) ||
          point.type === PointType.CURVE) {
          context.moveTo(point.x, point.y)
          context.lineTo(nextPoint.x, nextPoint.y)
          context.stroke()
        }
      }
      context.closePath()
      context.restore()

      // draw points
      if (self._state === Polygon2DState.DRAW) {
        const tmpPoint = new PathPoint2D(self._mouseCoord.x, self._mouseCoord.y)
        const tmpStyle = pointStyle
        tmpStyle.color = assignColor(numPoints + 1)
        tmpPoint.draw(context, ratio, tmpStyle)
        let numVertices = 1
        _.forEach(self._points, (point, index) => {
          if (point.type === PointType.VERTEX) {
            let style = pointStyle
            if (numVertices === self._highlightedHandle) {
              style = highPointStyle
            }
            style.color = assignColor(index + 1)
            point.draw(context, ratio, style)
            numVertices++
          }
        })
      } else if (self._state === Polygon2DState.FINISHED) {
        for (let i = 0; i < numPoints; ++i) {
          const point = self._points[i]
          let style = pointStyle
          if (i + 1 === self._highlightedHandle) {
            style = highPointStyle
          }
          style.color = assignColor(i + 1)
          point.draw(context, ratio, style)
        }
      }
    }
  }

  /**
   * Handle mouse down
   * @param coord
   */
  public onMouseDown (coord: Vector2D, handleIndex: number): boolean {
    this._mouseDown = true
    this._mouseCoord = coord.clone()
    if (this._selected) {
      this._mouseDownCoord = coord.clone()
      if (this._state === Polygon2DState.FINISHED &&
          this._highlightedHandle < 0) {
        // not click edge or point
        return true
      } else if (this._state === Polygon2DState.FINISHED &&
        this._highlightedHandle > 0) {
        // click point
        this._state = Polygon2DState.RESHAPE
        this.editing = true
        if (this.isKeyDown(Key.C_UP) || this.isKeyDown(Key.C_LOW)) {
          // convert line to bezier curve
          this.lineToCurve()
        } else if (this.isKeyDown(Key.D_UP) || this.isKeyDown(Key.D_LOW)) {
          // delete vertex
          this.toCache()
          this.deleteVertex()
        } else {
          // drag vertex or midpoint
          this.toCache()
          if (this._points[this._highlightedHandle - 1].type ===
              PointType.MID) {
            // drag midpoint: convert midpoint to vertex first
            this.midToVertex()
          }
        }
        this._labelList.addUpdatedLabel(this)
        return true
      } else if (this._state === Polygon2DState.FINISHED &&
        this._highlightedHandle === 0 && handleIndex === 0) {
        // drag edges
        this._state = Polygon2DState.MOVE
        this.editing = true
        this.toCache()
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
                      _labelIndex: number, handleIndex: number): boolean {
    if (this._state === Polygon2DState.DRAW) {
      // move to add vertex
      this._mouseCoord = coord.clone()
      this._highlightedHandle = handleIndex
    } else if (this._mouseDown === true &&
      this._state === Polygon2DState.RESHAPE) {
      // dragging point
      this.reshape(coord, _limit)
      this._labelList.addUpdatedLabel(this)
    } else if (this._mouseDown === true &&
      this._state === Polygon2DState.MOVE) {
      // dragging edges
      this.move(coord, _limit)
      this._labelList.addUpdatedLabel(this)
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
      const isFinished = this.addVertex(coord)
      if (isFinished) {
        // finish adding when it is FINISHED
        this._state = Polygon2DState.FINISHED
        this.editing = false
        if (this.isValid()) {
          this._labelList.addUpdatedLabel(this)
        }
      }
    } else if (this.editing === true &&
      this._state === Polygon2DState.RESHAPE) {
      // finish dragging point
      this._state = Polygon2DState.FINISHED
      this.editing = false
    } else if (this.editing === true &&
      this._state === Polygon2DState.MOVE) {
      // finish dragging edges
      this._state = Polygon2DState.FINISHED
      this.editing = false
    }
    this._mouseDown = false
    if (!this.isValid() && !this.editing && !this.temporary) {
      this._points = []
      for (const point of this._startingPoints) {
        this._points.push(point.clone())
      }
    }
    return true
  }

  /**
   * handle keyboard down event
   * @param e pressed key
   */
  public onKeyDown (e: string): boolean {
    this._keyDownMap[e] = true
    if ((e === Key.D_UP || e === Key.D_LOW) &&
      this._state === Polygon2DState.DRAW) {
      return this.deleteVertex()
    }
    return true
  }

  /**
   * handle keyboard up event
   * @param e pressed key
   */
  public onKeyUp (e: string): void {
    delete this._keyDownMap[e]
  }

  /**
   * to check whether the label is valid
   */
  public isValid (): boolean {
    if (this._state !== Polygon2DState.FINISHED) {
      return false
    }
    if (this._closed) {
      const lines: PathPoint2D[][] = []
      let l = 0
      let r = 1
      let maxx = Number.MIN_VALUE
      let minx = Number.MAX_VALUE
      let maxy = Number.MIN_VALUE
      let miny = Number.MAX_VALUE
      for (const item of this._points) {
        maxx = Math.max(maxx, item.x)
        minx = Math.min(minx, item.x)
        maxy = Math.max(maxy, item.y)
        miny = Math.min(miny, item.y)
      }
      if ((maxx - minx) * (maxy - miny) < MIN_SIZE) {
        return false
      }
      while (r < this._points.length) {
        if (this._points[r].type === PointType.VERTEX) {
          lines.push([this._points[l], this._points[r]])
          l = r
        }
        r++
      }
      if (this._state === Polygon2DState.FINISHED) {
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
    } else {
      // TODO: check polyline validation
      if (this._points.length <= 1) {
        return false
      }
    }
    return true
  }

  /** Get shape objects for committing to state */
  public shapes (): ShapeType[] {
    if (!this._label) {
      throw new Error('Uninitialized label')
    }
    /**
     * This is a temporary solution for assigning the correct ID to the shapes
     * We should initialize the shape when the temporary label is created.
     * Also store the shape id properly so that the generated shape state has
     * the right id directly.
     */
    const polygon = this.toPolygon()
    if (!this._temporary) {
      polygon.id = this._label.shapes[0]
    }
    return [polygon]
  }

  /**
   * to update the shape of polygon
   * @param _shapes
   */
  public updateShapes (shapes: ShapeType[]): void {
    if (this._label) {
      const polygon = shapes[0] as PolygonType
      if (!_.isEqual(this.toPolygon(), polygon)) {
        this._points = new Array()
        for (const point of polygon.points) {
          switch (point.pointType) {
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
        if (this._closed) {
          const tmp = this._points[this._points.length - 1]
          if (tmp.type === PointType.VERTEX) {
            this._points.push(this.getMidpoint(tmp, this._points[0]))
          }
        }
        this._state = Polygon2DState.FINISHED
      }
    }
  }

  /**
   * create new polygon label
   * @param _state
   * @param _start
   */
  protected initTempLabel (state: State, _start: Vector2D): LabelType {
    this.editing = true
    this._state = Polygon2DState.DRAW
    const itemIndex = state.user.select.item
    const labelType = this._closed ?
                LabelTypeName.POLYGON_2D : LabelTypeName.POLYLINE_2D
    const label = makeLabel({
      type: labelType, item: itemIndex,
      category: [state.user.select.category],
      order: this._order
    })
    this._highlightedHandle = 1
    return label
  }

  /**
   * Move the polygon
   * @param _end
   * @param _limit
   */
  private move (end: Vector2D, _limit: Size2D): void {
    if (this._highlightedHandle !== 0) {
      throw new Error(sprintf('not operation move'))
    }
    const delta = end.clone().subtract(this._mouseDownCoord)
    for (let i = 0; i < this._points.length; ++i) {
      this._points[i].x = this._startingPoints[i].x + delta.x
      this._points[i].y = this._startingPoints[i].y + delta.y
    }
  }

  /**
   * add a new vertex to polygon label
   * it will return whether it is FINISHED
   * @param _coord
   * @param _limit
   */
  private addVertex (coord: Vector2D): boolean {
    let closed = false
    if (this._points.length === 0) {
      // First point for the polygon
      const newPoint = new PathPoint2D(coord.x, coord.y, PointType.VERTEX)
      this._points.push(newPoint)
    } else if (this._highlightedHandle === 1) {
      // When the polygon is FINISHED, only add a new mid point
      if (this._closed) {
        const point1 = this._points[0]
        const point2 = this._points[this._points.length - 1]
        const midPoint = this.getMidpoint(point1, point2)
        this._points.push(midPoint)
      }
      closed = true
    } else {
      // add a new vertex and the new mid point on the new edge
      const point2 = this._points[this._points.length - 1]
      const midPoint = this.getMidpoint(point2, coord)
      this._points.push(midPoint)
      this._points.push(new PathPoint2D(coord.x, coord.y, PointType.VERTEX))
    }
    return closed
  }

  /**
   * delete one vertex in polygon
   */
  private deleteVertex (): boolean {
    const numPoints = this._points.length
    if (numPoints === 0) return false
    switch (this._state) {
      case Polygon2DState.DRAW:
        const indexGap = numPoints === 1 ? 1 : 2
        this._points.splice(numPoints - indexGap, indexGap)
        break
      case Polygon2DState.RESHAPE:
        let numVertices = 0
        for (const point of this._points) {
          if (point.type === PointType.VERTEX) {
            ++numVertices
          }
        }
        const minVertexNumber = this._closed ? 4 : 3
        if (numVertices >= minVertexNumber && this._highlightedHandle > 0 &&
          this._points[this._highlightedHandle - 1].type === PointType.VERTEX) {
          const selectedLabelIndex = this._highlightedHandle - 1
          let numPreviousPointsToDelete = 0
          let previousVertexIndex = 0
          for (
            let i = (selectedLabelIndex - 1 + numPoints) % numPoints;
            i >= 0;
            i--
          ) {
            if (this._points[i].type === PointType.VERTEX) {
              previousVertexIndex = i
              break
            }
            numPreviousPointsToDelete++
          }

          let nextVertexIndex = 0
          let numNextPointsToDelete = 0
          for (
            let i = (selectedLabelIndex + 1) % numPoints;
            i < this._points.length;
            i = (i + 1) % numPoints
          ) {
            if (this._points[i].type === PointType.VERTEX) {
              nextVertexIndex = i
              break
            }
            numNextPointsToDelete++
          }

          const prevPoint = this._points[previousVertexIndex]
          const nextPoint = this._points[nextVertexIndex]
          this._points[selectedLabelIndex].copy(
            this.getMidpoint(prevPoint, nextPoint))

          if (numNextPointsToDelete === 0) {
            this._points.splice(
              selectedLabelIndex,
              1
              )
            this._points.splice(
              previousVertexIndex + 1,
              numPreviousPointsToDelete
            )
          } else if (numPreviousPointsToDelete === 0) {
            this._points.splice(
              selectedLabelIndex + 1,
              numNextPointsToDelete
            )
            this._points.splice(
              selectedLabelIndex,
              1
            )
          } else if (previousVertexIndex > selectedLabelIndex) {
            this._points.splice(
              previousVertexIndex + 1,
              numPreviousPointsToDelete
            )
            this._points.splice(
              selectedLabelIndex + 1,
              numNextPointsToDelete
            )
          } else {
            this._points.splice(
              selectedLabelIndex + 1,
              numNextPointsToDelete
            )
            this._points.splice(
              previousVertexIndex + 1,
              numPreviousPointsToDelete
            )
          }

          while (this._points[0].type !== PointType.VERTEX) {
            const point1 = this._points.shift()
            if (point1) {
              this._points.push(point1)
            }
          }
        }
    }
    return (this._points.length !== 0)
  }

  /**
   * reshape the polygon: drag vertex or control points
   * @param _end
   * @param _limit
   */
  private reshape (end: Vector2D, _limit: Size2D): void {
    //
    if (this._highlightedHandle <= 0) {
      throw new Error(sprintf('not operation reshape'))
    }
    const point = this._points[this._highlightedHandle - 1]
    point.x = end.clone().x
    point.y = end.clone().y

    // change corresponding midpoint
    if (point.type === PointType.VERTEX) {
      const numPoints = this._points.length
      const selectedLabelIndex = this._highlightedHandle - 1
      const nextPoint = this.getNextIndex(selectedLabelIndex) === -1 ?
        null
        : this._points[(selectedLabelIndex + 1) % numPoints]
      const prevPoint = this.getPreviousIndex(selectedLabelIndex) === -1 ?
        null
        : this._points[(selectedLabelIndex + numPoints - 1) % numPoints]

      if (prevPoint && prevPoint.type !== PointType.CURVE) {
        const prevVertex = this._points[
          (selectedLabelIndex + numPoints - 2) % numPoints]
        prevPoint.copy(this.getMidpoint(prevVertex, point))
      }
      if (nextPoint && nextPoint.type !== PointType.CURVE) {
        const nextVertex = this._points[
          (selectedLabelIndex + 2) % numPoints]
        nextPoint.copy(this.getMidpoint(point, nextVertex))
      }
    }
  }

  /**
   * return the midpoint of the line
   * @param prev the previous vertex
   * @param next the next vertex
   */
  private getMidpoint (prev: Vector2D, next: Vector2D): PathPoint2D {
    const mid = prev.clone().add(next).scale(0.5)
    return new PathPoint2D(mid.x, mid.y, PointType.MID)
  }

  /**
   * return the control points of the bezier curve
   * @param src the source vertex
   * @param dest the next vertex
   */
  private getCurvePoints (src: Vector2D, dest: Vector2D): PathPoint2D[] {
    const first = src.clone().scale(2).add(dest).scale(1 / 3)
    const point1 = new PathPoint2D(first.x, first.y, PointType.CURVE)
    const second = dest.clone().scale(2).add(src).scale(1 / 3)
    const point2 = new PathPoint2D(second.x, second.y, PointType.CURVE)
    return [point1, point2]
  }

  /**
   * convert a midpoint to a vertex
   */
  private midToVertex (): void {
    const selectedLabelIndex = this._highlightedHandle - 1
    const point = this._points[selectedLabelIndex]
    if (point.type !== PointType.MID) {
      throw new Error(sprintf('not a midpoint'))
    }
    const prevPoint =
      this._points[this.getPreviousIndex(selectedLabelIndex)]
    const mid1 = this.getMidpoint(prevPoint, point)
    const mid2 = this.getMidpoint(
      point, this._points[this.getNextIndex(selectedLabelIndex)])
    this._points.splice(selectedLabelIndex, 0, mid1)
    this._points.splice(
      this.getNextIndex(this.getNextIndex(selectedLabelIndex)), 0, mid2
    )
    point.type = PointType.VERTEX
    this._highlightedHandle++
  }

  /**
   * convert a line to a curve and vice-versa
   */
  private lineToCurve (): void {
    const selectedLabelIndex = this._highlightedHandle - 1
    const point = this._points[selectedLabelIndex]
    const highlightedHandleIndex = this._highlightedHandle - 1
    switch (point.type) {
      case PointType.MID: // from midpoint to curve
        const prevPoint =
          this._points[this.getPreviousIndex(highlightedHandleIndex)]
        const nextPoint =
          this._points[this.getNextIndex(highlightedHandleIndex)]
        const controlPoints = this.getCurvePoints(prevPoint, nextPoint)
        this._points[highlightedHandleIndex] = controlPoints[0]
        this._points.splice(
          this.getNextIndex(highlightedHandleIndex), 0, controlPoints[1]
        )
        break
      case PointType.CURVE: // from curve to midpoint
        const newMidPointIndex =
          (this._points[highlightedHandleIndex - 1].type === PointType.CURVE) ?
            this.getPreviousIndex(highlightedHandleIndex) :
            highlightedHandleIndex
        this._points.splice(highlightedHandleIndex, 1)
        this._points[newMidPointIndex].copy(this.getMidpoint(
          this._points[this.getNextIndex(newMidPointIndex)],
          this._points[this.getPreviousIndex(newMidPointIndex)]
        ))
    }
  }

  /**
   * save current points to cache
   */
  private toCache (): void {
    this._startingPoints = []
    for (const point of this._points) {
      this._startingPoints.push(
        new PathPoint2D(point.x, point.y, point.type))
    }
  }

  /**
   * Given three collinear points p, q, r, the function checks if q lies
   * on line segment pr
   */
  private onSegment (
    p: PathPoint2D, q: PathPoint2D, r: PathPoint2D): boolean {
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
  private orientation (p: PathPoint2D, q: PathPoint2D, r: PathPoint2D):
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
  private intersect (a: PathPoint2D[], b: PathPoint2D[]): boolean {
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
   * Whether a specific key is pressed down
   * @param key - the key to check
   */
  private isKeyDown (key: Key): boolean {
    return this._keyDownMap[key]
  }

  /**
   * convert this drawable polygon to a polygon state
   */
  private toPolygon (): PolygonType {
    const pathPoints: PolyPathPoint2DType [] = new Array()
    for (const point of this._points) {
      if (point.type === PointType.MID) continue
      pathPoints.push(point.toPathPoint())
    }
    return makePolygon({ points: pathPoints })
  }

  /**
   * Get index of previous point, circular indexing
   * @param index
   */
  private getPreviousIndex (index: number): number {
    if (!this._closed && index === 0) {
      return -1
    } else {
      return (index - 1 + this._points.length) % this._points.length
    }
  }

  /**
   * Get index of previous point, circular indexing
   * @param index
   */
  private getNextIndex (index: number): number {
    if (!this._closed && index === this._points.length - 1) {
      return -1
    } else {
      return (index + 1) % this._points.length
    }
  }
}
