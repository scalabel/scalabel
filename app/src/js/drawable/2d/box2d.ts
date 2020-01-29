import _ from 'lodash'
import { Cursor, LabelTypeName, ShapeTypeName } from '../../common/types'
import { makeLabel } from '../../functional/states'
import { LabelType, ShapeType, State } from '../../functional/types'
import { Size2D } from '../../math/size2d'
import { Vector2D } from '../../math/vector2d'
import { blendColor, Context2D, encodeControlColor } from '../util'
import { DrawMode, Label2D } from './label2d'
import { Label2DList } from './label2d_list'
import { makePoint2DStyle, Point2D } from './point2d'
import { makeRect2DStyle, Rect2D } from './rect2d'

const DEFAULT_VIEW_RECT_STYLE = makeRect2DStyle({ lineWidth: 4 })
const DEFAULT_VIEW_POINT_STYLE = makePoint2DStyle({ radius: 8 })
const DEFAULT_VIEW_HIGH_POINT_STYLE = makePoint2DStyle({ radius: 12 })
const DEFAULT_CONTROL_RECT_STYLE = makeRect2DStyle({ lineWidth: 10 })
const DEFAULT_CONTROL_POINT_STYLE = makePoint2DStyle({ radius: 12 })
const MIN_AREA = 10

enum Handles {
  TOP_LEFT = 0,
  TOP_MIDDLE = 1,
  TOP_RIGHT = 2,
  RIGHT_MIDDLE = 3,
  BOTTOM_RIGHT = 4,
  BOTTOM_MIDDLE = 5,
  BOTTOM_LEFT = 6,
  LEFT_MIDDLE = 7,
  EDGE = 8
}

/**
 * Box2d Label
 */
export class Box2D extends Label2D {
  /** cache shape for moving */
  private _startingRect: Rect2D
  /** Corners and midpoints */
  private _points:
    [Point2D, Point2D, Point2D, Point2D, Point2D, Point2D, Point2D, Point2D]

  constructor (labelList: Label2DList) {
    super(labelList)
    this._points = [
      new Point2D(), new Point2D(), new Point2D(), new Point2D(),
      new Point2D(), new Point2D(), new Point2D(), new Point2D()
    ]

    this._startingRect = new Rect2D()
  }

  /** Get cursor for use when highlighting */
  public get highlightCursor (): string {
    switch (this._highlightedHandle) {
      case Handles.EDGE:
        return Cursor.MOVE
      case Handles.TOP_LEFT:
      case Handles.BOTTOM_RIGHT:
        return Cursor.NWSE_RESIZE
      case Handles.TOP_RIGHT:
      case Handles.BOTTOM_LEFT:
        return Cursor.NESW_RESIZE
      case Handles.TOP_MIDDLE:
      case Handles.BOTTOM_MIDDLE:
        return Cursor.NS_RESIZE
      case Handles.LEFT_MIDDLE:
      case Handles.RIGHT_MIDDLE:
        return Cursor.EW_RESIZE
    }

    return super.highlightCursor
  }

  /** Draw the label on viewing or control canvas */
  public draw (context: Context2D, ratio: number, mode: DrawMode): void {
    const self = this

    // Set proper drawing styles
    let pointStyle = makePoint2DStyle()
    let highPointStyle = makePoint2DStyle()
    let rectStyle = makeRect2DStyle()
    let assignColor: (i: number) => number[] = () => [0]
    switch (mode) {
      case DrawMode.VIEW:
        pointStyle = _.assign(pointStyle, DEFAULT_VIEW_POINT_STYLE)
        highPointStyle = _.assign(highPointStyle, DEFAULT_VIEW_HIGH_POINT_STYLE)
        rectStyle = _.assign(rectStyle, DEFAULT_VIEW_RECT_STYLE)
        assignColor = (i: number): number[] => {
          if (i % 2 === 1) {
            // midpoint
            return blendColor(self._color, [255, 255, 255], 0.7)
          } else {
            // vertex
            return self._color
          }
        }
        break
      case DrawMode.CONTROL:
        pointStyle = _.assign(pointStyle, DEFAULT_CONTROL_POINT_STYLE)
        highPointStyle = _.assign(
          highPointStyle, DEFAULT_CONTROL_POINT_STYLE)
        rectStyle = _.assign(rectStyle, DEFAULT_CONTROL_RECT_STYLE)
        assignColor = (i: number): number[] => {
          return encodeControlColor(self._index, i)
        }
        break
    }

    // Draw!!!
    const rect = Object.values(self._shapes)[0] as Rect2D
    rectStyle.color = assignColor(this._points.length)
    rect.draw(context, ratio, rectStyle)
    if (mode === DrawMode.VIEW) {
      self.drawTag(context, ratio, [rect.x, rect.y], self._color)
    }
    if (mode === DrawMode.CONTROL || this._selected || this._highlighted) {
      for (let i = 0; i < this._points.length; i += 1) {
        let style
        if (i === self._highlightedHandle) {
          style = highPointStyle
        } else {
          style = pointStyle
        }
        style.color = assignColor(i)
        const point = self._points[i]
        point.draw(context, ratio, style)
      }
    }
  }

  /**
   * Resize the box
   * @param {Vector2D} start: starting point
   * @param {Vector2D} end: ending point
   */
  public resize (end: Vector2D, _limit: Size2D): void {
    const c = end
    const x = c.x
    const y = c.y
    let x1
    let x2
    let y1
    let y2
    if (this._highlightedHandle % 2 === 1) {
      // move a midpoint
      const v1 = this._points[Handles.TOP_LEFT]
      const v2 = this._points[Handles.BOTTOM_RIGHT]
      if (this._highlightedHandle === Handles.TOP_MIDDLE) {
        v1.y = y
      } else if (this._highlightedHandle === Handles.RIGHT_MIDDLE) {
        v2.x = x
      } else if (this._highlightedHandle === Handles.BOTTOM_MIDDLE) {
        v2.y = y
      } else if (this._highlightedHandle === Handles.LEFT_MIDDLE) {
        v1.x = x
      }
      x1 = Math.min(v1.x, v2.x)
      x2 = Math.max(v1.x, v2.x)
      y1 = Math.min(v1.y, v2.y)
      y2 = Math.max(v1.y, v2.y)
      if (x === x1) {
        this._highlightedHandle = Handles.LEFT_MIDDLE
      } else if (x === x2) {
        this._highlightedHandle = Handles.RIGHT_MIDDLE
      } else if (y === y1) {
        this._highlightedHandle = Handles.TOP_MIDDLE
      } else if (y === y2) {
        this._highlightedHandle = Handles.BOTTOM_MIDDLE
      }
    } else {
      // move a vertex
      const oppVertex = this._points[(this._highlightedHandle + 4 + 8) % 8]
      x1 = Math.min(x, oppVertex.x)
      x2 = Math.max(x, oppVertex.x)
      y1 = Math.min(y, oppVertex.y)
      y2 = Math.max(y, oppVertex.y)
      if (oppVertex.x < x) {
        if (oppVertex.y < y) {
          this._highlightedHandle = Handles.BOTTOM_RIGHT
        } else {
          this._highlightedHandle = Handles.TOP_RIGHT
        }
      } else {
        if (oppVertex.y < y) {
          this._highlightedHandle = Handles.BOTTOM_LEFT
        } else {
          this._highlightedHandle = Handles.TOP_LEFT
        }
      }
    }
    // update the rectangle
    const rect = this._shapes[0] as Rect2D
    rect.x = Math.min(x1, x2)
    rect.y = Math.min(y1, y2)
    rect.w = Math.min(x2 - x1)
    rect.h = Math.min(y2 - y1)
    this.updatePoints()
  }

  /**
   * Move the box
   * @param {Vector2D} start: starting point
   * @param {Vector2D} delta: how far the handle has been dragged
   * @param {Vector2D} limit: limit of the canvas frame
   */
  public move (end: Vector2D, limit: Size2D): void {
    const [width, height] = [limit.width, limit.height]
    const rect = this._shapes[0] as Rect2D
    const delta = end.clone().subtract(this._mouseDownCoord)
    rect.x = this._startingRect.x + delta.x
    rect.y = this._startingRect.y + delta.y
    // The rect should not go outside the frame limit
    rect.x = Math.min(width - this._startingRect.w, Math.max(0, rect.x))
    rect.y = Math.min(height - this._startingRect.h, Math.max(0, rect.y))
    rect.w = this._startingRect.w
    rect.h = this._startingRect.h
    this.updatePoints()
  }

  /**
   * Handle mouse up
   * @param coord
   */
  public onMouseUp (_coord: Vector2D): boolean {
    this._mouseDown = false
    this.editing = false
    return true
  }

  /**
   * Handle mouse down
   * @param coord
   */
  public onMouseDown (coord: Vector2D, _handleIndex: number): boolean {
    this._mouseDown = true
    if (this._selected) {
      this.editing = true
      this._mouseDownCoord = coord.clone()
      this._startingRect.copy(this._shapes[0] as Rect2D)
      return true
    }
    return false
  }

/**
 * Drag the handle to a new position
 * @param {Vector2D} coord: current mouse position
 * @param {Vector2D} limit: limit of the canvas frame
 */
  public onMouseMove (coord: Vector2D, limit: Size2D,
                      _labelIndex: number, _handleIndex: number): boolean {
    if (this._selected && this._mouseDown && this.editing) {
      if (
        this._highlightedHandle >= 0 &&
        this._highlightedHandle < this._points.length
      ) {
        this.resize(coord, limit)
        this._labelList.addUpdatedLabel(this)
      } else if (
        this._highlightedHandle === this._points.length
      ) {
        this.move(coord, limit)
        this._labelList.addUpdatedLabel(this)
      }
      return true
    }

    return false
  }

  /**
   * handle keyboard down event
   * @param e pressed key
   */
  public onKeyDown (): boolean {
    return true
  }

  /**
   * handle keyboard up event
   * @param e pressed key
   */
  public onKeyUp (): void {
    return
  }

  /** Get shape objects for committing to state */
  public shapeStates (): [number[], ShapeTypeName[], ShapeType[]] {
    return [
      this._labelState.shapes,
      [ShapeTypeName.RECT],
      [this._shapes[0].toState().shape]
    ]
  }

  /** Initialize this label to be temporary */
  public initTemp (state: State, start: Vector2D): void {
    super.initTemp(state, start)
    const itemIndex = state.user.select.item
    this._labelState = makeLabel({
      type: LabelTypeName.BOX_2D, id: -1, item: itemIndex,
      category: [state.user.select.category],
      attributes: state.user.select.attributes,
      order: this.order
    })

    this._shapes = [new Rect2D(start.x, start.y, 0, 0)]
    this._highlightedHandle = Handles.BOTTOM_RIGHT
    this.updatePoints()
  }

  /**
   * to check whether the label is valid
   */
  public isValid (): boolean {
    const rect = this._shapes[0] as Rect2D
    const area = rect.w * rect.h
    if (area >= MIN_AREA) {
      return true
    } else {
      return false
    }
  }

  /** Convert label state to drawable */
  public updateState (
    labelState: LabelType
  ): void {
    super.updateState(labelState)
    this.updatePoints()
  }

  /** Update corner and midpoint positions */
  private updatePoints () {
    const [tl, tm, tr, rm, br, bm, bl, lm] = this._points
    const rect = this._shapes[0] as Rect2D
    const x = rect.x
    const y = rect.y
    const w = rect.w
    const h = rect.h

    // vertices
    tl.set(x, y)
    tr.set(x + w, y)
    bl.set(x, y + h)
    br.set(x + w, y + h)

    // midpoints
    tm.set(x + w / 2, y)
    bm.set(x + w / 2, y + h)
    lm.set(x, y + h / 2)
    rm.set(x + w, y + h / 2)
  }
}
