import _ from 'lodash'
import { Cursor, LabelTypeName } from '../../common/types'
import { LabelType } from '../../functional/types'
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

export enum Handles {
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
  /** Corners and midpoints */
  private _points:
    [Point2D, Point2D, Point2D, Point2D, Point2D, Point2D, Point2D, Point2D]

  constructor (labelList: Label2DList) {
    super(labelList)
    this._points = [
      new Point2D(), new Point2D(), new Point2D(), new Point2D(),
      new Point2D(), new Point2D(), new Point2D(), new Point2D()
    ]
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
    rectStyle.color = assignColor(-1)
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
  public resize (delta: Vector2D, _limit: Size2D): void {
    const rect = this._shapes[0] as Rect2D
    const highlightedPoint = this._points[this._highlightedHandle].toVector()
    if (this._highlightedHandle % 2 === 1) {
      // move a midpoint
      switch (this._highlightedHandle) {
        case Handles.LEFT_MIDDLE:
          highlightedPoint.x += delta.x
          rect.x += delta.x
          rect.w -= delta.x
          break
        case Handles.TOP_MIDDLE:
          highlightedPoint.y += delta.y
          rect.y += delta.y
          rect.h -= delta.y
          break
        case Handles.BOTTOM_MIDDLE:
          highlightedPoint.y += delta.y
          rect.h += delta.y
          break
        case Handles.RIGHT_MIDDLE:
          highlightedPoint.x += delta.x
          rect.w += delta.x
          break
      }
    } else {
      // move a vertex
      highlightedPoint.add(delta)
      const oppVertex = this._points[(this._highlightedHandle + 4 + 8) % 8]
      rect.x = Math.min(highlightedPoint.x, oppVertex.x)
      rect.y = Math.min(highlightedPoint.y, oppVertex.y)
      rect.w = Math.abs(highlightedPoint.x - oppVertex.x)
      rect.h = Math.abs(highlightedPoint.y - oppVertex.y)
    }
    this.updatePoints()
    let closestDistance = Infinity
    this._points.forEach((point, index) => {
      const difference = point.toVector().subtract(highlightedPoint)
      const distance = difference.dot(difference)
      if (distance < closestDistance) {
        closestDistance = distance
        this._highlightedHandle = index
      }
    })
    this.setAllShapesUpdated()
  }

  /**
   * Move the box
   * @param {Vector2D} start: starting point
   * @param {Vector2D} delta: how far the handle has been dragged
   * @param {Vector2D} limit: limit of the canvas frame
   */
  public move (delta: Vector2D, limit: Size2D): void {
    const [width, height] = [limit.width, limit.height]
    const rect = this._shapes[0] as Rect2D
    rect.x += delta.x
    rect.y += delta.y
    // The rect should not go outside the frame limit
    rect.x = Math.min(width - rect.w, Math.max(0, rect.x))
    rect.y = Math.min(height - rect.h, Math.max(0, rect.y))
    this.updatePoints()
    this.setAllShapesUpdated()
  }

/**
 * Drag the handle to a new position
 * @param {Vector2D} coord: current mouse position
 * @param {Vector2D} limit: limit of the canvas frame
 */
  public drag (delta: Vector2D, limit: Size2D): boolean {
    if (
      this._highlightedHandle >= 0 &&
      this._highlightedHandle < this._points.length
    ) {
      this.resize(delta, limit)
      this._labelList.addUpdatedShape(this._shapes[0])
      this._labelList.addUpdatedLabel(this)
    } else {
      this.move(delta, limit)
      this._labelList.addUpdatedShape(this._shapes[0])
      this._labelList.addUpdatedLabel(this)
    }
    return true
  }

  /** Click not supported */
  public click () {
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

  /** Initialize this label to be temporary */
  public initTemp (
    order: number,
    itemIndex: number,
    category: number[],
    attributes: { [key: number]: number[] },
    color: number[],
    start: Vector2D
  ): void {
    super.initTemp(order, itemIndex, category, attributes, color, start)

    const rect = new Rect2D()
    this._labelList.addTemporaryShape(rect)
    rect.associateLabel(this)
    rect.x = start.x
    rect.y = start.y
    rect.w = 0
    rect.h = 0
    this._shapes = [rect]

    this._labelState.type = LabelTypeName.BOX_2D
    this._labelState.shapes = [this._shapes[0].shapeId]
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
