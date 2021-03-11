import _ from "lodash"

import { Cursor, LabelTypeName } from "../../const/common"
import { makeLabel, makeRect } from "../../functional/states"
import { Size2D } from "../../math/size2d"
import { Vector2D } from "../../math/vector2d"
import { LabelType, RectType, ShapeType, State } from "../../types/state"
import { blendColor, Context2D, encodeControlColor } from "../util"
import { DrawMode, Label2D } from "./label2d"
import { Label2DList } from "./label2d_list"
import { makePoint2DStyle, Point2D } from "./point2d"
import { makeRect2DStyle, Rect2D } from "./rect2d"

const DEFAULT_VIEW_RECT_STYLE = makeRect2DStyle({ lineWidth: 4 })
const DEFAULT_VIEW_POINT_STYLE = makePoint2DStyle({ radius: 8 })
const DEFAULT_VIEW_HIGH_POINT_STYLE = makePoint2DStyle({ radius: 12 })
const DEFAULT_CONTROL_RECT_STYLE = makeRect2DStyle({ lineWidth: 10 })
const DEFAULT_CONTROL_POINT_STYLE = makePoint2DStyle({ radius: 12 })
const MIN_AREA = 10

enum Handles {
  EDGE = 0,
  TOP_LEFT = 1,
  TOP_MIDDLE = 2,
  TOP_RIGHT = 3,
  RIGHT_MIDDLE = 4,
  BOTTOM_RIGHT = 5,
  BOTTOM_MIDDLE = 6,
  BOTTOM_LEFT = 7,
  LEFT_MIDDLE = 8
}

/**
 * Compare two rectangles
 *
 * @param r1
 * @param r2
 */
function equalRects(r1: RectType, r2: RectType): boolean {
  return (
    r1.x1 === r2.x1 && r1.x2 === r2.x2 && r1.y1 === r2.y1 && r1.y2 === r2.y2
  )
}

/**
 * Box2d Label
 */
export class Box2D extends Label2D {
  /** rect shape of the box 2d */
  private readonly _rect: Rect2D
  /** 8 control points */
  private readonly _controlPoints: Point2D[]
  /** cache shape for moving */
  private _startingRect: Rect2D

  /**
   * Constructor
   *
   * @param labelList
   */
  constructor(labelList: Label2DList) {
    super(labelList)
    this._rect = new Rect2D()
    this._controlPoints = _.range(8).map(() => new Point2D())

    this._startingRect = new Rect2D()
  }

  /** Get cursor for use when highlighting */
  public get highlightCursor(): string {
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

  /**
   * Draw the label on viewing or control canvas
   *
   * @param context
   * @param ratio
   * @param mode
   */
  public draw(context: Context2D, ratio: number, mode: DrawMode): void {
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
          if (i % 2 === 0 && i > 0) {
            // Midpoint
            return blendColor(this._color, [255, 255, 255], 0.7)
          } else {
            // Vertex
            return this._color
          }
        }
        break
      case DrawMode.CONTROL:
        pointStyle = _.assign(pointStyle, DEFAULT_CONTROL_POINT_STYLE)
        highPointStyle = _.assign(highPointStyle, DEFAULT_CONTROL_POINT_STYLE)
        rectStyle = _.assign(rectStyle, DEFAULT_CONTROL_RECT_STYLE)
        assignColor = (i: number): number[] => {
          return encodeControlColor(this._index, i)
        }
        break
    }

    // Draw!!!
    const rect = this._rect
    rectStyle.color = assignColor(0)
    rect.draw(context, ratio, rectStyle)
    if (mode === DrawMode.VIEW) {
      this.drawTag(context, ratio, new Vector2D(rect.x1, rect.y1), this._color)
    }
    if (mode === DrawMode.CONTROL || this._selected || this._highlighted) {
      for (let i = 1; i <= 8; i += 1) {
        let style
        if (i === this._highlightedHandle) {
          style = highPointStyle
        } else {
          style = pointStyle
        }
        style.color = assignColor(i)
        const point = this._controlPoints[i - 1]
        point.draw(context, ratio, style)
      }
    }
  }

  /**
   * Resize the box
   *
   * @param {Vector2D} start: starting point
   * @param {Vector2D} end: ending point
   * @param end
   * @param _limit
   */
  public resize(end: Vector2D, _limit: Size2D): void {
    const c = end
    const x = c.x
    const y = c.y
    let x1
    let x2
    let y1
    let y2
    if (this._highlightedHandle % 2 === 0) {
      // Move a midpoint
      const v1 = this._controlPoints[0]
      const v2 = this._controlPoints[4]
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
      // Move a vertex
      const oppVertex = this._controlPoints[
        ((this._highlightedHandle + 12) % 8) - 1
      ]
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
    // Update the rectangle
    const rect = this._rect.shape()
    rect.x1 = x1
    rect.y1 = y1
    rect.x2 = x2
    rect.y2 = y2
    this.updateShapeValues(rect)
  }

  /**
   * Move the box
   *
   * @param {Vector2D} start: starting point
   * @param {Vector2D} delta: how far the handle has been dragged
   * @param {Vector2D} limit: limit of the canvas frame
   * @param end
   * @param limit
   */
  public move(end: Vector2D, limit: Size2D): void {
    const [width, height] = [limit.width, limit.height]
    const rect = this._rect.shape()
    const delta = end.clone().subtract(this._mouseDownCoord)
    rect.x1 = this._startingRect.x1 + delta.x
    rect.y1 = this._startingRect.y1 + delta.y
    // The rect should not go outside the frame limit
    rect.x1 = Math.min(width - this._startingRect.width(), Math.max(0, rect.x1))
    rect.y1 = Math.min(
      height - this._startingRect.height(),
      Math.max(0, rect.y1)
    )
    rect.x2 = rect.x1 + this._startingRect.width()
    rect.y2 = rect.y1 + this._startingRect.height()
    this.updateShapeValues(rect)
  }

  /**
   * Handle mouse up
   *
   * @param coord
   * @param _coord
   */
  public onMouseUp(_coord: Vector2D): boolean {
    this._mouseDown = false
    this.editing = false
    return true
  }

  /**
   * Handle mouse down
   *
   * @param coord
   * @param _handleIndex
   */
  public onMouseDown(coord: Vector2D, _handleIndex: number): boolean {
    this._mouseDown = true
    if (this._selected) {
      this.editing = true
      this._mouseDownCoord = coord.clone()
      this._startingRect = this._rect.clone()
      return true
    }
    return false
  }

  /**
   * Drag the handle to a new position
   *
   * @param {Vector2D} coord: current mouse position
   * @param {Vector2D} limit: limit of the canvas frame
   * @param coord
   * @param limit
   * @param _labelIndex
   * @param handleIndex
   */
  public onMouseMove(
    coord: Vector2D,
    limit: Size2D,
    _labelIndex: number,
    handleIndex: number
  ): boolean {
    if (this._selected && this._mouseDown && this.editing) {
      if (this._highlightedHandle > 0) {
        this.resize(coord, limit)
        this._labelList.addUpdatedLabel(this)
      } else if (
        this._highlightedHandle === Handles.EDGE &&
        handleIndex === 0
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
   *
   * @param e pressed key
   */
  public onKeyDown(): boolean {
    return true
  }

  /**
   * handle keyboard up event
   *
   * @param e pressed key
   */
  public onKeyUp(): void {}

  /** Get shape objects for committing to state */
  public shapes(): ShapeType[] {
    if (this._label === null) {
      throw new Error("Uninitialized label")
    }
    /**
     * This is a temporary solution for assigning the correct ID to the shapes
     * We should initialize the shape when the temporary label is created.
     * Also store the shape id properly so that the generated shape state has
     * the right id directly.
     */
    const box = this.toRect()
    return [box]
  }

  /** Get rect representation */
  public toRect(): RectType {
    return this._rect.shape()
  }

  /**
   * to check whether the label is valid
   */
  public isValid(): boolean {
    const rect = this._rect
    const area = rect.width() * rect.height()
    if (area >= MIN_AREA) {
      return true
    } else {
      return false
    }
  }

  /**
   * Convert label state to drawable
   *
   * @param shapes
   */
  public updateShapes(shapes: ShapeType[]): void {
    const rect = shapes[0] as RectType
    if (!equalRects(this.toRect(), rect)) {
      this.updateShapeValues(rect)
    }
  }

  /**
   * Initialize this label to be temporary
   *
   * @param state
   * @param start
   */
  protected initTempLabel(state: State, start: Vector2D): LabelType {
    const itemIndex = state.user.select.item
    const label = makeLabel({
      type: LabelTypeName.BOX_2D,
      item: itemIndex,
      category: [state.user.select.category],
      attributes: state.user.select.attributes,
      order: this._order
    })
    const rect = makeRect({
      x1: start.x,
      y1: start.y,
      x2: start.x,
      y2: start.y,
      label: [label.id]
    })
    label.shapes = [rect.id]
    this.updateShapes([rect])
    this._highlightedHandle = Handles.BOTTOM_RIGHT
    return label
  }

  /**
   * Update the values of the drawable shapes
   *
   * @param {RectType} rect
   */
  private updateShapeValues(rect: RectType): void {
    this._rect.set(rect)
    const [tl, tm, tr, rm, br, bm, bl, lm] = this._controlPoints
    const x = this._rect.x1
    const y = this._rect.y1
    const w = this._rect.width()
    const h = this._rect.height()
    // Vertices
    tl.set(x, y)
    tr.set(x + w, y)
    bl.set(x, y + h)
    br.set(x + w, y + h)

    // Midpoints
    tm.set(x + w / 2, y)
    bm.set(x + w / 2, y + h)
    lm.set(x, y + h / 2)
    rm.set(x + w, y + h / 2)
  }
}
