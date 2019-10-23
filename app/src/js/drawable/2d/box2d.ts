import _ from 'lodash'
import { addBox2dLabel } from '../../action/box2d'
import { changeLabelProps, changeLabelShape } from '../../action/common'
import Session from '../../common/session'
import { LabelTypeName } from '../../common/types'
import { makeLabel, makeRect } from '../../functional/states'
import { RectType, ShapeType, State } from '../../functional/types'
import { Size2D } from '../../math/size2d'
import { Vector2D } from '../../math/vector2d'
import { blendColor, Context2D, encodeControlColor, getColorById } from '../util'
import { DrawMode, Label2D } from './label2d'
import { makePoint2DStyle, Point2D } from './point2d'
import { makeRect2DStyle, Rect2D } from './rect2d'

type Shape = Rect2D | Point2D

const DEFAULT_VIEW_RECT_STYLE = makeRect2DStyle({ lineWidth: 4 })
const DEFAULT_VIEW_POINT_STYLE = makePoint2DStyle({ radius: 8 })
const DEFAULT_VIEW_HIGH_POINT_STYLE = makePoint2DStyle({ radius: 12 })
const DEFAULT_CONTROL_RECT_STYLE = makeRect2DStyle({ lineWidth: 10 })
const DEFAULT_CONTROL_POINT_STYLE = makePoint2DStyle({ radius: 12 })
const MIN_AREA = 10

/**
 * Box2d Label
 */
export class Box2D extends Label2D {
  /** list of shapes for this box 2d */
  private _shapes: Shape[]
  /** cache shape for moving */
  private _startingRect: Rect2D

  constructor () {
    super()
    this._shapes = [
      new Rect2D(),
      new Point2D(), new Point2D(), new Point2D(), new Point2D(),
      new Point2D(), new Point2D(), new Point2D(), new Point2D()
    ]

    this._startingRect = new Rect2D()
  }

  /**
   * Return a list of the shape for inspection and testing
   */
  public get shapes (): Array<Readonly<Shape>> {
    return this._shapes
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
          if (i % 2 === 0 && i > 0) {
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
    const rect = self._shapes[0] as Rect2D
    rectStyle.color = assignColor(0)
    rect.draw(context, ratio, rectStyle)
    if (mode === DrawMode.VIEW) {
      self.drawTag(context, ratio, [rect.x, rect.y], self._color)
    }
    if (mode === DrawMode.CONTROL || this._selected || this._highlighted) {
      for (let i = 1; i <= 8; i += 1) {
        let style
        if (i === self._highlightedHandle) {
          style = highPointStyle
        } else {
          style = pointStyle
        }
        style.color = assignColor(i)
        const point = self._shapes[i] as Point2D
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
    if (this._selectedHandle % 2 === 0) {
      // move a midpoint
      const v1 = this._shapes[1]
      const v2 = this._shapes[5]
      if (this._selectedHandle === 2) {
        v1.y = y
      } else if (this._selectedHandle === 4) {
        v2.x = x
      } else if (this._selectedHandle === 6) {
        v2.y = y
      } else if (this._selectedHandle === 8) {
        v1.x = x
      }
      x1 = Math.min(v1.x, v2.x)
      x2 = Math.max(v1.x, v2.x)
      y1 = Math.min(v1.y, v2.y)
      y2 = Math.max(v1.y, v2.y)
      if (x === x1) {
        this._selectedHandle = 8
      } else if (x === x2) {
        this._selectedHandle = 4
      } else if (y === y1) {
        this._selectedHandle = 2
      } else if (y === y2) {
        this._selectedHandle = 6
      }
    } else {
      // move a vertex
      const oppVertex = this._shapes[(this._selectedHandle + 4 + 8) % 8]
      x1 = Math.min(x, oppVertex.x)
      x2 = Math.max(x, oppVertex.x)
      y1 = Math.min(y, oppVertex.y)
      y2 = Math.max(y, oppVertex.y)
      if (oppVertex.x < x) {
        if (oppVertex.y < y) {
          this._selectedHandle = 5
        } else {
          this._selectedHandle = 3
        }
      } else {
        if (oppVertex.y < y) {
          this._selectedHandle = 7
        } else {
          this._selectedHandle = 1
        }
      }
    }
    // update the rectangle
    const rect = (this._shapes[0] as Rect2D).toRect()
    rect.x1 = x1
    rect.y1 = y1
    rect.x2 = x2
    rect.y2 = y2
    this.updateShapeValues(rect)
  }

  /**
   * Move the box
   * @param {Vector2D} start: starting point
   * @param {Vector2D} delta: how far the handle has been dragged
   * @param {Vector2D} limit: limit of the canvas frame
   */
  public move (end: Vector2D, limit: Size2D): void {
    const [width, height] = [limit.width, limit.height]
    const rect = (this._shapes[0] as Rect2D).toRect()
    const delta = end.clone().subtract(this._mouseDownCoord)
    rect.x1 = this._startingRect.x + delta.x
    rect.y1 = this._startingRect.y + delta.y
    // The rect should not go outside the frame limit
    rect.x1 = Math.min(width - this._startingRect.w, Math.max(0, rect.x1))
    rect.y1 = Math.min(height - this._startingRect.h, Math.max(0, rect.y1))
    rect.x2 = rect.x1 + this._startingRect.w
    rect.y2 = rect.y1 + this._startingRect.h
    this.updateShapeValues(rect)
  }

  /**
   * Handle mouse up
   * @param coord
   */
  public onMouseUp (_coord: Vector2D): boolean {
    this._mouseDown = false
    this.editing = false
    return this.commitLabel()
  }

  /**
   * Handle mouse down
   * @param coord
   */
  public onMouseDown (coord: Vector2D): boolean {
    this._mouseDown = true
    if (this._selected) {
      this.editing = true
      this._mouseDownCoord = coord.clone()
      this._startingRect = (this.shapes[0] as Rect2D).clone()
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
      if (this._selectedHandle > 0) {
        this.resize(coord, limit)
      } else if (this._selectedHandle === 0) {
        this.move(coord, limit)
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

  /** Update the shapes of the label to the state */
  public commitLabel (): boolean {
    if (!this._label) {
      return false
    }
    if (this._selected) {
      const valid = this.isValid()
      if (valid) {
        if (this._labelId < 0) {
          const r = this.toRect()
          if (Session.tracking && this._trackId in Session.tracks) {
            Session.tracks[this._trackId].onLabelCreated(
              this._label.item, this
            )
          } else {
            Session.dispatch(addBox2dLabel(
              this._label.item, this._label.category, this._label.attributes,
              r.x1, r.y1, r.x2, r.y2
            ))
          }
        } else {
          const rect = this.toRect()
          Session.dispatch(changeLabelShape(
            this._label.item, this._label.shapes[0], rect))
          Session.dispatch(changeLabelProps(
            this._label.item, this._labelId, { manual: true }
          ))
          if (Session.tracking && this._trackId in Session.tracks) {
            Session.tracks[this._trackId].onLabelUpdated(
              this._label.item, [rect]
            )
          }
        }
        return true
      }
    }
    return false
  }

  /** Initialize this label to be temporary */
  public initTemp (state: State, start: Vector2D): void {
    const itemIndex = state.user.select.item
    this._order = state.task.status.maxOrder + 1
    this._label = makeLabel({
      type: LabelTypeName.BOX_2D, id: -1, item: itemIndex,
      category: [state.user.select.category],
      attributes: state.user.select.attributes,
      order: this._order
    })
    this._labelId = -1
    this._color = getColorById(
      state.task.status.maxLabelId + 1,
      ((Session.tracking) ? (state.task.status.maxTrackId + 1) : -1)
    )
    const rect = makeRect({
      x1: start.x, y1: start.y, x2: start.x, y2: start.y
    })
    this.updateShapes([rect])
    this.setSelected(true, 5)
  }

  /** Get rect representation */
  public toRect (): RectType {
    return (this._shapes[0] as Rect2D).toRect()
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
  public updateShapes (shapes: ShapeType[]): void {
    if (this._label !== null) {
      const rect = shapes[0] as RectType
      if (!_.isEqual(this.toRect(), rect)) {
        this.updateShapeValues(rect)
      }
    }
  }

  /**
   * Update the values of the drawable shapes
   * @param {RectType} rect
   */
  private updateShapeValues (rect: RectType): void {
    const [rect2d, tl, tm, tr, rm, br, bm, bl, lm] = this._shapes
    const x = rect.x1
    const y = rect.y1
    const w = rect.x2 - rect.x1
    const h = rect.y2 - rect.y1
    rect2d.set(x, y, w, h)

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
