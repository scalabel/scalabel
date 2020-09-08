import _ from "lodash"

import { Cursor, Key } from "../../const/common"
import { makeLabel } from "../../functional/states"
import { Size2D } from "../../math/size2d"
import { Vector2D } from "../../math/vector2d"
import {
  Label2DTemplateType,
  LabelType,
  Node2DType,
  ShapeType,
  State
} from "../../types/state"
import { Context2D, encodeControlColor, toCssColor } from "../util"
import { DrawMode, Label2D } from "./label2d"
import { Label2DList } from "./label2d_list"
import { Node2D } from "./node2d"
import { makePoint2DStyle, Point2D } from "./point2d"
import { makeRect2DStyle, Rect2D } from "./rect2d"

const DEFAULT_VIEW_RECT_STYLE = makeRect2DStyle({ lineWidth: 4, dashed: true })
const DEFAULT_VIEW_POINT_STYLE = makePoint2DStyle({ radius: 8 })
const DEFAULT_VIEW_HIGH_POINT_STYLE = makePoint2DStyle({ radius: 12 })
const DEFAULT_CONTROL_RECT_STYLE = makeRect2DStyle({ lineWidth: 8 })
const DEFAULT_CONTROL_POINT_STYLE = makePoint2DStyle({ radius: 12 })
const lineWidth = 4

/** Class for templated user-defined labels */
export class CustomLabel2D extends Label2D {
  /** Label template */
  private readonly _template: Label2DTemplateType
  /** Shapes */
  private _shapes: Node2D[]
  /** Map between node names and colors */
  private readonly _colorMap: { [name: string]: number[] }
  /** The hashed list of keys currently down */
  private _keyDownMap: { [key: string]: boolean }

  /**
   * Corners for resizing,
   * order: top-left, top-right, bottom-right, bottom-left
   */
  private readonly _corners: Point2D[]
  /** Bounds of the shape */
  private _bounds: Rect2D

  /**
   * Constructor
   *
   * @param labelList
   * @param template
   */
  constructor(labelList: Label2DList, template: Label2DTemplateType) {
    super(labelList)
    this._template = template
    this._shapes = []
    this._bounds = new Rect2D()
    this._corners = [new Point2D(), new Point2D(), new Point2D(), new Point2D()]
    this._colorMap = {}
    for (const node of this._template.nodes) {
      if (node.color !== null && node.color !== undefined) {
        this._colorMap[node.name] = node.color
      }
    }
    this._keyDownMap = {}
  }

  /** Get cursor for highlighting */
  public get highlightCursor(): string {
    if (
      this._highlightedHandle >= 0 &&
      this._highlightedHandle < this._shapes.length
    ) {
      return Cursor.DEFAULT
    } else if (
      this._highlightedHandle >= this._shapes.length &&
      this._highlightedHandle < this._shapes.length + this._corners.length
    ) {
      const cornerIndex = this._highlightedHandle - this._shapes.length
      if (cornerIndex % 2 === 0) {
        return Cursor.NWSE_RESIZE
      } else {
        return Cursor.NESW_RESIZE
      }
    } else {
      return Cursor.MOVE
    }
  }

  /**
   * Draw according to template
   *
   * @param context
   * @param ratio
   * @param mode
   */
  public draw(context: Context2D, ratio: number, mode: DrawMode): void {
    // Set proper drawing styles
    let pointStyle = makePoint2DStyle()
    let rectStyle = makeRect2DStyle()
    let highPointStyle = makePoint2DStyle()
    let assignColor: (i: number) => number[] = () => [0]
    switch (mode) {
      case DrawMode.VIEW:
        pointStyle = _.assign(pointStyle, DEFAULT_VIEW_POINT_STYLE)
        highPointStyle = _.assign(highPointStyle, DEFAULT_VIEW_HIGH_POINT_STYLE)
        rectStyle = _.assign(rectStyle, DEFAULT_VIEW_RECT_STYLE)
        assignColor = (i: number): number[] => {
          // Vertex
          if (
            i < this._shapes.length &&
            this._shapes[i].name in this._colorMap
          ) {
            return this._colorMap[this._shapes[i].name]
          }
          return this._color
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

    // Draw bounding box if selected
    if (this._selected) {
      rectStyle.color = assignColor(this._shapes.length + this._corners.length)
      this._bounds.draw(context, ratio, rectStyle)

      for (let i = 0; i < this._corners.length; i++) {
        const style =
          i === this._highlightedHandle - this._shapes.length
            ? highPointStyle
            : pointStyle
        style.color = assignColor(i + this._shapes.length)
        this._corners[i].draw(context, ratio, style)
      }
    }

    // Draw edges
    for (const edge of this._template.edges) {
      const startPoint = this._shapes[edge[0]]
      const endPoint = this._shapes[edge[1]]

      if (startPoint.hidden || endPoint.hidden) {
        continue
      }

      const realStart = startPoint.clone().scale(ratio)
      const realEnd = endPoint.clone().scale(ratio)

      context.save()
      context.strokeStyle = toCssColor(
        assignColor(this._shapes.length + this._corners.length)
      )
      context.lineWidth = lineWidth
      context.beginPath()
      context.moveTo(realStart.x, realStart.y)
      context.lineTo(realEnd.x, realEnd.y)
      context.closePath()
      context.stroke()
      context.restore()
    }

    // Draw nodes
    for (let i = 0; i < this._shapes.length; i++) {
      if (this._shapes[i].hidden) {
        continue
      }
      const style = i === this._highlightedHandle ? highPointStyle : pointStyle
      style.color = assignColor(i)
      this._shapes[i].draw(context, ratio, style)
    }
  }

  /**
   * Override on mouse down
   *
   * @param coord
   * @param handleIndex
   */
  public onMouseDown(coord: Vector2D, handleIndex: number): boolean {
    const returnValue = super.onMouseDown(coord, handleIndex)
    this.editing = true

    if (
      (this._keyDownMap[Key.D_LOW] || this._keyDownMap[Key.D_UP]) &&
      this._highlightedHandle < this._shapes.length
    ) {
      // Delete highlighted handle if d key pressed
      this._shapes[this._highlightedHandle].hide()
      this._highlightedHandle = -1
    }

    return returnValue
  }

  /**
   * Handle mouse move
   *
   * @param coord
   * @param _limit
   * @param _labelIndex
   * @param _handleIndex
   */
  public onMouseMove(
    coord: Vector2D,
    _limit: Size2D,
    _labelIndex: number,
    _handleIndex: number
  ): boolean {
    if (
      this._temporary ||
      (this._highlightedHandle >= this._shapes.length &&
        this._highlightedHandle < this._shapes.length + this._corners.length)
    ) {
      // Calculate new scale by
      // comparing mouse coordinate against opposite corner
      const cornerIndex = this._highlightedHandle - this._shapes.length
      const corner = this._corners[cornerIndex]
      const oppositeCornerIndex = this.getOppositeCorner(cornerIndex)
      const oppositeCorner = this._corners[oppositeCornerIndex]
      const xScale =
        (coord.x - oppositeCorner.x) / (corner.x - oppositeCorner.x)
      const yScale =
        (coord.y - oppositeCorner.y) / (corner.y - oppositeCorner.y)
      this.scale(oppositeCorner, new Vector2D(xScale, yScale))
      this._labelList.addUpdatedLabel(this)
    } else if (this._highlightedHandle >= 0) {
      if (this._highlightedHandle < this._shapes.length) {
        // Move single point
        this._shapes[this._highlightedHandle].x = coord.x
        this._shapes[this._highlightedHandle].y = coord.y
        this.updateBounds()
      } else {
        // Drag shape
        const delta = coord.clone().subtract(this._mouseDownCoord)
        for (const shape of this._shapes) {
          shape.x += delta.x
          shape.y += delta.y
        }
      }
      this._labelList.addUpdatedLabel(this)
    }
    this._mouseDownCoord.x = coord.x
    this._mouseDownCoord.y = coord.y
    this.updateBounds()

    return false
  }

  /**
   * Override on mouse up
   *
   * @param coord
   */
  public onMouseUp(coord: Vector2D): boolean {
    const returnValue = super.onMouseUp(coord)
    this.editing = false
    return returnValue
  }

  /**
   * On key down
   *
   * @param key
   */
  public onKeyDown(key: string): boolean {
    this._keyDownMap[key] = true
    return true
  }

  /**
   * handle keyboard up event
   *
   * @param e pressed key
   * @param key
   */
  public onKeyUp(key: string): void {
    // eslint-disable-next-line @typescript-eslint/no-dynamic-delete
    delete this._keyDownMap[key]
  }

  /**
   * Expand the primitive shapes to drawable shapes
   *
   * @param {ShapeType[]} shapes
   */
  public updateShapes(shapes: ShapeType[]): void {
    if (shapes.length !== this._shapes.length) {
      this._shapes = []
      for (const node of shapes) {
        this._shapes.push(new Node2D(node as Node2DType))
      }
    } else {
      for (let i = 0; i < shapes.length; i++) {
        const node = shapes[i] as Node2DType
        this._shapes[i].x = node.x
        this._shapes[i].y = node.y
        this._shapes[i].name = node.name
        if (node.hidden !== null) {
          this._shapes[i].hide()
        }
      }
    }
    this.updateBounds()
  }

  /** Get shape id's and shapes for updating */
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
    const shapeStates: Node2DType[] = this._shapes.map((shape) =>
      shape.toState()
    )
    if (!this._temporary) {
      for (let i = 0; i < shapeStates.length; i++) {
        shapeStates[i].id = this._label.shapes[i]
      }
    }
    return shapeStates
  }

  /**
   * Temporary initialization on mouse down
   *
   * @param state
   * @param start
   */
  protected initTempLabel(state: State, start: Vector2D): LabelType {
    const itemIndex = state.user.select.item
    const templateName =
      state.task.config.labelTypes[state.user.select.labelType]
    const label = makeLabel({
      type: templateName,
      item: itemIndex,
      category: [state.user.select.category],
      attributes: state.user.select.attributes,
      order: this._order
    })

    // Initialize with template information
    this._shapes = []
    for (const node of this._template.nodes) {
      this._shapes.push(new Node2D(node))
    }

    // Get template bounds
    this.updateBounds()

    // Move to start
    for (const point of this._shapes) {
      point.x += start.x - this._bounds.x1
      point.y += start.y - this._bounds.y1
    }

    // Update bounds after moving
    this.updateBounds()

    this.setSelected(true)
    this._highlightedHandle = this._shapes.length + 2
    return label
  }

  /** update bounds to current points */
  private updateBounds(): void {
    const bounds = {
      x1: Infinity,
      y1: Infinity,
      x2: -Infinity,
      y2: -Infinity
    }

    for (const point of this._shapes) {
      bounds.x1 = Math.min(point.x, bounds.x1)
      bounds.y1 = Math.min(point.y, bounds.y1)
      bounds.x2 = Math.max(point.x, bounds.x2)
      bounds.y2 = Math.max(point.y, bounds.y2)
    }

    this._bounds.x1 = bounds.x1
    this._bounds.y1 = bounds.y1
    this._bounds.x2 = bounds.x2
    this._bounds.y2 = bounds.y2

    for (let i = 0; i < 4; i++) {
      this._corners[i].x = this._bounds.x1
      this._corners[i].y = this._bounds.y1
      this._corners[i].x +=
        this._bounds.width() * (1 - Math.floor(Math.abs(i - 1.5)))
      this._corners[i].y += this._bounds.height() * Math.floor(i / 2)
    }
  }

  /**
   * Scale
   *
   * @param anchor
   * @param scale
   */
  private scale(anchor: Vector2D, scale: Vector2D): void {
    // Scale minimum to 1 pixel wide/high
    let widthSign = Math.sign(scale.x)
    if (widthSign === 0) {
      widthSign = 1
    }
    const newWidth =
      Math.max(Math.abs(scale.x * this._bounds.width()), 1) * widthSign

    let heightSign = Math.sign(scale.y)
    if (heightSign === 0) {
      heightSign = 1
    }
    const newHeight =
      Math.max(Math.abs(scale.y * this._bounds.height()), 1) * heightSign

    scale.x = newWidth / this._bounds.width()
    scale.y = newHeight / this._bounds.height()

    for (const shape of this._shapes) {
      shape.x = (shape.x - anchor.x) * scale.x + anchor.x
      shape.y = (shape.y - anchor.y) * scale.y + anchor.y
    }

    // If the highlighted handle is one of the corners,
    // flip the handle if scale is negative
    if (
      this._highlightedHandle >= this._shapes.length &&
      this._highlightedHandle <= this._shapes.length + this._corners.length
    ) {
      const cornerIndex = this._highlightedHandle - this._shapes.length
      const cornerCoords = new Vector2D(
        1 - Math.floor(Math.abs(cornerIndex - 1.5)),
        Math.floor(cornerIndex / 2)
      )

      if (scale.x < 0) {
        cornerCoords.x = 1 - cornerCoords.x
      }

      if (scale.y < 0) {
        cornerCoords.y = 1 - cornerCoords.y
      }

      this._highlightedHandle =
        Math.abs(cornerCoords.y * 3 - cornerCoords.x) + this._shapes.length
    }

    this.updateBounds()
  }

  /**
   * Get index of opposite corner
   *
   * @param index
   */
  private getOppositeCorner(index: number): number {
    return (index + 2) % this._corners.length
  }
}
