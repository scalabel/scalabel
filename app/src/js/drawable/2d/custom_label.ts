import _ from 'lodash'
import { Cursor, Key } from '../../common/types'
import { makeLabel } from '../../functional/states'
import { Label2DTemplateType, LabelType, State } from '../../functional/types'
import { Size2D } from '../../math/size2d'
import { Vector2D } from '../../math/vector2d'
import { Context2D, encodeControlColor, getColorById, toCssColor } from '../util'
import { DrawMode, Label2D } from './label2d'
import { Label2DList } from './label2d_list'
import { Node2D } from './node2d'
import { makePoint2DStyle, Point2D } from './point2d'
import { makeRect2DStyle, Rect2D } from './rect2d'

const DEFAULT_VIEW_RECT_STYLE = makeRect2DStyle({ lineWidth: 4, dashed: true })
const DEFAULT_VIEW_POINT_STYLE = makePoint2DStyle({ radius: 8 })
const DEFAULT_VIEW_HIGH_POINT_STYLE = makePoint2DStyle({ radius: 12 })
const DEFAULT_CONTROL_RECT_STYLE = makeRect2DStyle({ lineWidth: 8 })
const DEFAULT_CONTROL_POINT_STYLE = makePoint2DStyle({ radius: 12 })
const lineWidth = 4

/** Class for templated user-defined labels */
export class CustomLabel2D extends Label2D {
  /** Nodes */
  private _nodes: Node2D[]
  /** Label template */
  private _template: Label2DTemplateType
  /** Map between node names and colors */
  private _colorMap: { [name: string]: number[] }
  /** The hashed list of keys currently down */
  private _keyDownMap: { [key: string]: boolean }

  /**
   * Corners for resizing,
   * order: top-left, top-right, bottom-right, bottom-left
   */
  private _corners: Point2D[]
  /** Bounds of the shape */
  private _bounds: Rect2D

  constructor (labelList: Label2DList, template: Label2DTemplateType) {
    super(labelList)
    this._nodes = []
    this._template = template
    this._bounds = new Rect2D(-1, -1, -1, -1)
    this._corners = [new Point2D(), new Point2D(), new Point2D(), new Point2D()]
    this._colorMap = {}
    for (const node of this._template.nodes) {
      if (node.color) {
        this._colorMap[node.name] = node.color
      }
    }
    this._keyDownMap = {}
  }

  /** Get cursor for highlighting */
  public get highlightCursor () {
    if (
      this._highlightedHandle >= 0 &&
      this._highlightedHandle < this._nodes.length
    ) {
      return Cursor.DEFAULT
    } else if (
      this._highlightedHandle >= this._nodes.length &&
      this._highlightedHandle < this._nodes.length + this._corners.length
    ) {
      const cornerIndex = this._highlightedHandle - this._nodes.length
      if (cornerIndex % 2 === 0) {
        return Cursor.NWSE_RESIZE
      } else {
        return Cursor.NESW_RESIZE
      }
    } else {
      return Cursor.MOVE
    }
  }

  /** Draw according to template */
  public draw (context: Context2D, ratio: number, mode: DrawMode): void {
    const self = this

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
          // vertex
          if (
            i < this._nodes.length &&
            this._nodes[i].name in this._colorMap
          ) {
            return this._colorMap[this._nodes[i].name]
          }
          return self._color
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

    // Draw bounding box if selected
    if (this._selected) {
      rectStyle.color = assignColor(this._nodes.length + this._corners.length)
      this._bounds.draw(context, ratio, rectStyle)

      for (let i = 0; i < this._corners.length; i++) {
        const style =
          (i === this._highlightedHandle - this._nodes.length) ?
            highPointStyle : pointStyle
        style.color = assignColor(i + this._nodes.length)
        this._corners[i].draw(context, ratio, style)
      }
    }

    // Draw edges
    for (const edge of this._template.edges) {
      const startPoint = this._nodes[edge[0]]
      const endPoint = this._nodes[edge[1]]

      if (startPoint.hidden || endPoint.hidden) {
        continue
      }

      context.save()
      context.strokeStyle = toCssColor(assignColor(
        this._nodes.length + this._corners.length
      ))
      context.lineWidth = lineWidth
      context.beginPath()
      context.moveTo(
        startPoint.x * ratio,
        startPoint.y * ratio
      )
      context.lineTo(
        endPoint.x * ratio,
        endPoint.y * ratio
      )
      context.closePath()
      context.stroke()
      context.restore()
    }

    // Draw nodes
    for (let i = 0; i < this._nodes.length; i++) {
      if (this._nodes[i].hidden) {
        continue
      }
      const style =
        (i === this._highlightedHandle) ? highPointStyle : pointStyle
      style.color = assignColor(i)
      this._nodes[i].draw(context, ratio, style)
    }
  }

  /** Temporary initialization on mouse down */
  public initTemp (
    state: State, start: Vector2D
  ) {
    super.initTemp(state, start)
    const itemIndex = state.user.select.item
    this.order = state.task.status.maxOrder + 1
    const templateName =
      state.task.config.labelTypes[state.user.select.labelType]
    this._labelState = makeLabel({
      type: templateName,
      id: -1,
      item: itemIndex,
      category: [state.user.select.category],
      attributes: state.user.select.attributes,
      order: this.order
    })
    this._color = getColorById(
      state.task.status.maxLabelId + 1,
      (state.task.config.tracking) ? state.task.status.maxTrackId + 1 : -1
    )

    // Initialize with template information
    this._nodes = []
    for (const node of this._template.nodes) {
      this._nodes.push(new Node2D(node))
    }

    // Get template bounds
    this.updateBounds()

    // Move to start
    for (const point of this._nodes) {
      point.x += start.x - this._bounds.x
      point.y += start.y - this._bounds.y
    }

    // Update bounds after moving
    this.updateBounds()

    this.selected = true
    this._highlightedHandle = this._nodes.length + 2
  }

  /** Override on mouse down */
  public onMouseDown (
    coord: Vector2D, labelIndex: number, handleIndex: number
  ): boolean {
    const returnValue = super.onMouseDown(coord, labelIndex, handleIndex)
    this.editing = true

    if (
      (this._keyDownMap[Key.D_LOW] || this._keyDownMap[Key.D_UP]) &&
      this._highlightedHandle < this._nodes.length
    ) {
      // Delete highlighted handle if d key pressed
      this._nodes[this._highlightedHandle].hide()
      this._highlightedHandle = -1
    }

    return returnValue
  }

  /** Handle mouse move */
  public onMouseMove (
    coord: Vector2D,
    _limit: Size2D,
    _labelIndex: number,
    _handleIndex: number
  ) {
    if (
      this.labelId < 0 || (
        this._highlightedHandle >= this._nodes.length &&
        this._highlightedHandle < this._nodes.length + this._corners.length
      )
    ) {
      // Calculate new scale by
      // comparing mouse coordinate against opposite corner
      const cornerIndex = this._highlightedHandle - this._nodes.length
      const corner = this._corners[cornerIndex]
      const oppositeCornerIndex = this.getOppositeCorner(cornerIndex)
      const oppositeCorner = this._corners[oppositeCornerIndex]
      const xScale =
          (coord.x - oppositeCorner.x) / (corner.x - oppositeCorner.x)
      const yScale =
          (coord.y - oppositeCorner.y) / (corner.y - oppositeCorner.y)
      this.scale(
        new Vector2D(oppositeCorner.x ,oppositeCorner.y),
        new Vector2D(xScale, yScale)
      )
      this.setAllShapesUpdated()
    } else if (this._highlightedHandle >= 0) {
      if (this._highlightedHandle < this._nodes.length) {
        // Move single point
        this._nodes[this._highlightedHandle].x = coord.x
        this._nodes[this._highlightedHandle].y = coord.y
        this.updateBounds()
      } else {
        // Drag shape
        const delta = coord.clone().subtract(this._mouseDownCoord)
        for (const shape of this._nodes) {
          shape.x += delta.x
          shape.y += delta.y
        }
      }
      this.setAllShapesUpdated()
    }
    this._mouseDownCoord.x = coord.x
    this._mouseDownCoord.y = coord.y
    this.updateBounds()

    return false
  }

  /** Override on mouse up */
  public onMouseUp (coord: Vector2D): boolean {
    const returnValue = super.onMouseUp(coord)
    this.editing = false
    if (this.labelId < 0) {
      this._shapes = [...this._nodes]
      for (const shape of this._shapes) {
        this._labelList.addTemporaryShape(shape)
        shape.associateLabel(this)
      }
      this._labelState.shapes = this._shapes.map((shape) => shape.shapeId)
      this._labelList.addUpdatedLabel(this)
    }
    return returnValue
  }

  /** On key down */
  public onKeyDown (key: string): boolean {
    this._keyDownMap[key] = true
    return true
  }

  /**
   * handle keyboard up event
   * @param e pressed key
   */
  public onKeyUp (key: string): void {
    delete this._keyDownMap[key]
  }

  /**
   * Expand the primitive shapes to drawable shapes
   * @param {ShapeType[]} shapes
   */
  public updateState (labelState: LabelType): void {
    super.updateState(labelState)
    this._nodes = [...this._shapes as Node2D[]]
    this.updateBounds()
  }

  /** update bounds to current points */
  private updateBounds () {
    const bounds = {
      x1: Infinity, y1: Infinity, x2: -Infinity, y2: -Infinity
    }

    for (const point of this._nodes) {
      bounds.x1 = Math.min(point.x, bounds.x1)
      bounds.y1 = Math.min(point.y, bounds.y1)
      bounds.x2 = Math.max(point.x, bounds.x2)
      bounds.y2 = Math.max(point.y, bounds.y2)
    }

    this._bounds.x = bounds.x1
    this._bounds.y = bounds.y1
    this._bounds.w = bounds.x2 - bounds.x1
    this._bounds.h = bounds.y2 - bounds.y1

    for (let i = 0; i < 4; i++) {
      this._corners[i].x = this._bounds.x
      this._corners[i].y = this._bounds.y
      this._corners[i].x += this._bounds.w * (1 - Math.floor(Math.abs(i - 1.5)))
      this._corners[i].y += this._bounds.h * Math.floor(i / 2)
    }
  }

  /** Scale */
  private scale (anchor: Vector2D, scale: Vector2D) {
    // Scale minimum to 1 pixel wide/high
    let widthSign = Math.sign(scale.x)
    if (widthSign === 0) {
      widthSign = 1
    }
    const newWidth = Math.max(Math.abs(scale.x * this._bounds.w), 1) * widthSign

    let heightSign = Math.sign(scale.y)
    if (heightSign === 0) {
      heightSign = 1
    }
    const newHeight =
      Math.max(Math.abs(scale.y * this._bounds.h), 1) * heightSign

    scale.x = newWidth / this._bounds.w
    scale.y = newHeight / this._bounds.h

    for (const shape of this._nodes) {
      shape.x = (shape.x - anchor.x) * scale.x + anchor.x
      shape.y = (shape.y - anchor.y) * scale.y + anchor.y
    }

    // If the highlighted handle is one of the corners,
    // flip the handle if scale is negative
    if (this._highlightedHandle >= this._nodes.length &&
        this._highlightedHandle <= this._nodes.length + this._corners.length) {
      const cornerIndex = this._highlightedHandle - this._nodes.length
      const cornerCoords = new Vector2D(
        (1 - Math.floor(Math.abs(cornerIndex - 1.5))),
        Math.floor(cornerIndex / 2)
      )

      if (scale.x < 0) {
        cornerCoords.x = 1 - cornerCoords.x
      }

      if (scale.y < 0) {
        cornerCoords.y = 1 - cornerCoords.y
      }

      this._highlightedHandle = Math.abs(cornerCoords.y * 3 - cornerCoords.x) +
        this._nodes.length
    }

    this.updateBounds()
  }

  /** Get index of opposite corner */
  private getOppositeCorner (index: number) {
    return (index + 2) % this._corners.length
  }
}
