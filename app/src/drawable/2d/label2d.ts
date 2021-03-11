import _ from "lodash"

import { Cursor, LabelTypeName } from "../../const/common"
import { getRootLabelId, getRootTrackId } from "../../functional/common"
import { makeTaskConfig, makeTrack } from "../../functional/states"
import { Size2D } from "../../math/size2d"
import { Vector2D } from "../../math/vector2d"
import {
  ConfigType,
  IdType,
  INVALID_ID,
  LabelType,
  ShapeType,
  State
} from "../../types/state"
import { Context2D, getColorById } from "../util"
import { Label2DList } from "./label2d_list"

export enum DrawMode {
  VIEW,
  CONTROL
}

export interface ViewMode {
  /** whether the label is dimmed to show the other labels more clearly */
  dimmed: boolean
}

/**
 * Abstract class for 2D drawable labels
 */
export abstract class Label2D {
  /* The members are public for testing purpose */
  /** label id in state */
  protected _labelId: IdType
  /** track id in state */
  protected _trackId: IdType
  /** index of the label */
  protected _index: number
  /** drawing order of the label */
  protected _order: number
  /** the corresponding label in the state */
  protected _label: LabelType | null
  /** drawing mode */
  protected _viewMode: ViewMode
  /** whether the label is selected */
  protected _selected: boolean
  /** whether the label is highlighted */
  protected _highlighted: boolean
  /** -1 means no handle is selected */
  protected _highlightedHandle: number
  /** rgba color decided by labelId */
  protected _color: number[]
  /** true if mouse down */
  protected _mouseDown: boolean
  /** mouse coordinate when pressed down */
  protected _mouseDownCoord: Vector2D
  /** whether the label is being editing */
  protected _editing: boolean
  /** config */
  protected _config: ConfigType
  /** label list */
  protected _labelList: Label2DList
  /** whether the label is temporary */
  protected _temporary: boolean

  /**
   * Constructor
   *
   * @param labelList
   */
  constructor(labelList: Label2DList) {
    this._index = -1
    this._labelId = INVALID_ID
    this._trackId = INVALID_ID
    this._selected = false
    this._highlighted = false
    this._highlightedHandle = -1
    this._order = -1
    this._label = null
    this._color = [0, 0, 0, 1]
    this._viewMode = {
      dimmed: false
    }
    this._mouseDownCoord = new Vector2D()
    this._mouseDown = false
    this._editing = false
    this._config = makeTaskConfig()
    this._labelList = labelList
    this._temporary = true
  }

  /**
   * Set index of this label
   */
  public set index(i: number) {
    this._index = i
  }

  /** get index */
  public get index(): number {
    return this._index
  }

  /** get category */
  public get category(): number[] {
    if (this._label?.category !== undefined) {
      return this._label.category
    }
    return []
  }

  /** get attributes */
  public get attributes(): { [key: number]: number[] } {
    if (this._label?.attributes !== undefined) {
      return this._label.attributes
    }
    return {}
  }

  /** get label type */
  public get type(): string {
    if (this._label !== null) {
      return this._label.type
    }
    return LabelTypeName.EMPTY
  }

  /** get label state */
  public get label(): LabelType {
    if (this._label === null) {
      throw new Error("Label uninitialized")
    }
    return this._label
  }

  /** get labelId */
  public get labelId(): IdType {
    return this._labelId
  }

  /** get track id */
  public get trackId(): IdType {
    if (this._label !== null) {
      return this._label.track
    }
    return INVALID_ID
  }

  /** get item index */
  public get item(): number {
    if (this._label !== null) {
      return this._label.item
    }
    return -1
  }

  /** get color */
  public get color(): number[] {
    return this._color
  }

  /** get whether highlighted */
  public get highlighted(): boolean {
    return this._highlighted || this._selected
  }

  /** Get cursor to use when highlighting */
  public get highlightCursor(): string {
    return Cursor.CROSSHAIR
  }

  /** Returns whether this label is selected */
  public get selected(): boolean {
    return this._selected
  }

  /** return order of this label */
  public get order(): number {
    return this._order
  }

  /** set the order of this label */
  public set order(o: number) {
    this._order = o
  }

  /** return the editing of this label */
  public get editing(): boolean {
    return this._editing
  }

  /** set the editing of this label */
  public set editing(e: boolean) {
    this._editing = e
  }

  /** Parent drawable */
  public get parent(): Label2D | null {
    return null
  }

  /**
   * Check whether the label is temporary
   */
  public get temporary(): boolean {
    return this._temporary
  }

  /**
   * Set whether the label is highlighted
   *
   * @param mode
   */
  public setViewMode(mode: Partial<ViewMode>): void {
    this._viewMode = _.assign(this._viewMode, mode)
  }

  /**
   * select the label
   *
   * @param s
   */
  public setSelected(s: boolean): void {
    this._selected = s
  }

  /**
   * highlight the label
   *
   * @param h
   * @param handleIndex
   */
  public setHighlighted(h: boolean, handleIndex: number = -1): void {
    if (h && handleIndex < 0) {
      throw Error("need to highlight handle as well")
    }
    this._highlighted = h
    this._highlightedHandle = handleIndex
  }

  /** Whether label valid */
  public isValid(): boolean {
    return true
  }

  /** Set to manual */
  public setManual(): void {
    if (this._label !== null) {
      this._label.manual = true
    }
  }

  /**
   * Draw the label on viewing or control canvas
   *
   * @param {Context2D} canvas
   * @param {number} ratio: display to image size ratio
   * @param {DrawMode} mode
   */
  public abstract draw(canvas: Context2D, ratio: number, mode: DrawMode): void

  /**
   * Draw the label tag on viewing or control canvas
   *
   * @param {Context2D} ctx
   * @param {Vector2D} position
   * @param {number} ratio
   * @param {number[]} fillStyle
   */
  public drawTag(
    ctx: Context2D,
    ratio: number,
    position: Vector2D,
    fillStyle: number[]
  ): void {
    const TAG_WIDTH = 50
    const TAG_HEIGHT = 28
    const [x, y] = position
    ctx.save()
    const config = this._config
    const category =
      this._label !== null &&
      this._label.category[0] < config.categories.length &&
      this._label.category[0] >= 0
        ? config.categories[this._label.category[0]]
        : ""
    const attributes =
      this._label?.attributes !== undefined ? this._label.attributes : {}
    const words = category.split(" ")
    let tw = TAG_WIDTH
    // Abbreviate tag as the first 3 chars of the last word
    let abbr = words[words.length - 1].substring(0, 3)

    for (const attributeId of Object.keys(attributes)) {
      const attribute = config.attributes[Number(attributeId)]
      if (attribute.toolType === "switch") {
        if (attributes[Number(attributeId)][0] > 0) {
          abbr += "," + attribute.tagText
          tw += 36
        }
      } else if (attribute.toolType === "list") {
        if (
          Number(attributeId) in attribute &&
          attributes[Number(attributeId)][0] > 0
        ) {
          abbr +=
            "," +
            attribute.tagText +
            ":" +
            attribute.tagSuffixes[attributes[Number(attributeId)][0]]
          tw += 72
        }
      }
    }

    ctx.fillStyle = `rgb(${fillStyle[0]}, ${fillStyle[1]}, ${fillStyle[2]})`
    ctx.fillRect(x * ratio, y * ratio - TAG_HEIGHT, tw, TAG_HEIGHT)
    ctx.fillStyle = "rgb(0,0,0)"
    ctx.font = `${20}px Verdana`
    ctx.fillText(abbr, x * ratio + 6, y * ratio - 6)
    ctx.restore()
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
      this._mouseDownCoord = coord.clone()
      return true
    }
    return false
  }

  /**
   * Handle mouse up
   *
   * @param coord
   * @param _coord
   */
  public onMouseUp(_coord: Vector2D): boolean {
    this._mouseDown = false
    return false
  }

  /**
   * Process mouse move
   *
   * @param {Vector2D} coord: mouse coordinate
   * @param {Size2D} limit: limit of the canvas frame
   */
  public abstract onMouseMove(
    coord: Vector2D,
    limit: Size2D,
    labelIndex: number,
    handleIndex: number
  ): boolean

  /**
   * handle keyboard down event
   *
   * @param e pressed key
   */
  public abstract onKeyDown(e: string): boolean

  /**
   * handle keyboard up event
   *
   * @param e pressed key
   */
  public abstract onKeyUp(e: string): void

  /**
   * Expand the primitive shapes to drawable shapes
   *
   * @param {ShapeType[]} shapes
   */
  public abstract updateShapes(shapes: ShapeType[]): void

  /** Get shape id's and shapes for updating */
  public abstract shapes(): ShapeType[]

  /**
   * Initialize this label to be temporary
   *
   * @param {State} state
   * @param {Vector2D} start: starting coordinate of the label
   * @param start
   */
  public initTemp(state: State, start: Vector2D): void {
    this._order = state.task.status.maxOrder + 1
    this._config = state.task.config
    this._selected = true
    this._temporary = true
    this._label = this.initTempLabel(state, start)
    this._labelId = this._label.id
    if (state.task.config.tracking) {
      const track = makeTrack()
      this._label.track = track.id
    }
    this._trackId = this._label.track
    this._color = getColorById(this._labelId, this._trackId)
  }

  /**
   * Convert label state to drawable
   *
   * @param state
   * @param itemIndex
   * @param labelId
   */
  public updateState(state: State, itemIndex: number, labelId: IdType): void {
    const item = state.task.items[itemIndex]
    this._label = _.cloneDeep(item.labels[labelId])
    this._order = this._label.order
    this._labelId = this._label.id
    this._trackId = this._label.track
    this._config = state.task.config
    // None of the labels in the state is temporary
    this._temporary = false
    const select = state.user.select
    this._color = getColorById(
      getRootLabelId(item, labelId),
      getRootTrackId(item, labelId)
    )
    this.setSelected(
      this._label.item in select.labels &&
        select.labels[this._label.item].includes(labelId)
    )
    this.updateShapes(this._label.shapes.map((i) => item.shapes[i]))
  }

  /**
   * Initialize the temp label content
   *
   * @param state
   * @param _start
   */
  protected abstract initTempLabel(state: State, _start: Vector2D): LabelType
}

export default Label2D
