import _ from 'lodash'
import { sprintf } from 'sprintf-js'
import { Cursor } from '../../common/types'
import { makeLabel } from '../../functional/states'
import { LabelType, State } from '../../functional/types'
import { Size2D } from '../../math/size2d'
import { Vector2D } from '../../math/vector2d'
import { Context2D, getColorById } from '../util'
import { Label2DList } from './label2d_list'
import { Shape2D } from './shape2d'

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
  /** index of the label */
  protected _index: number
  /** the corresponding label in the state */
  protected _labelState: LabelType
  /** whether the label is selected */
  protected _selected: boolean
  /** whether the label is highlighted */
  protected _highlighted: boolean
  /** -1 means no handle is selected */
  protected _highlightedHandle: number
  /** rgba color decided by labelId */
  protected _color: number[]
  /** whether the label is being editing */
  protected _editing: boolean
  /** shapes */
  protected _shapes: Shape2D[]
  /** label list */
  protected _labelList: Label2DList
  /** parent label */
  protected _parent: Label2D | null
  /** child labels */
  protected _children: Label2D []

  constructor (labelList: Label2DList) {
    this._index = -1
    this._selected = false
    this._highlighted = false
    this._highlightedHandle = -1
    this._labelState = makeLabel()
    this._color = [0, 0, 0, 1]
    this._editing = false
    this._shapes = []
    this._labelList = labelList
    this._parent = null
    this._children = []
  }

  /**
   * Set index of this label
   */
  public set index (i: number) {
    this._index = i
  }

  /** get index */
  public get index (): number {
    return this._index
  }

  /** get category */
  public get category (): number[] {
    if (this._labelState && this._labelState.category) {
      return this._labelState.category
    }
    return []
  }

  /** get attributes */
  public get attributes (): {[key: number]: number[]} {
    if (this._labelState && this._labelState.attributes) {
      return this._labelState.attributes
    }
    return {}
  }

  /** get label type */
  public get type (): string {
    return this._labelState.type
  }

  /** get label state */
  public get labelState (): LabelType {
    if (!this._labelState) {
      throw new Error('Label uninitialized')
    }
    return this._labelState
  }

  /** get label id */
  public get labelId (): number {
    return this._labelState.id
  }

  /** get track id */
  public get trackId (): number {
    return this._labelState.track
  }

  /** get item index */
  public get item (): number {
    return this._labelState.item
  }

  /** get color */
  public get color (): number[] {
    return this._color
  }

  /** get whether highlighted */
  public get highlighted (): boolean {
    return this._highlighted || this._selected
  }

  /** Get cursor to use when highlighting */
  public get highlightCursor (): string {
    return Cursor.CROSSHAIR
  }

  /** Get shape drawables */
  public shapes (): Shape2D[] {
    return this._shapes
  }

  /** Returns whether this label is selected */
  public get selected (): boolean {
    return this._selected
  }

  /** Set selected */
  public set selected (s: boolean) {
    this._selected = s
  }

  /** highlight the label */
  public setHighlighted (h: boolean, handleIndex: number = -1) {
    if (h && handleIndex < 0) {
      throw Error('need to highlight handle as well')
    }
    this._highlighted = h
    this._highlightedHandle = handleIndex
  }

  /** return order of this label */
  public get order (): number {
    return this._labelState.order
  }

  /** set the order of this label */
  public set order (o: number) {
    this._labelState.order = o
  }

  /** set parent */
  public set parent (parent: Label2D | null) {
    this._parent = parent
    const root = this.getRoot()
    this._color = getColorById(root.labelId, root.trackId)
  }

  /** get parent */
  public get parent (): Label2D | null {
    return this._parent
  }

  /** Get top level parent */
  public getRoot (): Label2D {
    let label: Label2D = this
    while (label.parent) {
      label = label.parent
    }
    return label
  }

  /** return whether label is being edited */
  public get editing (): boolean {
    return this._editing
  }

  /** set the editing of this label */
  public set editing (e: boolean) {
    this._editing = e
  }

  /** Whether label valid */
  public isValid (): boolean {
    return true
  }

  /** Set to manual */
  public setManual () {
    this._labelState.manual = true
  }

  /**
   * Draw the label on viewing or control canvas
   * @param {Context2D} canvas
   * @param {number} ratio: display to image size ratio
   * @param {DrawMode} mode
   */
  public abstract draw (
    canvas: Context2D, ratio: number, mode: DrawMode, mousePosition: Vector2D
  ): void

  /**
   * Draw the label tag on viewing or control canvas
   * @param {Context2D} ctx
   * @param {[number, number]} position
   * @param {number} ratio
   * @param {number[]} fillStyle
   */
  public drawTag (ctx: Context2D,
                  ratio: number,
                  position: [number, number],
                  fillStyle: number[]
                  ) {
    const TAG_WIDTH = 50
    const TAG_HEIGHT = 28
    const [x, y] = position
    const config = this._labelList.config
    const self = this
    ctx.save()
    const category = (
      self._labelState &&
      self._labelState.category[0] <
        this._labelList.config.categories.length &&
      self._labelState.category[0] >= 0
    ) ? config.categories[self._labelState.category[0]] : ''
    const attributes = self._labelState && self._labelState.attributes ?
                       self._labelState.attributes : {}
    const words = category.split(' ')
    let tw = TAG_WIDTH
    // abbreviate tag as the first 3 chars of the last word
    let abbr = words[words.length - 1].substring(0, 3)

    for (const attributeId of Object.keys(attributes)) {
      const attribute = config.attributes[Number(attributeId)]
      if (attribute.toolType === 'switch') {
        if (attributes[Number(attributeId)][0] > 0) {
          abbr += ',' + attribute.tagText
          tw += 36
        }
      } else if (attribute.toolType === 'list') {
        if (attribute &&
          attributes[Number(attributeId)][0] > 0) {
          abbr += ',' + attribute.tagText + ':' +
              attribute.tagSuffixes[attributes[Number(attributeId)][0]]
          tw += 72
        }
      }
    }

    ctx.fillStyle = sprintf('rgb(%d, %d, %d)',
      fillStyle[0], fillStyle[1], fillStyle[2])
    ctx.fillRect(x * ratio, y * ratio - TAG_HEIGHT,
                 tw, TAG_HEIGHT)
    ctx.fillStyle = 'rgb(0,0,0)'
    ctx.font = sprintf('%dpx Verdana', 20)
    ctx.fillText(abbr, (x * ratio + 6), (y * ratio - 6))
    ctx.restore()
  }

  /**
   * Handle mouse down
   * @param coord
   */
  public abstract drag (delta: Vector2D, limit: Size2D): boolean

  /**
   * Handle mouse up
   * @param coord
   */
  public abstract click (coord: Vector2D): boolean

  /**
   * handle keyboard down event
   * @param e pressed key
   */
  public abstract onKeyDown (e: string): boolean

  /**
   * handle keyboard up event
   * @param e pressed key
   */
  public abstract onKeyUp (e: string): void

  /**
   * Initialize this label to be temporary
   * @param {State} state
   * @param {Vector2D} start: starting coordinate of the label
   */
  public initTemp (state: State, _start: Vector2D): void {
    this.order = state.task.status.maxOrder + 1
    this._labelState.id = -1
    this._labelState.track = -1
    this._color = getColorById(
      state.task.status.maxLabelId + 1,
      (state.task.config.tracking) ? state.task.status.maxTrackId + 1 : -1
    )
    this._selected = true
    this.editing = true
  }

  /** Convert label state to drawable */
  public updateState (
    labelState: LabelType
  ): void {
    this._labelState = { ...labelState }
    this._shapes = []
    for (const shapeId of this._labelState.shapes) {
      const shape = this._labelList.getShape(shapeId)
      if (shape) {
        this._shapes.push(shape)
      } else {
        throw new Error(`Could not find shape with id ${shapeId}`)
      }
    }
  }

  /** Add all shapes as updated to label list */
  protected setAllShapesUpdated () {
    for (const shape of this._shapes) {
      this._labelList.addUpdatedShape(shape)
    }
    this._labelList.addUpdatedLabel(this)
  }
}

export default Label2D
