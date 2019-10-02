import _ from 'lodash'
import { LabelType, ShapeType, State } from '../../functional/types'
import { Size2D } from '../../math/size2d'
import { Vector2D } from '../../math/vector2d'
import { Context2D, getColorById } from '../util'

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
  protected _labelId: number
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
  /** -1 means no handle is selected */
  protected _selectedHandle: number
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

  constructor () {
    this._index = -1
    this._labelId = -1
    this._selected = false
    this._selectedHandle = -1
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
  }

  /** Set whether the label is highlighted */
  public setViewMode (mode: Partial<ViewMode>): void {
    this._viewMode = _.assign(this._viewMode, mode)
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

  /** get labelId */
  public get labelId (): number {
    return this._labelId
  }

  /** select the label */
  public setSelected (s: boolean, h: number = -1) {
    if (s && h < 0) {
      throw Error('need to select handle as well')
    }
    this._selected = s
    this._selectedHandle = h
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
    return this._order
  }

  /** set the order of this label */
  public set order (o: number) {
    this._order = o
  }

  /** return the editing of this label */
  public get editing (): boolean {
    return this._editing
  }

  /** set the editing of this label */
  public set editing (e: boolean) {
    this._editing = e
  }

  /**
   * Draw the label on viewing or control canvas
   * @param {Context2D} canvas
   * @param {number} ratio: display to image size ratio
   * @param {DrawMode} mode
   */
  public abstract draw (canvas: Context2D, ratio: number, mode: DrawMode): void

  /**
   * Handle mouse down
   * @param coord
   */
  public onMouseDown (coord: Vector2D): boolean {
    this._mouseDown = true
    if (this._selected) {
      this._mouseDownCoord = coord.clone()
      return true
    }
    return false
  }

  /**
   * Handle mouse up
   * @param coord
   */
  public onMouseUp (_coord: Vector2D): boolean {
    this._mouseDown = false
    return false
  }

  /**
   * Process mouse move
   * @param {Vector2D} coord: mouse coordinate
   * @param {Size2D} limit: limit of the canvas frame
   */
  public abstract onMouseMove (
    coord: Vector2D, limit: Size2D,
    labelIndex: number, handleIndex: number
  ): boolean

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
   * Expand the primitive shapes to drawable shapes
   * @param {ShapeType[]} shapes
   */
  public abstract updateShapes (shapes: ShapeType[]): void

  /** Update the shapes of the label to the state */
  public abstract commitLabel (): boolean

  /**
   * Initialize this label to be temporary
   * @param {State} state
   * @param {Vector2D} start: starting coordinate of the label
   */
  public abstract initTemp (
    state: State, start: Vector2D): void

  /** Convert label state to drawable */
  public updateState (
    state: State, itemIndex: number, labelId: number): void {
    const item = state.task.items[itemIndex]
    this._label = item.labels[labelId]
    this._order = this._label.order
    this._labelId = this._label.id
    this._color = getColorById(this._labelId)
    this.setSelected(labelId === state.user.select.label, 0)
    this.updateShapes(this._label.shapes.map((i) => item.shapes[i].shape))
  }
}

export default Label2D
