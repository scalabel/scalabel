import _ from 'lodash'
import { sprintf } from 'sprintf-js'
import Session from '../../common/session'
import { LabelTypes } from '../../common/types'
import { State } from '../../functional/types'
import { Size2D } from '../../math/size2d'
import { Vector2D } from '../../math/vector2d'
import { Context2D } from '../util'
import { Box2D } from './box2d'
import { DrawMode, Label2D } from './label2d'
import { Tag2D } from './tag2d'

/**
 * Make a new drawable label based on the label type
 * @param {string} labelType: type of the new label
 */
function makeDrawableLabel (labelType: string): Label2D {
  if (labelType === LabelTypes.BOX_2D) {
    return new Box2D()
  } else if (labelType === LabelTypes.TAG) {
    return new Tag2D()
  } else {
    throw new Error(sprintf('Undefined label type %s', labelType))
  }
}

/**
 * List of drawable labels
 * ViewController for the labels
 */
export class Label2DList {
  /** Label the labels */
  private _labels: {[labelId: number]: Label2D}
  /** list of the labels sorted by label order */
  private _labelList: Label2D[]
  /** Recorded state of last update */
  private _state: State
  /** selected label */
  private _selectedLabel: Label2D | null
  /** highlighted label */
  private _highlightedLabel: Label2D | null
  /** whether mouse is down */
  private _mouseDown: boolean

  constructor () {
    this._labels = {}
    this._labelList = []
    this._selectedLabel = null
    this._highlightedLabel = null
    this._mouseDown = false
    this._state = Session.getState()
    this.updateState(this._state, this._state.user.select.item)
  }

  /**
   * Access the drawable label by index
   */
  public get (index: number): Label2D {
    return this._labelList[index]
  }

  /** get readonly label list for state inspection */
  public getLabelList (): Array<Readonly<Label2D>> {
    return this._labelList
  }

  /**
   * Draw label and control context
   * @param {Context2D} labelContext
   * @param {Context2D} controlContext
   * @param {number} ratio: ratio: display to image size ratio
   */
  public redraw (
      labelContext: Context2D, controlContext: Context2D, ratio: number): void {
    this._labelList.forEach((v) => v.draw(labelContext, ratio, DrawMode.VIEW))
    if (!this._mouseDown) {
      this._labelList.forEach(
        (v) => v.draw(controlContext, ratio, DrawMode.CONTROL))
    }
  }

  /**
   * update labels from the state
   */
  public updateState (state: State, itemIndex: number): void {
    if (this._mouseDown) {
      // don't update the drawing state when the mouse is down
      return
    }
    const self = this
    self._state = state
    const item = state.task.items[itemIndex]
    // remove any label not in the state
    self._labels = Object.assign({} as typeof self._labels,
        _.pick(self._labels, _.keys(item.labels)))
    // update drawable label values
    _.forEach(item.labels, (label, key) => {
      const labelId = Number(key)
      if (!(labelId in self._labels)) {
        self._labels[labelId] = makeDrawableLabel(label.type)
      }
      self._labels[labelId].updateState(state, itemIndex, labelId)
    })
    // order the labels and assign order values
    self._labelList = _.sortBy(_.values(self._labels), [(label) => label.order])
    _.forEach(self._labelList,
      (l: Label2D, index: number) => { l.index = index })
    this._highlightedLabel = null
    if (state.user.select.label >= 0 &&
        (state.user.select.label in this._labels)) {
      this._selectedLabel = this._labels[state.user.select.label]
    } else {
      this._selectedLabel = null
    }
  }

  /**
   * Process mouse down action
   * @param coord
   * @param labelIndex
   * @param handleIndex
   */
  public onMouseDown (
      coord: Vector2D, labelIndex: number, handleIndex: number): boolean {
    this._mouseDown = true
    if (this._highlightedLabel !== null) {
      this._highlightedLabel.setHighlighted(false)
      this._highlightedLabel = null
    }
    if (this._selectedLabel !== null) {
      this._selectedLabel.setSelected(false)
      this._selectedLabel = null
    }
    if (labelIndex >= 0) {
      this._selectedLabel = this._labelList[labelIndex]
      this._selectedLabel.setSelected(true, handleIndex)
      this._selectedLabel.onMouseDown(coord)
      return true
    } else {
      const state = this._state
      const label = makeDrawableLabel(
        state.task.config.labelTypes[state.user.select.labelType])
      label.initTemp(state, coord)
      this._selectedLabel = label
      this._labelList.push(label)
      label.onMouseDown(coord)
      return true
    }
  }

  /**
   * Process mouse up action
   * @param coord
   * @param labelIndex
   * @param handleIndex
   */
  public onMouseUp (
      coord: Vector2D, _labelIndex: number, _handleIndex: number): void {
    this._mouseDown = false
    if (this._selectedLabel !== null) {
      this._selectedLabel.onMouseUp(coord)
      // If label did not commit remove from list
      if (!this._selectedLabel.commitLabel()) {
        this._labelList.splice(this._labelList.indexOf(this._selectedLabel), 1)
      }
    }
  }

  /**
   * Process mouse move action
   */
  public onMouseMove (
      coord: Vector2D, canvasLimit: Size2D,
      labelIndex: number, handleIndex: number): boolean {
    if (!this._selectedLabel ||
        !this._selectedLabel.onMouseMove(coord, canvasLimit)) {
      if (labelIndex >= 0) {
        if (this._highlightedLabel === null) {
          this._highlightedLabel = this._labelList[labelIndex]
        }
        if (this._highlightedLabel.index !== labelIndex) {
          this._highlightedLabel.setHighlighted(false)
          this._highlightedLabel = this._labelList[labelIndex]
        }
        this._highlightedLabel.setHighlighted(true, handleIndex)
      } else if (this._highlightedLabel !== null) {
        this._highlightedLabel.setHighlighted(false)
        this._highlightedLabel = null
      }
      return false
    }
    return true
  }
}
