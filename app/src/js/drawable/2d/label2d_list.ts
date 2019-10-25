import _ from 'lodash'
import { changeSelect, linkLabels } from '../../action/common'
import { selectLabel } from '../../action/select'
import Session from '../../common/session'
import { makeTrackPolicy, Track } from '../../common/track'
import { Key, LabelTypeName } from '../../common/types'
import { makeTrack } from '../../functional/states'
import { State } from '../../functional/types'
import { Size2D } from '../../math/size2d'
import { Vector2D } from '../../math/vector2d'
import { Context2D } from '../util'
import { Box2D } from './box2d'
import { DrawMode, Label2D } from './label2d'
import { Polygon2D } from './polygon2d'
import { Tag2D } from './tag2d'

/**
 * Make a new drawable label based on the label type
 * @param {string} labelType: type of the new label
 */
function makeDrawableLabel (labelType: string): Label2D | undefined {
  switch (labelType) {
    case LabelTypeName.BOX_2D:
      return new Box2D()
    case LabelTypeName.TAG:
      return new Tag2D()
    case LabelTypeName.POLYGON_2D:
      return new Polygon2D()
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
  private _selectedLabels: Label2D[]
  /** highlighted label */
  private _highlightedLabel: Label2D | null
  /** whether mouse is down */
  private _mouseDown: boolean
  /** The hashed list of keys currently down */
  private _keyDownMap: { [key: string]: boolean }

  constructor () {
    this._labels = {}
    this._labelList = []
    this._selectedLabels = []
    this._highlightedLabel = null
    this._mouseDown = false
    this._state = Session.getState()
    this._keyDownMap = {}
    this.updateState(this._state, this._state.user.select.item)
  }

  /**
   * Access the drawable label by index
   */
  public get (index: number): Label2D {
    return this._labelList[index]
  }

  /** get label list for state inspection */
  public get labelList (): Label2D[] {
    return this._labelList
  }

  /** get highlightedLabel for state inspection */
  public get highlightedLabel (): Label2D | null {
    return this._highlightedLabel
  }

  /** get selectedLabel for state inspection */
  public get selectedLabels (): Label2D[] {
    return this._selectedLabels
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
        const newLabel = makeDrawableLabel(label.type)
        if (newLabel) {
          self._labels[labelId] = newLabel
        }
      }
      if (labelId in self._labels) {
        self._labels[labelId].updateState(state, itemIndex, labelId)
      }
    })
    // order the labels and assign order values
    self._labelList = _.sortBy(_.values(self._labels), [(label) => label.order])
    _.forEach(self._labelList,
      (l: Label2D, index: number) => { l.index = index })
    this._highlightedLabel = null
    this._selectedLabels = []
    const select = state.user.select
    const selectedLabelItems = Object.keys(select.labels)
    for (const key of selectedLabelItems) {
      const index = Number(key)
      for (const id of select.labels[index]) {
        if (id in this._labels) {
          this._selectedLabels.push(this._labels[id])
        }
      }
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
    if (this._highlightedLabel !== null &&
      (this.isSelectedLabelsEmpty() ||
      this._selectedLabels[0].editing === false)) {
      this._highlightedLabel.setHighlighted(false)
      this._highlightedLabel = null
    }

    if (this.isKeyDown(Key.META) || this.isKeyDown(Key.CONTROL)) {
      // multi select
      if (labelIndex >= 0) {
        const label = this._labelList[labelIndex]
        const index = this.selectedLabels.indexOf(label)
        if (index === -1) {
          this._selectedLabels.push(label)
        } else {
          this._selectedLabels.splice(index, 1)
        }
        const selectedLabelIdArray = []
        for (const tmp of this._selectedLabels) {
          selectedLabelIdArray.push(tmp.labelId)
        }
        if (this.isSelectedLabelsEmpty()) {
          Session.dispatch(changeSelect(
            { category: undefined,
              attributes: undefined,
              labels: {}
            })
          )
        } else {
          Session.dispatch(changeSelect(
            {
              category: this._selectedLabels[0].category[0],
              attributes: this._selectedLabels[0].attributes,
              labels: { [this._state.user.select.item]: selectedLabelIdArray }
            })
          )
        }
      }
      return true
    } else if (!this.isSelectedLabelsEmpty() &&
      this._selectedLabels[0].editing === false) {
      for (const label of this._selectedLabels) {
        label.setSelected(false)
      }
      this._selectedLabels = []
    }

    this._mouseDown = true
    if (this.isSelectedLabelsEmpty()) {
      if (labelIndex >= 0) {
        this._selectedLabels.push(this._labelList[labelIndex])
        this._selectedLabels[0].setSelected(true, handleIndex)
        Session.dispatch(selectLabel(
          this._state,
          this._state.user.select.item,
          this._selectedLabels[0].labelId,
          this._selectedLabels[0].category[0],
          this._selectedLabels[0].attributes
        ))
      } else {
        const state = this._state
        const currentPolicyType =
          state.task.config.policyTypes[state.user.select.policyType]
        const newTrack = new Track()
        newTrack.updateState(
          makeTrack(-1), makeTrackPolicy(newTrack, currentPolicyType)
        )
        Session.tracks[-1] = newTrack

        const label = makeDrawableLabel(
          state.task.config.labelTypes[state.user.select.labelType])
        if (label) {
          label.initTemp(state, coord)
          this._selectedLabels.push(label)
          this._labelList.push(label)
        }
      }
    }
    this._selectedLabels[0].onMouseDown(coord)
    return true
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
    if (!this.isSelectedLabelsEmpty() && !this.isKeyDown(Key.META)) {
      const shouldDelete = !this._selectedLabels[0].onMouseUp(coord)
      if (shouldDelete) {
        this._labelList.splice(
          this._labelList.indexOf(this._selectedLabels[0]), 1
        )
      }
    }
  }

  /**
   * Process mouse move action
   */
  public onMouseMove (
      coord: Vector2D, canvasLimit: Size2D,
      labelIndex: number, handleIndex: number): boolean {
    if (!this.isSelectedLabelsEmpty() &&
      this._selectedLabels[0].editing === true) {
      this._selectedLabels[0].onMouseMove(
        coord, canvasLimit, labelIndex, handleIndex)
      return true
    } else {
      if (labelIndex >= 0) {
        if (!this._highlightedLabel) {
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
    }
    return false
  }

  /**
   * Handle keyboard down events
   * @param e
   */
  public onKeyDown (e: KeyboardEvent): void {
    this._keyDownMap[e.key] = true
    if (!this.isSelectedLabelsEmpty() &&
      !this._selectedLabels[0].onKeyDown(e.key)) {
      this._labelList.splice(
        this._labelList.indexOf(this._selectedLabels[0]), 1
      )
      this._selectedLabels = []
    }
    // linking
    if (this.isKeyDown(Key.L_LOW) || this.isKeyDown(Key.L_UP)) {
      this.linkLabels()
    }
  }

  /**
   * Handle keyboard up events
   * @param e
   */
  public onKeyUp (e: KeyboardEvent): void {
    delete this._keyDownMap[e.key]
    if (!this.isSelectedLabelsEmpty()) {
      this._selectedLabels[0].onKeyUp(e.key)
    }
  }

  /** judge whether selectedLabels is empty */
  private isSelectedLabelsEmpty (): boolean {
    return this._selectedLabels.length === 0
  }

  /**
   * Whether a specific key is pressed down
   * @param key - the key to check
   */
  private isKeyDown (key: Key): boolean {
    return this._keyDownMap[key]
  }

  /** link selected labels */
  private linkLabels (): void {
    if (this.selectedLabels.length < 2) {
      return
    }
    const selectedLabelIdArray = []
    for (const tmp of this._selectedLabels) {
      selectedLabelIdArray.push(tmp.labelId)
    }
    Session.dispatch(linkLabels(
      this._state.user.select.item, selectedLabelIdArray
    ))
  }
}
