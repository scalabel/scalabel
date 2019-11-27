import _ from 'lodash'
import { policyFromString } from '../../common/track_policies/track_policy'
import { LabelTypeName, TrackPolicyType } from '../../common/types'
import { makeState } from '../../functional/states'
import { State } from '../../functional/types'
import { Context2D } from '../util'
import { Box2D } from './box2d'
import { DrawMode, Label2D } from './label2d'
import { Polygon2D } from './polygon2d'
import { Tag2D } from './tag2d'

/**
 * Make a new drawable label based on the label type
 * @param {string} labelType: type of the new label
 */
export function makeDrawableLabel2D (labelType: string): Label2D | null {
  switch (labelType) {
    case LabelTypeName.BOX_2D:
      return new Box2D()
    case LabelTypeName.TAG:
      return new Tag2D()
    case LabelTypeName.POLYGON_2D:
      return new Polygon2D()
  }
  return null
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
  /** selected label */
  private _selectedLabels: Label2D[]
  /** state */
  private _state: State
  /** callbacks */
  private _callbacks: Array<() => void>

  constructor () {
    this._labels = {}
    this._labelList = []
    this._selectedLabels = []
    this._state = makeState()
    this._callbacks = []
  }

  /**
   * Access the drawable label by index
   */
  public get (index: number): Label2D {
    return this._labelList[index]
  }

  /** Subscribe callback for drawable update */
  public subscribe (callback: () => void) {
    this._callbacks.push(callback)
  }

  /** Unsubscribe callback for drawable update */
  public unsubscribe (callback: () => void) {
    const index = this._callbacks.indexOf(callback)
    if (index >= 0) {
      this._callbacks.splice(index, 1)
    }
  }

  /** Call when any drawable has been updated */
  public onDrawableUpdate (): void {
    for (const callback of this._callbacks) {
      callback()
    }
  }

  /**
   * Get label by id
   * @param id
   */
  public getLabelById (id: number): Label2D {
    return this._labels[id]
  }

  /** get label list for state inspection */
  public get labelList (): Label2D[] {
    return this._labelList
  }

  /** get selectedLabel for state inspection */
  public get selectedLabels (): Label2D[] {
    return this._selectedLabels
  }

  /**
   * Get id's of selected labels
   */
  public get selectedLabelIds (): {[index: number]: number[]} {
    return this._state.user.select.labels
  }

  /**
   * Get current policy type
   */
  public get policyType (): TrackPolicyType {
    return policyFromString(
      this._state.task.config.policyTypes[this._state.user.select.policyType]
    )
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
    this._labelList.forEach(
      (v) => v.draw(controlContext, ratio, DrawMode.CONTROL)
    )
  }

  /**
   * update labels from the state
   */
  public updateState (state: State): void {
    this._state = state
    const self = this
    const itemIndex = state.user.select.item
    const item = state.task.items[itemIndex]
    // remove any label not in the state
    self._labels = Object.assign({} as typeof self._labels,
        _.pick(self._labels, _.keys(item.labels)))
    // update drawable label values
    _.forEach(item.labels, (label, key) => {
      const labelId = Number(key)
      if (!(labelId in self._labels)) {
        const newLabel = makeDrawableLabel2D(label.type)
        if (newLabel) {
          self._labels[labelId] = newLabel
        }
      }
      if (labelId in self._labels) {
        const drawableLabel = self._labels[labelId]
        if (!drawableLabel.editing) {
          drawableLabel.updateState(state, itemIndex, labelId)
        }
      }
    })
    // order the labels and assign order values
    self._labelList = _.sortBy(_.values(self._labels), [(label) => label.order])
    _.forEach(self._labelList,
      (l: Label2D, index: number) => { l.index = index })
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

  /** check whether selectedLabels is empty */
  public selectedLabelsEmpty (): boolean {
    return this._selectedLabels.length === 0
  }
}
