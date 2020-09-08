import _ from "lodash"

import { policyFromString } from "../../common/track"
import { LabelTypeName, TrackPolicyType } from "../../const/common"
import { makeState } from "../../functional/states"
import { Label2DTemplateType, State } from "../../types/state"
import { Context2D } from "../util"
import { Box2D } from "./box2d"
import { CustomLabel2D } from "./custom_label"
import { DrawMode, Label2D } from "./label2d"
import { Polygon2D } from "./polygon2d"
import { Tag2D } from "./tag2d"

/**
 * Make a new drawable label based on the label type
 *
 * @param {string} labelType: type of the new label
 * @param labelList
 * @param labelType
 */
export function makeDrawableLabel2D(
  labelList: Label2DList,
  labelType: string,
  labelTemplates: { [name: string]: Label2DTemplateType }
): Label2D | null {
  if (labelType in labelTemplates) {
    return new CustomLabel2D(labelList, labelTemplates[labelType])
  }
  switch (labelType) {
    case LabelTypeName.BOX_2D:
      return new Box2D(labelList)
    case LabelTypeName.TAG:
      return new Tag2D(labelList)
    case LabelTypeName.POLYGON_2D:
      return new Polygon2D(labelList, true)
    case LabelTypeName.POLYLINE_2D:
      return new Polygon2D(labelList, false)
  }
  return null
}

/**
 * List of drawable labels
 * ViewController for the labels
 */
export class Label2DList {
  /** Label the labels */
  private _labels: { [labelId: string]: Label2D }
  /** list of the labels sorted by label order */
  private _labelList: Label2D[]
  /** selected label */
  private _selectedLabels: Label2D[]
  /** state */
  private _state: State
  /** label templates */
  private _labelTemplates: { [name: string]: Label2DTemplateType }
  /** callbacks */
  private readonly _callbacks: Array<() => void>
  /** New labels to be committed */
  private readonly _updatedLabels: Set<Label2D>

  /**
   * Constructor
   */
  constructor() {
    this._labels = {}
    this._labelList = []
    this._selectedLabels = []
    this._state = makeState()
    this._callbacks = []
    this._labelTemplates = {}
    this._updatedLabels = new Set()
  }

  /**
   * Access the drawable label by index
   *
   * @param index
   */
  public get(index: number): Label2D {
    return this._labelList[index]
  }

  /**
   * Subscribe callback for drawable update
   *
   * @param callback
   */
  public subscribe(callback: () => void): void {
    this._callbacks.push(callback)
  }

  /**
   * Unsubscribe callback for drawable update
   *
   * @param callback
   */
  public unsubscribe(callback: () => void): void {
    const index = this._callbacks.indexOf(callback)
    if (index >= 0) {
      this._callbacks.splice(index, 1)
    }
  }

  /** Call when any drawable has been updated */
  public onDrawableUpdate(): void {
    for (const callback of this._callbacks) {
      callback()
    }
  }

  /**
   * Get label by id
   *
   * @param id
   */
  public getLabelById(id: number): Label2D {
    return this._labels[id]
  }

  /** get label list for state inspection */
  public get labelList(): Label2D[] {
    return this._labelList
  }

  /** get selectedLabel for state inspection */
  public get selectedLabels(): Label2D[] {
    return this._selectedLabels
  }

  /**
   * Get id's of selected labels
   */
  public get selectedLabelIds(): { [index: number]: string[] } {
    return this._state.user.select.labels
  }

  /**
   * Get current policy type
   */
  public get policyType(): TrackPolicyType {
    return policyFromString(
      this._state.task.config.policyTypes[this._state.user.select.policyType]
    )
  }

  /**
   * Draw label and control context
   *
   * @param {Context2D} labelContext
   * @param {Context2D} controlContext
   * @param {number} ratio: ratio: display to image size ratio
   * @param ratio
   * @param hideLabels
   */
  public redraw(
    labelContext: Context2D,
    controlContext: Context2D,
    ratio: number,
    hideLabels?: boolean
  ): void {
    const labelsToDraw =
      hideLabels !== null && hideLabels !== undefined && hideLabels
        ? this._labelList.filter((label) => label.selected)
        : this._labelList
    labelsToDraw.forEach((v) => v.draw(labelContext, ratio, DrawMode.VIEW))
    labelsToDraw.forEach((v) => v.draw(controlContext, ratio, DrawMode.CONTROL))
  }

  /**
   * update labels from the state
   *
   * @param state
   */
  public updateState(state: State): void {
    // Don't interrupt ongoing editing
    if (this._selectedLabels.length > 0 && this.selectedLabels[0].editing) {
      return
    }

    this._state = state
    this._labelTemplates = state.task.config.label2DTemplates
    const itemIndex = state.user.select.item
    const item = state.task.items[itemIndex]
    // Remove any label not in the state
    this._labels = Object.assign({}, _.pick(this._labels, _.keys(item.labels)))
    // Update drawable label values
    _.forEach(item.labels, (label, labelId) => {
      if (!(labelId in this._labels)) {
        const newLabel = makeDrawableLabel2D(
          this,
          label.type,
          this._labelTemplates
        )
        if (newLabel !== null) {
          this._labels[labelId] = newLabel
        }
      }
      if (labelId in this._labels) {
        const drawableLabel = this._labels[labelId]
        if (!drawableLabel.editing) {
          drawableLabel.updateState(state, itemIndex, labelId)
        }
      }
    })
    // Order the labels and assign order values
    this._labelList = _.sortBy(_.values(this._labels), [(label) => label.order])
    _.forEach(this._labelList, (l: Label2D, index: number) => {
      l.index = index
    })
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

  /** Get uncommitted labels */
  public popUpdatedLabels(): Label2D[] {
    const labels = [...this._updatedLabels.values()]
    this._updatedLabels.clear()
    return labels
  }

  /**
   * Push updated label to array
   *
   * @param label
   */
  public addUpdatedLabel(label: Label2D): void {
    this._updatedLabels.add(label)
  }

  /** Clear uncommitted label list */
  public clearUpdatedLabels(): void {
    this._updatedLabels.clear()
  }
}
