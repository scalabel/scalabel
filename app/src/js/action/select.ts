import Session from '../common/session'
import { changeSelect } from './common'
import * as types from './types'

/**
 * Delete given label
 * @param {number} itemIndex
 * @param {number} labelId
 * @return {DeleteLabelAction}
 */
export function deleteSelectedLabels (): types.DeleteLabelsAction {
  const select = Session.getState().user.select
  return {
    type: types.DELETE_LABELS,
    sessionId: Session.id,
    itemIndices: [select.item],
    labelIds: [select.labels]
  }
}

/**
 * Change the properties of the label
 * @param {number} itemIndex
 * @param {number} labelId
 * @param {Partial<LabelType>}props
 * @return {ChangeLabelPropsAction}
 */
export function changeSelectedLabelsAttributes (
  attributes: {[key: number]: number[]}
  ): types.ChangeLabelsAction {
  const select = Session.getState().user.select
  const duplicatedAttributes = select.labels.map(((_id) => ({ attributes })))
  return {
    type: types.CHANGE_LABELS,
    sessionId: Session.id,
    itemIndices: [select.item],
    labelIds: [select.labels],
    props: [duplicatedAttributes]
  }
}

/**
 * Change the properties of the label
 * @param {number} itemIndex
 * @param {number} labelId
 * @param {Partial<LabelType>}props
 * @return {ChangeLabelPropsAction}
 */
export function changeSelectedLabelsCategories (
  category: number[]
  ): types.ChangeLabelsAction {
  const select = Session.getState().user.select
  const duplicatedCategories = select.labels.map(((_id) => ({ category })))
  return {
    type: types.CHANGE_LABELS,
    sessionId: Session.id,
    itemIndices: [select.item],
    labelIds: [select.labels],
    props: [duplicatedCategories]
  }
}

/**
 * Select label by ID
 * @param {number} labelId
 */
export function selectLabel (
  labelId: number,
  category?: number,
  attributes?: {[key: number]: number[]},
  append: boolean = false
): types.ChangeSelectAction {
  const select = Session.getState().user.select
  const labels = (append) ? select.labels : []
  if (labelId >= 0 && !labels.includes(labelId)) {
    labels.push(labelId)
  }
  return changeSelect({ labels, category, attributes })
}
