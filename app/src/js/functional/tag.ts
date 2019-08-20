import _ from 'lodash'
import * as types from '../action/types'
import { addLabel, changeLabel } from './common'
import { getCurrentItem } from './state_util'
import { makeLabel } from './states'
import { LabelType, State } from './types'

/**
 * Create a Tag label
 * @param {number} labelId
 * @param {number} itemId
 * @param {object} optionalAttributes
 * @return {LabelType}
 */
export function createTagLabel (
    labelId: number, itemId: number,
    optionalAttributes: { [key: number]: number[] }): LabelType {
  return makeLabel({ id: labelId, item: itemId,
    attributes: optionalAttributes })
}

/**
 * Image tagging
 * @param {State} state
 * @param {types.TagImageAction} action
 * @return {State}
 */
export function tagImage (
    state: State, action: types.TagImageAction): State {
  const [attributeIndex, attributeValue] =
    [action.attributeIndex, action.selectedIndex]
  const attributes = { [attributeIndex]: attributeValue }
  const item = getCurrentItem(state)
  if (_.size(item.labels) > 0) {
    const labelId = Number(_.findKey(item.labels))
    const newAction: types.ChangeLabelAction = {
      type: types.CHANGE_LABEL_PROPS,
      sessionId: action.sessionId,
      itemIndex: action.itemIndex,
      labelId,
      props: { attributes }
    }
    return changeLabel(state, newAction)
  } else {
    const label = createTagLabel(0, state.user.select.item, attributes)
    const newAction: types.AddLabelAction = {
      type: types.ADD_LABEL,
      sessionId: action.sessionId,
      itemIndex: action.itemIndex,
      label,
      shapes: []
    }
    return addLabel(state, newAction)
  }
}
