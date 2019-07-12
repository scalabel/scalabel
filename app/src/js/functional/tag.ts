import _ from 'lodash'
import { addLabel, changeLabelProps } from './common'
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
 * @param {number} attributeIndex
 * @param {number} attributeValue
 * @return {State}
 */
export function tagImage (
    state: State, attributeIndex: number,
    attributeValue: number[]): State {
  const attributes = { [attributeIndex]: attributeValue }
  const item = state.items[state.current.item]
  if (_.size(item.labels) > 0) {
    const labelId = parseInt(_.findKey(item.labels) as string, 10)
    return changeLabelProps(state, item.index, labelId, { attributes })
  }
  const label = createTagLabel(0, state.current.item, attributes)
  return addLabel(state, item.index, label, [])
}
