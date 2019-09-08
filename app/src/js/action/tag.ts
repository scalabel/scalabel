import _ from 'lodash'
import Session from '../common/session'
import { LabelTypes } from '../common/types'
import { makeLabel } from '../functional/states'
import { updateObject } from '../functional/util'
import * as actions from './common'
import { AddLabelAction, ChangeLabelAction } from './types'

/**
 * If tag exists for attribute, updates the label, else create a new label for
 * the attribute. This is used for both image tagging and updating attributes
 * for labels
 * @param itemIndex
 * @param attributeIndex
 * @param selectedIndex
 */
export function addLabelTag (attributeIndex: number,
                             selectedIndex: number):
  AddLabelAction | ChangeLabelAction {
  const state = Session.getState()
  const itemIndex = state.user.select.item
  const attribute = { [attributeIndex]: [selectedIndex] }
  const item = state.task.items[itemIndex]
  if (_.size(item.labels) > 0) {
    const labelId = Number(_.findKey(item.labels))
    const newAttributes = updateObject(item.labels[labelId].attributes,
      attribute)
    return actions.changeLabelProps(itemIndex, labelId,
      { attributes: newAttributes })
  } else {
    const label = makeLabel({ type: LabelTypes.TAG, attributes: attribute })
    return actions.addLabel(itemIndex, label, [], [])
  }
}
