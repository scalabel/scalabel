import _ from "lodash"

import Session from "../common/session"
import { LabelTypeName } from "../const/common"
import { makeLabel } from "../functional/states"
import { updateObject } from "../functional/util"
import { AddLabelsAction, ChangeLabelsAction } from "../types/action"
import * as actions from "./common"

/**
 * If tag exists for attribute, updates the label, else create a new label for
 * the attribute.
 *
 * @param itemIndex
 * @param attributeIndex
 * @param selectedIndex
 */
export function addLabelTag(
  attributeIndex: number,
  selectedIndex: number
): AddLabelsAction | ChangeLabelsAction {
  const state = Session.getState()
  const itemIndex = state.user.select.item
  const attribute = { [attributeIndex]: [selectedIndex] }
  const item = state.task.items[itemIndex]
  const labelId = _.findKey(item.labels)
  if (labelId !== undefined) {
    const newAttributes = updateObject(
      item.labels[labelId].attributes,
      attribute
    )
    return actions.changeLabelProps(itemIndex, labelId, {
      attributes: newAttributes
    })
  } else {
    const label = makeLabel({
      type: LabelTypeName.TAG,
      attributes: attribute,
      sensors: Object.keys(state.task.sensors).map((key) => Number(key))
    })
    return actions.addLabel(itemIndex, label, [])
  }
}
