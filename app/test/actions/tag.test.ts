import _ from "lodash"

import { goToItem } from "../../src/action/common"
import * as tag from "../../src/action/tag"
import Session from "../../src/common/session"
import { setupTestStore } from "../components/util"
import { testJson } from "../test_states/test_image_objects"

test("Add and change tag for image label", () => {
  setupTestStore(testJson)

  const itemIndex = 0
  Session.dispatch(goToItem(itemIndex))
  const attributeIndices = [0]
  const selectedIndices = [0]
  const actualAttributes: { [key: number]: [number] } = {}
  actualAttributes[attributeIndices[0]] = [selectedIndices[0]]
  Session.dispatch(tag.addLabelTag(attributeIndices[0], selectedIndices[0]))

  let state = Session.getState()
  let itemLabels = state.task.items[itemIndex].labels
  let sample = _.sample(itemLabels)
  expect(_.size(itemLabels)).toBe(1)
  if (sample !== undefined) {
    expect(sample.attributes).toStrictEqual(actualAttributes)
  }

  // Change selected value for attribute 0
  selectedIndices[0] = 1
  actualAttributes[attributeIndices[0]] = [selectedIndices[0]]
  Session.dispatch(tag.addLabelTag(attributeIndices[0], selectedIndices[0]))
  state = Session.getState()
  itemLabels = state.task.items[itemIndex].labels
  expect(_.size(itemLabels)).toBe(1)
  sample = _.sample(itemLabels)
  if (sample !== undefined) {
    expect(sample.attributes).toStrictEqual(actualAttributes)
  }
  // Add a new attribute
  selectedIndices[1] = 1
  attributeIndices[1] = 3
  actualAttributes[attributeIndices[1]] = [selectedIndices[1]]
  Session.dispatch(tag.addLabelTag(attributeIndices[1], selectedIndices[1]))
  state = Session.getState()
  itemLabels = state.task.items[itemIndex].labels
  expect(_.size(itemLabels)).toBe(1)
  sample = _.sample(itemLabels)
  if (sample !== undefined) {
    expect(sample.attributes).toStrictEqual(actualAttributes)
  }
})
