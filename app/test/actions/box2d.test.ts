import _ from "lodash"

import * as box2d from "../../src/action/box2d"
import * as action from "../../src/action/common"
import Session from "../../src/common/session"
import { LabelTypeName, ShapeTypeName } from "../../src/const/common"
import { RectType } from "../../src/types/state"
import { setupTestStore } from "../components/util"
import { testJson } from "../test_states/test_image_objects"

test("Add, change and delete box2d labels", () => {
  setupTestStore(testJson)

  // Parameters for item index and number of labels to add
  const itemIndex = 0
  const numLabels = 3

  // Add the labels to the item
  Session.dispatch(action.goToItem(itemIndex))
  for (let i = 0; i < numLabels; i++) {
    Session.dispatch(
      box2d.addBox2dLabel(
        itemIndex,
        -1,
        [0],
        {},
        { x1: 1, y1: 2, x2: 3, y2: 4 }
      )
    )
  }

  // Check the labels were added
  let item = Session.getState().task.items[itemIndex]
  expect(_.size(item.labels)).toBe(numLabels)
  expect(_.size(item.shapes)).toBe(numLabels)

  // Check a random label/shape for the correct properties
  const labelIds = _.map(item.labels, (l) => l.id)
  let label = item.labels[labelIds[0]]
  expect(label.item).toBe(itemIndex)
  expect(label.type).toBe(LabelTypeName.BOX_2D)

  const indexedShape = item.shapes[label.shapes[0]]
  expect(indexedShape.shapeType).toBe(ShapeTypeName.RECT)
  expect(indexedShape.label.length).toBe(1)
  expect(indexedShape.label[0]).toBe(label.id)

  let shape = indexedShape as RectType
  expect(shape).toMatchObject({ x1: 1, y1: 2, x2: 3, y2: 4 })

  // Check label ids
  _.forEach(item.labels, (v, i) => {
    expect(v.id).toBe(i)
  })
  // Check shape ids
  _.forEach(item.shapes, (v, i) => {
    expect(v.id).toBe(i)
  })

  Session.dispatch(
    box2d.changeBox2d(itemIndex, indexedShape.id, { x1: 2, x2: 7 })
  )
  item = Session.getState().task.items[itemIndex]
  label = item.labels[label.id]
  shape = item.shapes[label.shapes[0]] as RectType
  expect(shape).toMatchObject({ x1: 2, y1: 2, x2: 7, y2: 4 })

  Session.dispatch(action.deleteLabel(itemIndex, label.id))
  item = Session.getState().task.items[itemIndex]
  expect(_.size(item.labels)).toBe(numLabels - 1)
  expect(_.size(item.shapes)).toBe(numLabels - 1)
})
