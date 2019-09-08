import _ from 'lodash'
import * as action from '../../js/action/common'
import Session from '../../js/common/session'
import { initStore } from '../../js/common/session_init'
import { makeLabel } from '../../js/functional/states'
import { testJson } from '../test_image_objects'

test('Add and delete labels', () => {
  Session.devMode = false
  initStore(testJson)
  const itemIndex = 0
  Session.dispatch(action.goToItem(itemIndex))
  let label = makeLabel({ item: itemIndex })
  Session.dispatch(action.addLabel(itemIndex, label, [], []))
  Session.dispatch(action.addLabel(itemIndex, label, [], []))
  label = makeLabel({ item: itemIndex, manual: false })
  Session.dispatch(action.addLabel(itemIndex, label, [], []))
  let state = Session.getState()
  const labelId =
    state.task.status.maxLabelId

  // check setting of manual and order
  const label1 = state.task.items[0].labels[labelId - 1]
  expect(label1.manual).toBe(true)
  label = state.task.items[0].labels[labelId]
  expect(label.item).toBe(0)
  expect(label.manual).toBe(false)
  expect(label1.order < label.order).toBe(true)

  // check deleting label
  expect(_.size(state.task.items[0].labels)).toBe(3)
  Session.dispatch(action.deleteLabel(itemIndex, labelId))
  state = Session.getState()
  expect(_.size(state.task.items[0].labels)).toBe(2)
})

test('Change category', () => {
  Session.devMode = false
  initStore(testJson)
  const itemIndex = 0
  Session.dispatch(action.goToItem(itemIndex))
  Session.dispatch(action.addLabel(itemIndex, makeLabel(), [], []))
  Session.dispatch(action.changeLabelProps(itemIndex, 0, { category: [2] }))
  const state = Session.getState()
  expect(state.task.items[0].labels[0].category[0]).toBe(2)
})

test('Link labels', () => {
  Session.devMode = false
  initStore(testJson)
  const itemIndex = 0
  Session.dispatch(action.goToItem(itemIndex))
  Session.dispatch(action.addLabel(itemIndex, makeLabel(), [], []))
  const label1 = Session.getState().task.status.maxLabelId
  Session.dispatch(action.addLabel(itemIndex, makeLabel(), [], []))
  Session.dispatch(action.addLabel(itemIndex, makeLabel(), [], []))
  const label4 = Session.getState().task.status.maxLabelId
  Session.dispatch(action.addLabel(itemIndex, makeLabel(), [], []))
  const label2 = Session.getState().task.status.maxLabelId
  Session.dispatch(action.addLabel(itemIndex, makeLabel(), [], []))
  Session.dispatch(action.addLabel(itemIndex, makeLabel(), [], []))
  const label3 = Session.getState().task.status.maxLabelId
  Session.dispatch(action.addLabel(itemIndex, makeLabel(), [], []))
  Session.dispatch(action.addLabel(itemIndex, makeLabel(), [], []))

  // Link multiple labels
  let children = [label1, label2, label3]
  Session.dispatch(action.linkLabels(itemIndex, children))
  const parent1 = Session.getState().task.status.maxLabelId
  let state = Session.getState()
  let item = state.task.items[itemIndex]
  expect(item.labels[parent1].children).toEqual(children)
  for (const label of children) {
    expect(item.labels[label].parent).toEqual(parent1)
  }

  // test recursive linking
  children = [label1, label4]
  Session.dispatch(action.linkLabels(itemIndex, children))
  state = Session.getState()
  const parent2 = Session.getState().task.status.maxLabelId
  item = state.task.items[itemIndex]
  children = [parent1, label4]
  expect(item.labels[parent2].children).toEqual(children)
  for (const label of children) {
    expect(item.labels[label].parent).toEqual(parent2)
  }
  for (const label of [label1, label2, label3]) {
    expect(item.labels[label].parent).toEqual(parent1)
  }
})
