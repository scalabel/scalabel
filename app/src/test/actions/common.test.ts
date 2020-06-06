import _ from 'lodash'
import * as action from '../../js/action/common'
import Session from '../../js/common/session'
import { makeLabel } from '../../js/functional/states'
import { LabelType } from '../../js/functional/types'
import { setupTestStore } from '../components/util'
import { testJson } from '../test_states/test_image_objects'

test('Add and delete labels', () => {
  setupTestStore(testJson)

  const itemIndex = 0
  Session.dispatch(action.goToItem(itemIndex))
  Session.dispatch(
    action.addLabel(itemIndex, makeLabel()))
  const manualLabel = makeLabel()
  Session.dispatch(
    action.addLabel(itemIndex, manualLabel))
  let autoLabel = makeLabel({ item: itemIndex, manual: false })
  Session.dispatch(action.addLabel(itemIndex, autoLabel))
  let state = Session.getState()

  // check setting of manual and order
  const labels = state.task.items[itemIndex].labels
  const label1 = labels[manualLabel.id]
  expect(label1.manual).toBe(true)
  expect(label1.item).toBe(0)
  autoLabel = labels[autoLabel.id]
  expect(autoLabel.item).toBe(0)
  expect(autoLabel.manual).toBe(false)
  expect(label1.order < autoLabel.order).toBe(true)
  expect(_.size(labels)).toBe(3)

  // check deleting label
  Session.dispatch(action.deleteLabel(itemIndex, autoLabel.id))
  state = Session.getState()
  expect(_.size(state.task.items[itemIndex].labels)).toBe(2)
})

test('Change category', () => {
  setupTestStore(testJson)

  const itemIndex = 0
  Session.dispatch(action.goToItem(itemIndex))
  const label = makeLabel()
  Session.dispatch(action.addLabel(itemIndex, label))
  Session.dispatch(
    action.changeLabelProps(itemIndex, label.id, { category: [2] }))
  const state = Session.getState()
  expect(state.task.items[itemIndex].labels[label.id].category[0]).toBe(2)
})

test('Link labels', () => {
  setupTestStore(testJson)

  const itemIndex = 0
  Session.dispatch(action.goToItem(itemIndex))
  Session.dispatch(action.addLabel(itemIndex, makeLabel()))
  const label1 = makeLabel()
  Session.dispatch(action.addLabel(itemIndex, label1))
  Session.dispatch(action.addLabel(itemIndex, makeLabel()))
  const label4 = makeLabel()
  Session.dispatch(action.addLabel(itemIndex, label4))
  const label2 = makeLabel()
  Session.dispatch(action.addLabel(itemIndex, label2))
  Session.dispatch(action.addLabel(itemIndex, makeLabel()))
  const label3 = makeLabel()
  Session.dispatch(action.addLabel(itemIndex, label3))
  Session.dispatch(action.addLabel(itemIndex, makeLabel()))

  // Link multiple labels
  let children = [label1.id, label2.id, label3.id]
  Session.dispatch(action.linkLabels(itemIndex, children))
  let state = Session.getState()
  let item = state.task.items[itemIndex]
  const parent1Maybe = _.find(item.labels, (label) => label.children.length > 0)
  expect(parent1Maybe).not.toBe(undefined)
  const parent1 = parent1Maybe as LabelType
  expect(item.labels[parent1.id].children).toEqual(children)
  for (const label of children) {
    expect(item.labels[label].parent).toEqual(parent1.id)
  }

  // test recursive linking
  children = [label1.id, label4.id]
  Session.dispatch(action.linkLabels(itemIndex, children))
  state = Session.getState()

  item = state.task.items[itemIndex]
  const parent2Maybe = _.find(item.labels,
    (label) => label.children.length > 0 && label.id !== parent1.id)
  expect(parent2Maybe).not.toBe(undefined)
  const parent2 = parent2Maybe as LabelType
  children = [parent1.id, label4.id]
  expect(item.labels[parent2.id].children).toEqual(children)
  for (const label of children) {
    expect(item.labels[label].parent).toEqual(parent2.id)
  }
  for (const label of [label1, label2, label3]) {
    expect(item.labels[label.id].parent).toEqual(parent1.id)
  }
})

test('Submit task', () => {
  const constantDate = Date.now()
  Date.now = jest.fn(() => {
    return constantDate
  })
  setupTestStore(testJson)

  // first submission
  Session.dispatch(action.submit())
  let state = Session.getState()
  let submissions = state.task.progress.submissions
  expect(submissions.length).toBe(1)
  expect(submissions[0].time).toBe(constantDate)
  expect(submissions[0].user).toBe(state.user.id)

  // second submission
  Session.dispatch(action.submit())
  state = Session.getState()
  submissions = state.task.progress.submissions
  expect(submissions.length).toBe(2)
  expect(submissions[1].time).toBe(constantDate)
  expect(submissions[1].user).toBe(state.user.id)
})
