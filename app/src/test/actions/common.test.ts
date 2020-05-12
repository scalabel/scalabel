import _ from 'lodash'
import * as action from '../../js/action/common'
import Session from '../../js/common/session'
import { initStore } from '../../js/common/session_init'
import { makeLabel } from '../../js/functional/states'
import { LabelType } from '../../js/functional/types'
import { makeRandomRect } from '../server/util/util'
import { testJson } from '../test_states/test_image_objects'

test('Add and delete labels and their shapes', () => {
  Session.devMode = false
  initStore(testJson)
  const itemIndex = 0
  Session.dispatch(action.goToItem(itemIndex))
  Session.dispatch(
    action.addLabel(itemIndex, makeLabel(), [makeRandomRect()]))
  const manualLabel = makeLabel()
  Session.dispatch(
    action.addLabel(itemIndex, manualLabel, [makeRandomRect()]))
  let autoLabel = makeLabel({ item: itemIndex, manual: false })
  Session.dispatch(
    action.addLabel(itemIndex, autoLabel, [makeRandomRect()]))
  let state = Session.getState()

  // check setting of manual
  const label1 = state.task.items[0].labels[manualLabel.id]
  expect(label1.manual).toBe(true)
  expect(label1.item).toBe(0)
  autoLabel = state.task.items[0].labels[autoLabel.id]
  expect(autoLabel.item).toBe(0)
  expect(autoLabel.manual).toBe(false)

  // check deleting label
  expect(_.size(state.task.items[0].labels)).toBe(3)
  expect(_.size(state.task.items[0].shapes)).toBe(3)
  Session.dispatch(action.deleteLabel(itemIndex, autoLabel.id))
  state = Session.getState()
  expect(_.size(state.task.items[0].labels)).toBe(2)
  expect(_.size(state.task.items[0].shapes)).toBe(2)
})

test('Change category', () => {
  Session.devMode = false
  initStore(testJson)
  const itemIndex = 0
  Session.dispatch(action.goToItem(itemIndex))
  const label = makeLabel()
  Session.dispatch(action.addLabel(itemIndex, label))
  Session.dispatch(
    action.changeLabelProps(itemIndex, label.id, { category: [2] }))
  const state = Session.getState()
  expect(state.task.items[0].labels[label.id].category[0]).toBe(2)
})

test('Link labels', () => {
  Session.devMode = false
  initStore(testJson)
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
  Session.devMode = false
  initStore(testJson)
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
