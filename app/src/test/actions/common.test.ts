import _ from 'lodash'
import * as action from '../../js/action/common'
import Session from '../../js/common/session'
import { initStore } from '../../js/common/session_init'
import { makeLabel } from '../../js/functional/states'
import { testJson } from '../test_objects'

test('Add and delete labels', () => {
  Session.devMode = false
  initStore(testJson)
  const itemIndex = 0
  Session.dispatch(action.goToItem(itemIndex))
  const label = makeLabel({ item: itemIndex })
  Session.dispatch(action.addLabel(itemIndex, label, []))
  Session.dispatch(action.addLabel(itemIndex, label, []))
  let state = Session.getState()
  const labelId =
    state.task.status.maxLabelId
  expect(_.size(state.task.items[0].labels)).toBe(2)
  expect(state.task.items[0].labels[labelId].item).toBe(0)
  Session.dispatch(action.deleteLabel(itemIndex, labelId))
  state = Session.getState()
  expect(_.size(state.task.items[0].labels)).toBe(1)
})

test('Change category', () => {
  Session.devMode = false
  initStore(testJson)
  const itemIndex = 0
  Session.dispatch(action.goToItem(itemIndex))
  Session.dispatch(action.addLabel(itemIndex, makeLabel(), []))
  Session.dispatch(action.changeLabelProps(itemIndex, 0, { category: [2] }))
  const state = Session.getState()
  expect(state.task.items[0].labels[0].category[0]).toBe(2)
})
