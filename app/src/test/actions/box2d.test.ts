import _ from 'lodash'
import * as box2d from '../../js/action/box2d'
import * as action from '../../js/action/common'
import * as labels from '../../js/common/label_types'
import Session from '../../js/common/session'
import { initStore } from '../../js/common/session_init'
import { RectType } from '../../js/functional/types'
import { testJson } from '../test_objects'

test('Add, change and delete box2d labels', () => {
  Session.devMode = false
  initStore(testJson)
  const itemIndex = 0
  Session.dispatch(action.goToItem(itemIndex))
  Session.dispatch(box2d.addBox2dLabel(itemIndex, [0], 1, 2, 3, 4))
  Session.dispatch(box2d.addBox2dLabel(itemIndex, [0], 1, 2, 3, 4))
  Session.dispatch(box2d.addBox2dLabel(itemIndex, [0], 1, 2, 3, 4))
  let state = Session.getState()
  expect(_.size(state.items[0].labels)).toBe(3)
  expect(_.size(state.items[0].shapes)).toBe(3)
  const labelIds: number[] = _.map(state.items[0].labels, (l) => l.id)
  let label = state.items[0].labels[labelIds[0]]
  expect(label.item).toBe(0)
  expect(label.type).toBe(labels.BOX_2D)
  let shape = state.items[0].shapes[label.shapes[0]] as RectType
  // Check label ids
  let index = 0
  _.forEach(state.items[0].labels, (v, i) => {
    expect(v.id).toBe(Number(i))
    expect(v.id).toBe(index)
    index += 1
  })
  // Check shape ids
  index = 0
  _.forEach(state.items[0].shapes, (v, i) => {
    expect(v.id).toBe(Number(i))
    expect(v.id).toBe(index)
    index += 1
  })
  expect(shape.x).toBe(1)
  expect(shape.y).toBe(2)
  expect(shape.w).toBe(3)
  expect(shape.h).toBe(4)

  Session.dispatch(action.changeLabelShape(itemIndex, shape.id, { x: 2, w: 5 }))
  state = Session.getState()
  label = state.items[0].labels[label.id]
  shape = state.items[0].shapes[label.shapes[0]] as RectType
  // console.log(label, shape, state.items[0].shapes);
  expect(shape.x).toBe(2)
  expect(shape.y).toBe(2)
  expect(shape.w).toBe(5)
  expect(shape.h).toBe(4)

  Session.dispatch(action.deleteLabel(itemIndex, label.id))
  state = Session.getState()
  expect(_.size(state.items[itemIndex].labels)).toBe(2)
  expect(_.size(state.items[itemIndex].shapes)).toBe(2)
})
