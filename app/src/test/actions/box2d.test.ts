import _ from 'lodash'
import * as box2d from '../../js/action/box2d'
import * as action from '../../js/action/common'
import Session from '../../js/common/session'
import { initStore } from '../../js/common/session_init'
import { LabelTypeName, ShapeTypeName } from '../../js/common/types'
import { RectType } from '../../js/functional/types'
import { testJson } from '../test_states/test_image_objects'

test('Add, change and delete box2d labels', () => {
  Session.devMode = false
  initStore(testJson)
  const itemIndex = 0
  Session.dispatch(action.goToItem(itemIndex))
  Session.dispatch(box2d.addBox2dLabel(itemIndex, -1, [0], {}, 1, 2, 3, 4))
  Session.dispatch(box2d.addBox2dLabel(itemIndex, -1, [0], {}, 1, 2, 3, 4))
  Session.dispatch(box2d.addBox2dLabel(itemIndex, -1, [0], {}, 1, 2, 3, 4))
  let state = Session.getState()
  expect(_.size(state.task.items[0].labels)).toBe(3)
  expect(_.size(state.task.items[0].shapes)).toBe(3)
  const labelIds = _.map(state.task.items[0].labels, (l) => l.id)
  let label = state.task.items[0].labels[labelIds[0]]
  expect(label.item).toBe(0)
  expect(label.type).toBe(LabelTypeName.BOX_2D)
  const indexedShape = state.task.items[0].shapes[label.shapes[0]]
  expect(indexedShape.shapeType).toBe(ShapeTypeName.RECT)
  expect(indexedShape.label.length).toBe(1)
  expect(indexedShape.label[0]).toBe(label.id)
  let shape = indexedShape as RectType
  // Check label ids
  _.forEach(state.task.items[0].labels, (v, i) => {
    expect(v.id).toBe(i)
  })
  // Check shape ids
  _.forEach(state.task.items[0].shapes, (v, i) => {
    expect(v.id).toBe(i)
  })
  expect(shape.x1).toBe(1)
  expect(shape.y1).toBe(2)
  expect(shape.x2).toBe(3)
  expect(shape.y2).toBe(4)

  Session.dispatch(
    box2d.changeBox2d(itemIndex, indexedShape.id, { x1: 2, x2: 7 }))
  state = Session.getState()
  label = state.task.items[0].labels[label.id]
  shape = state.task.items[0].shapes[label.shapes[0]] as RectType
  expect(shape.x1).toBe(2)
  expect(shape.y1).toBe(2)
  expect(shape.x2).toBe(7)
  expect(shape.y2).toBe(4)

  Session.dispatch(action.deleteLabel(itemIndex, label.id))
  state = Session.getState()
  expect(_.size(state.task.items[itemIndex].labels)).toBe(2)
  expect(_.size(state.task.items[itemIndex].shapes)).toBe(2)
})
