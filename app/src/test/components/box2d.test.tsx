import _ from 'lodash'
import * as React from 'react'
import * as action from '../../js/action/common'
import Session from '../../js/common/session'
import { initStore } from '../../js/common/session_init'
import { Label2dCanvas } from '../../js/components/label2d_canvas'
import { getShape } from '../../js/functional/state_util'
import { RectType } from '../../js/functional/types'
import { testJson } from '../test_states/test_image_objects'
import { drag, drawBox2D, setUpLabel2dCanvas } from './label2d_canvas_util'

const canvasRef: React.RefObject<Label2dCanvas> = React.createRef()

const getState = Session.getState.bind(Session)
const dispatch = Session.dispatch.bind(Session)

beforeEach(() => {
  expect(canvasRef.current).not.toBeNull()
  canvasRef.current?.clear()
  initStore(testJson)
  Session.subscribe(() => {
    Session.label2dList.updateState(getState())
    canvasRef.current?.updateState(getState())
  })
})

beforeAll(() => {
  Session.devMode = false
  initStore(testJson)
  Session.images.length = 0
  Session.images.push({ [-1]: new Image(1000, 1000) })
  for (let i = 0; i < getState().task.items.length; i++) {
    dispatch(action.loadItem(i, -1))
  }
  setUpLabel2dCanvas(dispatch, canvasRef, 1000, 1000)
})

test('Draw 2d boxes to label2d list', () => {
  const labelIds = []
  // It has been checked that canvasRef.current is not null
  const label2d = canvasRef.current as Label2dCanvas
  // Draw first box
  labelIds.push(drawBox2D(label2d, getState, 1, 1, 50, 50))
  let state = getState()
  expect(_.size(state.task.items[0].labels)).toEqual(1)
  let rect = getShape(state, 0, labelIds[0], 0) as RectType
  expect(rect.x1).toEqual(1)
  expect(rect.y1).toEqual(1)
  expect(rect.x2).toEqual(50)
  expect(rect.y2).toEqual(50)

  // Second box
  labelIds.push(drawBox2D(label2d, getState, 25, 20, 70, 85))

  state = getState()
  expect(_.size(state.task.items[0].labels)).toEqual(2)
  rect = getShape(state, 0, labelIds[1], 0) as RectType
  expect(rect.x1).toEqual(25)
  expect(rect.y1).toEqual(20)
  expect(rect.x2).toEqual(70)
  expect(rect.y2).toEqual(85)

  // third box
  labelIds.push(drawBox2D(label2d, getState, 15, 10, 60, 70))
  state = getState()
  expect(_.size(state.task.items[0].labels)).toEqual(3)
  rect = getShape(state, 0, labelIds[2], 0) as RectType
  expect(rect.x1).toEqual(15)
  expect(rect.y1).toEqual(10)
  expect(rect.x2).toEqual(60)
  expect(rect.y2).toEqual(70)

  // resize the second box
  drag(label2d, 25, 20, 30, 34)
  state = getState()
  expect(_.size(state.task.items[0].labels)).toEqual(3)
  rect = getShape(state, 0, labelIds[1], 0) as RectType
  expect(rect.x1).toEqual(30)
  expect(rect.y1).toEqual(34)

  // move the resized second box
  drag(label2d, 30, 50, 40, 60)
  state = getState()
  rect = getShape(state, 0, labelIds[1], 0) as RectType
  expect(rect.x1).toEqual(40)
  expect(rect.y1).toEqual(44)

  // flip top left and bottom right corner
  drag(label2d, 40, 44, 100, 100)
  state = getState()
  rect = getShape(state, 0, labelIds[1], 0) as RectType
  expect(rect.x1).toEqual(80)
  expect(rect.y1).toEqual(95)
  expect(rect.x2).toEqual(100)
  expect(rect.y2).toEqual(100)

  // move the third box
  drag(label2d, 30, 10, 40, 15)
  state = getState()
  rect = getShape(state, 0, labelIds[2], 0) as RectType
  expect(rect.x1).toEqual(25)
  expect(rect.y1).toEqual(15)
  expect(rect.x2).toEqual(70)
  expect(rect.y2).toEqual(75)
})

// test('Switch box order', () => {
//   const label2d = canvasRef.current as Label2dCanvas
//   drawBox2D(label2d, getState, 1, 1, 50, 50)
//   // TODO: add the test
// })
