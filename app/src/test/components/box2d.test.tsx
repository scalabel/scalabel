import _ from 'lodash'
import * as React from 'react'
import * as action from '../../js/action/common'
import Session from '../../js/common/session'
import { initStore } from '../../js/common/session_init'
import { Label2dCanvas } from '../../js/components/label2d_canvas'
import { getShape } from '../../js/functional/state_util'
import { RectType } from '../../js/functional/types'
import { testJson } from '../test_image_objects'
import { LabelCollector } from '../util/label_collector'
import { drawBox2D, mouseDown, mouseMove, mouseUp, setUpLabel2dCanvas } from './label2d_canvas_util'

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
  const labelIds = new LabelCollector(getState)
  // It has been checked that canvasRef.current is not null
  const label2d = canvasRef.current as Label2dCanvas
  // Draw first box
  drawBox2D(label2d, 1, 1, 50, 50)
  let state = getState()
  labelIds.collect()
  expect(_.size(state.task.items[0].labels)).toEqual(1)
  let rect = getShape(state, 0, labelIds[0], 0) as RectType
  expect(rect.x1).toEqual(1)
  expect(rect.y1).toEqual(1)
  expect(rect.x2).toEqual(50)
  expect(rect.y2).toEqual(50)

  // Second box
  drawBox2D(label2d, 25, 20, 70, 85)

  state = getState()
  labelIds.collect()
  expect(_.size(state.task.items[0].labels)).toEqual(2)
  rect = getShape(state, 0, labelIds[1], 0) as RectType
  expect(rect.x1).toEqual(25)
  expect(rect.y1).toEqual(20)
  expect(rect.x2).toEqual(70)
  expect(rect.y2).toEqual(85)

  // third box
  drawBox2D(label2d, 15, 10, 60, 70)
  state = getState()
  labelIds.collect()
  expect(_.size(state.task.items[0].labels)).toEqual(3)
  rect = getShape(state, 0, labelIds[2], 0) as RectType
  expect(rect.x1).toEqual(15)
  expect(rect.y1).toEqual(10)
  expect(rect.x2).toEqual(60)
  expect(rect.y2).toEqual(70)

  // resize the second box
  mouseMove(label2d, 25, 20)
  mouseDown(label2d, 25, 20)
  mouseMove(label2d, 15, 18)
  mouseMove(label2d, 30, 34)
  mouseUp(label2d, 30, 34)
  state = getState()
  expect(_.size(state.task.items[0].labels)).toEqual(3)
  rect = getShape(state, 0, labelIds[1], 0) as RectType
  expect(rect.x1).toEqual(30)
  expect(rect.y1).toEqual(34)

  // flip top left and bottom right corner
  mouseMove(label2d, 30, 34)
  mouseDown(label2d, 30, 34)
  mouseMove(label2d, 90, 90)
  mouseUp(label2d, 90, 90)
  state = getState()
  rect = getShape(state, 0, labelIds[1], 0) as RectType
  expect(rect.x1).toEqual(70)
  expect(rect.y1).toEqual(85)

  expect(rect.x2).toEqual(90)
  expect(rect.y2).toEqual(90)

  // move
  mouseMove(label2d, 30, 10)
  mouseDown(label2d, 30, 10)
  mouseMove(label2d, 40, 15)
  mouseUp(label2d, 40, 15)
  state = getState()
  rect = getShape(state, 0, labelIds[2], 0) as RectType
  expect(rect.x1).toEqual(25)
  expect(rect.y1).toEqual(15)
  expect(rect.x2).toEqual(70)
  expect(rect.y2).toEqual(75)
})

test('Switch box order', () => {
  const label2d = canvasRef.current as Label2dCanvas
  drawBox2D(label2d, 1, 1, 50, 50)
  // TODO: add the test
})
