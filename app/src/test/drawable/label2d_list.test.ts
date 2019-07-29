import { createCanvas } from 'canvas'
import _ from 'lodash'
import * as action from '../../js/action/common'
import Session from '../../js/common/session'
import { initStore } from '../../js/common/session_init'
import { Label2DList } from '../../js/drawable/label2d_list'
import { getShape } from '../../js/functional/state_util'
import { RectType } from '../../js/functional/types'
import { Size2D } from '../../js/math/size2d'
import { Vector2D } from '../../js/math/vector2d'
import { testJson } from '../test_objects'

test('Draw 2d boxes to label2d list', () => {
  Session.devMode = false
  initStore(testJson)
  const itemIndex = 0
  Session.dispatch(action.goToItem(itemIndex))
  const label2dList = new Label2DList()
  Session.subscribe(() => {
    label2dList.updateState(Session.getState(), Session.getState().current.item)
  })

  // Draw first box
  const canvasSize = new Size2D(100, 100)
  label2dList.onMouseDown(new Vector2D(1, 1), -1, 0)
  label2dList.onMouseMove(new Vector2D(10, 10), canvasSize, -1, 0)
  label2dList.onMouseUp(new Vector2D(10, 10), -1, 0)
  let state = Session.getState()
  expect(_.size(state.items[0].labels)).toEqual(1)
  let rect = getShape(state, 0, 0, 0) as RectType
  expect(rect.x).toEqual(1)
  expect(rect.y).toEqual(1)
  expect(rect.w).toEqual(9)
  expect(rect.h).toEqual(9)

  // Second box
  label2dList.onMouseDown(new Vector2D(19, 20), -1, 0)
  label2dList.onMouseMove(new Vector2D(25, 25), canvasSize, -1, 0)
  label2dList.onMouseMove(new Vector2D(30, 29), canvasSize, -1, 0)
  label2dList.onMouseUp(new Vector2D(30, 29), -1, 0)

  state = Session.getState()
  expect(_.size(state.items[0].labels)).toEqual(2)
  rect = getShape(state, 0, 1, 0) as RectType
  expect(rect.x).toEqual(19)
  expect(rect.y).toEqual(20)
  expect(rect.w).toEqual(11)
  expect(rect.h).toEqual(9)

  // third box
  label2dList.onMouseDown(new Vector2D(4, 5), -1, 0)
  label2dList.onMouseMove(new Vector2D(15, 15), canvasSize, -1, 0)
  label2dList.onMouseMove(new Vector2D(23, 24), canvasSize, -1, 0)
  label2dList.onMouseUp(new Vector2D(23, 24), -1, 0)
  state = Session.getState()
  expect(_.size(state.items[0].labels)).toEqual(3)
  rect = getShape(state, 0, 2, 0) as RectType
  expect(rect.x).toEqual(4)
  expect(rect.y).toEqual(5)
  expect(rect.w).toEqual(19)
  expect(rect.h).toEqual(19)

  // resize the second box
  label2dList.onMouseDown(new Vector2D(19, 20), 1, 1)
  label2dList.onMouseMove(new Vector2D(15, 18), canvasSize, -1, 0)
  label2dList.onMouseMove(new Vector2D(16, 17), canvasSize, -1, 0)
  label2dList.onMouseUp(new Vector2D(16, 17), -1, 0)
  state = Session.getState()
  expect(_.size(state.items[0].labels)).toEqual(3)
  rect = getShape(state, 0, 1, 0) as RectType
  expect(rect.x).toEqual(16)
  expect(rect.y).toEqual(17)

  // flip top left and bottom right corner
  label2dList.onMouseDown(new Vector2D(16, 17), 1, 1)
  label2dList.onMouseMove(new Vector2D(42, 43), canvasSize, -1, 0)
  label2dList.onMouseUp(new Vector2D(40, 41), -1, 0)
  state = Session.getState()
  rect = getShape(state, 0, 1, 0) as RectType
  expect(rect.x).toEqual(30)
  expect(rect.y).toEqual(29)
  expect(rect.w).toEqual(12)
  expect(rect.h).toEqual(14)

  // move
  label2dList.onMouseDown(new Vector2D(32, 31), 1, 0)
  label2dList.onMouseMove(new Vector2D(36, 32), canvasSize, -1, 0)
  label2dList.onMouseUp(new Vector2D(36, 32), -1, 0)
  state = Session.getState()
  rect = getShape(state, 0, 1, 0) as RectType
  expect(rect.x).toEqual(34)
  expect(rect.y).toEqual(30)
  expect(rect.w).toEqual(12)
  expect(rect.h).toEqual(14)

  // delete label
  Session.dispatch(action.deleteLabel(0, 1))
  expect(label2dList.getLabelList().length).toEqual(2)
  expect(label2dList.getLabelList()[0].index).toEqual(0)
  expect(label2dList.getLabelList()[0].labelId).toEqual(0)
  expect(label2dList.getLabelList()[1].index).toEqual(1)
  expect(label2dList.getLabelList()[1].labelId).toEqual(2)
})

test('Draw label2d list to canvas', () => {
  const labelCanvas = createCanvas(200, 200)
  const labelContext = labelCanvas.getContext('2d')
  const controlCanvas = createCanvas(200, 200)
  const controlContext = controlCanvas.getContext('2d')

  Session.devMode = false
  initStore(testJson)
  const itemIndex = 0
  Session.dispatch(action.goToItem(itemIndex))
  const label2dList = new Label2DList()
  Session.subscribe(() => {
    label2dList.updateState(Session.getState(), Session.getState().current.item)
  })

  // Draw first box
  const canvasSize = new Size2D(100, 100)
  label2dList.onMouseDown(new Vector2D(1, 1), -1, 0)
  for (let i = 1; i <= 10; i += 1) {
    label2dList.onMouseMove(new Vector2D(i, i), canvasSize, -1, 0)
    label2dList.redraw(labelContext, controlContext, 1)
  }
  label2dList.onMouseUp(new Vector2D(10, 10), -1, 0)
  label2dList.redraw(labelContext, controlContext, 1)

  const state = Session.getState()
  expect(_.size(state.items[0].labels)).toEqual(1)
  const rect = getShape(state, 0, 0, 0) as RectType
  expect(rect.x).toEqual(1)
  expect(rect.y).toEqual(1)
  expect(rect.w).toEqual(9)
  expect(rect.h).toEqual(9)
})
