import _ from 'lodash'
import * as React from 'react'
import * as action from '../../js/action/common'
import Session from '../../js/common/session'
import { initStore } from '../../js/common/session_init'
import { Label2dCanvas } from '../../js/components/label2d_canvas'
import { getShape } from '../../js/functional/state_util'
import { IdType, RectType } from '../../js/functional/types'
import { testJson } from '../test_image_objects'
import { findNewLabelsFromState } from '../util'
import { drawBox2D, setUpLabel2dCanvas } from './label2d_canvas_util'

const canvasRef: React.RefObject<Label2dCanvas> = React.createRef()

beforeEach(() => {
  expect(canvasRef.current).not.toBeNull()
  canvasRef.current?.clear()
  initStore(testJson)
})

beforeAll(() => {
  Session.devMode = false
  Session.subscribe(() => Session.label2dList.updateState(Session.getState()))
  initStore(testJson)
  Session.images.length = 0
  Session.images.push({ [-1]: new Image(1000, 1000) })
  for (let i = 0; i < Session.getState().task.items.length; i++) {
    Session.dispatch(action.loadItem(i, -1))
  }
  setUpLabel2dCanvas(Session.dispatch.bind(Session), canvasRef, 1000, 1000)
  console.log(Session.getState().session.itemStatuses)
})

test('Draw 2d boxes to label2d list', () => {
  if (canvasRef.current) {
    const labelIds: IdType[] = []
    const label2d = canvasRef.current
    // Draw first box
    drawBox2D(label2d, 1, 1, 50, 50)
    let state = Session.getState()
    labelIds.push(findNewLabelsFromState(state, 0, labelIds)[0])
    expect(_.size(state.task.items[0].labels)).toEqual(1)
    console.log(state.task.items[0].labels)
    console.log(state.task.items[0].shapes)
    let rect = getShape(state, 0, labelIds[0], 0) as RectType
    expect(rect.x1).toEqual(1)
    expect(rect.y1).toEqual(1)
    expect(rect.x2).toEqual(50)
    expect(rect.y2).toEqual(50)

    // Second box
    drawBox2D(label2d, 25, 20, 70, 85)

    state = Session.getState()
    labelIds.push(findNewLabelsFromState(state, 0, labelIds)[0])
    expect(_.size(state.task.items[0].labels)).toEqual(2)
    rect = getShape(state, 0, labelIds[1], 0) as RectType
    expect(rect.x1).toEqual(25)
    expect(rect.y1).toEqual(20)
    expect(rect.x2).toEqual(70)
    expect(rect.y2).toEqual(85)

    // third box
    drawBox2D(label2d, 15, 10, 60, 70)
    state = Session.getState()
    labelIds.push(findNewLabelsFromState(state, 0, labelIds)[0])
    expect(_.size(state.task.items[0].labels)).toEqual(3)
    rect = getShape(state, 0, labelIds[2], 0) as RectType
    expect(rect.x1).toEqual(15)
    expect(rect.y1).toEqual(10)
    expect(rect.x2).toEqual(60)
    expect(rect.y2).toEqual(70)

    // resize the second box
    drawBox2D(label2d, 25, 20, 30, 34)
    state = Session.getState()
    expect(_.size(state.task.items[0].labels)).toEqual(3)
    rect = getShape(state, 0, labelIds[1], 0) as RectType
    expect(rect.x1).toEqual(30)
    expect(rect.y1).toEqual(34)

    // flip top left and bottom right corner
    drawBox2D(label2d, 30, 34, 90, 90)
    state = Session.getState()
    rect = getShape(state, 0, labelIds[1], 0) as RectType
    expect(rect.x1).toEqual(70)
    expect(rect.y1).toEqual(85)
    expect(rect.x2).toEqual(90)
    expect(rect.y2).toEqual(90)

    // move
    drawBox2D(label2d, 30, 10, 40, 15)
    state = Session.getState()
    rect = getShape(state, 0, labelIds[2], 0) as RectType
    expect(rect.x1).toEqual(25)
    expect(rect.y1).toEqual(15)
    expect(rect.x2).toEqual(70)
    expect(rect.y2).toEqual(75)
  }
})
