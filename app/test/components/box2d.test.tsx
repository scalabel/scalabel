import _ from "lodash"
import * as React from "react"

import * as action from "../../src/action/common"
import Session, { dispatch, getState } from "../../src/common/session"
import { Label2dCanvas } from "../../src/components/label2d_canvas"
import { testJson } from "../test_states/test_image_objects"
import { checkBox2D } from "../util/shape"
import { drag, drawBox2D, setUpLabel2dCanvas } from "./canvas_util"
import { setupTestStore } from "./util"

const canvasRef: React.RefObject<Label2dCanvas> = React.createRef()

beforeEach(() => {
  expect(canvasRef.current).not.toBeNull()
  canvasRef.current?.clear()
  setupTestStore(testJson)
  Session.subscribe(() => {
    Session.label2dList.updateState(getState())
    canvasRef.current?.updateState(getState())
  })
})

beforeAll(() => {
  setupTestStore(testJson)
  Session.images.length = 0
  Session.images.push({ [-1]: new Image(1000, 1000) })
  for (let i = 0; i < getState().task.items.length; i++) {
    dispatch(action.loadItem(i, -1))
  }
  setUpLabel2dCanvas(dispatch, canvasRef, 1000, 1000)
})

test("Draw 2d boxes to label2d list", () => {
  const labelIds = []
  // It has been checked that canvasRef.current is not null
  const label2d = canvasRef.current as Label2dCanvas
  // Draw first box
  labelIds.push(drawBox2D(label2d, getState, 1, 1, 50, 50))
  let state = getState()
  expect(_.size(state.task.items[0].labels)).toEqual(1)
  checkBox2D(labelIds[0], { x1: 1, y1: 1, x2: 50, y2: 50 })

  // Second box
  labelIds.push(drawBox2D(label2d, getState, 25, 20, 70, 85))

  state = getState()
  expect(_.size(state.task.items[0].labels)).toEqual(2)
  checkBox2D(labelIds[1], { x1: 25, y1: 20, x2: 70, y2: 85 })

  // Third box
  labelIds.push(drawBox2D(label2d, getState, 15, 10, 60, 70))
  state = getState()
  expect(_.size(state.task.items[0].labels)).toEqual(3)
  checkBox2D(labelIds[2], { x1: 15, y1: 10, x2: 60, y2: 70 })

  // Resize the second box
  drag(label2d, 25, 20, 30, 34)
  state = getState()
  expect(_.size(state.task.items[0].labels)).toEqual(3)
  checkBox2D(labelIds[1], { x1: 30, y1: 34 })

  // Move the resized second box
  drag(label2d, 30, 50, 40, 60)
  checkBox2D(labelIds[1], { x1: 40, y1: 44 })

  // Flip top left and bottom right corner
  drag(label2d, 40, 44, 100, 100)
  checkBox2D(labelIds[1], { x1: 80, y1: 95, x2: 100, y2: 100 })

  // Move the third box
  drag(label2d, 30, 10, 40, 15)
  checkBox2D(labelIds[2], { x1: 25, y1: 15, x2: 70, y2: 75 })
})

// Test('Switch box order', () => {
//   const label2d = canvasRef.current as Label2dCanvas
//   drawBox2D(label2d, getState, 1, 1, 50, 50)
//   // TODO: add the test
// })
