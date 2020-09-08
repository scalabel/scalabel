import { createCanvas } from "canvas"
import _ from "lodash"

import { addPolygon2dLabel } from "../../src/action/polygon2d"
import { selectLabel } from "../../src/action/select"
import Session, { dispatch, getState } from "../../src/common/session"
import { Key } from "../../src/const/common"
import { getShape } from "../../src/functional/state_util"
import { makeSimplePathPoint2D } from "../../src/functional/states"
import { Size2D } from "../../src/math/size2d"
import { IdType, PathPointType } from "../../src/types/state"
import { findNewLabels } from "../util/state"
import {
  initializeTestingObjects,
  keyClick,
  mouseDown,
  mouseMove,
  mouseUp
} from "./util"

test("Draw label2d list to canvas", () => {
  const labelCanvas = createCanvas(200, 200)
  const labelContext = labelCanvas.getContext("2d")
  const controlCanvas = createCanvas(200, 200)
  const controlContext = controlCanvas.getContext("2d")
  const labelIds: IdType[] = []

  const [label2dHandler] = initializeTestingObjects()

  // Draw first box
  const canvasSize = new Size2D(100, 100)
  mouseDown(label2dHandler, 1, 1, -1, 0)
  for (let i = 1; i <= 10; i += 1) {
    mouseMove(label2dHandler, i, i, canvasSize, -1, 0)
    Session.label2dList.redraw(labelContext, controlContext, 1)
  }
  mouseUp(label2dHandler, 10, 10, -1, 0)
  Session.label2dList.redraw(labelContext, controlContext, 1)

  const state = getState()
  expect(_.size(state.task.items[0].labels)).toEqual(1)
  labelIds.push(findNewLabels(getState().task.items[0].labels, labelIds)[0])
  const rect = getShape(state, 0, labelIds[0], 0)
  expect(rect).toMatchObject({ x1: 1, y1: 1, x2: 10, y2: 10 })
})

test("Change label ordering", () => {
  const [label2dHandler] = initializeTestingObjects()
  dispatch(
    addPolygon2dLabel(
      0,
      -1,
      [0],
      [
        makeSimplePathPoint2D(0, 1, PathPointType.LINE),
        makeSimplePathPoint2D(1, 1, PathPointType.LINE),
        makeSimplePathPoint2D(1, 2, PathPointType.CURVE),
        makeSimplePathPoint2D(0, 2, PathPointType.CURVE)
      ],
      true
    )
  )
  dispatch(
    addPolygon2dLabel(
      0,
      -1,
      [0],
      [
        makeSimplePathPoint2D(3, 4, PathPointType.LINE),
        makeSimplePathPoint2D(4, 4, PathPointType.LINE),
        makeSimplePathPoint2D(4, 5, PathPointType.CURVE),
        makeSimplePathPoint2D(3, 5, PathPointType.CURVE)
      ],
      false
    )
  )
  dispatch(
    addPolygon2dLabel(
      0,
      -1,
      [0],
      [
        makeSimplePathPoint2D(10, 11, PathPointType.LINE),
        makeSimplePathPoint2D(11, 11, PathPointType.LINE),
        makeSimplePathPoint2D(11, 12, PathPointType.CURVE),
        makeSimplePathPoint2D(10, 12, PathPointType.CURVE)
      ],
      true
    )
  )

  let state = getState()
  let labels = state.task.items[0].labels
  const labelIds = Object.keys(labels)
  expect(labelIds.length).toEqual(3)
  expect(labels[labelIds[0]].order).toEqual(0)
  expect(labels[labelIds[1]].order).toEqual(1)
  expect(labels[labelIds[2]].order).toEqual(2)

  // Move last label back
  dispatch(selectLabel(state.user.select.labels, 0, labelIds[2]))

  keyClick(label2dHandler, Key.ARROW_DOWN)

  state = getState()
  labels = state.task.items[0].labels
  expect(labels[labelIds[0]].order).toEqual(0)
  expect(labels[labelIds[1]].order).toEqual(2)
  expect(labels[labelIds[2]].order).toEqual(1)

  // Move first label forward
  dispatch(selectLabel(state.user.select.labels, 0, labelIds[0]))
  keyClick(label2dHandler, Key.ARROW_UP)

  state = getState()
  labels = state.task.items[0].labels
  expect(labels[labelIds[0]].order).toEqual(1)
  expect(labels[labelIds[1]].order).toEqual(2)
  expect(labels[labelIds[2]].order).toEqual(0)

  // Move label in front to back
  dispatch(selectLabel(state.user.select.labels, 0, labelIds[1]))
  keyClick(label2dHandler, Key.B_LOW)

  state = getState()
  labels = state.task.items[0].labels
  expect(labels[labelIds[0]].order).toEqual(2)
  expect(labels[labelIds[1]].order).toEqual(0)
  expect(labels[labelIds[2]].order).toEqual(1)

  // Move label in back to front
  dispatch(selectLabel(state.user.select.labels, 0, labelIds[1]))
  keyClick(label2dHandler, Key.F_LOW)

  state = getState()
  labels = state.task.items[0].labels
  expect(labels[labelIds[0]].order).toEqual(1)
  expect(labels[labelIds[1]].order).toEqual(2)
  expect(labels[labelIds[2]].order).toEqual(0)
})
