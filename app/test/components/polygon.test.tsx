import _ from "lodash"
import * as React from "react"

import * as action from "../../src/action/common"
import Session, { dispatch, getState } from "../../src/common/session"
import { Label2dCanvas } from "../../src/components/label2d_canvas"
import { getShapes } from "../../src/functional/state_util"
import { PathPoint2DType } from "../../src/types/state"
import { testJson } from "../test_states/test_image_objects"
import { LabelCollector } from "../util/label_collector"
import {
  drawPolygon,
  keyDown,
  keyUp,
  mouseDown,
  mouseMove,
  mouseMoveClick,
  mouseUp,
  setUpLabel2dCanvas
} from "./canvas_util"
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

test("Draw 2d polygons to label2d list", () => {
  Session.dispatch(action.changeSelect({ labelType: 1 }))
  const labelIds = new LabelCollector(getState)
  // It has been checked that canvasRef.current is not null
  const label2d = canvasRef.current as Label2dCanvas
  // Draw the first polygon
  mouseMoveClick(label2d, 10, 10)
  mouseMoveClick(label2d, 100, 100)
  mouseMoveClick(label2d, 200, 100)
  /**
   * drawing the first polygon
   * polygon 1: (10, 10) (100, 100) (200, 100)
   */
  let state = Session.getState()
  expect(_.size(state.task.items[0].labels)).toEqual(0)

  // Drag when drawing
  mouseMoveClick(label2d, 100, 0)
  mouseMoveClick(label2d, 10, 10)

  /**
   * polygon 1: (10, 10) (100, 100) (200, 100) (100, 0)
   */
  state = Session.getState()
  expect(_.size(state.task.items[0].labels)).toEqual(1)
  labelIds.collect()
  let points = getShapes(state, 0, labelIds[0]) as PathPoint2DType[]
  expect(points.length).toEqual(4)
  expect(points[0]).toMatchObject({ x: 10, y: 10, pointType: "line" })
  expect(points[1]).toMatchObject({ x: 100, y: 100, pointType: "line" })
  expect(points[2]).toMatchObject({ x: 200, y: 100, pointType: "line" })
  expect(points[3]).toMatchObject({ x: 100, y: 0, pointType: "line" })

  // Draw second polygon
  drawPolygon(label2d, [
    [500, 500],
    [600, 400],
    [700, 700]
  ])
  /**
   * polygon 1: (10, 10) (100, 100) (200, 100) (100, 0)
   * polygon 2: (500, 500) (600, 400) (700, 700)
   */

  state = Session.getState()
  labelIds.collect()
  expect(_.size(state.task.items[0].labels)).toEqual(2)
  points = getShapes(state, 0, labelIds[1]) as PathPoint2DType[]
  expect(points.length).toEqual(3)
  expect(points[0]).toMatchObject({ x: 500, y: 500, pointType: "line" })
  expect(points[1]).toMatchObject({ x: 600, y: 400, pointType: "line" })
  expect(points[2]).toMatchObject({ x: 700, y: 700, pointType: "line" })
})

test("2d polygons highlighted and selected", () => {
  Session.dispatch(action.changeSelect({ labelType: 1 }))
  const labelIds = new LabelCollector(getState)
  // It has been checked that canvasRef.current is not null
  const label2d = canvasRef.current as Label2dCanvas
  // Draw first polygon
  drawPolygon(label2d, [
    [120, 120],
    [210, 210],
    [310, 260],
    [410, 210],
    [210, 110]
  ])
  /**
   * polygon 1: (120, 120) (210, 210) (310, 260) (410, 210) (210, 110)
   */
  labelIds.collect()
  let selected = Session.label2dList.selectedLabels
  expect(selected[0].labelId).toEqual(labelIds[0])

  // Draw second polygon
  drawPolygon(label2d, [
    [500, 500],
    [600, 400],
    [700, 700]
  ])
  /**
   * polygon 1: (120, 120) (210, 210) (310, 260) (410, 210) (210, 110)
   * polygon 2: (500, 500) (600, 400) (700, 700)
   */

  // Change selected label
  mouseMove(label2d, 120, 120)
  mouseDown(label2d, 120, 120)
  mouseMove(label2d, 140, 140)
  mouseUp(label2d, 140, 140)
  selected = Session.label2dList.selectedLabels
  labelIds.collect()
  expect(selected[0].labelId).toEqual(labelIds[0])
})

test("validation check for polygon2d", () => {
  Session.dispatch(action.changeSelect({ labelType: 1 }))
  const labelIds = new LabelCollector(getState)
  // It has been checked that canvasRef.current is not null
  const label2d = canvasRef.current as Label2dCanvas
  // Draw a valid polygon
  drawPolygon(label2d, [
    [120, 120],
    [210, 210],
    [310, 260],
    [410, 210],
    [210, 110]
  ])
  labelIds.collect()

  /**
   * polygon 1: (120, 120) (210, 210) (310, 260) (410, 210) (210, 110)
   */

  // Draw one invalid polygon
  drawPolygon(label2d, [
    [200, 100],
    [400, 300],
    [300, 200],
    [300, 0]
  ])

  /**
   * polygon 1: (120, 120) (210, 210) (310, 260) (410, 210) (210, 110)
   * polygon 2: (200, 100) (400, 300) (300, 200) (300, 0) invalid
   */

  let state = Session.getState()
  expect(_.size(state.task.items[0].labels)).toEqual(1)
  expect(Session.label2dList.labelList.length).toEqual(1)

  // Drag the polygon to an invalid shape
  mouseMove(label2d, 310, 260)
  mouseDown(label2d, 310, 260)
  mouseMove(label2d, 310, 0)
  mouseUp(label2d, 310, 0)

  /**
   * polygon 1: (120, 120) (210, 210) (310, 0) (410, 210) (210, 110)
   * polygon 1 is an invalid shape
   */

  state = Session.getState()
  const points = getShapes(state, 0, labelIds[0]) as PathPoint2DType[]
  expect(points[2]).toMatchObject({ x: 310, y: 260, pointType: "line" })
  /**
   * polygon 1: (120, 120) (210, 210) (310, 260) (410, 210) (210, 110)
   */

  // Draw a too small polygon
  drawPolygon(label2d, [
    [0, 0],
    [1, 0],
    [0, 1]
  ])

  /**
   * polygon 1: (120, 120) (210, 210) (310, 260) (410, 210) (210, 110)
   * polygon 2: (0, 0) (1, 0) (0, 1) too small, invalid
   */

  state = Session.getState()
  expect(_.size(state.task.items[0].labels)).toEqual(1)
  expect(Session.label2dList.labelList.length).toEqual(1)
})

test("2d polygons drag vertices, midpoints and edges", () => {
  Session.dispatch(action.changeSelect({ labelType: 1 }))
  const labelIds = new LabelCollector(getState)
  // It has been checked that canvasRef.current is not null
  const label2d = canvasRef.current as Label2dCanvas
  drawPolygon(label2d, [
    [10, 10],
    [100, 100],
    [200, 100],
    [100, 0]
  ])
  /**
   * polygon 1: (10, 10) (100, 100) (200, 100) (100, 0)
   */
  labelIds.collect()

  // Drag a vertex
  mouseMove(label2d, 200, 100)
  mouseDown(label2d, 200, 100)
  mouseMove(label2d, 300, 100)
  mouseUp(label2d, 300, 100)
  let state = Session.getState()
  let points = getShapes(state, 0, labelIds[0]) as PathPoint2DType[]
  expect(points[2]).toMatchObject({ x: 300, y: 100, pointType: "line" })

  /**
   * polygon 1: (10, 10) (100, 100) (300, 100) (100, 0)
   */

  // Drag midpoints
  mouseMove(label2d, 200, 100)
  mouseDown(label2d, 200, 100)
  mouseMove(label2d, 200, 150)
  mouseUp(label2d, 200, 150)
  state = Session.getState()
  points = getShapes(state, 0, labelIds[0]) as PathPoint2DType[]
  expect(points[2]).toMatchObject({ x: 200, y: 150, pointType: "line" })
  expect(points.length).toEqual(5)
  /**
   * polygon 1: (10, 10) (100, 100) (200, 150) (300, 100) (100, 0)
   */

  // Drag edges
  mouseMove(label2d, 20, 20)
  mouseDown(label2d, 20, 20)
  mouseMove(label2d, 120, 120)
  mouseUp(label2d, 120, 120)
  state = Session.getState()
  points = getShapes(state, 0, labelIds[0]) as PathPoint2DType[]
  expect(points[0]).toMatchObject({ x: 110, y: 110, pointType: "line" })
  expect(points.length).toEqual(5)
  /**
   * polygon 1: (110, 110) (200, 200) (300, 250) (400, 200) (200, 100)
   */
})

test("2d polygons delete vertex and draw bezier curve", () => {
  Session.dispatch(action.changeSelect({ labelType: 1 }))
  const labelIds = new LabelCollector(getState)
  // It has been checked that canvasRef.current is not null
  const label2d = canvasRef.current as Label2dCanvas
  // Draw a polygon and delete vertex when drawing
  mouseMoveClick(label2d, 200, 100)
  keyDown(label2d, "d")
  keyUp(label2d, "d")
  mouseMoveClick(label2d, 250, 100)
  mouseMoveClick(label2d, 300, 0)
  mouseMoveClick(label2d, 350, 100)
  mouseMoveClick(label2d, 300, 200)
  mouseMove(label2d, 320, 130)
  keyDown(label2d, "d")
  keyUp(label2d, "d")
  mouseDown(label2d, 320, 130)
  mouseUp(label2d, 320, 130)
  mouseMoveClick(label2d, 300, 150)
  mouseMoveClick(label2d, 250, 100)

  /**
   * polygon: (250, 100) (300, 0) (350, 100) (320, 130) (300, 150)
   */
  labelIds.collect()

  let state = Session.getState()
  expect(_.size(state.task.items[0].labels)).toEqual(1)
  expect(Session.label2dList.labelList.length).toEqual(1)

  let points = getShapes(state, 0, labelIds[0]) as PathPoint2DType[]
  expect(points.length).toEqual(5)
  expect(points[0]).toMatchObject({ x: 250, y: 100, pointType: "line" })
  expect(points[1]).toMatchObject({ x: 300, y: 0, pointType: "line" })
  expect(points[2]).toMatchObject({ x: 350, y: 100, pointType: "line" })
  expect(points[3]).toMatchObject({ x: 320, y: 130, pointType: "line" })
  expect(points[4]).toMatchObject({ x: 300, y: 150, pointType: "line" })

  // Delete vertex when closed
  keyDown(label2d, "d")
  mouseMove(label2d, 275, 125)
  mouseDown(label2d, 275, 125)
  mouseUp(label2d, 2750, 1250)
  mouseMoveClick(label2d, 300, 150)
  keyUp(label2d, "d")
  /**
   * polygon: (250, 100) (300, 0) (350, 100) (320, 130)
   */

  state = Session.getState()
  points = getShapes(state, 0, labelIds[0]) as PathPoint2DType[]
  expect(points.length).toEqual(4)
  expect(points[3]).toMatchObject({ x: 320, y: 130, pointType: "line" })

  // Draw bezier curve
  keyDown(label2d, "c")
  mouseMoveClick(label2d, 335, 115)
  keyUp(label2d, "c")
  /**
   * polygon: (250, 100) (300, 0) (350, 100)
   * [ (340, 110) (330, 120) <bezier curve control points>]
   * (320, 130)
   */

  state = Session.getState()
  points = getShapes(state, 0, labelIds[0]) as PathPoint2DType[]
  expect(points.length).toEqual(6)
  expect(points[3]).toMatchObject({ x: 340, y: 110, pointType: "bezier" })
  expect(points[4]).toMatchObject({ x: 330, y: 120, pointType: "bezier" })

  // Drag bezier curve control points
  mouseMove(label2d, 340, 110)
  mouseDown(label2d, 340, 110)
  mouseMove(label2d, 340, 90)
  mouseUp(label2d, 340, 90)
  /**
   * polygon: (250, 100) (300, 0) (350, 100)
   * [ (340, 90) (330, 120) <bezier curve control points>]
   * (320, 130)
   */

  state = Session.getState()
  points = getShapes(state, 0, labelIds[0]) as PathPoint2DType[]
  expect(points.length).toEqual(6)
  expect(points[2]).toMatchObject({ x: 350, y: 100, pointType: "line" })
  expect(points[3]).toMatchObject({ x: 340, y: 90, pointType: "bezier" })
  expect(points[4]).toMatchObject({ x: 330, y: 120, pointType: "bezier" })

  // Delete vertex on bezier curve
  keyDown(label2d, "d")
  mouseMoveClick(label2d, 350, 100)
  keyUp(label2d, "d")
  /**
   * polygon: (250, 100) (300, 0) (320, 130)
   */

  state = Session.getState()
  points = getShapes(state, 0, labelIds[0]) as PathPoint2DType[]
  expect(points.length).toEqual(3)
})

test("2d polygons multi-select and multi-label moving", () => {
  Session.dispatch(action.changeSelect({ labelType: 1 }))
  const labelIds = new LabelCollector(getState)
  // It has been checked that canvasRef.current is not null
  const label2d = canvasRef.current as Label2dCanvas
  // Draw first polygon
  drawPolygon(label2d, [
    [10, 10],
    [100, 100],
    [200, 100]
  ])
  /**
   * polygon 1: (10, 10) (100, 100) (200, 100)
   */
  labelIds.collect()

  // Draw second polygon
  drawPolygon(label2d, [
    [500, 500],
    [600, 400],
    [700, 700]
  ])
  /**
   * polygon 1: (10, 10) (100, 100) (200, 100)
   * polygon 2: (500, 500) (600, 400) (700, 700)
   */
  labelIds.collect()

  // Draw third polygon
  drawPolygon(label2d, [
    [250, 250],
    [300, 250],
    [350, 350]
  ])

  /**
   * polygon 1: (10, 10) (100, 100) (200, 100)
   * polygon 2: (500, 500) (600, 400) (700, 700)
   * polygon 3: (250, 250) (300, 250) (350, 350)
   */
  labelIds.collect()

  let state = Session.getState()
  expect(_.size(state.task.items[0].labels)).toEqual(3)

  // Select label 1
  mouseMoveClick(label2d, 600, 600)

  state = Session.getState()
  expect(state.user.select.labels[0].length).toEqual(1)
  expect(Session.label2dList.selectedLabels.length).toEqual(1)
  expect(Session.label2dList.selectedLabels[0].labelId).toEqual(
    state.user.select.labels[0][0]
  )

  // Select label 1, 2, 3
  keyDown(label2d, "Meta")
  mouseMoveClick(label2d, 300, 250)
  mouseMoveClick(label2d, 50, 50)
  keyUp(label2d, "Meta")

  state = Session.getState()
  expect(state.user.select.labels[0].length).toEqual(3)
  expect(Session.label2dList.selectedLabels.length).toEqual(3)

  // Unselect label 3
  keyDown(label2d, "Meta")
  mouseMoveClick(label2d, 300, 250)
  keyUp(label2d, "Meta")
  mouseMove(label2d, 0, 0)

  state = Session.getState()
  expect(state.user.select.labels[0].length).toEqual(2)
  expect(Session.label2dList.selectedLabels.length).toEqual(2)
  expect(Session.label2dList.labelList[2].highlighted).toEqual(false)

  // Select three labels
  keyDown(label2d, "Meta")
  mouseMoveClick(label2d, 300, 250)
  keyUp(label2d, "Meta")

  state = Session.getState()
  expect(state.user.select.labels[0].length).toEqual(3)
  expect(Session.label2dList.selectedLabels.length).toEqual(3)

  // Move
  mouseMove(label2d, 20, 20)
  mouseDown(label2d, 20, 20)
  mouseMove(label2d, 60, 60)
  mouseMove(label2d, 120, 120)
  mouseUp(label2d, 120, 120)
  /**
   * polygon 1: (110, 110) (200, 200) (300, 200)
   * polygon 2: (600, 600) (700, 500) (800, 800)
   * polygon 3: (350, 350) (400, 350) (450, 450)
   */

  state = Session.getState()
  let points = getShapes(state, 0, labelIds[0]) as PathPoint2DType[]
  expect(points[0]).toMatchObject({ x: 110, y: 110 })
  points = getShapes(state, 0, labelIds[1]) as PathPoint2DType[]
  expect(points[0]).toMatchObject({ x: 600, y: 600 })
  points = getShapes(state, 0, labelIds[2]) as PathPoint2DType[]
  expect(points[0]).toMatchObject({ x: 350, y: 350 })
})

test("2d polygons linking labels and moving", () => {
  Session.dispatch(action.changeSelect({ labelType: 1 }))
  const labelIds = new LabelCollector(getState)
  // It has been checked that canvasRef.current is not null
  const label2d = canvasRef.current as Label2dCanvas
  // Draw first polygon
  drawPolygon(label2d, [
    [10, 10],
    [100, 100],
    [200, 100]
  ])
  /**
   * polygon 1: (10, 10) (100, 100) (200, 100)
   */
  labelIds.collect()

  // Draw second polygon
  drawPolygon(label2d, [
    [500, 500],
    [600, 400],
    [700, 700]
  ])
  /**
   * polygon 1: (10, 10) (100, 100) (200, 100)
   * polygon 2: (500, 500) (600, 400) (700, 700)
   */
  labelIds.collect()

  // Draw third polygon
  drawPolygon(label2d, [
    [250, 250],
    [300, 250],
    [350, 350]
  ])
  /**
   * polygon 1: (10, 10) (100, 100) (200, 100)
   * polygon 2: (500, 500) (600, 400) (700, 700)
   * polygon 3: (250, 250) (300, 250) (350, 350)
   */
  labelIds.collect()

  let state = Session.getState()
  expect(_.size(state.task.items[0].labels)).toEqual(3)

  // Select label 2 and 0
  mouseMoveClick(label2d, 300, 300)
  keyDown(label2d, "Meta")
  mouseMoveClick(label2d, 100, 100)
  keyUp(label2d, "Meta")
  /**
   * polygon 1: (10, 10) (100, 100) (200, 100)
   * polygon 2: (500, 500) (600, 400) (700, 700)
   * polygon 3: (250, 250) (300, 250) (350, 350)
   */

  state = Session.getState()
  expect(state.user.select.labels[0].length).toEqual(2)
  expect(Session.label2dList.selectedLabels.length).toEqual(2)
  expect(Session.label2dList.selectedLabels[0].labelId).toEqual(
    state.user.select.labels[0][0]
  )
  expect(Session.label2dList.selectedLabels[1].labelId).toEqual(
    state.user.select.labels[0][1]
  )

  // Select label 1 and 2
  mouseMoveClick(label2d, 600, 600)
  keyDown(label2d, "Meta")
  mouseMoveClick(label2d, 50, 50)
  keyUp(label2d, "Meta")

  // Link label 1 and 2
  keyDown(label2d, "l")
  keyUp(label2d, "l")
  /**
   * polygon 1: (10, 10) (100, 100) (200, 100)
   * polygon 2: (500, 500) (600, 400) (700, 700)
   * polygon 3: (250, 250) (300, 250) (350, 350)
   * group 1: 1, 2
   */

  state = Session.getState()
  expect(_.size(state.task.items[0].labels)).toEqual(4)
  expect(_.size(Session.label2dList.labelList)).toEqual(3)
  expect(Session.label2dList.labelList[0].color).toEqual(
    Session.label2dList.labelList[1].color
  )

  // Reselect label 1 and 2
  mouseMoveClick(label2d, 300, 250)
  mouseMoveClick(label2d, 50, 50)

  state = Session.getState()
  expect(state.user.select.labels[0].length).toEqual(2)
  expect(Session.label2dList.selectedLabels.length).toEqual(2)

  // Moving group 1
  mouseMove(label2d, 20, 20)
  mouseDown(label2d, 20, 20)
  mouseMove(label2d, 60, 60)
  mouseMove(label2d, 120, 120)
  mouseUp(label2d, 120, 120)
  /**
   * polygon 1: (110, 110) (200, 200) (300, 200)
   * polygon 2: (600, 600) (700, 500) (800, 800)
   * polygon 3: (250, 250) (300, 250) (350, 350)
   * group 1: 1, 2
   */

  state = Session.getState()
  let points = getShapes(state, 0, labelIds[0]) as PathPoint2DType[]
  expect(points[0]).toMatchObject({ x: 110, y: 110 })
  points = getShapes(state, 0, labelIds[1]) as PathPoint2DType[]
  expect(points[0]).toMatchObject({ x: 600, y: 600 })
  points = getShapes(state, 0, labelIds[2]) as PathPoint2DType[]
  expect(points[0]).toMatchObject({ x: 250, y: 250 })
})

test("2d polygons unlinking", () => {
  Session.dispatch(action.changeSelect({ labelType: 1 }))

  // It has been checked that canvasRef.current is not null
  const label2d = canvasRef.current as Label2dCanvas
  // Draw first polygon
  drawPolygon(label2d, [
    [10, 10],
    [100, 100],
    [200, 100]
  ])
  /**
   * polygon 1: (10, 10) (100, 100) (200, 100)
   */

  // Draw second polygon
  drawPolygon(label2d, [
    [500, 500],
    [600, 400],
    [700, 700]
  ])
  /**
   * polygon 1: (10, 10) (100, 100) (200, 100)
   * polygon 2: (500, 500) (600, 400) (700, 700)
   */

  // Draw third polygon
  drawPolygon(label2d, [
    [250, 250],
    [300, 250],
    [350, 350]
  ])
  /**
   * polygon 1: (10, 10) (100, 100) (200, 100)
   * polygon 2: (500, 500) (600, 400) (700, 700)
   * polygon 3: (250, 250) (300, 250) (350, 350)
   */

  let state = Session.getState()
  expect(_.size(state.task.items[0].labels)).toEqual(3)

  // Select polygon 1 and 3
  keyDown(label2d, "Meta")
  mouseMoveClick(label2d, 100, 100)
  keyUp(label2d, "Meta")

  // Link polygon 1 and 3
  keyDown(label2d, "l")
  keyUp(label2d, "l")

  state = Session.getState()
  expect(_.size(state.task.items[0].labels)).toEqual(4)
  expect(_.size(Session.label2dList.labelList)).toEqual(3)
  expect(Session.label2dList.labelList[0].color).toEqual(
    Session.label2dList.labelList[2].color
  )
  /**
   * polygon 1: (10, 10) (100, 100) (200, 100)
   * polygon 2: (500, 500) (600, 400) (700, 700)
   * polygon 3: (250, 250) (300, 250) (350, 350)
   * group 1: polygon 1, 3
   */

  // Select polygon 1, 2, 3
  mouseMoveClick(label2d, 550, 550)
  keyDown(label2d, "Meta")
  mouseMoveClick(label2d, 100, 100)
  keyUp(label2d, "Meta")

  // Unlink polygon 1 and 3
  keyDown(label2d, "L")
  keyUp(label2d, "L")
  /**
   * polygon 1: (10, 10) (100, 100) (200, 100)
   * polygon 2: (500, 500) (600, 400) (700, 700)
   * polygon 3: (250, 250) (300, 250) (350, 350)
   */

  state = Session.getState()
  expect(_.size(state.task.items[0].labels)).toEqual(3)
  expect(_.size(Session.label2dList.labelList)).toEqual(3)

  // Unselect polygon 1
  keyDown(label2d, "Meta")
  mouseMoveClick(label2d, 100, 100)
  keyUp(label2d, "Meta")

  // Link polygon 2 and 3
  keyDown(label2d, "l")
  keyUp(label2d, "l")
  /**
   * polygon 1: (10, 10) (100, 100) (200, 100)
   * polygon 2: (500, 500) (600, 400) (700, 700)
   * polygon 3: (250, 250) (300, 250) (350, 350)
   * group 1: polygon 2, 3
   */

  state = Session.getState()
  expect(_.size(state.task.items[0].labels)).toEqual(4)
  expect(_.size(Session.label2dList.labelList)).toEqual(3)
  expect(Session.label2dList.labelList[1].color).toEqual(
    Session.label2dList.labelList[2].color
  )
})
