import React from "react"

import * as action from "../../src/action/common"
import Session, { dispatch, getState, getStore } from "../../src/common/session"
import { updateTracks } from "../../src/common/session_setup"
import { Label2dCanvas } from "../../src/components/label2d_canvas"
// Import { TrackCollector } from '../server/util/track_collector'
import { emptyTrackingTask } from "../test_states/test_track_objects"
import {
  drawPolygon2DTracks,
  keyDown,
  keyUp,
  mouseDown,
  mouseMove,
  mouseMoveClick,
  mouseUp,
  setUpLabel2dCanvas
} from "./canvas_util"
import { setupTestStore } from "./util"
import { getShapes } from "../../src/functional/state_util"
import { PathPoint2DType } from "../../src/types/state"

const canvasRef: React.RefObject<Label2dCanvas> = React.createRef()

beforeEach(() => {
  // TODO: Find the reason why 'canvasRef.current' is a null object.
  // and remove these code borrowed from beforeAll().
  setupTestStore(emptyTrackingTask)
  Session.images.length = 0
  Session.images.push({ [-1]: new Image(1000, 1000) })
  // Mock loading every item to make sure the canvas can be successfully
  // initialized
  for (let i = 0; i < getState().task.items.length; i++) {
    dispatch(action.loadItem(i, -1))
  }
  setUpLabel2dCanvas(dispatch, canvasRef, 1000, 1000, true)
  // original code
  expect(canvasRef.current).not.toBeNull()
  canvasRef.current?.clear()
  setupTestStore(emptyTrackingTask)
  Session.subscribe(() => {
    Session.label2dList.updateState(getState())
    canvasRef.current?.updateState(getState())
    updateTracks(getState())
  })
})

beforeAll(() => {
  setupTestStore(emptyTrackingTask)
  Session.images.length = 0
  Session.images.push({ [-1]: new Image(1000, 1000) })
  // Mock loading every item to make sure the canvas can be successfully
  // initialized
  for (let i = 0; i < getState().task.items.length; i++) {
    dispatch(action.loadItem(i, -1))
  }
  setUpLabel2dCanvas(dispatch, canvasRef, 1000, 1000, true)
})

test("Vertex deletion and addition in polygon tracking", () => {
  Session.dispatch(action.changeSelect({ labelType: 1 }))
  const label2d = canvasRef.current as Label2dCanvas

  const itemIndices = [0]
  const polygons = [
    [
      [10, 10],
      [10, 50],
      [50, 50],
      [50, 10]
    ]
  ]
  /**
   * polygon: (10, 10) (10, 50) (50, 50) (50, 10)
   */

  // Test adding tracks
  const trackIds = drawPolygon2DTracks(
    label2d,
    getStore(),
    itemIndices,
    polygons
  )

  dispatch(action.goToItem(2))
  // Delete vertex of a polygon in tracking mode
  keyDown(label2d, "d")
  mouseMove(label2d, 50, 50)
  mouseDown(label2d, 50, 50)
  mouseUp(label2d, 2750, 1250)
  keyUp(label2d, "d")

  /**
   * polygon modified: (10, 10) (10, 50) (50, 10)
   */

  let state = getState()
  let points = getShapes(
    state,
    2,
    state.task.tracks[trackIds[0]].labels[2]
  ) as PathPoint2DType[]
  expect(points.length).toEqual(3)
  expect(points[0]).toMatchObject({ x: 10, y: 10, pointType: "line" })
  expect(points[1]).toMatchObject({ x: 10, y: 50, pointType: "line" })
  expect(points[2]).toMatchObject({ x: 50, y: 10, pointType: "line" })
  expect(
    state.task.items[2].labels[state.task.tracks[trackIds[0]].labels[2]].changed
  ).toBe(true)

  // Add vertex by dragging midpoint
  mouseMove(label2d, 30, 30)
  mouseDown(label2d, 30, 30)
  mouseMove(label2d, 40, 40)
  mouseUp(label2d, 40, 40)

  /**
   * polygon: (10, 10) (10, 50) (40, 40) (50, 10)
   */

  state = getState()
  points = getShapes(
    state,
    2,
    state.task.tracks[trackIds[0]].labels[2]
  ) as PathPoint2DType[]
  expect(points.length).toEqual(4)
  expect(points[0]).toMatchObject({ x: 10, y: 10, pointType: "line" })
  expect(points[1]).toMatchObject({ x: 10, y: 50, pointType: "line" })
  expect(points[2]).toMatchObject({ x: 40, y: 40, pointType: "line" })
  expect(points[3]).toMatchObject({ x: 50, y: 10, pointType: "line" })
  expect(
    state.task.items[2].labels[state.task.tracks[trackIds[0]].labels[2]].changed
  ).toBe(true)

  // polygon of the same track of other frames should not be modified,
  // They should be (10, 10) (10, 50) (50, 50) (50, 10)
  points = getShapes(
    state,
    1,
    state.task.tracks[trackIds[0]].labels[1]
  ) as PathPoint2DType[]
  expect(points.length).toEqual(4)
  expect(points[0]).toMatchObject({ x: 10, y: 10, pointType: "line" })
  expect(points[1]).toMatchObject({ x: 10, y: 50, pointType: "line" })
  expect(points[2]).toMatchObject({ x: 50, y: 50, pointType: "line" })
  expect(points[3]).toMatchObject({ x: 50, y: 10, pointType: "line" })
  expect(
    state.task.items[1].labels[state.task.tracks[trackIds[0]].labels[1]].changed
  ).toBe(false)
})

test("Bezier curve in polygon tracking", () => {
  Session.dispatch(action.changeSelect({ labelType: 1 }))
  const label2d = canvasRef.current as Label2dCanvas

  const itemIndices = [0]
  const polygons = [
    [
      [250, 100],
      [300, 0],
      [350, 100],
      [320, 130]
    ]
  ]
  /**
   * polygon: (250, 100) (300, 0) (350, 100) (320, 130)
   */

  // Test adding tracks
  const trackIds = drawPolygon2DTracks(
    label2d,
    getStore(),
    itemIndices,
    polygons
  )

  dispatch(action.goToItem(2))
  // Delete vertex of a polygon in tracking mode
  keyDown(label2d, "c")
  mouseMoveClick(label2d, 335, 115)
  keyUp(label2d, "c")
  /**
   * polygon: (250, 100) (300, 0) (350, 100)
   * [ (340, 110) (330, 120) <bezier curve control points>]
   * (320, 130)
   */

  let state = getState()
  let points = getShapes(
    state,
    2,
    state.task.tracks[trackIds[0]].labels[2]
  ) as PathPoint2DType[]
  expect(points.length).toEqual(6)
  expect(points[3]).toMatchObject({ x: 340, y: 110, pointType: "bezier" })
  expect(points[4]).toMatchObject({ x: 330, y: 120, pointType: "bezier" })
  expect(
    state.task.items[2].labels[state.task.tracks[trackIds[0]].labels[2]].changed
  ).toBe(true)

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

  state = getState()
  points = getShapes(
    state,
    2,
    state.task.tracks[trackIds[0]].labels[2]
  ) as PathPoint2DType[]
  expect(points.length).toEqual(6)
  expect(points[2]).toMatchObject({ x: 350, y: 100, pointType: "line" })
  expect(points[3]).toMatchObject({ x: 340, y: 90, pointType: "bezier" })
  expect(points[4]).toMatchObject({ x: 330, y: 120, pointType: "bezier" })
  expect(
    state.task.items[2].labels[state.task.tracks[trackIds[0]].labels[2]].changed
  ).toBe(true)

  // polygon of the same track of other frames should not be modified,
  // They should be (250, 100) (300, 0) (350, 100) (320, 130)
  points = getShapes(
    state,
    1,
    state.task.tracks[trackIds[0]].labels[1]
  ) as PathPoint2DType[]
  expect(points.length).toEqual(4)
  expect(points[0]).toMatchObject({ x: 250, y: 100, pointType: "line" })
  expect(points[1]).toMatchObject({ x: 300, y: 0, pointType: "line" })
  expect(points[2]).toMatchObject({ x: 350, y: 100, pointType: "line" })
  expect(points[3]).toMatchObject({ x: 320, y: 130, pointType: "line" })
  expect(
    state.task.items[1].labels[state.task.tracks[trackIds[0]].labels[1]].changed
  ).toBe(false)
})
