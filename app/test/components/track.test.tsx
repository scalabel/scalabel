import { fireEvent, render } from "@testing-library/react"
import _ from "lodash"
import React from "react"

import * as action from "../../src/action/common"
import { selectLabel } from "../../src/action/select"
import Session, { dispatch, getState, getStore } from "../../src/common/session"
import { updateTracks } from "../../src/common/session_setup"
import { Label2dCanvas } from "../../src/components/label2d_canvas"
import { ToolBar } from "../../src/components/toolbar"
import { IdType, State } from "../../src/types/state"
// Import { TrackCollector } from '../server/util/track_collector'
import { emptyTrackingTask } from "../test_states/test_track_objects"
import { checkBox2D } from "../util/shape"
import {
  drag,
  drawBox2DTracks,
  mouseMoveClick,
  setUpLabel2dCanvas
} from "./canvas_util"
import { setupTestStore } from "./util"

const canvasRef: React.RefObject<Label2dCanvas> = React.createRef()

beforeEach(() => {
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

test("Adding and deleting tracks", () => {
  const label2d = canvasRef.current as Label2dCanvas
  const numItems = getState().task.items.length

  const toolbarRef: React.Ref<ToolBar> = React.createRef()
  const { getByText } = render(
    <ToolBar
      ref={toolbarRef}
      categories={null}
      attributes={[]}
      labelType={"labelType"}
    />
  )
  expect(toolbarRef.current).not.toBeNull()
  expect(toolbarRef.current).not.toBeUndefined()
  if (toolbarRef.current !== null) {
    toolbarRef.current.componentDidMount()
  }

  const itemIndices = [0, 2, 4, 6]
  const numLabels = [1, 1, 2, 2, 3, 3, 4, 4]
  const boxes = [
    [1, 1, 50, 50],
    [19, 20, 30, 29],
    [100, 20, 80, 100],
    [500, 500, 80, 100]
  ]

  // Test adding tracks
  const trackIds = drawBox2DTracks(label2d, getStore(), itemIndices, boxes)
  let state = getState()
  const shapeIds = new Set<IdType>()
  expect(_.size(state.task.tracks)).toEqual(4)
  itemIndices.forEach((itemIndex, i) => {
    expect(_.size(state.task.tracks[trackIds[i]].labels)).toEqual(
      numItems - itemIndex
    )
    state.task.items.forEach((item, index) => {
      expect(_.size(item.labels)).toEqual(numLabels[index])
      expect(_.size(item.shapes)).toEqual(numLabels[index])
    })
  })
  // Check all the shapes have unique IDs
  state.task.items.forEach((item) => {
    _.forEach(item.shapes, (_s, id) => {
      expect(shapeIds.has(id)).toBeFalsy()
      shapeIds.add(id)
    })
  })

  // Terminate the track by button
  dispatch(action.goToItem(2))
  mouseMoveClick(label2d, 1, 30)
  fireEvent(
    getByText("Delete"),
    new MouseEvent("click", {
      bubbles: true,
      cancelable: true
    })
  )
  state = getState()
  expect(_.size(state.task.items[2].labels)).toEqual(1)
  expect(_.size(state.task.items[1].labels)).toEqual(1)
  expect(_.size(state.task.tracks[trackIds[0]].labels)).toEqual(2)

  // Delete the track by button
  dispatch(action.goToItem(6))
  expect(_.size(state.task.items[6].labels)).toEqual(3)
  Session.dispatch(
    selectLabel(
      state.user.select.labels,
      6,
      state.task.tracks[trackIds[3]].labels[6]
    )
  )
  fireEvent(
    getByText("Delete"),
    new MouseEvent("click", {
      bubbles: true,
      cancelable: true
    })
  )
  state = getState()
  expect(_.size(state.task.items[6].labels)).toEqual(2)
  expect(_.size(state.task.tracks)).toEqual(3)

  // Terminate the track by key
  dispatch(action.goToItem(1))
  Session.dispatch(
    selectLabel(
      state.user.select.labels,
      1,
      state.task.tracks[trackIds[0]].labels[1]
    )
  )
  fireEvent.keyDown(document, { key: "Backspace" })
  state = getState()
  expect(_.size(state.task.items[1].labels)).toEqual(0)
  expect(_.size(state.task.items[0].labels)).toEqual(1)
  expect(_.size(state.task.tracks[trackIds[0]].labels)).toEqual(1)

  // Delete the track by key
  dispatch(action.goToItem(0))
  Session.dispatch(
    selectLabel(
      state.user.select.labels,
      0,
      state.task.tracks[trackIds[0]].labels[0]
    )
  )
  fireEvent.keyDown(document, { key: "Backspace" })
  state = getState()
  expect(_.size(state.task.items[0].labels)).toEqual(0)
  expect(_.size(state.task.tracks)).toEqual(2)
})

test("Linking tracks", () => {
  const label2d = canvasRef.current as Label2dCanvas

  const toolbarRef: React.Ref<ToolBar> = React.createRef()
  const { getAllByText } = render(
    <ToolBar
      ref={toolbarRef}
      categories={null}
      attributes={[]}
      labelType={"labelType"}
    />
  )
  expect(toolbarRef.current).not.toBeNull()
  expect(toolbarRef.current).not.toBeUndefined()
  if (toolbarRef.current !== null) {
    toolbarRef.current.componentDidMount()
  }

  const itemIndices = [0, 2, 4, 6]
  const boxes = [
    [1, 1, 50, 50],
    [19, 20, 30, 29],
    [100, 20, 80, 100],
    [500, 500, 80, 100]
  ]

  const trackIds = drawBox2DTracks(label2d, getStore(), itemIndices, boxes)

  // Terminate the track by button
  let state = getState()
  dispatch(action.goToItem(2))
  Session.dispatch(
    selectLabel(
      state.user.select.labels,
      2,
      state.task.tracks[trackIds[0]].labels[2]
    )
  )
  fireEvent(
    getAllByText("Delete")[0],
    new MouseEvent("click", {
      bubbles: true,
      cancelable: true
    })
  )
  dispatch(action.goToItem(1))
  state = getState()
  Session.dispatch(
    selectLabel(
      state.user.select.labels,
      1,
      state.task.tracks[trackIds[0]].labels[1]
    )
  )
  fireEvent(
    getAllByText("Link Tracks")[0],
    new MouseEvent("click", {
      bubbles: true,
      cancelable: true
    })
  )

  dispatch(action.goToItem(4))
  state = getState()
  Session.dispatch(
    selectLabel(
      state.user.select.labels,
      4,
      state.task.tracks[trackIds[2]].labels[4]
    )
  )
  fireEvent(
    getAllByText("Finish")[0],
    new MouseEvent("click", {
      bubbles: true,
      cancelable: true
    })
  )
  state = getState()
  expect(_.size(state.task.tracks)).toEqual(3)
})

test("Changing attributes and categories of tracks", () => {
  const label2d = canvasRef.current as Label2dCanvas

  const toolbarRef: React.Ref<ToolBar> = React.createRef()
  const { getByText, getAllByRole } = render(
    <ToolBar
      ref={toolbarRef}
      categories={(emptyTrackingTask as State).task.config.categories}
      attributes={(emptyTrackingTask as State).task.config.attributes}
      labelType={"labelType"}
    />
  )
  expect(toolbarRef.current).not.toBeNull()
  expect(toolbarRef.current).not.toBeUndefined()
  if (toolbarRef.current !== null) {
    toolbarRef.current.componentDidMount()
  }

  const itemIndices = [0]
  const boxes = [[1, 1, 50, 50]]

  const trackIds = drawBox2DTracks(label2d, getStore(), itemIndices, boxes)

  // Changing category
  dispatch(action.goToItem(2))
  mouseMoveClick(label2d, 1, 30)
  fireEvent(
    getByText("car"),
    new MouseEvent("click", {
      bubbles: true,
      cancelable: true
    })
  )
  dispatch(action.goToItem(1))
  let state = getState()
  const labelIdIn2 = state.task.tracks[trackIds[0]].labels[2]
  const labelIdIn3 = state.task.tracks[trackIds[0]].labels[3]
  expect(state.task.items[3].labels[labelIdIn3].category).toEqual([2])

  // Changing attributes
  // Attribute should be propagated to the end of each track
  dispatch(action.goToItem(2))
  mouseMoveClick(label2d, 1, 30)
  const switchBtn = getAllByRole("checkbox")[0]
  switchBtn.click()
  state = getState()
  expect(state.task.items[2].labels[labelIdIn2].attributes[0]).toEqual([1])
  expect(state.task.items[3].labels[labelIdIn3].attributes[0]).toEqual([1])
  expect(state.task.items[2].labels[labelIdIn2].attributes[1]).toEqual([0])
  expect(state.task.items[2].labels[labelIdIn2].attributes[2]).toEqual([0])
})

test("Changing shapes and locations of tracks", () => {
  const label2d = canvasRef.current as Label2dCanvas

  const itemIndices = [0, 2]
  const boxes = [
    [10, 20, 50, 60],
    [100, 110, 200, 300]
  ]

  const trackIds = drawBox2DTracks(label2d, getStore(), itemIndices, boxes)

  // Changing shape
  dispatch(action.goToItem(4))
  drag(label2d, 10, 20, 15, 25)
  let state = getState()
  // Shapes starting from item 4 should change
  for (let i = 4; i < 8; ++i) {
    checkBox2D(
      state.task.tracks[trackIds[0]].labels[i],
      { x1: 15, y1: 25, x2: 50, y2: 60 },
      i
    )
  }
  // Shapes before item 4 should be linearly interpolated
  for (let i = 0; i < 4; ++i) {
    const diff = i * 1.25
    checkBox2D(
      state.task.tracks[trackIds[0]].labels[i],
      { x1: 10 + diff, y1: 20 + diff, x2: 50, y2: 60 },
      i
    )
  }
  dispatch(action.goToItem(6))
  drag(label2d, 50, 60, 55, 65)
  state = getState()
  // Shapes should change only between item 6 to 8
  for (let i = 6; i < 8; ++i) {
    checkBox2D(
      state.task.tracks[trackIds[0]].labels[i],
      { x1: 15, y1: 25, x2: 55, y2: 65 },
      i
    )
  }
  for (let i = 4; i < 6; ++i) {
    const diff = (i - 4) * 2.5
    checkBox2D(
      state.task.tracks[trackIds[0]].labels[i],
      { x1: 15, y1: 25, x2: 50 + diff, y2: 60 + diff },
      i
    )
  }
  // Changing location
  dispatch(action.goToItem(5))
  drag(label2d, 110, 110, 200, 200) // (90, 90) translation
  state = getState()
  // Locations starting from item 5 should change
  for (let i = 5; i < 8; ++i) {
    checkBox2D(
      state.task.tracks[trackIds[1]].labels[i],
      { x1: 190, y1: 200, x2: 290, y2: 390 },
      i
    )
  }
  // Locations before item 5 should be interpolated
  for (let i = 2; i < 5; ++i) {
    const diff = (i - 2) * 30
    checkBox2D(
      state.task.tracks[trackIds[1]].labels[i],
      { x1: 100 + diff, y1: 110 + diff, x2: 200 + diff, y2: 300 + diff },
      i
    )
  }
})
