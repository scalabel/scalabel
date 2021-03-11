import _ from "lodash"

import * as action from "../../src/action/common"
import Session, { dispatch, getState } from "../../src/common/session"
import { updateTracks } from "../../src/common/session_setup"
import {
  Label2DList,
  makeDrawableLabel2D
} from "../../src/drawable/2d/label2d_list"
import { commit2DLabels } from "../../src/drawable/states"
import { makeImageViewerConfig } from "../../src/functional/states"
import { Size2D } from "../../src/math/size2d"
import { Vector2D } from "../../src/math/vector2d"
import { RectType } from "../../src/types/state"
import { setupTestStore } from "../components/util"
import { testJson } from "../test_states/test_track_objects"

beforeAll(() => {
  setupTestStore(testJson)

  Session.images.length = 0
  for (let i = 0; i < getState().task.items.length; i++) {
    Session.images.push({ [-1]: new Image(1000, 1000) })
    dispatch(action.loadItem(i, -1))
  }
  dispatch(action.addViewerConfig(0, makeImageViewerConfig(0)))
  dispatch(action.goToItem(0))
})

beforeEach(() => {
  setupTestStore(testJson)
})

test("Add new valid drawable track", () => {
  dispatch(action.goToItem(0))
  const state = getState()
  expect(_.size(state.task.items[0].labels)).toEqual(3)
  expect(_.size(state.task.tracks)).toEqual(4)
  const label2dlist = new Label2DList()
  const label = makeDrawableLabel2D(label2dlist, "box2d", {})
  expect(label).not.toBeNull()
  if (label !== null) {
    label.initTemp(state, new Vector2D(10, 10))
    label.onMouseDown(new Vector2D(10, 10), 1)
    label.onMouseMove(new Vector2D(20, 20), new Size2D(1000, 1000), 1, 2)
    label.onMouseUp(new Vector2D(20, 20))

    commit2DLabels([label])

    const currentState = getState()
    expect(_.size(currentState.task.items[0].labels)).toEqual(4)
    const savedLabels = currentState.task.items[0].labels
    let savedLabelId = ""
    for (const labelId of Object.keys(savedLabels)) {
      if (labelId.length === 16) {
        savedLabelId = labelId
      }
    }
    // Verify label is added
    const savedLabel = currentState.task.items[0].labels[savedLabelId]
    // Verify label's shape is added
    expect(
      currentState.task.items[0].shapes[savedLabel.shapes[0]]
    ).not.toBeUndefined()
    // Verify label's track is added
    expect(_.size(currentState.task.tracks)).toEqual(5)
    expect(currentState.task.tracks[savedLabel.track]).not.toBeUndefined()
    const savedTrack = currentState.task.tracks[savedLabel.track]
    // Verify all labels in the track is added
    Object.keys(savedTrack.labels).forEach((itemIndexStr) => {
      const itemIndex = Number(itemIndexStr)
      expect(
        currentState.task.items[itemIndex].labels[savedTrack.labels[itemIndex]]
      ).not.toBeUndefined()
    })
  }
})

test("Add new invalid drawable track", () => {
  dispatch(action.goToItem(0))
  const state = getState()
  expect(_.size(state.task.items[0].labels)).toEqual(3)
  expect(_.size(state.task.tracks)).toEqual(4)
  const label2dlist = new Label2DList()
  const label = makeDrawableLabel2D(label2dlist, "box2d", {})
  expect(label).not.toBeNull()
  if (label != null) {
    label.initTemp(state, new Vector2D(10, 10))
    label.onMouseDown(new Vector2D(10, 10), 1)
    label.onMouseMove(new Vector2D(12, 12), new Size2D(1000, 1000), 1, 2)
    label.onMouseUp(new Vector2D(12, 12))

    commit2DLabels([label])

    const currentState = getState()
    expect(_.size(currentState.task.items[0].labels)).toEqual(3)
    expect(_.size(currentState.task.items[0].shapes)).toEqual(
      _.size(state.task.items[0].shapes)
    )
    expect(_.size(currentState.task.tracks)).toEqual(4)
  }
})

test("Update existing drawable of a track", () => {
  updateTracks(getState())

  dispatch(action.goToItem(1))
  const state = getState()
  expect(_.size(state.task.items[1].labels)).toEqual(3)
  // Label coord is [835, 314][861, 406]
  const label2dList = new Label2DList()
  label2dList.updateState(state)
  const label = label2dList.get(2)
  // Existing label must call selected and highlighted property
  label.setSelected(true)
  label.setHighlighted(true, 5) // Handles.BOTTOM_RIGHT
  label.onMouseDown(new Vector2D(861, 406), 1)
  // Mouse move is essential
  label.onMouseMove(new Vector2D(850, 350), new Size2D(1200, 1200), 1, 2)
  label.onMouseUp(new Vector2D(850, 350))

  commit2DLabels([label])

  const currentState = getState()
  const newLabel = currentState.task.items[1].labels["70"]
  const rect = currentState.task.items[1].shapes[newLabel.shapes[0]] as RectType
  // Expect the label resized is changed
  expect(rect.x2).toEqual(850)
  expect(rect.y2).toEqual(350)
  const trackLabels = currentState.task.tracks[newLabel.track].labels
  for (const itemId of Object.keys(trackLabels)) {
    const itemIdIdx = Number(itemId)
    if (itemIdIdx >= 1) {
      const item = currentState.task.items[itemIdIdx]
      const pageLabel = item.labels[trackLabels[itemIdIdx]]
      const pageRect = item.shapes[pageLabel.shapes[0]] as RectType
      // Expect all labels in the track afterward is changed
      expect(pageRect.x2).toEqual(850)
      expect(pageRect.y2).toEqual(350)
    }
  }
})

test("Update existing drawable of a track to invalid, from page 1", () => {
  updateTracks(getState())

  // Terminate current track
  dispatch(action.goToItem(1))
  const state = getState()
  expect(_.size(state.task.items[1].labels)).toEqual(3)
  // Label coord is [835, 314][861, 406]
  const label2dList = new Label2DList()
  label2dList.updateState(state)
  const label = label2dList.get(2)
  // Existing label must call selected and highlighted property
  label.setSelected(true)
  label.setHighlighted(true, 5) // Handles.BOTTOM_RIGHT
  label.onMouseDown(new Vector2D(861, 406), 1)
  // Mouse move is essential
  label.onMouseMove(new Vector2D(837, 315), new Size2D(1200, 1200), 1, 2)
  label.onMouseUp(new Vector2D(837, 315))
  const trackId = label.trackId
  const oldTrackLabels = state.task.tracks[trackId].labels

  commit2DLabels([label])

  const currentState = getState()
  const newLabel = currentState.task.items[1].labels["70"]
  expect(newLabel).toBeUndefined() // Expect the resized label is gone
  for (const itemId of Object.keys(oldTrackLabels)) {
    const itemIdIdx = Number(itemId)
    if (itemIdIdx >= 1) {
      const item = currentState.task.items[itemIdIdx]
      const pageLabel = item.labels[oldTrackLabels[itemIdIdx]]
      // Expect every label in the track afterward is gone
      expect(pageLabel).toBeUndefined()
    }
  }
  const currentTrack = currentState.task.tracks[trackId]
  // Expect the label in first page is still exists
  expect(_.size(currentTrack.labels)).toEqual(1)
  expect(
    currentState.task.items[0].labels[oldTrackLabels[0]]
  ).not.toBeUndefined()
})

test("Update existing drawable of a track to invalid, from page 0", () => {
  updateTracks(getState())

  // Terminate current track
  dispatch(action.goToItem(0))
  const state = getState()
  expect(_.size(state.task.items[0].labels)).toEqual(3)
  // Label coord is [835, 314][861, 406]
  const label2dList = new Label2DList()
  label2dList.updateState(state)
  const label = label2dList.get(2)
  // Existing label must call selected and highlighted property
  label.setSelected(true)
  label.setHighlighted(true, 5) // Handles.BOTTOM_RIGHT
  label.onMouseDown(new Vector2D(861, 406), 1)
  // Mouse move is essential
  label.onMouseMove(new Vector2D(837, 315), new Size2D(1200, 1200), 1, 2)
  label.onMouseUp(new Vector2D(837, 315))
  const trackId = label.trackId
  const oldTrackLabels = state.task.tracks[trackId].labels

  commit2DLabels([label])

  const currentState = getState()
  const newLabel = currentState.task.items[0].labels["69"]
  expect(newLabel).toBeUndefined() // Expect the resized label is gone
  for (const itemId of Object.keys(oldTrackLabels)) {
    const itemIdIdx = Number(itemId)
    const item = currentState.task.items[itemIdIdx]
    const pageLabel = item.labels[oldTrackLabels[itemIdIdx]]
    // Expect every label in the track afterward is gone
    expect(pageLabel).toBeUndefined()
  }
  const currentTrack = currentState.task.tracks[trackId]
  // Expect the label in first page is still exists
  expect(currentTrack).toBeUndefined()
})
