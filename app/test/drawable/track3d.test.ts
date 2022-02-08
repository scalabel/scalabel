import _ from "lodash"

import * as action from "../../src/action/common"
import Session, { dispatch, getState } from "../../src/common/session"
import {
  Label3DList,
  createBox3dLabel,
  createPlaneLabel
} from "../../src/drawable/3d/label3d_list"
import { commitLabels } from "../../src/drawable/states"
import { makePointCloudViewerConfig } from "../../src/functional/states"
import { Vector3D } from "../../src/math/vector3d"
import { setupTestStore } from "../components/util"
import { testJson3DTracking } from "../test_states/test_track_objects"

beforeAll(() => {
  setupTestStore(testJson3DTracking)

  Session.images.length = 0
  dispatch(action.loadItem(0, -1))
  dispatch(action.addViewerConfig(0, makePointCloudViewerConfig(0)))
  dispatch(action.goToItem(0))
})

beforeEach(() => {
  setupTestStore(testJson3DTracking)
})

test("Add new valid 3d box drawable track", () => {
  dispatch(action.goToItem(0))
  const state = getState()
  expect(_.size(state.task.items[0].labels)).toEqual(0)
  expect(_.size(state.task.tracks)).toEqual(0)
  const label3dList = new Label3DList()
  const label = createBox3dLabel(
    label3dList,
    0,
    [0],
    0,
    new Vector3D(10, 0, 0),
    new Vector3D(2, 2, 2),
    new Vector3D()
  )
  expect(label).not.toBeNull()
  if (label !== null) {
    commitLabels([label], true)

    const currentState = getState()
    expect(_.size(currentState.task.items[0].labels)).toEqual(1)
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
    expect(_.size(currentState.task.tracks)).toEqual(1)
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

test("Add new valid 3d plane drawable track", () => {
  dispatch(action.goToItem(0))
  const state = getState()
  expect(_.size(state.task.items[0].labels)).toEqual(0)
  expect(_.size(state.task.tracks)).toEqual(0)
  const label3dList = new Label3DList()
  const label = createPlaneLabel(
    label3dList,
    0,
    0,
    new Vector3D(10, 0, 0),
    new Vector3D(),
    [0]
  )
  expect(label).not.toBeNull()
  if (label !== null) {
    commitLabels([label], true)

    const currentState = getState()
    expect(_.size(currentState.task.items[0].labels)).toEqual(1)
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
    expect(_.size(currentState.task.tracks)).toEqual(1)
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
