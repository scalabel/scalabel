import fs from "fs-extra"

import * as action from "../../src/action/common"
import Session from "../../src/common/session"
import { makeLabel } from "../../src/functional/states"
import { setupTestStore } from "../components/util"
import { testJson } from "../test_states/test_track_objects"

const getState = Session.getState.bind(Session)
const dispatch = Session.dispatch.bind(Session)
const data = JSON.parse(
  fs.readFileSync("./app/test/test_states/sample_state.json", "utf8")
)

beforeEach(() => {
  setupTestStore(testJson)

  Session.subscribe(() => {
    Session.label2dList.updateState(getState())
  })
})

beforeAll(() => {
  setupTestStore(testJson)
  Session.images.length = 0
  Session.images.push({ [-1]: new Image(1000, 1000) })
  for (let i = 0; i < getState().task.items.length; i++) {
    dispatch(action.loadItem(i, -1))
  }
})

describe("Test link labels", () => {
  test("Link labels without track", () => {
    setupTestStore(data)
    dispatch(action.goToItem(1))
    let label1 = makeLabel()
    let label2 = makeLabel()
    const labelId1 = label1.id
    const labelId2 = label2.id
    dispatch(action.addLabel(1, label1))
    dispatch(action.addLabel(1, label2))
    let state = getState()
    label1 = state.task.items[1].labels[labelId1]
    label2 = state.task.items[1].labels[labelId2]
    expect(label1.parent).toBe("")
    expect(label2.parent).toBe("")
    dispatch(action.linkLabels(1, [labelId1, labelId2]))
    state = getState()
    label1 = state.task.items[1].labels[labelId1]
    label2 = state.task.items[1].labels[labelId2]
    expect(label1.parent).not.toBe("")
    expect(label2.parent).not.toBe("")
    expect(label1.parent).toBe(label2.parent)
    const parentLabel = state.task.items[1].labels[label1.parent]
    expect(parentLabel).not.toBeUndefined()
    expect(parentLabel.children.length).toBe(2)
    expect(parentLabel.children).toContain(labelId1)
    expect(parentLabel.children).toContain(labelId2)
  })

  test("Link labels with multiple tracks", () => {
    dispatch(action.goToItem(1))
    let state = getState()
    let item = state.task.items[1]
    let label1 = item.labels["24"]
    let label2 = item.labels["47"]
    const track1 = label1.track
    const track1Labels = state.task.tracks[track1].labels
    const track2 = label2.track
    const track2Labels = state.task.tracks[track2].labels
    dispatch(action.linkLabels(1, ["24", "47"]))
    state = getState()
    expect(state.task.tracks[track2]).toBeUndefined()
    expect(state.task.tracks[track1]).not.toBeUndefined()

    item = state.task.items[1]
    label1 = item.labels["24"]
    label2 = item.labels["47"]
    expect(label1.track).toBe(label2.track)
    expect(label1.parent).toBe(label2.parent)
    const parentLabelInItem1 = item.labels[label1.parent]
    const newTrackLabels = state.task.tracks[track1].labels
    expect(newTrackLabels[1]).toBe(parentLabelInItem1.id)

    for (let i = 0; i < state.task.items.length; i++) {
      if (i in track1Labels) {
        expect(newTrackLabels[i]).not.toEqual(track1Labels[i])
        label1 = state.task.items[i].labels[track1Labels[i]]
        expect(newTrackLabels[i]).toEqual(label1.parent)
      }
      if (i in track2Labels) {
        expect(newTrackLabels[i]).not.toEqual(track2Labels[i])
        label2 = state.task.items[i].labels[track2Labels[i]]
        expect(newTrackLabels[i]).toEqual(label2.parent)
      }
    }
  })
})
