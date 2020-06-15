import _ from 'lodash'
import * as action from '../../js/action/common'
import * as track from '../../js/action/track'
import Session from '../../js/common/session'
import { initStore } from '../../js/common/session_init'
import { testJson } from '../test_states/test_track_objects'

const getState = Session.getState.bind(Session)
const dispatch = Session.dispatch.bind(Session)

beforeAll(() => {
  Session.devMode = false
  initStore(testJson)
  Session.images.length = 0
  Session.images.push({ [-1]: new Image(1000, 1000) })
  for (let i = 0; i < getState().task.items.length; i++) {
    dispatch(action.loadItem(i, -1))
  }
})

test('Test tracks ops', () => {
  // Terminate tracks
  const itemIndex = 1
  Session.dispatch(action.goToItem(itemIndex))
  let state = Session.getState()
  expect(_.size(state.task.items[2].labels)).toBe(3)
  expect(_.size(state.task.items[2].shapes)).toBe(3)
  const trackIdx = 3
  expect(_.size(state.task.tracks[trackIdx].labels)).toBe(6)
  Session.dispatch(
    track.terminateTracks([state.task.tracks[trackIdx]],
      itemIndex, _.size(state.task.items)))
  state = Session.getState()
  expect(_.size(state.task.tracks[trackIdx].labels)).toBe(1)
  expect(_.size(state.task.items[1].labels)).toBe(2)
  expect(_.size(state.task.items[1].shapes)).toBe(2)
  expect(_.size(state.task.items[0].labels)).toBe(3)
  expect(_.size(state.task.items[0].shapes)).toBe(3)

  // Merge tracks
  const toMergeTrack1 = '2'
  const toMergeTrack2 = '9'
  const continueItemIdx = 5
  expect(_.size(state.task.tracks)).toBe(4)
  const labelId = state.task.tracks[toMergeTrack2].labels[continueItemIdx]
  Session.dispatch(action.mergeTracks([toMergeTrack1, toMergeTrack2]))
  state = Session.getState()
  expect(_.size(state.task.tracks)).toBe(3)
  expect(state.task.tracks[toMergeTrack1].labels[continueItemIdx]).toBe(labelId)
  // ^^ make these have different categories before merging

  // Test selection is maintained
  // The merged track has labels at items 0-3, and 5
  Session.dispatch(action.goToItem(2))
  state = getState()
  const labelId2 = state.task.tracks[toMergeTrack1].labels[2]
  const labelId3 = state.task.tracks[toMergeTrack1].labels[3]
  const labelId5 = state.task.tracks[toMergeTrack1].labels[5]
  Session.dispatch(action.changeSelect(
    { labels: { 2: [labelId2] } }
  ))
  expect(getState().user.select.labels).toStrictEqual({ 2: [labelId2] })
  Session.dispatch(action.goToItem(3))
  expect(getState().user.select.labels).toStrictEqual({ 3: [labelId3] })
  Session.dispatch(action.goToItem(4))
  expect(getState().user.select.labels).toStrictEqual({})
  Session.dispatch(action.goToItem(5))
  expect(getState().user.select.labels).toStrictEqual({ 5: [labelId5] })
})
