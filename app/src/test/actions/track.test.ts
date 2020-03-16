import _ from 'lodash'
import * as action from '../../js/action/common'
import * as track from '../../js/action/track'
import Session from '../../js/common/session'
import { initStore, loadImages } from '../../js/common/session_init'
import { testJson } from '../test_track_objects'

test('Test tracks ops', () => {
  Session.devMode = false
  initStore(testJson)
  loadImages()

  // terminate tracks
  const itemIndex = 1
  Session.dispatch(action.goToItem(itemIndex))
  let state = Session.getState()
  expect(_.size(state.task.items[2].labels)).toBe(3)
  expect(_.size(state.task.items[2].shapes)).toBe(3)
  const trackIdx = 3
  expect(_.size(state.task.tracks[trackIdx].labels)).toBe(6)
  Session.dispatch(
    track.terminateTracks([state.task.tracks[trackIdx]], itemIndex))
  state = Session.getState()
  expect(_.size(state.task.tracks[trackIdx].labels)).toBe(2)
  expect(_.size(state.task.items[2].labels)).toBe(2)
  expect(_.size(state.task.items[2].shapes)).toBe(2)

  // merge tracks
  const toMergeTrack1 = 2
  const toMergeTrack2 = 9
  const continueItemIdx = 5
  expect(_.size(state.task.tracks)).toBe(4)
  const labelId = state.task.tracks[toMergeTrack2].labels[continueItemIdx]
  Session.dispatch(action.mergeTracks([toMergeTrack1, toMergeTrack2]))
  state = Session.getState()
  expect(_.size(state.task.tracks)).toBe(3)
  expect(state.task.tracks[toMergeTrack1].labels[continueItemIdx]).toBe(labelId)
})
