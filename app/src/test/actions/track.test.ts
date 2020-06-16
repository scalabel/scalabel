import _ from 'lodash'
import * as action from '../../js/action/common'
import * as track from '../../js/action/track'
import Session from '../../js/common/session'
import { setupTestStore } from '../components/util'
import { testJson } from '../test_states/test_track_objects'

const getState = Session.getState.bind(Session)
const dispatch = Session.dispatch.bind(Session)

beforeAll(() => {
  Session.images.length = 0
  Session.images.push({ [-1]: new Image(1000, 1000) })
  for (let i = 0; i < getState().task.items.length; i++) {
    dispatch(action.loadItem(i, -1))
  }
})

test('Test tracks ops', () => {
  setupTestStore(testJson)

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
})
