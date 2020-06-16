import _ from 'lodash'
import * as action from '../../js/action/common'
import { changeSelectedLabelsCategories } from '../../js/action/select'
import * as track from '../../js/action/track'
import Session from '../../js/common/session'
import {
  getCategory, getLabelInTrack, getNumItems, getNumLabels, getNumLabelsForTrack,
  getNumLabelsForTrackId,
  getNumShapes, getNumTracks, getSelectedLabels, getTrack
} from '../../js/functional/state_util'
import { makeLabel, makeShape } from '../../js/functional/states'
import { IdType } from '../../js/functional/types'
import { setupTestStore } from '../components/util'
import { findNewTracksFromState } from '../server/util/util'
import { testJson } from '../test_states/test_track_objects'

const getState = Session.getState.bind(Session)
const dispatch = Session.dispatch.bind(Session)
let originalNumTracks: number

beforeEach(() => {
  setupTestStore(testJson)
  Session.images.length = 0
  Session.images.push({ [-1]: new Image(1000, 1000) })
  for (let i = 0; i < getState().task.items.length; i++) {
    dispatch(action.loadItem(i, -1))
  }
  originalNumTracks = 4 // The original number of tracks in the test file
})

/**
 * Helper function to change the category of the label
 * @param itemIndex
 * @param labelId
 * @param category
 */
function changeCategory (itemIndex: number, labelId: IdType, category: number) {
  // First select the label
  dispatch(action.goToItem(itemIndex))
  dispatch(action.changeSelect({ labels: { [itemIndex]: [labelId] } }))

  // Then change its category
  dispatch(action.changeSelect({ category }))
  dispatch(changeSelectedLabelsCategories(getState(), [category]))
}

describe('Test tracking operations', () => {
  test('Track termination', () => {
    // Check the initial state
    const itemIndex = 1
    dispatch(action.goToItem(itemIndex))
    let state = getState()
    expect(getNumLabels(state, 2)).toBe(3)
    expect(getNumShapes(state, 2)).toBe(3)

    // Terminate a track
    const trackId = '3'
    let track3 = getTrack(state, trackId)
    expect(getNumLabelsForTrack(track3)).toBe(6)
    dispatch(
      track.terminateTracks([track3],
        itemIndex, getNumItems(state)))

    // Check that the track was terminated
    state = getState()
    track3 = getTrack(state, trackId)
    expect(getNumLabelsForTrack(track3)).toBe(1)
    expect(getNumLabels(state, 1)).toBe(2)
    expect(getNumShapes(state, 1)).toBe(2)
    expect(getNumLabels(state, 0)).toBe(3)
    expect(getNumShapes(state, 0)).toBe(3)
  })

  test('Track linking', () => {
    // Check initial state
    const toMergeTrack1 = '2' // Has labels at items 0-3
    const toMergeTrack2 = '9' // Has a label at item 5
    const continueItemIdx = 5
    let state = getState()
    expect(getNumTracks(state)).toBe(originalNumTracks)
    const labelId = getLabelInTrack(state, toMergeTrack2, continueItemIdx)

    dispatch(action.mergeTracks([toMergeTrack1, toMergeTrack2]))

    // Check that the tracks were merged
    // The 1st track should have the 2nd track's label
    state = getState()
    expect(getNumTracks(state)).toBe(originalNumTracks - 1)
    expect(
      getLabelInTrack(state, toMergeTrack1, continueItemIdx)).toBe(labelId)
  })

  test('Selecting after linking', () => {
    // First merge the tracks
    const track1 = '2' // Has labels at items 0-3
    const track2 = '9' // Has a label at item 5
    expect(getNumTracks(getState())).toBe(originalNumTracks)
    dispatch(action.mergeTracks([track1, track2]))

    // Get the label ids of the merged track
    const state = getState()
    const labelId2 = getLabelInTrack(state, track1, 2)
    const labelId3 = getLabelInTrack(state, track1, 3)
    const labelId5 = getLabelInTrack(state, track1, 5)

    // Select the label in the track at item 2
    dispatch(action.goToItem(2))
    dispatch(action.changeSelect({ labels: { 2: [labelId2] } }))

    // Test that the selection is maintained while going through items
    expect(getSelectedLabels(getState())).toStrictEqual({ 2: [labelId2] })
    dispatch(action.goToItem(3))
    expect(getSelectedLabels(getState())).toStrictEqual({ 3: [labelId3] })
    dispatch(action.goToItem(4))
    expect(getSelectedLabels(getState())).toStrictEqual({})
    dispatch(action.goToItem(5))
    expect(getSelectedLabels(getState())).toStrictEqual({ 5: [labelId5] })
  })

  test('Changing category after linking', () => {
    // First merge the tracks
    const track1 = '2' // Has labels at items 0-3
    const track1ItemIndex = 3
    const track2 = '9' // Has a label at item 5
    const track2ItemIndex = 5
    expect(getNumTracks(getState())).toBe(originalNumTracks)
    dispatch(action.mergeTracks([track1, track2]))

    // Check the initial categories
    let state = getState()
    const track1LabelId = getLabelInTrack(state, track1, track1ItemIndex)
    const track2LabelId = getLabelInTrack(state, track1, track2ItemIndex)
    const initialCategory = 0
    expect(getCategory(state, track1ItemIndex, track1LabelId)).
      toStrictEqual([initialCategory])
    expect(getCategory(state, track2ItemIndex, track2LabelId))
      .toStrictEqual([initialCategory])

    const newCategory = 1
    changeCategory(track1ItemIndex, track1LabelId, newCategory)

    // Check category of labels in merged track
    state = getState()
    expect(getCategory(state, track1ItemIndex, track1LabelId)).
      toStrictEqual([newCategory])
    expect(getCategory(state, track2ItemIndex, track2LabelId)).
      toStrictEqual([newCategory])
  })

  test('Linking with different categories', () => {
    // Check the initial categories
    const track1 = '2' // Has labels at items 0-3
    const track1ItemIndex = 3
    const track2 = '9' // Has a label at item 5
    const track2ItemIndex = 5

    let state = getState()
    expect(getNumTracks(getState())).toBe(originalNumTracks)
    const track1LabelId = getLabelInTrack(state, track1, track1ItemIndex)
    const track2LabelId = getLabelInTrack(state, track2, track2ItemIndex)
    const initialCategory = 0
    expect(getCategory(state, track1ItemIndex, track1LabelId)).
      toStrictEqual([initialCategory])
    expect(getCategory(state, track2ItemIndex, track2LabelId)).
      toStrictEqual([initialCategory])

    // Change the category for the first track
    const newCategory = 1
    changeCategory(track1ItemIndex, track1LabelId, newCategory)
    state = getState()
    expect(getCategory(state, track1ItemIndex, track1LabelId)).
      toStrictEqual([newCategory])
    expect(getCategory(state, track2ItemIndex, track2LabelId)).
      toStrictEqual([initialCategory])

    // Merge the tracks
    dispatch(action.mergeTracks([track1, track2]))
    state = getState()
    expect(getCategory(state, track1ItemIndex, track1LabelId)).
      toStrictEqual([newCategory])
    expect(getCategory(state, track2ItemIndex, track2LabelId)).
      toStrictEqual([newCategory])
  })

  test('Linking single frame tracks', () => {
    const originalTrackIds = ['1', '2', '3', '9']
    const newTrackIds = []

    dispatch(action.goToItem(0))
    for (let itemIndex = 0; itemIndex < 5; itemIndex++) {
      // Start and terminate a track on succesive frames
      const range = _.range(itemIndex, 6)
      const labels = range.map(() => makeLabel({ track: `id${itemIndex}` }))
      const shapes = range.map(() => [makeShape()])
      dispatch(action.addTrack(
        range, '', labels, shapes
      ))
      dispatch(action.goToItem(itemIndex + 1))

      const trackId: string = findNewTracksFromState(getState(),
        newTrackIds.concat(originalTrackIds))[0]
      newTrackIds.push(trackId)
      const currentTrack = getTrack(getState(), trackId)
      expect(getNumLabelsForTrackId(getState(), trackId)).toBe(6 - itemIndex)
      dispatch(
        track.terminateTracks([currentTrack],
          itemIndex + 1, getNumItems(getState())))
      expect(getNumLabelsForTrackId(getState(), trackId)).toBe(1)
    }
    expect(getNumTracks(getState())).toBe(originalNumTracks + 5)

    // Test linking all the 1 frame tracks
    dispatch(action.mergeTracks(newTrackIds))

    const state = getState()
    expect(getNumTracks(state)).toBe(originalNumTracks + 1)
    expect(getNumLabelsForTrackId(state, newTrackIds[0])).toBe(5)
  })
})
