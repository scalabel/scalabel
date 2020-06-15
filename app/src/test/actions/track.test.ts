import _ from 'lodash'
import * as action from '../../js/action/common'
import { changeSelectedLabelsCategories } from '../../js/action/select'
import * as track from '../../js/action/track'
import Session from '../../js/common/session'
import { initStore } from '../../js/common/session_init'
import {
  getCategory, getLabelInTrack, getNumItems, getNumLabels, getNumLabelsForTrack,
  getNumShapes, getNumTracks, getSelectedLabels, getTrack
} from '../../js/functional/state_util'
import { IdType } from '../../js/functional/types'
import { testJson } from '../test_states/test_track_objects'

const getState = Session.getState.bind(Session)
const dispatch = Session.dispatch.bind(Session)

beforeEach(() => {
  Session.devMode = false
  initStore(testJson)
  Session.images.length = 0
  Session.images.push({ [-1]: new Image(1000, 1000) })
  for (let i = 0; i < getState().task.items.length; i++) {
    dispatch(action.loadItem(i, -1))
  }
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

  test('Track merging', () => {
    // Check initial state
    const toMergeTrack1 = '2' // Has labels at items 0-3
    const toMergeTrack2 = '9' // Has a label at item 5
    const continueItemIdx = 5
    let state = getState()
    expect(getNumTracks(state)).toBe(4)
    const labelId = getLabelInTrack(state, toMergeTrack2, continueItemIdx)

    dispatch(action.mergeTracks([toMergeTrack1, toMergeTrack2]))

    // Check that the tracks were merged
    // The 1st track should have the 2nd track's label
    state = getState()
    expect(getNumTracks(state)).toBe(3)
    expect(
      getLabelInTrack(state, toMergeTrack1, continueItemIdx)).toBe(labelId)
  })

  test('Selecting after merging', () => {
    // First merge the tracks
    const track1 = '2' // Has labels at items 0-3
    const track2 = '9' // Has a label at item 5
    expect(getNumTracks(getState())).toBe(4)
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

  test('Changing category after merging', () => {
    // First merge the tracks
    const track1 = '2' // Has labels at items 0-3
    const track2 = '9' // Has a label at item 5
    expect(getNumTracks(getState())).toBe(4)
    dispatch(action.mergeTracks([track1, track2]))

    // Check the initial categories
    let state = getState()
    const labelId3 = getLabelInTrack(state, track1, 3)
    const labelId5 = getLabelInTrack(state, track1, 5)
    const initialCategory = 0
    expect(getCategory(state, 3, labelId3)).toStrictEqual([initialCategory])
    expect(getCategory(state, 5, labelId5)).toStrictEqual([initialCategory])

    const newCategory = 1
    changeCategory(3, labelId3, newCategory)

    // Check category of labels in merged track
    state = getState()
    expect(getCategory(state, 3, labelId3)).toStrictEqual([newCategory])
    expect(getCategory(state, 5, labelId5)).toStrictEqual([newCategory])
  })

  test('Merging with different categories', () => {
    // Check the initial categories
    const track1 = '2' // Has labels at items 0-3
    const track2 = '9' // Has a label at item 5

    let state = getState()
    expect(getNumTracks(getState())).toBe(4)
    const labelId3 = getLabelInTrack(state, track1, 3)
    const labelId5 = getLabelInTrack(state, track2, 5)
    const initialCategory = 0
    expect(getCategory(state, 3, labelId3)).toStrictEqual([initialCategory])
    expect(getCategory(state, 5, labelId5)).toStrictEqual([initialCategory])

    // Change the category for the first track
    const newCategory = 1
    changeCategory(3, labelId3, newCategory)
    state = getState()
    expect(getCategory(state, 3, labelId3)).toStrictEqual([newCategory])
    expect(getCategory(state, 5, labelId5)).toStrictEqual([initialCategory])

    // Merge the tracks
    dispatch(action.mergeTracks([track1, track2]))
    state = getState()
    expect(getCategory(state, 3, labelId3)).toStrictEqual([newCategory])
    expect(getCategory(state, 5, labelId5)).toStrictEqual([newCategory])
  })
})
