import _ from 'lodash'
import React from 'react'
import * as action from '../../js/action/common'
import Session from '../../js/common/session'
import { initStore } from '../../js/common/session_init'
import { Label2dCanvas } from '../../js/components/label2d_canvas'
// import { TrackCollector } from '../server/util/track_collector'
import { emptyTrackingTask } from '../test_states/test_track_objects'
import { drawBox2DTracks, keyClick, mouseClick, setUpLabel2dCanvas } from './label2d_canvas_util'

const canvasRef: React.RefObject<Label2dCanvas> = React.createRef()

const store = Session.getSimpleStore()
const getState = store.getter()
const dispatch = store.dispatcher()

beforeEach(() => {
  expect(canvasRef.current).not.toBeNull()
  canvasRef.current?.clear()
  initStore(emptyTrackingTask)
  Session.subscribe(() => {
    Session.label2dList.updateState(getState())
    canvasRef.current?.updateState(getState())
  })
})

beforeAll(() => {
  Session.devMode = false
  initStore(emptyTrackingTask)
  Session.images.length = 0
  Session.images.push({ [-1]: new Image(1000, 1000) })
  // mock loading every item to make sure the canvas can be successfully
  // initialized
  for (let i = 0; i < getState().task.items.length; i++) {
    dispatch(action.loadItem(i, -1))
  }
  setUpLabel2dCanvas(dispatch, canvasRef, 1000, 1000)
})

test('Basic track operations', () => {
  const label2d = canvasRef.current as Label2dCanvas
  const numItems = getState().task.items.length

  const itemIndices = [0, 2, 4, 6]
  const numLabels = [1, 1, 2, 2, 3, 3, 4, 4]
  const boxes = [
    [1, 1, 50, 50],
    [19, 20, 30, 29],
    [100, 20, 80, 100],
    [500, 500, 80, 100]
  ]

  // test adding tracks
  const trackIds = drawBox2DTracks(label2d, store, itemIndices, boxes)
  let state = getState()
  expect(_.size(state.task.tracks)).toEqual(4)
  itemIndices.forEach((itemIndex, i) => {
    expect(_.size(state.task.tracks[trackIds[i]].labels)).toEqual(
      numItems - itemIndex)
    state.task.items.forEach((item, index) => {
      expect(_.size(item.labels)).toEqual(numLabels[index])
      expect(_.size(item.shapes)).toEqual(numLabels[index])
    })
  })

  // Terminate the track by key
  dispatch(action.goToItem(1))
  mouseClick(label2d, 1, 30)
  keyClick(label2d, ['Control', 'E'])
  state = getState()
  expect(_.size(state.task.tracks[trackIds[0]].labels)).toEqual(2)
})

// test('Terminate track by key', () => {
//   initStore(testJson)
//   const label2d = canvasRef.current as Label2dCanvas
//   Session.dispatch(action.goToItem(1))
//   let state = Session.getState()
//   const trackLabels = state.task.tracks[3].labels
//   const lblInItm2 = trackLabels[2]
//   const lblInItm3 = trackLabels[3]
//   const lblInItm4 = trackLabels[4]
//   const lblInItm5 = trackLabels[5]
//   expect(_.size(state.task.tracks[3].labels)).toBe(6)
//   expect(_.size(state.task.items[2].labels)).toBe(3)
//   expect(_.size(state.task.items[2].shapes)).toBe(3)
//   label2d.onMouseDown(mouseDownEvent(835, 314))
//   label2d.onMouseUp(mouseUpEvent(835, 314))
//   label2d.onKeyDown(keyDownEvent('Control'))
//   label2d.onKeyDown(keyDownEvent('E'))
//   label2d.onKeyUp(keyUpEvent('E'))
//   label2d.onKeyUp(keyUpEvent('Control'))

//   state = Session.getState()
//   expect(_.size(state.task.tracks[3].labels)).toBe(2)
//   expect(state.task.items[2].labels[lblInItm2]).toBeUndefined()
//   expect(state.task.items[2].labels[lblInItm3]).toBeUndefined()
//   expect(state.task.items[2].labels[lblInItm4]).toBeUndefined()
//   expect(state.task.items[2].labels[lblInItm5]).toBeUndefined()
// })

// test('Merge track by key', () => {
//   initStore(testJson)
//   const label2d = canvasRef.current as Label2dCanvas
//   Session.dispatch(action.goToItem(3))
//   let state = Session.getState()
//   expect(_.size(state.task.tracks[2].labels)).toBe(4)
//   expect(_.size(state.task.tracks[9].labels)).toBe(1)
//   expect(state.task.items[5].labels[203].track).toEqual(9)

//   label2d.onMouseDown(mouseDownEvent(925, 397))
//   label2d.onMouseUp(mouseUpEvent(925, 397))
//   label2d.onKeyDown(keyDownEvent('Control'))
//   label2d.onKeyDown(keyDownEvent('L'))
//   label2d.onKeyUp(keyUpEvent('L'))
//   label2d.onKeyUp(keyUpEvent('Control'))

//   Session.dispatch(action.goToItem(5))
//   label2d.onMouseDown(mouseDownEvent(931, 300))
//   label2d.onMouseUp(mouseUpEvent(931, 300))
//   label2d.onKeyDown(keyDownEvent('Enter'))
//   label2d.onKeyUp(keyUpEvent('Enter'))

//   state = Session.getState()
//   expect(_.size(state.task.tracks[2].labels)).toBe(5)
//   expect(state.task.tracks[9]).toBeUndefined()
//   expect(state.task.items[5].labels[203].track).toEqual(2)
// })
