import { fireEvent, render } from '@testing-library/react'
import _ from 'lodash'
import React from 'react'
import * as action from '../../js/action/common'
import { selectLabel } from '../../js/action/select'
import Session from '../../js/common/session'
import { updateTracks } from '../../js/common/session_setup'
import { Label2dCanvas } from '../../js/components/label2d_canvas'
import { ToolBar } from '../../js/components/toolbar'
import { getShape } from '../../js/functional/state_util'
import { RectType, State } from '../../js/functional/types'
// Import { TrackCollector } from '../server/util/track_collector'
import { emptyTrackingTask } from '../test_states/test_track_objects'
import { drag, drawBox2DTracks, mouseMoveClick, setUpLabel2dCanvas } from './label2d_canvas_util'
import { setupTestStore } from './util'

const canvasRef: React.RefObject<Label2dCanvas> = React.createRef()

const store = Session.getSimpleStore()
const getState = store.getter()
const dispatch = store.dispatcher()

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

describe('basic track ops', () => {
  test('Adding and deleting tracks', () => {
    const label2d = canvasRef.current as Label2dCanvas
    const numItems = getState().task.items.length

    const toolbarRef: React.Ref<ToolBar> = React.createRef()
    const { getByText } = render(
      <ToolBar
        ref={toolbarRef}
        categories={null}
        attributes={[]}
        labelType={'labelType'}
      />
    )
    expect(toolbarRef.current).not.toBeNull()
    expect(toolbarRef.current).not.toBeUndefined()
    if (toolbarRef.current) {
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

    // Terminate the track by button
    dispatch(action.goToItem(2))
    mouseMoveClick(label2d, 1, 30)
    fireEvent(
      getByText('Delete'),
      new MouseEvent('click', {
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
    Session.dispatch(selectLabel(
      state.user.select.labels, 6,
      state.task.tracks[trackIds[3]].labels[6]))
    fireEvent(
      getByText('Delete'),
      new MouseEvent('click', {
        bubbles: true,
        cancelable: true
      })
    )
    state = getState()
    expect(_.size(state.task.items[6].labels)).toEqual(2)
    expect(_.size(state.task.tracks)).toEqual(3)

    // Terminate the track by key
    dispatch(action.goToItem(1))
    Session.dispatch(selectLabel(
      state.user.select.labels, 1,
      state.task.tracks[trackIds[0]].labels[1]))
    fireEvent.keyDown(document, { key: 'Backspace' })
    state = getState()
    expect(_.size(state.task.items[1].labels)).toEqual(0)
    expect(_.size(state.task.items[0].labels)).toEqual(1)
    expect(_.size(state.task.tracks[trackIds[0]].labels)).toEqual(1)

    // Delete the track by key
    dispatch(action.goToItem(0))
    Session.dispatch(selectLabel(
      state.user.select.labels, 0,
      state.task.tracks[trackIds[0]].labels[0]))
    fireEvent.keyDown(document, { key: 'Backspace' })
    state = getState()
    expect(_.size(state.task.items[0].labels)).toEqual(0)
    expect(_.size(state.task.tracks)).toEqual(2)
  })

  test('Linking tracks', () => {
    const label2d = canvasRef.current as Label2dCanvas

    const toolbarRef: React.Ref<ToolBar> = React.createRef()
    const { getAllByText } = render(
      <ToolBar
        ref={toolbarRef}
        categories={null}
        attributes={[]}
        labelType={'labelType'}
      />
    )
    expect(toolbarRef.current).not.toBeNull()
    expect(toolbarRef.current).not.toBeUndefined()
    if (toolbarRef.current) {
      toolbarRef.current.componentDidMount()
    }

    const itemIndices = [0, 2, 4, 6]
    const boxes = [
      [1, 1, 50, 50],
      [19, 20, 30, 29],
      [100, 20, 80, 100],
      [500, 500, 80, 100]
    ]

    const trackIds = drawBox2DTracks(label2d, store, itemIndices, boxes)

    // Terminate the track by button
    let state = getState()
    dispatch(action.goToItem(2))
    Session.dispatch(selectLabel(
      state.user.select.labels, 2,
      state.task.tracks[trackIds[0]].labels[2]))
    fireEvent(
      getAllByText('Delete')[0],
      new MouseEvent('click', {
        bubbles: true,
        cancelable: true
      })
    )
    dispatch(action.goToItem(1))
    state = getState()
    Session.dispatch(selectLabel(
      state.user.select.labels, 1,
      state.task.tracks[trackIds[0]].labels[1]))
    fireEvent(
      getAllByText('Track-Link')[0],
      new MouseEvent('click', {
        bubbles: true,
        cancelable: true
      })
    )

    dispatch(action.goToItem(4))
    state = getState()
    Session.dispatch(selectLabel(
      state.user.select.labels, 4,
      state.task.tracks[trackIds[2]].labels[4]))
    fireEvent(
      getAllByText('Finish Track-Link')[0],
      new MouseEvent('click', {
        bubbles: true,
        cancelable: true
      })
    )
    state = getState()
    expect(_.size(state.task.tracks)).toEqual(3)
  })

  test('Changing attributes and categories of tracks', () => {
    const label2d = canvasRef.current as Label2dCanvas

    const toolbarRef: React.Ref<ToolBar> = React.createRef()
    const { getByText, getAllByRole } = render(
      <ToolBar
        ref={toolbarRef}
        categories={(emptyTrackingTask as State).task.config.categories}
        attributes={(emptyTrackingTask as State).task.config.attributes}
        labelType={'labelType'}
      />
    )
    expect(toolbarRef.current).not.toBeNull()
    expect(toolbarRef.current).not.toBeUndefined()
    if (toolbarRef.current) {
      toolbarRef.current.componentDidMount()
    }

    const itemIndices = [0]
    const boxes = [
      [1, 1, 50, 50]
    ]

    const trackIds = drawBox2DTracks(label2d, store, itemIndices, boxes)

    // Changing category
    dispatch(action.goToItem(2))
    mouseMoveClick(label2d, 1, 30)
    fireEvent(
      getByText('car'),
      new MouseEvent('click', {
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
    const switchBtn = getAllByRole('checkbox')[0]
    switchBtn.click()
    state = getState()
    expect(state.task.items[2].labels[labelIdIn2].attributes[0]).toEqual([1])
    expect(state.task.items[3].labels[labelIdIn3].attributes[0]).toEqual([1])
    expect(state.task.items[2].labels[labelIdIn2].attributes[1]).toEqual([0])
    expect(state.task.items[2].labels[labelIdIn2].attributes[2]).toEqual([0])
  })

  test('Changing shapes and locations of tracks', () => {
    const label2d = canvasRef.current as Label2dCanvas

    const itemIndices = [0, 2]
    const boxes = [
      [10, 20, 50, 60],
      [100, 110, 200, 300]
    ]

    const trackIds = drawBox2DTracks(label2d, store, itemIndices, boxes)

    // Changing shape
    dispatch(action.goToItem(4))
    drag(label2d, 10, 20, 15, 25)
    let state = getState()
    // Shapes starting from item 4 should change
    for (let i = 4; i < 8; ++i) {
      const labelIdInk = state.task.tracks[trackIds[0]].labels[i]
      const rect = getShape(state, i, labelIdInk, 0) as RectType
      expect(rect.x1).toEqual(15)
      expect(rect.y1).toEqual(25)
      expect(rect.x2).toEqual(50)
      expect(rect.y2).toEqual(60)
    }
    // Shapes before item 4 should not change
    for (let i = 0; i < 4; ++i) {
      const labelIdInk = state.task.tracks[trackIds[0]].labels[i]
      const rect = getShape(state, i, labelIdInk, 0) as RectType
      expect(rect.x1).toEqual(10)
      expect(rect.y1).toEqual(20)
      expect(rect.x2).toEqual(50)
      expect(rect.y2).toEqual(60)
    }
    dispatch(action.goToItem(6))
    drag(label2d, 50, 60, 55, 65)
    state = getState()
    // Shapes should change only between item 6 to 8
    for (let i = 6; i < 8; ++i) {
      const labelIdInk = state.task.tracks[trackIds[0]].labels[i]
      const rect = getShape(state, i, labelIdInk, 0) as RectType
      expect(rect.x1).toEqual(15)
      expect(rect.y1).toEqual(25)
      expect(rect.x2).toEqual(55)
      expect(rect.y2).toEqual(65)
    }
    for (let i = 4; i < 6; ++i) {
      const labelIdInk = state.task.tracks[trackIds[0]].labels[i]
      const rect = getShape(state, i, labelIdInk, 0) as RectType
      expect(rect.x1).toEqual(15)
      expect(rect.y1).toEqual(25)
      expect(rect.x2).toEqual(50)
      expect(rect.y2).toEqual(60)
    }
    // Changing location
    dispatch(action.goToItem(5))
    drag(label2d, 110, 110, 210, 210)  // (100, 100) translation
    state = getState()
    // Locations starting from item 5 should change
    for (let i = 5; i < 8; ++i) {
      const labelIdInk = state.task.tracks[trackIds[1]].labels[i]
      const rect = getShape(state, i, labelIdInk, 0) as RectType
      expect(rect.x1).toEqual(200)
      expect(rect.y1).toEqual(210)
      expect(rect.x2).toEqual(300)
      expect(rect.y2).toEqual(400)
    }
    // Locations before item 5 should not change
    for (let i = 2; i < 5; ++i) {
      const labelIdInk = state.task.tracks[trackIds[1]].labels[i]
      const rect = getShape(state, i, labelIdInk, 0) as RectType
      expect(rect.x1).toEqual(100)
      expect(rect.y1).toEqual(110)
      expect(rect.x2).toEqual(200)
      expect(rect.y2).toEqual(300)
    }

  })
})
