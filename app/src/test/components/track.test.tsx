import { fireEvent, render } from '@testing-library/react'
import _ from 'lodash'
import React from 'react'
import * as action from '../../js/action/common'
import { selectLabel } from '../../js/action/select'
import Session from '../../js/common/session'
import { initStore } from '../../js/common/session_init'
import { Label2dCanvas } from '../../js/components/label2d_canvas'
import { ToolBar } from '../../js/components/toolbar'
import { Attribute } from '../../js/functional/types'
// import { TrackCollector } from '../server/util/track_collector'
import { emptyTrackingTask } from '../test_states/test_track_objects'
import { drawBox2DTracks, mouseMoveClick, setUpLabel2dCanvas } from './label2d_canvas_util'

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
        categories={emptyTrackingTask.task.config.categories}
        attributes={emptyTrackingTask.task.config.attributes as Attribute[]}
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
})
