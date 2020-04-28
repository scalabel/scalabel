import { render } from '@testing-library/react'
import _ from 'lodash'
import React from 'react'
import * as action from '../../js/action/common'
import Session from '../../js/common/session'
import { initStore } from '../../js/common/session_init'
import { Label2dCanvas } from '../../js/components/label2d_canvas'
import { makeImageViewerConfig } from '../../js/functional/states'
import { TrackCollector } from '../server/util/track_collector'
import { emptyTrackingTask, testJson } from '../test_states/test_track_objects'
import { drawBox2D } from './label2d_canvas_util'

const canvasRef: React.RefObject<Label2dCanvas> = React.createRef()

beforeEach(() => {
  expect(canvasRef.current).not.toBeNull()
  canvasRef.current?.clear()
})

beforeAll(() => {
  Session.devMode = false
  Session.subscribe(() => Session.label2dList.updateState(Session.getState()))
  initStore(testJson)
  Session.images.length = 0
  Session.images.push({ [-1]: new Image(1000, 1000) })
  setUpLabel2dCanvas(1000, 1000)
})

/** Set up component for testing */
function setUpLabel2dCanvas (width: number, height: number) {
  Session.dispatch(action.addViewerConfig(0, makeImageViewerConfig(0)))

  const display = document.createElement('div')
  display.getBoundingClientRect = () => {
    return {
      width,
      height,
      top: 0,
      bottom: 0,
      left: 0,
      right: 0,
      x: 0,
      y: 0,
      toJSON: () => {
        return {
          width,
          height,
          top: 0,
          bottom: 0,
          left: 0,
          right: 0,
          x: 0,
          y: 0
        }
      }
    }
  }

  render(
    <div style={{ width: `${width}px`, height: `${height}px` }}>
      <Label2dCanvas
        classes={{
          label2d_canvas: 'label2dcanvas',
          control_canvas: 'controlcanvas'
        }}
        id={0}
        display={display}
        ref={canvasRef}
        shouldFreeze={false}
      />
    </div>
  )

  const itemIndex = 0
  Session.dispatch(action.goToItem(itemIndex))

  expect(canvasRef.current).not.toBeNull()
  expect(canvasRef.current).not.toBeUndefined()
}

/** Create mouse down event */
function mouseDownEvent (
  x: number, y: number
): React.MouseEvent<HTMLCanvasElement> {
  return new MouseEvent(
    'mousedown',
    { clientX: x, clientY: y }
  ) as unknown as React.MouseEvent<HTMLCanvasElement>
}

/** Create mouse down event */
function mouseUpEvent (
  x: number, y: number
): React.MouseEvent<HTMLCanvasElement> {
  return new MouseEvent(
    'mouseup',
    { clientX: x, clientY: y }
  ) as unknown as React.MouseEvent<HTMLCanvasElement>
}

/** Create key down event */
function keyDownEvent (key: string): KeyboardEvent {
  return new KeyboardEvent('keydown', { key })
}

/** Create key up event */
function keyUpEvent (key: string): KeyboardEvent {
  return new KeyboardEvent('keyup', { key })
}

describe('Test track', () => {
  test('Add track', () => {
    initStore(emptyTrackingTask)
    const getState = Session.getSimpleStore().getState
    const dispatch = Session.getSimpleStore().dispatch

    const label2d = canvasRef.current as Label2dCanvas
    const trackIds = new TrackCollector(getState)
    const numItems = getState().task.items.length

    dispatch(action.goToItem(0))
    // Draw first box
    drawBox2D(label2d, 1, 1, 50, 50)
    let state = getState()
    trackIds.collect()
    expect(_.size(state.task.tracks)).toEqual(1)
    // check whether the track is propagated to the other frames
    expect(_.size(state.task.tracks[trackIds[0]].labels)).toEqual(numItems)
    expect(_.size(state.task.items[numItems - 1].labels)).toEqual(1)
    expect(_.size(state.task.items[numItems - 1].shapes)).toEqual(1)

    drawBox2D(label2d, 19, 20, 30, 29)
    state = getState()
    trackIds.collect()
    expect(_.size(state.task.tracks)).toEqual(2)
    expect(_.size(state.task.items[numItems - 2].labels)).toEqual(2)
    expect(_.size(state.task.items[numItems - 2].shapes)).toEqual(2)

    drawBox2D(label2d, 100, 20, 80, 100)
    state = getState()
    trackIds.collect()
    expect(_.size(state.task.tracks)).toEqual(3)
    expect(_.size(state.task.items[numItems - 3].labels)).toEqual(3)
    expect(_.size(state.task.items[numItems - 3].shapes)).toEqual(3)

    drawBox2D(label2d, 500, 500, 80, 100)
    state = getState()
    trackIds.collect()
    expect(_.size(state.task.tracks)).toEqual(4)
    expect(_.size(state.task.items[numItems - 4].labels)).toEqual(4)
    expect(_.size(state.task.items[numItems - 4].shapes)).toEqual(4)

    dispatch(action.goToItem(1))
  })

  test('Terminate track by key', () => {
    initStore(testJson)
    const label2d = canvasRef.current as Label2dCanvas
    Session.dispatch(action.goToItem(1))
    let state = Session.getState()
    const trackLabels = state.task.tracks[3].labels
    const lblInItm2 = trackLabels[2]
    const lblInItm3 = trackLabels[3]
    const lblInItm4 = trackLabels[4]
    const lblInItm5 = trackLabels[5]
    expect(_.size(state.task.tracks[3].labels)).toBe(6)
    expect(_.size(state.task.items[2].labels)).toBe(3)
    expect(_.size(state.task.items[2].shapes)).toBe(3)
    label2d.onMouseDown(mouseDownEvent(835, 314))
    label2d.onMouseUp(mouseUpEvent(835, 314))
    label2d.onKeyDown(keyDownEvent('Control'))
    label2d.onKeyDown(keyDownEvent('E'))
    label2d.onKeyUp(keyUpEvent('E'))
    label2d.onKeyUp(keyUpEvent('Control'))

    state = Session.getState()
    expect(_.size(state.task.tracks[3].labels)).toBe(2)
    expect(state.task.items[2].labels[lblInItm2]).toBeUndefined()
    expect(state.task.items[2].labels[lblInItm3]).toBeUndefined()
    expect(state.task.items[2].labels[lblInItm4]).toBeUndefined()
    expect(state.task.items[2].labels[lblInItm5]).toBeUndefined()
  })

  test('Merge track by key', () => {
    initStore(testJson)
    const label2d = canvasRef.current as Label2dCanvas
    Session.dispatch(action.goToItem(3))
    let state = Session.getState()
    expect(_.size(state.task.tracks[2].labels)).toBe(4)
    expect(_.size(state.task.tracks[9].labels)).toBe(1)
    expect(state.task.items[5].labels[203].track).toEqual(9)

    label2d.onMouseDown(mouseDownEvent(925, 397))
    label2d.onMouseUp(mouseUpEvent(925, 397))
    label2d.onKeyDown(keyDownEvent('Control'))
    label2d.onKeyDown(keyDownEvent('L'))
    label2d.onKeyUp(keyUpEvent('L'))
    label2d.onKeyUp(keyUpEvent('Control'))

    Session.dispatch(action.goToItem(5))
    label2d.onMouseDown(mouseDownEvent(931, 300))
    label2d.onMouseUp(mouseUpEvent(931, 300))
    label2d.onKeyDown(keyDownEvent('Enter'))
    label2d.onKeyUp(keyUpEvent('Enter'))

    state = Session.getState()
    expect(_.size(state.task.tracks[2].labels)).toBe(5)
    expect(state.task.tracks[9]).toBeUndefined()
    expect(state.task.items[5].labels[203].track).toEqual(2)
  })
})
