import { cleanup, render } from '@testing-library/react'
import _ from 'lodash'
import React from 'react'
import * as action from '../../js/action/common'
import Session from '../../js/common/session'
import { initStore, loadImages } from '../../js/common/session_init'
import { Label2dCanvas } from '../../js/components/label2d_canvas'
import { makeImageViewerConfig } from '../../js/functional/states'
import { testJson } from '../test_track_objects'

const canvasRef: React.RefObject<Label2dCanvas> = React.createRef()

beforeEach(() => {
  cleanup()
  initStore(testJson)
  loadImages()
})

afterEach(cleanup)

beforeAll(() => {
  Session.devMode = false
  Session.subscribe(() => Session.label2dList.updateState(Session.getState()))
  initStore(testJson)
  loadImages()
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
      toJSON:  () => {
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
  test('Terminate track by key', () => {
    if (canvasRef.current) {
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
      canvasRef.current.onMouseDown(mouseDownEvent(835, 314))
      canvasRef.current.onMouseUp(mouseUpEvent(835, 314))
      canvasRef.current.onKeyDown(keyDownEvent('Control'))
      canvasRef.current.onKeyDown(keyDownEvent('E'))
      canvasRef.current.onKeyUp(keyUpEvent('E'))
      canvasRef.current.onKeyUp(keyUpEvent('Control'))

      state = Session.getState()
      expect(_.size(state.task.tracks[3].labels)).toBe(2)
      expect(state.task.items[2].labels[lblInItm2]).toBeUndefined()
      expect(state.task.items[2].labels[lblInItm3]).toBeUndefined()
      expect(state.task.items[2].labels[lblInItm4]).toBeUndefined()
      expect(state.task.items[2].labels[lblInItm5]).toBeUndefined()
    }
  })

  test('Merge track by key', () => {
    if (canvasRef.current) {
      Session.dispatch(action.goToItem(3))
      let state = Session.getState()
      expect(_.size(state.task.tracks[2].labels)).toBe(4)
      expect(_.size(state.task.tracks[9].labels)).toBe(1)
      expect(state.task.items[5].labels[203].track).toEqual(9)

      canvasRef.current.onMouseDown(mouseDownEvent(925, 397))
      canvasRef.current.onMouseUp(mouseUpEvent(925, 397))
      canvasRef.current.onKeyDown(keyDownEvent('Control'))
      canvasRef.current.onKeyDown(keyDownEvent('L'))
      canvasRef.current.onKeyUp(keyUpEvent('L'))
      canvasRef.current.onKeyUp(keyUpEvent('Control'))

      Session.dispatch(action.goToItem(5))
      canvasRef.current.onMouseDown(mouseDownEvent(931, 300))
      canvasRef.current.onMouseUp(mouseUpEvent(931, 300))
      canvasRef.current.onKeyDown(keyDownEvent('Enter'))
      canvasRef.current.onKeyUp(keyUpEvent('Enter'))

      state = Session.getState()
      expect(_.size(state.task.tracks[2].labels)).toBe(5)
      expect(state.task.tracks[9]).toBeUndefined()
      expect(state.task.items[5].labels[203].track).toEqual(2)
    }
  })
})
