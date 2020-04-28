import { render } from '@testing-library/react'
import * as React from 'react'
import * as action from '../../js/action/common'
import { ActionType } from '../../js/action/types'
import { Label2dCanvas } from '../../js/components/label2d_canvas'
import { makeImageViewerConfig } from '../../js/functional/states'

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

/** Create mouse down event */
function mouseMoveEvent (
  x: number, y: number
): React.MouseEvent<HTMLCanvasElement> {
  return new MouseEvent(
    'mousemove',
    { clientX: x, clientY: y }
  ) as unknown as React.MouseEvent<HTMLCanvasElement>
}

/**
 * Move mouse wrapper
 * @param label2d
 * @param x
 * @param y
 */
export function mouseMove (label2d: Label2dCanvas, x: number, y: number) {
  label2d.onMouseMove(mouseMoveEvent(x, y))
}

/**
 * Mouse down wrapper
 * @param label2d
 * @param x
 * @param y
 */
export function mouseDown (label2d: Label2dCanvas, x: number, y: number) {
  label2d.onMouseDown(mouseDownEvent(x, y))
}

/**
 * Mouse up wrapper
 * @param label2d
 * @param x
 * @param y
 */
export function mouseUp (label2d: Label2dCanvas, x: number, y: number) {
  label2d.onMouseUp(mouseUpEvent(x, y))
}

/**
 * Test driver function for mouse click
 * @param label2d
 * @param x
 * @param y
 */
export function mouseClick (label2d: Label2dCanvas, x: number, y: number) {
  mouseDown(label2d, x, y)
  mouseUp(label2d, x, y)
}

/**
 * Test driver function for mouse move and click
 * @param label2d
 * @param x
 * @param y
 */
export function mouseMoveClick (label2d: Label2dCanvas, x: number, y: number) {
  mouseMove(label2d, x, y)
  mouseClick(label2d, x, y)
}

/**
 * Driver to press the key
 * @param label2d
 * @param key
 */
export function keyDown (label2d: Label2dCanvas, key: string) {
  label2d.onKeyDown(new KeyboardEvent('keydown', { key }))
}

/**
 * Driver to release the key
 * @param label2d
 * @param key
 */
export function keyUp (label2d: Label2dCanvas, key: string) {
  label2d.onKeyUp(new KeyboardEvent('keyup', { key }))
}

/**
 * Test driver to draw a polygon on label2d canvas
 * @param label2d
 * @param points
 */
export function drawPolygon (label2d: Label2dCanvas, points: number[][]) {
  for (const p of points) {
    mouseMoveClick(label2d, p[0], p[1])
  }
  mouseMoveClick(label2d, points[0][0], points[0][1])
}

/**
 * Test driver to draw a box
 * @param label2d
 * @param x1
 * @param y1
 * @param x2
 * @param y2
 */
export function drawBox2D (
  label2d: Label2dCanvas, x1: number, y1: number, x2: number, y2: number) {
  mouseMove(label2d, x1, y1)
  mouseDown(label2d, x1, y1)
  // move to a middle point first for more testing
  mouseMove(label2d, (x1 + x2) / 2, (y1 + y2) / 2)
  mouseMove(label2d, x2, y2)
  mouseUp(label2d, x2, y2)
}

/** Set up component for testing */
export function setUpLabel2dCanvas (
    dispatch: (actoin: ActionType) => void,
    canvasRef: React.RefObject<Label2dCanvas>,
    width: number, height: number) {
  dispatch(action.addViewerConfig(0, makeImageViewerConfig(0)))

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
  dispatch(action.goToItem(itemIndex))

  expect(canvasRef.current).not.toBeNull()
  expect(canvasRef.current).not.toBeUndefined()
}
