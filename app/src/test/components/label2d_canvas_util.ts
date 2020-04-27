import * as React from 'react'
import { Label2dCanvas } from '../../js/components/label2d_canvas'

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
