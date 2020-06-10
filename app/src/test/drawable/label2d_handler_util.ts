import * as action from '../../js/action/common'
import Session from '../../js/common/session'
import { Label2DHandler } from '../../js/drawable/2d/label2d_handler'
import { RectCoords } from '../../js/functional/types'
import { Size2D } from '../../js/math/size2d'
import { Vector2D } from '../../js/math/vector2d'

/**
 * Create a polygon based on the input vertices
 * @param label2dHandler
 * @param canvasSize
 * @param points
 */
export function drawPolygon (
  label2dHandler: Label2DHandler, canvasSize: Size2D, points: number[][]) {
  for (const p of points) {
    mouseMoveClick(label2dHandler, p[0], p[1], canvasSize, -1, 0)
  }
  mouseMoveClick(label2dHandler, points[0][0], points[0][1], canvasSize, -1, 1)
}

/**
 * Make mouse movements to do some 2d box operation
 * @param coords: coords for starting and ending clicks
 * @param labelIndex: if not -1, specifies changing an existing label
 * @param interrupt: if enabled, interrupt the operation with another action
 */
function do2DBoxOperation (
  label2dHandler: Label2DHandler, canvasSize: Size2D,
  coords: RectCoords, labelIndex: number, handleIndex: number,
  interrupt: boolean= false) {
  const x1 = coords.x1
  const x2 = coords.x2
  const y1 = coords.y1
  const y2 = coords.y2

  // First point
  mouseMove(label2dHandler, x1, y1, canvasSize, labelIndex, handleIndex)
  mouseDown(label2dHandler, x1, y1, labelIndex, handleIndex)

  // Some random intermediate move
  mouseMove(label2dHandler,
    x1 + Math.random() * 5, y1 - Math.random() * 5, canvasSize, -1, 0)

  if (interrupt) {
    Session.dispatch(action.setStatusToSaved())
  }

  // Last point
  mouseMove(label2dHandler, x2, y2, canvasSize, -1, 0)
  // Mouse up doesn't need to be exactly at the point
  mouseUp(label2dHandler,
    x2 + Math.random(), y2 - Math.random(), -1, 0)
}

/**
 * Make mouse movements to add the 2D box
 * @param coords: the coordinates of the new box
 */
export function draw2DBox (
  label2dHandler: Label2DHandler, canvasSize: Size2D,
  coords: RectCoords, interrupt: boolean= false) {
  do2DBoxOperation(label2dHandler, canvasSize, coords, -1, 0, interrupt)
}

/**
 * Make mouse movements to resize the 2D box
 * @param coords: x1/y1 is the existing point, x2/y2 is the new point
 * @param labelIndex: the index of the label to move
 */
export function resize2DBox (
  label2dHandler: Label2DHandler, canvasSize: Size2D,
  coords: RectCoords, labelIndex: number, interrupt: boolean= false) {
  do2DBoxOperation(
    label2dHandler, canvasSize, coords, labelIndex, labelIndex, interrupt)
}

/**
 * Make mouse movements to move the 2D box
 * @param coords: x1/y1 is the existing point, x2/y2 is the new point
 * @param labelIndex: the index of the label to move
 */
export function move2DBox (
  label2dHandler: Label2DHandler, canvasSize: Size2D,
  coords: RectCoords, labelIndex: number, interrupt: boolean= false) {
  // Handle index of 0 represents a move instead of a resize
  do2DBoxOperation(
    label2dHandler, canvasSize, coords, labelIndex, 0, interrupt)
}

/**
 * Driver function for mouse move
 * @param label2d
 * @param x
 * @param y
 * @param canvasSize
 * @param labelIndex
 * @param handleIndex
 */
export function mouseMove (
  label2d: Label2DHandler, x: number, y: number,
  canvasSize: Size2D, labelIndex: number, handleIndex: number) {
  label2d.onMouseMove(new Vector2D(x, y), canvasSize, labelIndex, handleIndex)
}

/**
 * Driver function for mouse down
 * @param label2d
 * @param x
 * @param y
 * @param labelIndex
 * @param handleIndex
 */
export function mouseDown (label2d: Label2DHandler, x: number, y: number,
                           labelIndex: number, handleIndex: number) {
  label2d.onMouseDown(new Vector2D(x, y), labelIndex, handleIndex)
}

/**
 * Driver function for mouse up
 * @param label2d
 * @param x
 * @param y
 * @param labelIndex
 * @param handleIndex
 */
export function mouseUp (label2d: Label2DHandler, x: number, y: number,
                         labelIndex: number, handleIndex: number) {
  label2d.onMouseUp(new Vector2D(x, y), labelIndex, handleIndex)
}

/**
 * Driver function for mouse click
 * @param label2d
 * @param x
 * @param y
 * @param labelIndex
 * @param handleIndex
 */
export function mouseClick (label2d: Label2DHandler, x: number, y: number,
                            labelIndex: number, handleIndex: number) {
  mouseDown(label2d, x, y, labelIndex, handleIndex)
  mouseUp(label2d, x, y, labelIndex, handleIndex)
}

/**
 * Driver function for mouse move and click
 * @param label2d
 * @param x
 * @param y
 * @param canvasSize
 * @param labelIndex
 * @param handleIndex
 */
export function mouseMoveClick (
  label2d: Label2DHandler, x: number, y: number,
  canvasSize: Size2D, labelIndex: number, handleIndex: number) {
  mouseMove(label2d, x, y, canvasSize, labelIndex, handleIndex)
  mouseClick(label2d, x, y, labelIndex, handleIndex)
}

/**
 * Driver function for key down
 * @param label2d
 * @param key
 */
export function keyDown (label2d: Label2DHandler, key: string) {
  label2d.onKeyDown(new KeyboardEvent('keydown', { key }))
}

/**
 * Driver function for key up
 * @param label2d
 * @param key
 */
export function keyUp (label2d: Label2DHandler, key: string) {
  label2d.onKeyUp(new KeyboardEvent('keyup', { key }))
}

/**
 * Driver function for key click
 * @param label2d
 * @param key
 */
export function keyClick (label2d: Label2DHandler, key: string) {
  keyDown(label2d, key)
  keyUp(label2d, key)
}
