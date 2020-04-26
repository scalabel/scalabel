import { Label2DHandler } from '../../js/drawable/2d/label2d_handler'
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
  let v: Vector2D
  for (const p of points) {
    v = new Vector2D(p[0], p[1])
    label2dHandler.onMouseMove(v, canvasSize, -1, 0)
    label2dHandler.onMouseDown(v, -1, 0)
    label2dHandler.onMouseUp(v, -1, 0)
  }
  v = new Vector2D(points[0][0], points[0][1])
  label2dHandler.onMouseMove(v, canvasSize, -1, 1)
  label2dHandler.onMouseDown(v, -1, 1)
  label2dHandler.onMouseUp(v, -1, 1)
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
