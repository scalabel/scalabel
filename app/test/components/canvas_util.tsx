import { render } from "@testing-library/react"
import * as React from "react"

import * as action from "../../src/action/common"
import { GetStateFunc, SimpleStore } from "../../src/common/simple_store"
import { Label2dCanvas } from "../../src/components/label2d_canvas"
import { makeImageViewerConfig } from "../../src/functional/states"
import { ActionType } from "../../src/types/action"
import { IdType } from "../../src/types/state"
import { LabelCollector } from "../util/label_collector"
import { TrackCollector } from "../util/track_collector"

/**
 * Create mouse down event
 *
 * @param x
 * @param y
 */
function mouseDownEvent(
  x: number,
  y: number
): React.MouseEvent<HTMLCanvasElement> {
  return (new MouseEvent("mousedown", {
    clientX: x,
    clientY: y
  }) as unknown) as React.MouseEvent<HTMLCanvasElement>
}

/**
 * Create mouse down event
 *
 * @param x
 * @param y
 */
function mouseUpEvent(
  x: number,
  y: number
): React.MouseEvent<HTMLCanvasElement> {
  return (new MouseEvent("mouseup", {
    clientX: x,
    clientY: y
  }) as unknown) as React.MouseEvent<HTMLCanvasElement>
}

/**
 * Create mouse down event
 *
 * @param x
 * @param y
 */
function mouseMoveEvent(
  x: number,
  y: number
): React.MouseEvent<HTMLCanvasElement> {
  return (new MouseEvent("mousemove", {
    clientX: x,
    clientY: y
  }) as unknown) as React.MouseEvent<HTMLCanvasElement>
}

/**
 * Move mouse wrapper
 *
 * @param label2d
 * @param x
 * @param y
 */
export function mouseMove(label2d: Label2dCanvas, x: number, y: number): void {
  label2d.onMouseMove(mouseMoveEvent(x, y))
}

/**
 * Mouse down wrapper
 *
 * @param label2d
 * @param x
 * @param y
 */
export function mouseDown(label2d: Label2dCanvas, x: number, y: number): void {
  label2d.onMouseDown(mouseDownEvent(x, y))
}

/**
 * Mouse up wrapper
 *
 * @param label2d
 * @param x
 * @param y
 */
export function mouseUp(label2d: Label2dCanvas, x: number, y: number): void {
  label2d.onMouseUp(mouseUpEvent(x, y))
}

/**
 * Test driver function for mouse click
 *
 * @param label2d
 * @param x
 * @param y
 */
export function mouseClick(label2d: Label2dCanvas, x: number, y: number): void {
  mouseDown(label2d, x, y)
  mouseUp(label2d, x, y)
}

/**
 * Test driver function for mouse move and click
 *
 * @param label2d
 * @param x
 * @param y
 */
export function mouseMoveClick(
  label2d: Label2dCanvas,
  x: number,
  y: number
): void {
  mouseMove(label2d, x, y)
  mouseClick(label2d, x, y)
}

/**
 * Driver to press the key
 *
 * @param label2d
 * @param key
 */
export function keyDown(label2d: Label2dCanvas, key: string): void {
  label2d.onKeyDown(new KeyboardEvent("keydown", { key }))
}

/**
 * Driver to release the key
 *
 * @param label2d
 * @param key
 */
export function keyUp(label2d: Label2dCanvas, key: string): void {
  label2d.onKeyUp(new KeyboardEvent("keyup", { key }))
}

/**
 * Test driver to draw a polygon on label2d canvas
 *
 * @param label2d
 * @param points
 */
export function drawPolygon(label2d: Label2dCanvas, points: number[][]): void {
  points.forEach((p) => mouseMoveClick(label2d, p[0], p[1]))
  mouseMoveClick(label2d, points[0][0], points[0][1])
}

/**
 * Drag from point to point
 *
 * @param label2d
 * @param x1
 * @param y1
 * @param x2
 * @param y2
 */
export function drag(
  label2d: Label2dCanvas,
  x1: number,
  y1: number,
  x2: number,
  y2: number
): void {
  mouseMove(label2d, x1, y1)
  mouseDown(label2d, x1, y1)
  // Move to a middle point first for more testing
  mouseMove(label2d, (x1 + x2) / 2, (y1 + y2) / 2)
  mouseMove(label2d, x2, y2)
  mouseUp(label2d, x2, y2)
}

/**
 * Test driver to draw a box
 *
 * @param label2d
 * @param getState state accessor. it can be null, where the return vaule is ''
 * @param x1
 * @param y1
 * @param x2
 * @param y2
 * @returns The id of the new label drawn on thecanvas
 */
export function drawBox2D(
  label2d: Label2dCanvas,
  getState: GetStateFunc | null,
  x1: number,
  y1: number,
  x2: number,
  y2: number
): IdType {
  if (getState !== null) {
    const boxIds = new LabelCollector(getState)
    const start = boxIds.collect()
    drag(label2d, x1, y1, x2, y2)
    boxIds.collect()
    expect(boxIds.length).toEqual(start + 1)
    return boxIds[start]
  } else {
    drag(label2d, x1, y1, x2, y2)
    return ""
  }
}

/**
 * Draw a sequences of bounding box 2D tracks
 *
 * @param label2d 2d canvas
 * @param state state store
 * @param store
 * @param itemIndices the item indices of those boses
 * @param boxes corners coordinates in [x1, y1, x2, y2] for each box
 */
export function drawBox2DTracks(
  label2d: Label2dCanvas,
  store: SimpleStore,
  itemIndices: number[],
  boxes: number[][]
): TrackCollector {
  const trackIds = new TrackCollector(store.getter())
  itemIndices.forEach((itemIndex, boxIndex) => {
    store.dispatch(action.goToItem(itemIndex))
    const b = boxes[boxIndex]
    drawBox2D(label2d, null, b[0], b[1], b[2], b[3])
    trackIds.collect()
  })
  expect(trackIds.length).toEqual(itemIndices.length)
  return trackIds
}

/**
 * Set up component for testing
 *
 * @param dispatch
 * @param canvasRef
 * @param width
 * @param height
 * @param tracking
 */
export function setUpLabel2dCanvas(
  dispatch: (action: ActionType) => void,
  canvasRef: React.RefObject<Label2dCanvas>,
  width: number,
  height: number,
  tracking: boolean = false
): void {
  dispatch(action.addViewerConfig(0, makeImageViewerConfig(0)))

  const display = document.createElement("div")
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
          label2d_canvas: "label2dcanvas",
          control_canvas: "controlcanvas"
        }}
        id={0}
        display={display}
        ref={canvasRef}
        shouldFreeze={false}
        tracking={tracking}
      />
    </div>
  )

  const itemIndex = 0
  dispatch(action.goToItem(itemIndex))

  expect(canvasRef.current).not.toBeNull()
  expect(canvasRef.current).not.toBeUndefined()
}
