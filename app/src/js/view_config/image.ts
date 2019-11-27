import Session from '../common/session'
import { decodeControlIndex, rgbToIndex } from '../drawable/util'
import {
  getCurrentItem
} from '../functional/state_util'
import { ImageViewerConfigType, State } from '../functional/types'
import { Size2D } from '../math/size2d'
import { Vector2D } from '../math/vector2d'

// display export constants
/** The maximum scale */
export const MAX_SCALE = 3.0
/** The minimum scale */
export const MIN_SCALE = 1.0
/** The boosted ratio to draw shapes sharper */
export const UP_RES_RATIO = 2
/** The zoom ratio */
export const ZOOM_RATIO = 1.05
/** The scroll-zoom ratio */
export const SCROLL_ZOOM_RATIO = 1.03

/**
 * Get the current item in the state
 * @return {Size2D}
 */
export function getCurrentImageSize (state: State, viewerId: number): Size2D {
  const item = getCurrentItem(state)
  const sensor = state.user.viewerConfigs[viewerId].sensor
  if (sensor in Session.images[item.index]) {
    const image = Session.images[item.index][sensor]
    return new Size2D(image.width, image.height)
  }
  return new Size2D(0, 0)
}

/**
 * Convert image coordinate to canvas coordinate.
 * If affine, assumes values to be [x, y]. Otherwise
 * performs linear transformation.
 * @param {Vector2D} values - the values to convert.
 * @param {boolean} upRes
 * @return {Vector2D} - the converted values.
 */
export function toCanvasCoords (
  values: Vector2D, upRes: boolean, displayToImageRatio: number): Vector2D {
  const out = values.clone().scale(displayToImageRatio)
  if (upRes) {
    out.scale(UP_RES_RATIO)
  }
  return out
}

/**
 * Convert canvas coordinate to image coordinate.
 * If affine, assumes values to be [x, y]. Otherwise
 * performs linear transformation.
 * @param {Vector2D} values - the values to convert.
 * @param {boolean} upRes - whether the canvas has higher resolution
 * @return {Vector2D} - the converted values.
 */
export function toImageCoords (
  values: Vector2D,
  upRes: boolean = true,
  displayToImageRatio: number): Vector2D {
  const up = (upRes) ? 1 / UP_RES_RATIO : 1
  return values.clone().scale(displayToImageRatio * up)
}

/**
 * Draw image on canvas
 * @param canvas
 * @param context
 * @param image
 */
export function drawImageOnCanvas (
  canvas: HTMLCanvasElement,
  context: CanvasRenderingContext2D,
  image: HTMLImageElement): void {
  clearCanvas(canvas, context)
  context.drawImage(image, 0, 0, image.width, image.height,
    0, 0, canvas.width, canvas.height)
}

/**
 * Clear the canvas
 * @param {HTMLCanvasElement} canvas - the canvas to redraw
 * @param {any} context - the context to redraw
 * @return {boolean}
 */
export function clearCanvas (canvas: HTMLCanvasElement,
                             context: CanvasRenderingContext2D) {
  // clear context
  context.clearRect(0, 0, canvas.width, canvas.height)
}

/**
 * Get the coordinates of the upper left corner of the image canvas
 * @return {Vector2D} the x and y coordinates
 */
export function getVisibleCanvasCoords (
  display: HTMLDivElement,
  canvas: HTMLCanvasElement
): Vector2D {
  if (display && canvas) {
    const displayRect = display.getBoundingClientRect()
    const imgRect = canvas.getBoundingClientRect()
    if (imgRect.x && imgRect.y) {
      return new Vector2D(displayRect.x - imgRect.x, displayRect.y - imgRect.y)
    }

    return new Vector2D(displayRect.x, displayRect.y)
  }
  return new Vector2D(0, 0)
}

/**
 * Normalize mouse x & y to canvas coordinates
 * @param display
 * @param canvas
 * @param canvasWidth
 * @param canvasHeight
 * @param displayToImageRatio
 * @param clientX
 * @param clientY
 */
export function normalizeMouseCoordinates (
  display: HTMLDivElement,
  canvas: HTMLCanvasElement,
  canvasWidth: number,
  canvasHeight: number,
  displayToImageRatio: number,
  clientX: number,
  clientY: number
) {
  const [offsetX, offsetY] =
    getVisibleCanvasCoords(display, canvas)
  const displayRect = display.getBoundingClientRect()
  let x = clientX - displayRect.x + offsetX
  let y = clientY - displayRect.y + offsetY

  // limit the mouse within the image
  x = Math.max(0, Math.min(x, canvasWidth))
  y = Math.max(0, Math.min(y, canvasHeight))

  // return in the image coordinates
  return new Vector2D(x / displayToImageRatio,
    y / displayToImageRatio)
}

/**
 * Function to find mode of a number array.
 * @param {number[]} arr - the array.
 * @return {number} the mode of the array.
 */
export function mode (arr: number[]) {
  return arr.sort((a, b) =>
    arr.filter((v) => v === a).length
    - arr.filter((v) => v === b).length
  ).pop()
}

/**
 * Get handle id from image color
 * @param color
 */
export function imageDataToHandleId (data: Uint8ClampedArray) {
  const arr = []
  for (let i = 0; i < 16; i++) {
    const color = rgbToIndex(Array.from(data.slice(i * 4, i * 4 + 3)))
    arr.push(color)
  }
  // finding the mode of the data array to deal with anti-aliasing
  const hoveredIndex = mode(arr) as number
  return decodeControlIndex(hoveredIndex)
}

/**
 * Update canvas scale
 * @param display
 * @param canvas
 * @param context
 * @param config
 * @param zoomRatio
 * @param upRes
 */
export function updateCanvasScale (
  state: State,
  display: HTMLDivElement,
  canvas: HTMLCanvasElement,
  context: CanvasRenderingContext2D | null,
  config: ImageViewerConfigType,
  zoomRatio: number,
  upRes: boolean
): number[] {
  const displayRect = display.getBoundingClientRect()

  if (context) {
    context.scale(zoomRatio, zoomRatio)
  }

  // resize canvas
  const item = getCurrentItem(state)
  const image = Session.images[item.index][config.sensor]
  const ratio = image.width / image.height
  let canvasHeight
  let canvasWidth
  let displayToImageRatio
  if (displayRect.width / displayRect.height > ratio) {
    canvasHeight = displayRect.height * config.viewScale
    canvasWidth = canvasHeight * ratio
    displayToImageRatio = canvasHeight
      / image.height
  } else {
    canvasWidth = displayRect.width * config.viewScale
    canvasHeight = canvasWidth / ratio
    displayToImageRatio = canvasWidth / image.width
  }

  // set canvas resolution
  if (upRes) {
    canvas.height = canvasHeight * UP_RES_RATIO
    canvas.width = canvasWidth * UP_RES_RATIO
  } else {
    canvas.height = canvasHeight
    canvas.width = canvasWidth
  }

  // set canvas size
  canvas.style.height = canvasHeight + 'px'
  canvas.style.width = canvasWidth + 'px'

  // set padding
  const padding = new Vector2D(
    Math.max(0, (displayRect.width - canvasWidth) / 2),
    Math.max(0, (displayRect.height - canvasHeight) / 2))
  const padX = padding.x
  const padY = padding.y

  canvas.style.left = padX + 'px'
  canvas.style.top = padY + 'px'
  canvas.style.right = 'auto'
  canvas.style.bottom = 'auto'

  return [canvasWidth, canvasHeight, displayToImageRatio, config.viewScale]
}
