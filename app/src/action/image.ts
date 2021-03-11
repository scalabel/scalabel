import * as actionTypes from "../types/action"
import { ImageViewerConfigType } from "../types/state"
import { MAX_SCALE, MIN_SCALE } from "../view_config/image"
import { changeViewerConfig } from "./common"

/**
 * Zoom the image
 *
 * @param zoomRatio
 * @param offsetX
 * @param offsetY
 * @param display
 * @param canvas
 * @param viewerId
 * @param config
 * @param canvasWidth
 * @param canvasHeight
 * @param displayToImageRatio
 */
export function zoomImage(
  zoomRatio: number,
  viewerId: number,
  config: ImageViewerConfigType
): actionTypes.ChangeViewerConfigAction | null {
  const newScale = config.viewScale * zoomRatio
  if (newScale >= MIN_SCALE && newScale <= MAX_SCALE) {
    const newConfig = {
      ...config,
      viewScale: newScale
    }
    return changeViewerConfig(viewerId, newConfig)
  }
  return null
}
