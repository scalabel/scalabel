import { ImageViewerConfigType } from '../functional/types'
import {
  MAX_SCALE,
  MIN_SCALE
} from '../view_config/image'
import { changeViewerConfig } from './common'
import * as types from './types'

/**
 * Zoom the image
 * @param zoomRatio
 * @param offsetX
 * @param offsetY
 * @param display
 * @param canvas
 * @param config
 * @param canvasWidth
 * @param canvasHeight
 * @param displayToImageRatio
 */
export function zoomImage (
  zoomRatio: number,
  viewerId: number,
  config: ImageViewerConfigType
): types.ChangeViewerConfigAction | null {
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
