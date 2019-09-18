import Session from '../common/session'
import { ImageViewerConfigType } from '../functional/types'
import {
  MAX_SCALE,
  MIN_SCALE
} from '../view/image'
import * as types from './types'

/**
 * Update viewer config
 * @param newFields
 */
export function updateImageViewerConfig (
  newFields: Partial<ImageViewerConfigType>
): types.UpdateImageViewerConfigAction {
  return {
    type: types.UPDATE_IMAGE_VIEWER_CONFIG,
    sessionId: Session.id,
    newFields
  }
}

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
  config: ImageViewerConfigType
): types.UpdateImageViewerConfigAction | null {
  const newScale = config.viewScale * zoomRatio
  if (newScale >= MIN_SCALE && newScale <= MAX_SCALE) {
    return updateImageViewerConfig({
      viewScale: newScale
    })
  }
  return null
}
