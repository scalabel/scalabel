import Session from '../common/session'
import * as types from './types'

/**
 * create image zoom action
 * @param {number} ratio: zoom ratio
 * @param {number} offsetX: view offset x
 * @param {number} offsetY: vew offset y
 */
export function zoomImage (
    ratio: number, offsetX: number, offsetY: number): types.ImageZoomAction {
  return {
    type: types.IMAGE_ZOOM,
    sessionId: Session.id,
    ratio,
    viewOffsetX: offsetX,
    viewOffsetY: offsetY
  }
}
