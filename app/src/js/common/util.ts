import { sprintf } from 'sprintf-js'
import { ADD_LABELS } from '../action/types'
import { ActionPacketType } from '../server/types'
import { ItemTypeName, LabelTypeName } from './types'

/**
 * Get whether tracking is on
 * Also get the new item type
 */
export function getTracking (itemType: string): [string, boolean] {
  switch (itemType) {
    case ItemTypeName.VIDEO:
      return [ItemTypeName.IMAGE, true]
    case ItemTypeName.POINT_CLOUD_TRACKING:
      return [ItemTypeName.POINT_CLOUD, true]
    case ItemTypeName.FUSION:
      return [ItemTypeName.FUSION, true]
    default:
      return [itemType, false]
  }
}

/**
 * Create the link to the labeling instructions
 */
function makeInstructionUrl (pageName: string) {
  return sprintf('https://www.scalabel.ai/doc/instructions/%s.html', pageName)
}

/**
 * Select the correct instruction url for the given label type
 */
export function getInstructionUrl (labelType: string) {
  switch (labelType) {
    case LabelTypeName.BOX_2D: {
      return makeInstructionUrl('bbox')
    }
    case LabelTypeName.POLYGON_2D:
    case LabelTypeName.POLYLINE_2D: {
      return makeInstructionUrl('segmentation')
    }
    default: {
      return ''
    }
  }
}

/**
 * Select the correct page title for given label type
 */
export function getPageTitle (labelType: string, itemType: string) {
  const [, tracking] = getTracking(itemType)

  let title: string
  switch (labelType) {
    case LabelTypeName.TAG:
      title = 'Image Tagging'
      break
    case LabelTypeName.BOX_2D:
      title = '2D Bounding Box'
      break
    case LabelTypeName.POLYGON_2D:
      title = '2D Segmentation'
      break
    case LabelTypeName.POLYLINE_2D:
      title = '2D Lane'
      break
    case LabelTypeName.BOX_3D:
      title = '3D Bounding Box'
      break
    default:
      title = ''
      break
  }
  if (tracking) {
    title = sprintf('%s Tracking', title)
  }
  return title
}

/**
 * Converts index into a filename of size 6 with
 * trailing zeroes
 */
export function index2str (index: number) {
  return index.toString().padStart(6, '0')
}

/**
 * Checks if the action packet contains
 * any actions that would trigger a model query
 */
export function doesPacketTriggerModel (
  actionPacket: ActionPacketType, bots: boolean): boolean {
  if (!bots) {
    return false
  }
  for (const action of actionPacket.actions) {
    if (action.type === ADD_LABELS) {
      return true
    }
  }
  return false
}
