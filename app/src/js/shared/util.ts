import { sprintf } from 'sprintf-js'
import { LabelTypeName } from '../common/types'

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
export function getPageTitle (labelType: string) {
  switch (labelType) {
    case LabelTypeName.TAG: {
      return 'Image Tagging'
    }
    case LabelTypeName.BOX_2D: {
      return '2D Bounding Box'
    }
    case LabelTypeName.POLYGON_2D: {
      return '2D Segmentation'
    }
    case LabelTypeName.POLYLINE_2D: {
      return '2D Lane'
    }
    case LabelTypeName.BOX_3D: {
      return '3D Bounding Box'
    }
    default: {
      return ''
    }
  }
}
