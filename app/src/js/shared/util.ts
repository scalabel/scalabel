import { sprintf } from 'sprintf-js'
import { LabelTypeName } from '../common/types'

/**
 * Create the link to the labeling instructions
 */
function makeInstructionsLink (pageName: string) {
  return sprintf('https://www.scalabel.ai/doc/instructions/%s.html', pageName)
}

/**
 * Select the correct instructions link for the given label type
 */
export function getInstructionsLink (labelType: string) {
  switch (labelType) {
    case LabelTypeName.BOX_2D: {
      return makeInstructionsLink('bbox')
    }
    case LabelTypeName.POLYGON_2D:
    case LabelTypeName.POLYLINE_2D: {
      return makeInstructionsLink('segmentation')
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
