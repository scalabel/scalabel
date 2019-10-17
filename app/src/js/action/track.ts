import _ from 'lodash'
import Session from '../common/session'
import { LabelType, ShapeType } from '../functional/types'
import { addTrack } from './common'
import { AddTrackAction } from './types'

/**
 * Add track by duplicating label from startIndex to stopIndex
 * @param label
 * @param shapeTypes
 * @param shapes
 * @param startIndex
 * @param stopIndex
 */
export function addDuplicatedTrack (
  label: LabelType,
  shapeTypes: string[],
  shapes: ShapeType[],
  startIndex?: number,
  stopIndex?: number
): AddTrackAction {
  const trackLabels: LabelType[] = []
  const trackShapeTypes: string[][] = []
  const trackShapes: ShapeType[][] = []
  const itemIndices: number[] = []

  const itemLength = (Session.itemType === 'image') ? Session.images.length :
    Session.pointClouds.length

  if (!startIndex) {
    startIndex = 0
  }

  if (!stopIndex) {
    stopIndex = itemLength
  }
  const end = Math.min(stopIndex, itemLength)

  for (let index = startIndex; index < end; index += 1) {
    trackLabels.push(_.cloneDeep(label))
    trackShapeTypes.push(shapeTypes)
    trackShapes.push(shapes)
    itemIndices.push(index)
    if (index > startIndex) {
      trackLabels[trackLabels.length - 1].manual = false
    }
  }

  return addTrack(
    itemIndices, trackLabels, trackShapeTypes, trackShapes
  )
}
