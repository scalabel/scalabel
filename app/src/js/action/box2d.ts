import { LabelTypes, ShapeTypes } from '../common/types'
import { makeLabel, makeRect } from '../functional/states'
import * as actions from './common'
import { AddLabelsAction } from './types'

/**
 * Create AddLabelAction to create a box2d label
 * @param {number} itemIndex
 * @param {number[]} category: list of category ids
 * @param {[key: number]: number[]} attributes
 * @param {number} x
 * @param {number} y
 * @param {number} w
 * @param {number} h
 * @return {AddLabelAction}
 */
export function addBox2dLabel (
  itemIndex: number, category: number[], attributes: {[key: number]: number[]},
  x: number, y: number, w: number, h: number): AddLabelsAction {
  // create the rect object
  const rect = makeRect({ x1: x, y1: y, x2: w, y2: h })
  const label = makeLabel({ type: LabelTypes.BOX_2D, category, attributes })
  return actions.addLabel(itemIndex, label, [ShapeTypes.RECT], [rect])
}
