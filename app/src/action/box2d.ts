import { LabelTypeName } from "../const/common"
import { makeLabel, makeRect } from "../functional/states"
import { AddLabelsAction, ChangeShapesAction } from "../types/action"
import { IdType, RectType, ShapeType, SimpleRect } from "../types/state"
import * as actions from "./common"

/**
 * Create AddLabelAction to create a box2d label
 *
 * @param {number} itemIndex
 * @param {number[]} category: list of category ids
 * @param {[key: number]: number[]} attributes
 * @param {number} x
 * @param {number} y
 * @param {number} w
 * @param {number} h
 * @param sensor
 * @param category
 * @param box
 * @returns {AddLabelAction}
 */
export function addBox2dLabel(
  itemIndex: number,
  sensor: number,
  category: number[],
  attributes: { [key: number]: number[] },
  box: SimpleRect
): AddLabelsAction {
  // Create the rect object
  const label = makeLabel({
    type: LabelTypeName.BOX_2D,
    category,
    attributes,
    sensors: [sensor]
  })
  const rect = makeRect({ ...box, label: [label.id] })
  label.shapes = [rect.id]
  return actions.addLabel(itemIndex, label, [rect])
}

/**
 * A simple wrapper for changing box2d shapes
 *
 * @param itemIndex
 * @param shapeId
 * @param shape
 */
export function changeBox2d(
  itemIndex: number,
  shapeId: IdType,
  shape: Partial<RectType>
): ChangeShapesAction {
  return actions.changeLabelShape(
    itemIndex,
    shapeId,
    shape as Partial<ShapeType>
  )
}
