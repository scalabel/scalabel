import { makeLabel, makeRect } from '../functional/states';
import { AddLabelAction } from './types';
import * as actions from './creators';
import * as labels from '../common/label_types';

/**
 * Create AddLabelAction to create a box2d label
 * @param {number[]} category: list of category ids
 * @param {number} x
 * @param {number} y
 * @param {number} w
 * @param {number} h
 * @return {AddLabelAction}
 */
export function addBox2dLabel(
  category: number[],
  x: number, y: number, w: number, h: number): AddLabelAction {
  // create the rect object
  const rect = makeRect({ x, y, w, h });
  const label = makeLabel({ type: labels.BOX_2D, category });
  return actions.addLabel(label, [rect]);
}
