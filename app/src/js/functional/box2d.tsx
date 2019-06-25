import {makeLabel, makeRect} from './states';
import {LabelType, State} from './types';
import {addLabel, updateLabelShape} from './common';

/**
 * Create new Box2d label
 * @param {number} labelId: label id
 * @param {number} itemId: item id
 * @param {Object} optionalAttributes
 * @return {LabelType}
 */
export function createBox2DLabel(labelId: number,
                                 itemId: number,
                                 optionalAttributes: any = {}): LabelType {
  return makeLabel({id: labelId, item: itemId,
    shapes: optionalAttributes.shapes});
}

/**
 * Create new box2d label
 * @param {State} state: current state
 * @param {Object} optionalAttributes: contains shapeId, x, y, h, w
 * @return {State}
 */
export function addImageBox2DLabel(
  state: State,
  optionalAttributes: any = {}): State {
  // create the rect object
  const rect = makeRect({id: 0, ...optionalAttributes});
  return addLabel(state, makeLabel({item: state.current.item}), [rect]);
}

/**
 * assign new values to rectangle
 * @param {State} state
 * @param {number} shapeId
 * @param {Object} targetBoxAttributes: including x, y, w, h
 * @return {State}
 */
export function changeRect(state: State, shapeId: number,
                           targetBoxAttributes: any): State {
    return updateLabelShape(state, shapeId, targetBoxAttributes);
}
