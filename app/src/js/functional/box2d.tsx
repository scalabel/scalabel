
import {makeLabel, makeRect} from './states';
import {LabelType, StateType} from './types';
import {updateListItem, updateObject} from './util';

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
 * @param {StateType} state: current state
 * @param {number} itemId
 * @param {Object} optionalAttributes: contains shapeId, x, y, h, w
 * @return {StateType}
 */
export function newImageBox2DLabel(
  state: StateType,
  itemId: number,
  optionalAttributes: any = {}): StateType {
  // get the labelId
  const labelId = state.current.maxObjectId + 1;
  // put the labelId inside item
  const item = updateObject(state.items[itemId],
    {labels: state.items[itemId].labels.concat([labelId])});
  // put updated item inside items
  const items = updateListItem(state.items, itemId, item);
  // get the shape Id
  const shapeId = labelId + 1;
  // create the rect object
  const rect = makeRect({id: shapeId, ...optionalAttributes});
  // put rect inside shapes
  const shapes = updateObject(state.shapes, {[shapeId]: rect});
  // create the actual label with the labelId and shapeId and put inside labels
  const labels = updateObject(state.labels,
    {[labelId]: createBox2DLabel(labelId, itemId, {shapes: [shapeId]})});
  const current = updateObject(state.current,
          {label: labelId, maxObjectId: shapeId});
  return {
    ...state,
    items,
    labels,
    shapes,
    current
  };
}

/**
 * assign new values to rectangle
 * @param {StateType} state
 * @param {number} shapeId
 * @param {Object} targetBoxAttributes: including x, y, w, h
 * @return {Object}
 */
export function changeRect(state: StateType, shapeId: number,
                           targetBoxAttributes: any) {
    const shapes = state.shapes;
    const newRect = updateObject(shapes[shapeId], targetBoxAttributes);
    const newShapes = updateObject(shapes, {[shapeId]: newRect});
    return {...state, shapes: newShapes};
}
