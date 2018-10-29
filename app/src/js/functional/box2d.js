
import {makeLabel, makeRect} from './states';
import type {LabelType, StateType} from './types';
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
                            optionalAttributes: Object = {}): LabelType {
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
  optionalAttributes: Object = {}): StateType {
  // get the labelId
  let labelId = state.current.maxObjectId + 1;
  // put the labelId inside item
  let item = updateObject(state.items[itemId],
    {labels: state.items[itemId].labels.concat([labelId])});
  // put updated item inside items
  let items = updateListItem(state.items, itemId, item);
  // get the shape Id
  let shapeId = labelId + 1;
  // create the rect object
  let rect = makeRect({id: shapeId, ...optionalAttributes});
  // put rect inside shapes
  let shapes = updateObject(state.shapes, {[shapeId]: rect});
  // create the actual label with the labelId and shapeId and put inside labels
  let labels = updateObject(state.labels,
    {[labelId]: createBox2DLabel(labelId, itemId, {shapes: [shapeId]})});
  let current = updateObject(state.current,
          {label: labelId, maxObjectId: shapeId});
  return {
    ...state,
    items: items,
    labels: labels,
    shapes: shapes,
    current: current,
  };
}

/**
 * assign new values to rectangle
 * @param {Object} state
 * @param {number} shapeId
 * @param {number} x
 * @param {number} y
 * @param {number} w
 * @param {number} h
 * @return {Object}
 */
export function changeRect(state: Object, shapeId: number,
                           x: number, y: number, w: number, h: number) {
    let shapes = state.shapes;
    let newRect = makeRect({id: shapeId, x: x, y: y, w: w, h: h});
    let newShapes = updateObject(shapes, {[shapeId]: newRect});
    return {...state, shapes: newShapes};
}
