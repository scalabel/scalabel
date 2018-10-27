
import type {LabelType, StateType} from './types';
import {makeLabel} from './states';
import {newLabel} from './common';
import _ from 'lodash';

/**
 * Create a Tag label
 * @param {number} labelId
 * @param {number} itemId
 * @param {Object} optionalAttributes
 * @return {LabelType}
 */
export function createTagLabel(labelId: number, itemId: number,
                               optionalAttributes: Object): LabelType {
  return makeLabel({id: labelId, item: itemId, attributes: optionalAttributes});
}

/**
 *Image tagging
 * @param {StateType} state
 * @param {number} itemId
 * @param {number} attributeIndex
 * @param {number} attributeValue
 * @return {StateType}
 */
export function tagImage(
    state: StateType, itemId: number, attributeIndex: number,
    attributeValue: Array<number>): StateType {
  let attributes = {[attributeIndex]: attributeValue};
  let item = state.items[itemId];
  if (item.labels.length > 0) {
    let labelId = item.labels[0];
    // be careful about this merge
    return _.merge({}, state, {labels: {[labelId]: {attributes: attributes}}});
  }
  return newLabel(state, itemId, createTagLabel, attributes);
}
