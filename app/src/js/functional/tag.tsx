
import {LabelType, StateType} from './types';
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
                               optionalAttributes: any): LabelType {
  return makeLabel({id: labelId, item: itemId, attributes: optionalAttributes});
}

/**
 * Image tagging
 * @param {StateType} state
 * @param {number} itemId
 * @param {number} attributeIndex
 * @param {number} attributeValue
 * @return {StateType}
 */
export function tagImage(
    state: StateType, itemId: number, attributeIndex: number,
    attributeValue: number[]): StateType {
  const attributes = {[attributeIndex]: attributeValue};
  const item = state.items[itemId];
  if (item.labels.length > 0) {
    const labelId = item.labels[0];
    // be careful about this merge
    return _.merge({}, state, {labels: {[labelId]: {attributes}}});
  }
  return newLabel(state, itemId, createTagLabel, attributes);
}
