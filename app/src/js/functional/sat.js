/* @flow */

import type {ItemType, LabelType, SatType} from '../types';
import {updateObject, updateListItem} from './util';

/**
 * Create new label
 * @param {SatType} state: current state
 * @param {number} itemId
 * @param {Function} createLabel: label creation function
 * @param {Object} optionalAttributes
 * @return {SatType}
 */
export function newLabel(
    state: SatType,
    itemId: number,
    createLabel: (number, number, Object) => LabelType,
    optionalAttributes: Object = {}): SatType {
  let labelId = state.current.maxObjectId + 1;
  let item = updateObject(state.items[itemId],
    {labels: state.items[itemId].labels.concat([labelId])});
  let items = updateListItem(state.items, itemId, item);
  let labels = updateObject(state.labels,
    {[labelId]: createLabel(labelId, itemId, optionalAttributes)});
  let current = updateObject(state.current, {maxObjectId: labelId});
  return {
    ...state,
    items: items,
    labels: labels,
    current: current,
  };
}

/**
 * Create new Item
 * @param {SatType} state
 * @param {Function} createItem
 * @param {string} url
 * @return {SatType}
 */
export function newItem(
  state: SatType, createItem: (number, string) => ItemType,
  url: string): SatType {
  let id = state.items.length;
  let item = createItem(id, url);
  let items = state.items.slice();
  items.push(item);
  return {
    ...state,
    items: items,
  };
}
