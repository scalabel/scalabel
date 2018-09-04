/* @flow */

import type {LabelType, SatType} from '../types';

/**
 * Create new label
 * @param {SatType} state: current state
 * @param {Function} createLabel: label creation function
 * @return {SatType}
 */
export function newLabel(
    state: SatType, createLabel: (number) => LabelType): SatType {
  let labelId = state.current.maxObjectId + 1;
  let items = state.items;
  if (state.current.item > 0) {
    items = items.slice();
    let item = items[state.current.item];
    items[state.current.item] = {
      ...item, labels: item.labels.concat([labelId]),
    };
  }
  return {
    ...state,
    items: items,
    labels: {...state.labels, labelId: createLabel(labelId)},
    current: {...state.current, maxObjectId: labelId},
  };
}

/**
 * Update configurations in Sat
 * @param {SatType} sat: Sat State
 * @param {Object} config: new configuration
 * @return {SatType}
 */
export function updateConfig(sat: SatType, config: {}): SatType {
  return {...sat, config: {...sat.config, config}};
}
