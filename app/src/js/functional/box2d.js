/* @flow */

import {makeLabel, makeRect} from './states';
import type {LabelType, LabelFunctionalType} from './types';

/**
 * Create new Box2d label
 * @param {number} labelId: label id
 * @param {number} itemId: item id
 * @param {Object} optionalAttributes
 * @return {LabelType}
 */
export function createLabel(labelId: number,
                            itemId: number,
                            optionalAttributes: Object = {}): LabelType {
  let rect = makeRect();
  if (optionalAttributes) {
    // TODO: do something to rect here
  }
  return makeLabel({id: labelId, item: itemId, shapes: [rect]});
}

// This is necessary for different label types
export const Box2dF: LabelFunctionalType = {
  createLabel: createLabel,
};
