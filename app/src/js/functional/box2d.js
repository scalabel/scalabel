/* @flow */

import {makeLabel, makeRect} from '../states';
import type {LabelType, LabelFunctionalType} from '../types';

/**
 * Create new Box2d label
 * @param {number} id: label id
 * @param {Object} optionalAttributes
 * @return {LabelType}
 */
export function createLabel(id: number,
                            optionalAttributes: Object = {}): LabelType {
  let rect = makeRect();
  if (optionalAttributes) {
    // TODO: do something to rect here
  }
  return makeLabel({id: id, shapes: [rect]});
}

// This is necessary for different label types
export const Box2dF: LabelFunctionalType = {
  createLabel: createLabel,
};
