/* @flow */

import {makeLabel, makeRect} from '../states';
import type {LabelType, LabelFunctionalType} from '../types';

/**
 * Create new Box2d label
 * @param {number} id: label id
 * @return {LabelType}
 */
function createLabel(id: number): LabelType {
  return makeLabel({id: id, shapes: [makeRect()]});
}

// This is necessary for different label types
export const Box2dF: LabelFunctionalType = {
  createLabel: createLabel,
};
