/* @flow */

import {makeSat} from '../states';
import * as Sat from '../functional/sat';
import {Box2dF} from '../functional/box2d';
import _ from 'lodash/fp';

describe('Sat Functional Tests', function() {
  it('Sat', function() {
    let sat = makeSat();
    sat = Sat.newLabel(sat, Box2dF.createLabel);
    expect(sat.current.maxObjectId).toBe(0);
    expect(_.size(sat.labels)).toBe(1);
  });
});
