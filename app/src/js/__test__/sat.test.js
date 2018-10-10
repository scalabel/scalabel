/* @flow */

import {makeSat} from '../states';
import * as Sat from '../functional/sat';
import {Box2dF} from '../functional/box2d';
import {ImageF} from '../functional/image';
import _ from 'lodash/fp';

describe('Sat Functional Tests', function() {
  it('Sat', function() {
    let sat = makeSat();
    sat = Sat.newLabel(sat, Box2dF.createLabel);
    expect(sat.current.maxObjectId).toBe(0);
    expect(_.size(sat.labels)).toBe(1);
  });
  it('SatItem', function() {
    let sat = makeSat();
    sat = Sat.newItem(sat, ImageF.createItem, 'testurl');
    expect(sat.items.length).toBe(1);
    expect(sat.items[0].id).toBe(0);
    expect(sat.items[0].index).toBe(0);
    expect(sat.items[0].url).toBe('testurl');
  });
});
