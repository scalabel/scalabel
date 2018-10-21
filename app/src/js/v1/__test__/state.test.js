/* @flow */

import {makeLabel} from '../../functional/states';
import _ from 'lodash/fp';

describe('Label Tests', function() {
  it('Shape', function() {
    let label = makeLabel({id: 10});
    let label1 = makeLabel({id: 10});
    expect(label !== label1).toBe(true);
    let oldId = label.id;
    label = _.update('id', (id) => id + 10, label);
    label = _.update('shapes', (shapes) => shapes.concat(1), label);
    expect(label.id).toBe(20);
    expect(label.id === oldId).toBe(false);
  });
});
