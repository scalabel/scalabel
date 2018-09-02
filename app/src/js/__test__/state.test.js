/* @flow */
import {newLabel} from '../state';

describe('Label Tests', function() {
  it('Shape', function() {
    let label = newLabel({id: 10});
    let label1 = newLabel({id: 10});
    expect(label !== label1).toBe(true);
    let oldId = label.id;
    label = label.update('shapes', (shapes) => shapes.push(1));
    label = label.update('id', (id) => id + 10);
    expect(label.id).toBe(20);
    expect(label.id === oldId).toBe(false);
  });
});
