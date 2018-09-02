import {LabelS} from '../state';

describe('Label Tests', function() {
  it('Shape', function() {
    let label = new LabelS({id: 10});
    let oldId = label.id;
    label = label.update('shapes', (shapes) => shapes.push('1'));
    label = label.update('id', (id) => id + 10);
    expect(label.id).toBe(20);
    expect(label.id === oldId).toBe(false);
  });
});
