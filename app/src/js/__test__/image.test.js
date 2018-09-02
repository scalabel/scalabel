import {Sat} from '../sat';
import {Box2d} from '../box2d';
import {SatImage} from '../image';

describe('Label Tests', function() {
  it('Label', function() {
    let sat = new Sat(SatImage, Box2d, false);
    let state = sat.state;
    state = sat.newLabelF(state);
    expect(state.maxObjectId).toBe(0);
    expect(state.labels.size).toBe(1);
  });
});
