import * as common from '../functional/common';
import {testJson} from './test_objects';
import Session from '../common/session_single';
import {makeLabel} from '../functional/states';
import {_} from 'lodash';

describe('Functional Common Tests', function() {
  it('NewAndDelete', function() {
    Session.devMode = false;
    Session.initStore(testJson);
    let state = Session.getState();
    state = common.newLabel(state, 0,
        (labelId, itemId, ignored) => makeLabel({id: labelId, item: itemId}));
    let labelId = state.current.maxObjectId;
    expect(state.items[0].labels.length).toBe(1);
    expect(_.size(state.labels)).toBe(1);
    expect(state.labels[labelId].item).toBe(0);
    state = common.deleteLabel(state, 0, labelId);
    expect(state.items[0].labels.length).toBe(0);
    expect(_.size(state.labels)).toBe(0);
  });
  it('Change Category', function() {
    Session.devMode = false;
    Session.initStore(testJson);
    let state = Session.getState();
    state = common.changeCategory(state, 0, [2]);
    expect(state.labels[0].category[0]).toBe(2);
  });
});
