import * as common from '../functional/common';
import {testJson} from './test_objects';
import Session from '../common/session';
import {makeLabel} from '../functional/states';
import * as _ from 'lodash';

test('New And Delete', () => {
    Session.devMode = false;
    Session.initStore(testJson);
    let state = Session.getState();
    state = common.goToItem(state, 0);
    const label = makeLabel({item: state.current.item});
    state = common.addLabel(state, label);
    const labelId = state.current.maxObjectId;
    expect(_.size(state.items[0].labels)).toBe(1);
    expect(state.items[0].labels[labelId].item).toBe(0);
    state = common.deleteLabel(state, labelId);
    expect(_.size(state.items[0].labels)).toBe(0);
});

test('Change Category', () => {
    Session.devMode = false;
    Session.initStore(testJson);
    let state = Session.getState();
    state = common.goToItem(state, 0);
    state = common.changeCategory(state, 0, [2]);
    expect(state.items[0].labels[0].category[0]).toBe(2);
});
