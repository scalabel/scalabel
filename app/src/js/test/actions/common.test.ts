import * as action from '../../action/creators';
import {testJson} from '../test_objects';
import Session from '../../common/session';
import {makeLabel} from '../../functional/states';
import * as _ from 'lodash';

test('Add and delete labels', () => {
    Session.devMode = false;
    Session.initStore(testJson);
    Session.dispatch(action.goToItem(0));
    const label = makeLabel({item: 0});
    Session.dispatch(action.addLabel(label, []));
    Session.dispatch(action.addLabel(label, []));
    let state = Session.getState();
    const labelId =
        state.current.maxObjectId;
    expect(_.size(state.items[0].labels)).toBe(2);
    expect(state.items[0].labels[labelId].item).toBe(0);
    Session.dispatch(action.deleteLabel(labelId));
    state = Session.getState();
    expect(_.size(state.items[0].labels)).toBe(1);
});

test('Change category', () => {
    Session.devMode = false;
    Session.initStore(testJson);
    Session.dispatch(action.goToItem(0));
    Session.dispatch(action.addLabel(makeLabel(), []));
    Session.dispatch(action.changeLabelProps(0, {category: [2]}));
    const state = Session.getState();
    expect(state.items[0].labels[0].category[0]).toBe(2);
});
