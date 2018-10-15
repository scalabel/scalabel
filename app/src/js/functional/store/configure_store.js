/* @flow */
import {createStore} from 'redux';
import undoable, {includeAction} from 'redux-undo';
import {makeSat} from '../../states';
import reducer from '../reducers/reducer';

import {
// SAT specific actions
  NEW_ITEM,
  GO_TO_ITEM,
  NEW_LABEL,
  DELETE_LABEL,
  TAG_IMAGE,
  CHANGE_ATTRIBUTE,
  CHANGE_CATEGORY,
} from '../actions/action_types';

const configureStore = (json: Object = {}, devMode: boolean = false) => {
  let store;
  const initialHistory = {
    past: [],
    present: makeSat(json),
    future: [],
  };

  store = createStore(undoable(reducer, {
    limit: 20, // add a limit to history
    filter: includeAction([
      // undoable actions
      NEW_ITEM,
      GO_TO_ITEM,
      NEW_LABEL,
      DELETE_LABEL,
      TAG_IMAGE,
      CHANGE_ATTRIBUTE,
      CHANGE_CATEGORY,
    ]),
    debug: devMode,
  }), initialHistory);

  return store;
};

export default configureStore;
