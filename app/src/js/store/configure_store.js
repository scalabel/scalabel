import {createStore} from 'redux';
import undoable, {includeAction} from 'redux-undo';
import {makeState} from '../functional/states';
import reducer from '../reducers/reducer';

import {
// SAT specific actions
  NEW_ITEM,
  GO_TO_ITEM,
  IMAGE_ZOOM,
  NEW_LABEL,
  DELETE_LABEL,
  TAG_IMAGE,
  CHANGE_ATTRIBUTE,
  CHANGE_CATEGORY,
  TOGGLE_ASSISTANT_VIEW,
} from '../actions/action_types';

const configureStore = (json: Object = {}, devMode: boolean = false) => {
  let store;
  const initialHistory = {
    past: [],
    present: makeState(json),
    future: [],
  };

  store = createStore(undoable(reducer, {
    limit: 20, // add a limit to history
    filter: includeAction([
      // undoable actions
      NEW_ITEM,
      GO_TO_ITEM,
      IMAGE_ZOOM,
      NEW_LABEL,
      DELETE_LABEL,
      TAG_IMAGE,
      CHANGE_ATTRIBUTE,
      CHANGE_CATEGORY,
      TOGGLE_ASSISTANT_VIEW,
    ]),
    debug: devMode,
  }), initialHistory);

  return store;
};

export default configureStore;
