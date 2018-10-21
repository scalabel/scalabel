
import {makeState} from '../functional/states';
import * as types from '../actions/action_types';
import * as common from '../functional/common';
import * as tag from '../functional/tag';
import type {StateType} from '../functional/types';

/**
 * Reducer
 * @param {StateType} currentState
 * @param {object} action
 * @return {StateType}
 */
export default function(
    currentState: StateType = makeState(),
    action: Object): StateType {
  // Appending actions to action array
  let newActions = currentState.actions.slice();
  newActions.push(action);
  let state = {...currentState, actions: newActions};
  // Apply reducers to state
  switch (action.type) {
    case types.INIT_SESSION:
      return common.initSession(state);
    case types.NEW_ITEM:
      return common.newItem(state, action.createItem, action.url);
    case types.GO_TO_ITEM:
      return common.goToItem(state, action.index);
    case types.NEW_LABEL:
      return common.newLabel(state, action.itemId,
          action.createLabel, action.optionalAttributes);
    case types.DELETE_LABEL:
      return common.deleteLabel(state, action.itemId, action.labelId);
    case types.TAG_IMAGE:
      return tag.tagImage(state, action.itemId,
          action.attributeName, action.selectedIndex);
    case types.CHANGE_ATTRIBUTE:
      return common.changeAttribute(state, action.labelId,
          action.attributeOptions);
    case types.CHANGE_CATEGORY:
      return common.changeCategory(state, action.labelId,
          action.categoryOptions);
    default:
  }
  return state;
}
