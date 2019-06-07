import {makeState} from '../functional/states';
import * as types from '../actions/action_types';
import * as common from '../functional/common';
import * as image from '../functional/image';
import * as tag from '../functional/tag';
import * as box2d from '../functional/box2d';
import * as pointCloud from '../functional/point_cloud';
import {StateType} from '../functional/types';
import {ActionTypes} from '../actions/action_types';

/**
 * Reducer
 * @param {StateType} currentState
 * @param {object} action
 * @return {StateType}
 */
export function reducer(
    currentState: StateType = makeState(),
    action: ActionTypes): StateType {
  // Appending actions to action array
  const newActions = currentState.actions.slice();
  newActions.push(action);
  const state = {...currentState, actions: newActions};
  // Apply reducers to state
  switch (action.type) {
    case types.INIT_SESSION:
      return common.initSession(state);
    case types.NEW_ITEM:
      return common.newItem(state, action.createItem, action.url);
    case types.GO_TO_ITEM:
      return common.goToItem(state, action.index);
    case types.LOAD_ITEM:
      return common.loadItem(state, action.index, action.config);
    case types.UPDATE_ALL:
      return common.updateAll(state);
    case types.IMAGE_ZOOM:
      return image.zoomImage(state, action.ratio);
    case types.NEW_LABEL:
      return common.newLabel(state, action.itemId,
          action.createLabel, action.optionalAttributes);
    case types.NEW_IMAGE_BOX2D_LABEL:
      return box2d.newImageBox2DLabel(state, action.itemId,
        action.optionalAttributes);
    case types.DELETE_LABEL:
      return common.deleteLabel(state, action.itemId, action.labelId);
    case types.TAG_IMAGE:
      return tag.tagImage(state, action.itemId,
          action.attributeIndex, action.selectedIndex);
    case types.CHANGE_ATTRIBUTE:
      return common.changeAttribute(state, action.labelId,
          action.attributeOptions);
    case types.CHANGE_CATEGORY:
      return common.changeCategory(state, action.labelId,
          action.categoryOptions);
    case types.TOGGLE_ASSISTANT_VIEW:
      return common.toggleAssistantView(state);

    case types.CHANGE_RECT:
      return box2d.changeRect(state, action.shapeId,
          action.targetBoxAttributes);

    case types.MOVE_CAMERA:
      return pointCloud.moveCamera(state, action.newPosition);
    case types.MOVE_CAMERA_AND_TARGET:
      return pointCloud.moveCameraAndTarget(state, action.newPosition,
        action.newTarget);
    default:
  }
  return state;
}
