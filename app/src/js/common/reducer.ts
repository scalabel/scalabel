import { AnyAction, Reducer } from 'redux'
import * as types from '../action/types'
import * as common from '../functional/common'
import * as image from '../functional/image'
import * as pointCloud from '../functional/point_cloud'
import { makeState } from '../functional/states'
import * as tag from '../functional/tag'
import { State } from '../functional/types'

/**
 * Reducer
 * @param {State} currentState
 * @param {AnyAction} action
 * @return {State}
 */
export const reducer: Reducer<State> = (
    currentState: State = makeState(),
    action: AnyAction): State => {
  // Appending actions to action array
  // const newActions = currentState.actions.slice();
  // newActions.push(action);
  // const state = {...currentState, actions: newActions};
  const state = currentState
  // Apply reducers to state
  switch (action.type) {
    case types.INIT_SESSION:
      return common.initSession(state)
    case types.NEW_ITEM:
      return common.newItem(state, action as types.NewItemAction)
    case types.GO_TO_ITEM:
      return common.goToItem(state, action as types.GoToItemAction)
    case types.LOAD_ITEM:
      return common.loadItem(state, action as types.LoadItemAction)
    case types.UPDATE_ALL:
      return common.updateAll(state)
    case types.IMAGE_ZOOM:
      return image.zoomImage(state, action as types.ImageZoomAction)
    case types.ADD_LABEL:
      return common.addLabel(state, action as types.AddLabelAction)
    case types.CHANGE_LABEL_SHAPE:
      return common.changeShape(state, action as types.ChangeShapeAction)
    case types.CHANGE_LABEL_PROPS:
      return common.changeLabel(
        state, action as types.ChangeLabelAction)
    case types.DELETE_LABEL:
      return common.deleteLabel(state, action as types.DeleteLabelAction)
    case types.TAG_IMAGE:
      return tag.tagImage(state, action as types.TagImageAction)
    case types.TOGGLE_ASSISTANT_VIEW:
      return common.toggleAssistantView(state)
    case types.MOVE_CAMERA_AND_TARGET:
      return pointCloud.moveCameraAndTarget(
        state, action as types.MoveCameraAndTargetAction)
    default:
  }
  return state
}
