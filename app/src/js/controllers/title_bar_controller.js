import * as types from '../action/types';
import * as actions from '../action/common';
import {
    BaseController
} from './base_controller';

/**
 * TitleBarController provides callback functions for TitleBarViewer
 */
export class TitleBarController extends BaseController {
    /**
     * Go to the previous Item
     */
    goToPreviousItem() {
        let index = this.getState().current.item;
        this.dispatch(actions.goToItem(index - 1));
    }

    /**
     * Go to the next Item
     */
    goToNextItem() {
        let index = this.getState().current.item;
        this.dispatch(actions.goToItem(index + 1));
    }

    /**
     * Save the current state to the server
     */
    save() {
        let state = this.getState();
        let xhr = new XMLHttpRequest();
        xhr.open('POST', './postSaveV2');
        xhr.send(JSON.stringify(state));
    }

    /**
     * turn assistant view on/off
     */
    toggleAssistantView() {
        this.dispatch({
            type: types.TOGGLE_ASSISTANT_VIEW
        });
    }
}