import {BaseController} from '../controllers/base_controller';
import type {StateType, ViewerConfigType, ItemType} from '../functional/types';
import {makeState} from '../functional/states';

/**
 * BaseViewer interface
 */
export class BaseViewer {
  // TODO: support temporary objects
  state: StateType;
  controller: BaseController;
  /**
   * General viewer constructor to initialize the viewer state
   * @param {BaseController} controller: controller object to listen to events
   */
  constructor(controller: BaseController) {
    this.state = makeState();
    this.controller = controller;
    controller.addViewer(this);
  }

  /**
   * map state of the store and the state of the controller
   * to the actual values needed by the render, so that
   * the render function does not need to fetch the values itself
   * @param {Object} state: partial state definition
   */
  updateState(state: StateType): void {
    this.state = state;
    this.redraw();
  }

  /**
   * Retrieve the current state
   * @return {StateType}
   */
  getState(): StateType {
    return this.state;
  }

  /**
   * Retrieve the current viewer configuration
   * @return {ViewerConfigType}
   */
  getCurrentViewerConfig(): ViewerConfigType {
    return this.state.items[this.state.current.item].viewerConfig;
  }

  /**
   * Get the current item in the state
   * @return {ItemType}
   */
  getCurrentItem(): ItemType {
    return this.state.items[this.state.current.item];
  }

  /**
   * Render the view
   * @return {boolean}: whether redraw is successful
   */
  redraw(): boolean {
    return true;
  }
}
