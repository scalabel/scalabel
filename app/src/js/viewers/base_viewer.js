/* :: import {BaseController} from '../controllers/base_controller'; */
import type {
  State,
  ViewerConfigType,
  ItemType,
  LabelType,
} from '../functional/types';
import {makeState} from '../functional/states';

/**
 * BaseViewer interface
 */
export class BaseViewer {
  // TODO: support temporary objects
  state: State;
  fastState: State;
  /* :: controller: $Subtype<BaseController>; */
  /**
   * General viewer constructor to initialize the viewer state
   * @param {BaseController} controller: controller object to listen to events
   */
  constructor(controller/* : $Subtype<BaseController> */) {
    this.state = makeState();
    this.fastState = makeState();
    this.controller = controller;
    controller.addViewer(this);
  }

  /**
   * map state of the store and the state of the controller
   * to the actual values needed by the render, so that
   * the render function does not need to fetch the values itself
   * @param {State} state: state definition
   */
  updateState(state: State): void {
    this.state = state;
    this.redraw();
  }

  /**
   * Update the fast state. Some of the viewers like ImageViewer will ignore it.
   * So the state change is ignored by default.
   * @param {State} state: state definition
   */
  updateFastState(state: State): void {
    this.fastState = state;
  }

  /**
   * Retrieve the current state
   * @return {State}
   */
  getState(): State {
    return this.state;
  }

  /**
   * Get the label from the combination of state and fastState
   * @param {number} labelId: id of target label
   * @return {LabelType}
   */
  getLabel(labelId: number): LabelType {
    if ('labels' in this.fastState && labelId in this.fastState.labels) {
      return this.fastState.labels[labelId];
    } else {
      return this.state.labels[labelId];
    }
  }

  /**
   * Get the shape from the combination of state and fastState
   * @param {number} shapeId: id of target shape
   * @return {Object}
   */
  getShape(shapeId: number): Object {
    if ('shapes' in this.fastState && shapeId in this.fastState.shapes) {
      return this.fastState.shapes[shapeId];
    } else {
      return this.state.shapes[shapeId];
    }
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
