
/**
 * Viewer base class
 */
export class Viewer {
  /**
   * map state of the store and the state of the controller
   * to the actual values needed by the redraw, so that
   * the redraw function does not need to fetch the values itself
   * @param {Object} ignoredState
   * @param {Object} ignoredControllerState
   */
  getState(ignoredState: Object, ignoredControllerState: Object) {}

  /**
   * Should call mapStateToProps first to fetch necessary values
   * @param {Object} ignoredState
   * @param {Object} ignoredControllerState
   */
  redraw(ignoredState: Object, ignoredControllerState: Object) {}
}
