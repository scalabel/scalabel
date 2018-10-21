import * as types from '../actions/action_types';

/**
 * Initialize Page Control
 * @param {Object} store
 */
export function PageControl(store: Object) {
  let self = this;
  this.store = store;
  // buttons
  let prevBtn = document.getElementById('prev_btn');
  if (prevBtn) {
    prevBtn.onclick = function() {
      self._prevHandler();
    };
  }
  let nextBtn = document.getElementById('next_btn');
  if (nextBtn) {
    nextBtn.onclick = function() {
      self._nextHandler();
    };
  }
  let saveBtn = document.getElementById('save_btn');
  if (saveBtn) {
    saveBtn.onclick = function() {
      self.save();
    };
  }
}

/**
 * Prev button handler
 */
PageControl.prototype._prevHandler = function() {
  let index = this.store.getState().present.current.item;
  this.store.dispatch({
    type: types.GO_TO_ITEM,
    index: index - 1,
  });
};

/**
 * Next button handler
 */
PageControl.prototype._nextHandler = function() {
  let index = this.store.getState().present.current.item;
  this.store.dispatch({
    type: types.GO_TO_ITEM,
    index: index + 1,
  });
};

PageControl.prototype.save = function() {
  let state = this.store.getState().present;
  let xhr = new XMLHttpRequest();
  xhr.open('POST', './postSaveV2');
  xhr.send(JSON.stringify(state));
};
