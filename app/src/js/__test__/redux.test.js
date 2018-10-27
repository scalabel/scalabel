import configureStore from '../store/configure_store';
import * as types from '../actions/action_types';
import {ActionCreators} from 'redux-undo';
import {testJson} from './test_objects';

describe('Sat Redux Tests', function() {
  it('Initialize', function() {
    let store = configureStore(testJson, false);
    store.dispatch({type: types.INIT_SESSION});
    let state = store.getState().present;
    for (let i = 0; i < state.items.length; i++) {
      expect(state.items[i].id).toBe(i);
      expect(state.items[i].index).toBe(i);
    }
    expect(state.current.item).toBe(0);
  });
  it('Go to item', function() {
    let store = configureStore(testJson, false);
    store.dispatch({type: types.INIT_SESSION});
    let state = store.getState().present;
    expect(state.current.item).toBe(0);
    for (let i = 0; i < state.items.length; i++) {
      if (i === 0) {
        expect(state.items[i].active).toBe(true);
      } else {
        expect(state.items[i].active).toBe(false);
      }
    }
    store.dispatch({
      type: types.GO_TO_ITEM,
      index: 2,
    });
    state = store.getState().present;
    expect(state.current.item).toBe(2);
    for (let i = 0; i < state.items.length; i++) {
      if (i === 2) {
        expect(state.items[i].active).toBe(true);
      } else {
        expect(state.items[i].active).toBe(false);
      }
    }
    store.dispatch(ActionCreators.undo());
    state = store.getState().present;
    expect(state.current.item).toBe(0);
    for (let i = 0; i < state.items.length; i++) {
      if (i === 0) {
        expect(state.items[i].active).toBe(true);
      } else {
        expect(state.items[i].active).toBe(false);
      }
    }
    store.dispatch(ActionCreators.redo());
    state = store.getState().present;
    expect(state.current.item).toBe(2);
    for (let i = 0; i < state.items.length; i++) {
      if (i === 2) {
        expect(state.items[i].active).toBe(true);
      } else {
        expect(state.items[i].active).toBe(false);
      }
    }
  });
  it('Image Tagging', function() {
    let store = configureStore(testJson, false);
    store.dispatch({type: types.INIT_SESSION});
    let state = store.getState().present;
    let itemId = state.current.item;
    store.dispatch({
      type: types.TAG_IMAGE,
      itemId: itemId,
      attributeIndex: 0, // Weather
      selectedIndex: 2, // Snowy
    });
    state = store.getState().present;
    expect(state.labels[itemId].attributes[0]).toBe(2);
    store.dispatch({
      type: types.TAG_IMAGE,
      itemId: state.current.item,
      attributeIndex: 0, // Weather
      selectedIndex: 3, // Snowy
    });
    state = store.getState().present;
    expect(state.labels[itemId].attributes[0]).toBe(3);
    store.dispatch(ActionCreators.undo());
    state = store.getState().present;
    expect(state.labels[itemId].attributes[0]).toBe(2);
    store.dispatch(ActionCreators.redo());
    state = store.getState().present;
    expect(state.labels[itemId].attributes[0]).toBe(3);
  });
});
