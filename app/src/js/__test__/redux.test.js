/* @flow */
import configureStore from '../functional/store/configure_store';
import * as types from '../functional/actions/action_types';
import {ActionCreators} from 'redux-undo';

let testJson = {
  'config': {
    'assignmentId': 'e6015077-aad9-4e60-a5ed-dbccf931a049',
    'projectName': 'Redux0',
    'itemType': 'image',
    'labelType': 'tag',
    'taskSize': 5,
    'handlerUrl': 'label2dv2',
    'pageTitle': 'Image Tagging Labeling Tool',
    'instructionPage': 'undefined',
    'demoMode': false,
    'bundleFile': 'image_v2.js',
    'categories': null,
    'attributes': [
      {
        'name': 'Weather',
        'toolType': 'list',
        'tagText': '',
        'tagPrefix': 'w',
        'tagSuffixes': [
          '',
          'r',
          's',
          'c',
          'o',
          'p',
          'f',
        ],
        'values': [
          'NA',
          'Rainy',
          'Snowy',
          'Clear',
          'Overcast',
          'Partly Cloudy',
          'Foggy',
        ],
        'buttonColors': [
          'white',
          'white',
          'white',
          'white',
          'white',
          'white',
          'white',
        ],
      },
      {
        'name': 'Scene',
        'toolType': 'list',
        'tagText': '',
        'tagPrefix': 's',
        'tagSuffixes': [
          '',
          't',
          'r',
          'p',
          'c',
          'g',
          'h',
        ],
        'values': [
          'NA',
          'Tunnel',
          'Residential',
          'Parking Lot',
          'City Street',
          'Gas Stations',
          'Highway',
        ],
        'buttonColors': [
          'white',
          'white',
          'white',
          'white',
          'white',
          'white',
          'white',
        ],
      },
      {
        'name': 'Timeofday',
        'toolType': 'list',
        'tagText': '',
        'tagPrefix': 't',
        'tagSuffixes': [
          '',
          'day',
          'n',
          'daw',
        ],
        'values': [
          'NA',
          'Daytime',
          'Night',
          'Dawn/Dusk',
        ],
        'buttonColors': [
          'white',
          'white',
          'white',
          'white',
        ],
      },
    ],
    'taskId': '000000',
    'workerId': 'default_worker',
    'startTime': 1539820189,
    'submitTime': 0,
  },
  'current': {
    'item': -1,
    'label': -1,
    'maxObjectId': -1,
  },
  'items': [
    {
      'id': 0,
      'index': 0,
      'url': 'https://s3-us-west-2.amazonaws.com/scalabel-public/demo/frames/intersection-0000051.jpg',
      'active': false,
      'loaded': false,
      'labels': [],
    },
    {
      'id': 1,
      'index': 1,
      'url': 'https://s3-us-west-2.amazonaws.com/scalabel-public/demo/frames/intersection-0000052.jpg',
      'active': false,
      'loaded': false,
      'labels': [],
    },
    {
      'id': 2,
      'index': 2,
      'url': 'https://s3-us-west-2.amazonaws.com/scalabel-public/demo/frames/intersection-0000053.jpg',
      'active': false,
      'loaded': false,
      'labels': [],
    },
    {
      'id': 3,
      'index': 3,
      'url': 'https://s3-us-west-2.amazonaws.com/scalabel-public/demo/frames/intersection-0000054.jpg',
      'active': false,
      'loaded': false,
      'labels': [],
    },
    {
      'id': 4,
      'index': 4,
      'url': 'https://s3-us-west-2.amazonaws.com/scalabel-public/demo/frames/intersection-0000055.jpg',
      'active': false,
      'loaded': false,
      'labels': [],
    },
  ],
  'labels': {},
  'tracks': {},
  'shapes': {},
  'actions': [],
};

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
      attributeName: 'Weather', // Weather
      selectedIndex: 2, // Snowy
    });
    state = store.getState().present;
    expect(state.labels[itemId].attributes['Weather']).toBe(2);
    store.dispatch({
      type: types.TAG_IMAGE,
      itemId: state.current.item,
      attributeName: 'Weather', // Weather
      selectedIndex: 3, // Snowy
    });
    state = store.getState().present;
    expect(state.labels[itemId].attributes['Weather']).toBe(3);
    store.dispatch(ActionCreators.undo());
    state = store.getState().present;
    expect(state.labels[itemId].attributes['Weather']).toBe(2);
    store.dispatch(ActionCreators.redo());
    state = store.getState().present;
    expect(state.labels[itemId].attributes['Weather']).toBe(3);
  });
});
