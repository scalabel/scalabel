/* @flow */
import configureStore from '../functional/store/configure_store';
import * as types from '../functional/actions/action_types';
import {ActionCreators} from 'redux-undo';

let testJson = {
  'actions': [
    {
      'type': '@@redux/INIT4.d.1.9.1.p',
    },
    {
      'attributeName': 'Weather',
      'itemId': 0,
      'selectedIndex': 2,
      'type': 'TAG_IMAGE',
    },
    {
      'attributeName': 'Scene',
      'itemId': 0,
      'selectedIndex': 1,
      'type': 'TAG_IMAGE',
    },
    {
      'attributeName': 'Timeofday',
      'itemId': 0,
      'selectedIndex': 2,
      'type': 'TAG_IMAGE',
    },
  ],
  'config': {
    'assignmentId': '65d9e070-163e-4dc4-8cbe-d049a4d05a32',
    'projectName': 'Redux0',
    'itemType': 'image',
    'labelType': 'tag',
    'taskSize': 5,
    'handlerUrl': 'label2d',
    'pageTitle': 'Image Tagging Labeling Tool',
    'instructionPage': 'undefined',
    'demoMode': false,
    'bundleFile': 'image_v2.js',
    'categories': [
      'person',
      'rider',
      'car',
      'truck',
      'bus',
      'train',
      'motor',
      'bike',
      'traffic sign',
      'traffic light',
    ],
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
    'startTime': 1538885790,
    'submitTime': 1538885819,
  },
  'current': {
    'item': 0,
    'label': 0,
    'maxObjectId': 0,
  },
  'items': [
    {
      'id': 0,
      'index': 0,
      'url': 'https://s3-us-west-2.amazonaws.com/scalabel-public/demo/frames/intersection-0000051.jpg',
      'active': true,
      'loaded': false,
      'labels': null,
      'attributes': {
        'Scene': 1,
        'Timeofday': 2,
        'Weather': 2,
      },
    },
    {
      'id': 1,
      'index': 1,
      'url': 'https://s3-us-west-2.amazonaws.com/scalabel-public/demo/frames/intersection-0000052.jpg',
      'active': false,
      'loaded': false,
      'labels': null,
      'attributes': null,
    },
    {
      'id': 2,
      'index': 2,
      'url': 'https://s3-us-west-2.amazonaws.com/scalabel-public/demo/frames/intersection-0000053.jpg',
      'active': false,
      'loaded': false,
      'labels': null,
      'attributes': null,
    },
    {
      'id': 3,
      'index': 3,
      'url': 'https://s3-us-west-2.amazonaws.com/scalabel-public/demo/frames/intersection-0000054.jpg',
      'active': false,
      'loaded': false,
      'labels': null,
      'attributes': null,
    },
    {
      'id': 4,
      'index': 4,
      'url': 'https://s3-us-west-2.amazonaws.com/scalabel-public/demo/frames/intersection-0000055.jpg',
      'active': false,
      'loaded': false,
      'labels': null,
      'attributes': null,
    },
  ],
  'Labels': null,
  'Shapes': null,
  'Tracks': null,
};

describe('Sat Redux Tests', function() {
  it('Initialize', function() {
    let store = configureStore(testJson, false);
    let state = store.getState().present;
    for (let i = 0; i < state.items.length; i++) {
      expect(state.items[i].id).toBe(i);
      expect(state.items[i].index).toBe(i);
    }
    expect(state.current.item).toBe(0);
  });
  it('Go to item', function() {
    let store = configureStore(testJson, false);
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
    let state = store.getState().present;
    store.dispatch({
      type: types.TAG_IMAGE,
      itemId: state.current.item,
      attributeName: 'Weather', // Weather
      selectedIndex: 2, // Snowy
    });
    state = store.getState().present;
    expect(state.items[0].attributes['Weather']).toBe(2);
    store.dispatch({
      type: types.TAG_IMAGE,
      itemId: state.current.item,
      attributeName: 'Weather', // Weather
      selectedIndex: 3, // Snowy
    });
    state = store.getState().present;
    expect(state.items[0].attributes['Weather']).toBe(3);
    store.dispatch(ActionCreators.undo());
    state = store.getState().present;
    expect(state.items[0].attributes['Weather']).toBe(2);
    store.dispatch(ActionCreators.redo());
    state = store.getState().present;
    expect(state.items[0].attributes['Weather']).toBe(3);
  });
});
