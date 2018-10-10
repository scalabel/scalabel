/* @flow */

import type {ItemType, LabelType, SatType} from '../types';
import {makeLabel, makeItem} from '../states';
import {updateObject, updateListItem} from './util';
import _ from 'lodash/fp';

/**
 * Create new label
 * @param {SatType} state: current state
 * @param {Function} createLabel: label creation function
 * @param {Object} optionalAttributes
 * @return {SatType}
 */
export function newLabel(
    state: SatType, createLabel: (number, Object) => LabelType,
    optionalAttributes: Object = {}): SatType {
  let labelId = state.current.maxObjectId + 1;
  let items = state.items;
  if (state.current.item > 0) {
    items = items.slice(); // shallow copy all elements
    let item = items[state.current.item];
    items[state.current.item] = {
      ...item, labels: item.labels.concat([labelId]),
    };
  }
  return {
    ...state,
    items: items,
    labels: {...state.labels,
      labelId: createLabel(labelId, optionalAttributes)},
    current: {...state.current, maxObjectId: labelId},
  };
}

/**
 * Create new Item
 * @param {SatType} state
 * @param {Function} createItem
 * @param {string} url
 * @return {SatType}
 */
export function newItem(
  state: SatType, createItem: (number, string) => ItemType,
  url: string): SatType {
  let id = state.items.length;
  let item = createItem(id, url);
  let items = state.items.slice();
  items.push(item);
  return {
    ...state,
    items: items,
  };
}

/**
 * load Sat State from json
 * @param {SatType} state
 * @param {Object} json
 * @return {SatType}
 */
export function decodeBaseJson(state: SatType, json: Object): SatType {
  let config = updateObject(state.config, {
                        assignmentId: json.id, // id
                        projectName: json.task.projectOptions.name,
                        itemType: json.task.projectOptions.itemType,
                        labelType: json.task.projectOptions.labelType,
                        taskSize: json.task.projectOptions.taskSize,
                        handlerUrl: json.task.projectOptions.handlerUrl,
                        pageTitle: json.task.projectOptions.pageTitle,
                        instructionPage: json.task.projectOptions.instructions,
                        demoMode: json.task.projectOptions.demoMode,
                        bundleFile: json.task.projectOptions.bundleFile,
                        categories: json.task.projectOptions.categories,
                        attributes: json.task.projectOptions.attributes,
                        taskId: json.task.index,
                        workerId: json.workerId,
                        startTime: json.startTime});
  let items = [];
  for (let i = 0; i < json.task.items.length; i++) {
    items.push(makeItem({id: json.task.items[i].index, ...json.task.items[i]}));
  }
  let labels = [];
  // TODO: add decode label here
  for (let i = 0; json.labels && i < json.labels.length; i++) {
    labels.push(makeLabel(json.labels[i]));
  }
  items = updateListItem(items, 0,
    {...items[0], active: true});
  let current = state.current;
  if (state.current.item === -1) {
    current = {...current, item: 0};
  }
  return updateObject(state, {
    config: config,
    items: items,
    labels: labels,
    current: current});
}

/**
 * Encode the Sat State
 * @param {SatType} state
 * @return {Object} - JSON representation of the base functionality in this
 *   SAT state.
 */
export function encodeBaseJson(state: SatType): Object {
  let items = [];
  let labeledItemsCount = 0;
  for (let i = 0; i < _.size(state.items); i++) {
    items.push(itemToJson(state, i));
    if (state.items[i].labels.length > 0) {
      labeledItemsCount++;
    }
  }
  let labels = [];
  for (let i = 0; i < _.size(state.labels); i++) {
    if (state.labels[i].valid) {
      labels.push(labelToJson(state, i));
    }
  }
  return {
    id: state.config.assignmentId,
    task: {
      projectOptions: {
        name: state.config.projectName,
        itemType: state.config.itemType,
        labelType: state.config.labelType,
        taskSize: state.config.taskSize,
        handlerUrl: state.config.handlerUrl,
        pageTitle: state.config.pageTitle,
        categories: state.config.categories,
        attributes: state.config.attributes,
        instructions: state.config.instructionPage,
        demoMode: state.config.demoMode,
        bundleFile: state.config.bundleFile, //
      },
      index: state.config.taskId, //
      items: items,
    },
    workerId: state.config.workerId,
    labels: labels,
    startTime: state.config.startTime,
    numLabeledItems: labeledItemsCount,
    userAgent: navigator.userAgent,
  };
}

/**
 * Get this item's JSON representation.
 * @param {SatType} state
 * @param {number} itemId
 * @return {object} JSON representation of this item
 */
export function itemToJson(state: SatType, itemId: number) {
  let item = state.items[itemId];
  return {url: item.url, index: item.index,
    labelIds: item.labels, attributes: item.attributes,
    labelImport: null, groundTruth: null,
    data: null};
  // TODO: change backend to have identical naming convention
  // TODO: instead of having to do `labelIds: item.labels`
  // TODO: so that arbitrary item json can be directly saved
}

/**
 * Return json object encoding the label information
 * @param {SatType} state
 * @param {number} labelId
 * @return {object} JSON representation of this label
 */
export function labelToJson(state: SatType, labelId: number) {
  let label = state.labels[labelId];
  let json = {
    id: label.id, categoryPath: label.categoryPath,
    attributes: label.attributes, parentId: label.parent,
    childrenIds: label.children,
  };
  return json;
}
