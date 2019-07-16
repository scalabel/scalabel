import _ from 'lodash'
import * as types from '../action/types'
import {
  State
} from './types'
import {removeObjectFields, updateListItem, updateListItems,
  updateObject
} from './util'

/**
 * Initialize state
 * @param {State} state
 * @return {State}
 */
export function initSession (state: State): State {
  // initialize state
  let items = state.items.slice()
  for (let i = 0; i < items.length; i++) {
    items[i] = updateObject(items[i], { loaded: false })
  }
  state = updateObject(state, { items })
  if (state.current.item === -1) {
    const current = updateObject(state.current, { item: 0 })
    items = updateListItem(
        state.items, 0, updateObject(state.items[0], { active: true }))
    return updateObject(state, { current, items })
  } else {
    return state
  }
}

/**
 * Add new label. The ids of label and shapes will be updated according to
 * the current state.
 * @param {State} state: current state
 * @param {types.AddLabelAction} action
 * @return {State}
 */
export function addLabel (state: State, action: types.AddLabelAction): State {
  const itemIndex = action.itemIndex
  let label = action.label
  const shapes = action.shapes
  const newShapeId = state.current.maxShapeId + 1
  const labelId = state.current.maxLabelId + 1
  const shapeIds = _.range(shapes.length).map((i) => i + newShapeId)
  const newShapes = shapes.map(
    (s, i) => updateObject(s, { label: labelId, id: shapeIds[i] }))
  const order = state.current.maxOrder + 1
  label = updateObject(label, {id: labelId, item: itemIndex, order,
    shapes: label.shapes.concat(shapeIds)})
  let item = state.items[itemIndex]
  const labels = updateObject(
      item.labels,
      { [labelId]: label })
  const allShapes = updateObject(item.shapes, _.zipObject(shapeIds, newShapes))
  item = updateObject(item, { labels, shapes: allShapes })
  const items = updateListItem(state.items, itemIndex, item)
  const selectedLabelId = (action.sessionId === state.config.sessionId) ?
    labelId : state.current.label
  const current = updateObject(
      state.current,
    {
      label: selectedLabelId,
      maxLabelId: labelId,
      maxShapeId: shapeIds[shapeIds.length - 1],
      maxOrder: order
    })
  return {
    ...state,
    items,
    current
  }
}

/**
 * Update the properties of a shape
 * @param {State} state
 * @param {types.ChangeShapeAction} action
 * @return {State}
 */
export function changeShape (
  state: State, action: types.ChangeShapeAction): State {
  const itemIndex = action.itemIndex
  const shapeId = action.shapeId
  let item = state.items[itemIndex]
  let shape = item.shapes[shapeId]
  const props = updateObject(action.props, { id: shapeId, label: shape.label })
  shape = updateObject(shape, props)
  item = updateObject(
      item, { shapes: updateObject(item.shapes, { [shapeId]: shape }) })
  const selectedLabelId = (action.sessionId === state.config.sessionId) ?
    shape.label : state.current.label
  const current = updateObject(state.current, { label: selectedLabelId })
  const items = updateListItem(state.items, itemIndex, item)
  return { ...state, items, current }
}

/**
 * Update label properties except shapes
 * @param {State} state
 * @param {types.ChangeLabelAction} action
 * @return {State}
 */
export function changeLabel (
    state: State, action: types.ChangeLabelAction): State {
  const itemIndex = action.itemIndex
  const labelId = action.labelId
  const props = action.props
  let item = state.items[itemIndex]
  const label = updateObject(item.labels[labelId], props)
  item = updateObject(
      item, { labels: updateObject(item.labels, { [labelId]: label }) })
  const items = updateListItem(state.items, itemIndex, item)
  const selectedLabelId = (action.sessionId === state.config.sessionId) ?
    labelId : state.current.label
  const current = updateObject(state.current, { label: selectedLabelId })
  return { ...state, items, current }
}

/**
 * Create Item from url with provided creator
 * @param {State} state
 * @param {types.NewItemAction} action
 * @return {State}
 */
export function newItem (state: State, action: types.NewItemAction): State {
  const [createItem, url] = [action.createItem, action.url]
  const id = state.items.length
  const item = createItem(id, url)
  const items = state.items.slice()
  items.push(item)
  return {
    ...state,
    items
  }
}

/**
 * Go to item at index
 * @param {State} state
 * @param {types.GoToItemAction} actoin
 * @return {State}
 */
export function goToItem (state: State, action: types.GoToItemAction): State {
  const index = action.itemIndex
  if (index < 0 || index >= state.items.length) {
    return state
  }
  // TODO: don't do circling when no image number is shown
  // index = (index + state.items.length) % state.items.length;
  if (index === state.current.item) {
    return state
  }
  const deactivatedItem = updateObject(state.items[state.current.item],
      { active: false })
  const activatedItem = updateObject(state.items[index], { active: true })
  const items = updateListItems(state.items,
      [state.current.item, index],
      [deactivatedItem, activatedItem])
  const current = { ...state.current, item: index }
  return updateObject(state, { items, current })
}

/**
 * Signify a new item is loaded
 * @param {State} state
 * @param {number} itemIndex
 * @param {ViewerConfigType} viewerConfig
 * @return {State}
 */
export function loadItem (state: State, action: types.LoadItemAction): State {
  const itemIndex = action.itemIndex
  const viewerConfig = action.config
  return updateObject(
      state, {items: updateListItem(
          state.items, itemIndex,
            updateObject(state.items[itemIndex],
                { viewerConfig, loaded: true }))})
}

// TODO: now we are using redux, we have all the history anyway,
// TODO: do we still need to keep around all labels in current state?
/**
 * Deconstruct given label
 * @param {State} state
 * @param {number} itemIndex
 * @param {number} labelId
 * @return {State}
 */
export function deleteLabel (
    state: State, action: types.DeleteLabelAction): State {
  const itemIndex = action.itemIndex
  const labelId = action.labelId
  const item = state.items[itemIndex]
  const label = item.labels[labelId]
  const labels = removeObjectFields(item.labels, [labelId])
  // TODO: should we remove shapes?
  // depending on how boundary sharing is implemented.
  // remove labels
  const shapes = removeObjectFields(item.shapes, label.shapes)
  const items = updateListItem(state.items, itemIndex,
  updateObject(item, { labels, shapes }))
  // Reset selected object
  let current = state.current
  if (current.label === labelId) {
    current = updateObject(current, { label: -1 })
  }
  return updateObject(
  state, { current, items })
}

/**
 * assign Attribute to a label
 * @param {State} state
 * @param {number} _labelId
 * @param {object} _attributeOptions
 * @return {State}
 */
export function changeAttribute (state: State, _labelId: number,
                                 _attributeOptions: object): State {
  return state
}

/**
 * Notify all the subscribers to update. it is an no-op now.
 * @param {State} state
 * @return {State}
 */
export function updateAll (state: State): State {
  return state
}

/**
 * turn on/off assistant view
 * @param {State} state
 * @return {State}
 */
export function toggleAssistantView (state: State): State {
  return updateObject(state, {layout:
            updateObject(state.layout, {assistantView:
                  !state.layout.assistantView})})
}
