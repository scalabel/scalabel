import _ from 'lodash'
import * as types from '../action/types'
import { makeIndexedShape } from './states'
import {
  Select, State, UserType
} from './types'
import {
  removeObjectFields, updateListItem,
  updateObject
} from './util'

/**
 * Initialize state
 * @param {State} state
 * @return {State}
 */
export function initSession (state: State): State {
  // initialize state
  let session = state.session
  const items = session.items.slice()
  for (let i = 0; i < items.length; i++) {
    items[i] = updateObject(items[i], { loaded: false })
  }
  session = updateObject(session, { items })
  return updateObject(state, { session })
}

/**
 * Update the selected label in user
 * @param {UserType} user
 * @param {Partial<Select>} pselect partial selection
 */
function updateUserSelect (user: UserType, pselect: Partial<Select>): UserType {
  const select = updateObject(user.select, pselect)
  return updateObject(user, { select })
}

/**
 * Add new label. The ids of label and shapes will be updated according to
 * the current state.
 * @param {State} state: current state
 * @param {types.AddLabelAction} action
 * @return {State}
 */
export function addLabel (state: State, action: types.AddLabelAction): State {
  let { task, user } = state
  const session = state.session
  const itemIndex = action.itemIndex
  let label = action.label
  const shapes = action.shapes
  const newShapeId = task.status.maxShapeId + 1
  const labelId = task.status.maxLabelId + 1
  const shapeIds = _.range(shapes.length).map((i) => i + newShapeId)
  const newShapes = shapes.map(
    (s, i) => makeIndexedShape(shapeIds[i], labelId, true, s))
  const order = task.status.maxOrder + 1
  label = updateObject(label, {
    id: labelId, item: itemIndex, order,
    shapes: label.shapes.concat(shapeIds)
  })
  let item = state.task.items[itemIndex]
  const labels = updateObject(
    item.labels,
    { [labelId]: label })
  const allShapes = updateObject(item.shapes, _.zipObject(shapeIds, newShapes))
  item = updateObject(item, { labels, shapes: allShapes })
  const items = updateListItem(state.task.items, itemIndex, item)
  if (action.sessionId === session.id) {
    user = updateUserSelect(user, { label: labelId })
  }
  const status = updateObject(
    task.status,
    {
      maxLabelId: labelId,
      maxShapeId: shapeIds[shapeIds.length - 1],
      maxOrder: order
    })
  task = updateObject(task, { status, items })
  return { task, user, session }
}

/**
 * Update the properties of a shape
 * @param {State} state
 * @param {types.ChangeShapeAction} action
 * @return {State}
 */
export function changeShape (
  state: State, action: types.ChangeShapeAction): State {
  let { task, user } = state
  const itemIndex = action.itemIndex
  const shapeId = action.shapeId
  let item = state.task.items[itemIndex]
  let indexedShape = item.shapes[shapeId]
  indexedShape = updateObject(
    indexedShape, { shape: updateObject(indexedShape.shape, action.props) })
  item = updateObject(
    item, { shapes: updateObject(item.shapes, { [shapeId]: indexedShape }) })
  const selectedLabelId = (action.sessionId === state.session.id) ?
    indexedShape.label[0] : user.select.label
  const select = updateObject(user.select, { label: selectedLabelId })
  user = updateObject(user, { select })
  const items = updateListItem(state.task.items, itemIndex, item)
  task = updateObject(task, { items })
  return { ...state, task, user }
}

/**
 * Update label properties except shapes
 * @param {State} state
 * @param {types.ChangeLabelAction} action
 * @return {State}
 */
export function changeLabel (
  state: State, action: types.ChangeLabelAction): State {
  let { task, user } = state
  const itemIndex = action.itemIndex
  const labelId = action.labelId
  const props = action.props
  let item = task.items[itemIndex]
  const label = updateObject(item.labels[labelId], props)
  item = updateObject(
    item, { labels: updateObject(item.labels, { [labelId]: label }) })
  const items = updateListItem(task.items, itemIndex, item)
  task = updateObject(task, { items })
  if (action.sessionId === state.session.id) {
    user = updateUserSelect(user, { label: labelId })
  }
  return { ...state, user, task }
}

/**
 * Update the user selection
 * @param {State} state
 * @param {types.UpdateSelectAction} action
 */
export function updateSelect (
    state: State, action: types.UpdateSelectAction): State {
  const newSelect = updateObject(state.user.select, action.select)
  if (newSelect.item < 0 || newSelect.item >= state.task.items.length) {
    newSelect.item = state.user.select.item
  }
  return updateObject(state, { user: updateUserSelect(state.user, newSelect) })
}

/**
 * Signify a new item is loaded
 * @param {State} state
 * @param {types.LoadItemAction} action
 * @return {State}
 */
export function loadItem (state: State, action: types.LoadItemAction): State {
  const itemIndex = action.itemIndex
  let session = state.session
  session = updateObject(session, {
    items:
      updateListItem(session.items, itemIndex,
        updateObject(session.items[itemIndex], { loaded: true }))
  })
  return updateObject(state, { session })
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
  let { task, user } = state
  const itemIndex = action.itemIndex
  const labelId = action.labelId
  const item = state.task.items[itemIndex]
  const label = item.labels[labelId]
  const labels = removeObjectFields(item.labels, [labelId])
  // TODO: should we remove shapes?
  // depending on how boundary sharing is implemented.
  // remove labels
  const shapes = removeObjectFields(item.shapes, label.shapes)
  const items = updateListItem(state.task.items, itemIndex,
    updateObject(item, { labels, shapes }))
  task = updateObject(task, { items })
  // Reset selected object
  if (user.select.label === labelId) {
    user = updateUserSelect(user, { label: -1 })
  }
  return updateObject(state, { user, task })
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
  let user = state.user
  user = updateObject(user, {
    layout:
      updateObject(user.layout, {
        assistantView:
          !user.layout.assistantView
      })
  })
  return updateObject(state, { user })
}
