/**
 * Main functions for transforming state
 * NOTE: All the functions should be pure
 * Pure Function: https://en.wikipedia.org/wiki/Pure_function
 */
import _ from 'lodash'
import * as types from '../action/types'
import { makeIndexedShape, makeTrack } from './states'
import {
  IndexedShapeType, ItemType, LabelType, Select, ShapeType, State,
  TaskStatus, TaskType, TrackMapType, TrackType, UserType
} from './types'
import {
  assignToArray, getObjectKeys,
  pickArray,
  pickObject,
  removeListItems,
  removeObjectFields,
  updateListItem,
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
 * Update the task in state
 * @param {State} state: current state
 * @param {types.UpdateTaskAction} action
 */
export function updateTask (
  state: State,
  action: types.UpdateTaskAction): State {
  return updateObject(state, { task: _.cloneDeep(action.newTask) })
}

/**
 * Add new label. The ids of label and shapes will be updated according to
 * the current state.
 * @param {State} state: current state
 * @param {types.AddLabelAction} action
 * @return {State}
 */
export function addLabel (
  state: State, sessionId: string, itemIndex: number, label: LabelType,
  shapeTypes: string[] = [], shapes: ShapeType[] = []): State {
  const addLabelsAction: types.AddLabelsAction = {
    type: types.ADD_LABELS,
    sessionId,
    itemIndices: [itemIndex],
    labels: [[label]],
    shapeTypes: [[shapeTypes]],
    shapes: [[shapes]]
  }
  return addLabels(state, addLabelsAction)
}

/**
 * Add news labels to one item
 * @param item
 * @param taskStatus
 * @param label
 * @param shapeTypes
 * @param shapes
 */
function addLabelsToItem (
    item: ItemType, taskStatus: TaskStatus, newLabels: LabelType[],
    shapeTypes: string[][], shapes: ShapeType[][]
  ): [ItemType, LabelType[], TaskStatus] {
  newLabels = [...newLabels]
  const newLabelIds: number[] = []
  const newShapeIds: number[] = []
  const newShapes: IndexedShapeType[] = []
  newLabels.forEach((label, index) => {
    const newShapeId = taskStatus.maxShapeId + 1 + newShapes.length
    const labelId = taskStatus.maxLabelId + 1 + index
    const shapeIds = _.range(shapes[index].length).map((i) => i + newShapeId)
    const newLabelShapes = shapes[index].map((s, i) =>
        makeIndexedShape(shapeIds[i], [labelId], shapeTypes[index][i], s)
      )
    const order = taskStatus.maxOrder + 1 + index
    label = updateObject(label, {
      id: labelId,
      item: item.index,
      order,
      shapes: label.shapes.concat(shapeIds)
    })
    newLabels[index] = label
    newLabelIds.push(labelId)
    newShapes.push(...newLabelShapes)
    newShapeIds.push(...shapeIds)
  })
  const labels = updateObject(
    item.labels, _.zipObject(newLabelIds, newLabels))
  const allShapes = updateObject(
    item.shapes,
    _.zipObject(newShapeIds, newShapes)
  )
  item = updateObject(item, { labels, shapes: allShapes })
  taskStatus = updateObject(
    taskStatus,
    {
      maxLabelId: newLabelIds[newLabelIds.length - 1],
      maxOrder: taskStatus.maxOrder + newLabels.length
    })
  if (newShapeIds.length !== 0) {
    taskStatus = updateObject(
      taskStatus,
      {
        maxShapeId: newShapeIds[newShapeIds.length - 1]
      })
  }
  return [item, newLabels, taskStatus]
}

/**
 * Add labels to multiple items
 * @param item
 * @param taskStatus
 * @param newLabels
 * @param shapeTypes
 * @param shapes
 */
function addLabelstoItems (
  items: ItemType[], taskStatus: TaskStatus, labelsToAdd: LabelType[][],
  shapeTypes: string[][][], shapes: ShapeType[][][]
  ): [ItemType[], LabelType[], TaskStatus] {
  const allNewLabels: LabelType[] = []
  items = [...items]
  items.forEach((item, index) => {
    const [newItem, newLabels, newStatus] = addLabelsToItem(
      item, taskStatus, labelsToAdd[index], shapeTypes[index], shapes[index])
    items[index] = newItem
    taskStatus = newStatus
    allNewLabels.push(...newLabels)
  })
  return [items, allNewLabels, taskStatus]
}

/**
 * Add new label. The ids of label and shapes will be updated according to
 * the current state.
 * @param {State} state: current state
 * @param {types.AddLabelsAction} action
 * @return {State}
 */
export function addLabels (state: State, action: types.AddLabelsAction): State {
  let { task, user } = state
  const session = state.session
  let items = [...task.items]
  const selectedItems = pickArray(items, action.itemIndices)
  const [newItems, newLabels, status] = addLabelstoItems(
    selectedItems, task.status, action.labels, action.shapeTypes, action.shapes)
  items = assignToArray(items, newItems, action.itemIndices)
  // Find the first new label in the selected item if the labels are created
  // by this session.
  if (action.sessionId === session.id) {
    for (const label of newLabels) {
      if (label.item === user.select.item) {
        user = updateUserSelect(user, {
          labels: [label.id],
          category: label.category[0],
          attributes: label.attributes
        })
        break
      }
    }
  }
  task = updateObject(task, { status, items })
  return { task, user, session }
}

/**
 * Add one track to task
 * @param task
 * @param itemIndices
 * @param labels
 * @param shapeTypes
 * @param shapes
 */
function addTrackToTask (
  task: TaskType,
  itemIndices: number[],
  labels: LabelType[],
  shapeTypes: string[][],
  shapes: ShapeType[][]
): [TaskType, TrackType, LabelType[]] {
  const track = makeTrack(task.status.maxTrackId + 1, {})
  for (const label of labels) {
    label.track = track.id
  }
  const labelList = labels.map((l) => [l])
  const shapeTypeList = shapeTypes.map((s) => [s])
  const shapeList = shapes.map((s) => [s])
  const [newItems, newLabels, status] = addLabelstoItems(
    pickArray(task.items, itemIndices),
    task.status, labelList, shapeTypeList, shapeList)
  const items = assignToArray(task.items, newItems, itemIndices)
  newLabels.map((l) => {
    track.labels[l.item] = l.id
  })
  const tracks = updateObject(task.tracks, { [track.id]: track })
  status.maxTrackId += 1
  task = { ...task, items, status, tracks }
  return [ task, track, newLabels ]
}

/**
 * Add track action
 * @param {State} state
 * @param {types.AddTrackAction} action
 */
export function addTrack (state: State, action: types.AddTrackAction): State {
  let { user } = state
  const [task,, newLabels] = addTrackToTask(
    state.task, action.itemIndices, action.labels,
    action.shapeTypes, action.shapes
  )
  // select the label on the current item
  if (action.sessionId === state.session.id) {
    for (const l of newLabels) {
      if (l.item === user.select.item) {
        user = updateUserSelect(user, { labels: [l.id] })
        break
      }
    }
  }
  return { ...state, user, task }
}

/**
 * Merge tracks and items
 * @param tracks
 * @param items
 */
function mergeTracksInItems (
  tracks: TrackType[], items: ItemType[]): [TrackType, ItemType[]] {
  tracks = [...tracks]
  const labelIds: number[][] = _.range(items.length).map(() => [])
  const props: Array<Array<Partial<LabelType>>> = _.range(
    items.length).map(() => [])
  const prop: Partial<LabelType> = { track: tracks[0].id }
  const track = _.cloneDeep(tracks[0])
  for (let i = 1; i < tracks.length; i += 1) {
    _.forEach(tracks[i].labels, (labelId, itemIndex) => {
      labelIds[Number(itemIndex)].push(labelId)
      props[Number(itemIndex)].push(prop)
    })
    track.labels = { ...track.labels, ...tracks[i].labels }
  }
  items = changeLabelsInItems(items, labelIds, props)
  return [track, items]
}

/**
 * Merge tracks action
 * @param state
 * @param action
 */
export function mergeTracks (
  state: State, action: types.MergeTrackAction): State {
  let { task } = state
  const mergedTracks = action.trackIds.map((trackId) => task.tracks[trackId])
  const tracks = removeObjectFields(task.tracks, action.trackIds)
  const [track, items] = mergeTracksInItems(mergedTracks, task.items)
  tracks[track.id] = track
  task = updateObject(task, { items, tracks })
  return { ...state, task }
}

/**
 * update shapes in an item
 * @param item
 * @param shapeIds
 * @param shapes
 */
function changeShapesInItem (
  item: ItemType, shapeIds: number[],
  shapes: Array<Partial<ShapeType>>): ItemType {
  const newShapes = { ...item.shapes }
  shapeIds.forEach((shapeId, index) => {
    newShapes[shapeId] = updateObject(newShapes[shapeId],
      { shape: updateObject(newShapes[shapeId].shape, shapes[index]) })
  })
  return { ...item, shapes: newShapes }
}

/**
 * changes shapes in items
 * @param items
 * @param shapeIds
 * @param shapes
 */
function changeShapesInItems (
  items: ItemType[], shapeIds: number[][],
  shapes: Array<Array<Partial<ShapeType>>>): ItemType[] {
  items = [...items]
  items.forEach((item, index) => {
    items[index] = changeShapesInItem(item, shapeIds[index], shapes[index])
  })
  return items
}

/**
 * Change shapes action
 * @param state
 * @param action
 */
export function changeShapes (
    state: State, action: types.ChangeShapesAction): State {
  let { task, user } = state
  const shapeIds = action.shapeIds
  const newItems = changeShapesInItems(
    pickArray(task.items, action.itemIndices), shapeIds, action.shapes)
  const items = assignToArray(task.items, newItems, action.itemIndices)
  // select the label of the first shape on the current item
  if (action.sessionId === state.session.id) {
    const index = _.find(action.itemIndices, (v) => v === user.select.item)
    if (index !== undefined) {
      const labelId = items[index].shapes[shapeIds[0][0]].label[0]
      user = updateUserSelect(user, { labels: [labelId] })
    }
  }
  task = updateObject(task, { items })
  return { ...state, task, user }
}

/**
 * Change properties of labels in one item
 * @param item
 * @param labels
 */
function changeLabelsInItem (
  item: ItemType,
  labelIds: number[],
  props: Array<Partial<LabelType>>
  ): ItemType {
  const newLabels: {[key: number]: LabelType} = {}
  labelIds.forEach((labelId, index) => {
    const oldLabel = item.labels[labelId]
    // avoid changing the shape field in the label
    newLabels[labelId] = updateObject(
        oldLabel, { ..._.cloneDeep(props[index]), shapes: oldLabel.shapes })
  })
  item = updateObject(item, { labels: updateObject(item.labels, newLabels) })
  return item
}

/**
 * Change properties of labels in one item
 * @param items
 * @param labels
 */
function changeLabelsInItems (
  items: ItemType[], labelIds: number[][],
  props: Array<Array<Partial<LabelType>>>): ItemType[] {
  items = [...items]
  items.forEach((item, index) => {
    items[index] = changeLabelsInItem(item, labelIds[index], props[index])
  })
  return items
}

/**
 * Change labels action
 * @param state
 * @param action
 */
export function changeLabels (
  state: State, action: types.ChangeLabelsAction): State {
  let items = pickArray(state.task.items, action.itemIndices)
  items = changeLabelsInItems(items, action.labelIds, action.props)
  items = assignToArray(state.task.items, items, action.itemIndices)
  const task = updateObject(state.task, { items })
  return { ...state, task }
}

/**
 * Get the label id of the root of a label by tracing its ancestors
 * @param item
 * @param labelId
 */
export function getRootLabelId (item: ItemType, labelId: number): number {
  let parent = item.labels[labelId].parent
  while (parent >= 0) {
    labelId = parent
    parent = item.labels[labelId].parent
  }
  return labelId
}

/**
 * Get the trackId of the root of a label by tracing its ancestors
 * @param item
 * @param labelId
 */
export function getRootTrackId (item: ItemType, labelId: number): number {
  let parent = item.labels[labelId].parent
  while (parent >= 0) {
    labelId = parent
    parent = item.labels[labelId].parent
  }
  return item.labels[labelId].track
}

/**
 * Link two labels on the same item
 * The new label properties are the same as label1 in action
 * @param {State} state
 * @param {types.LinkLabelsAction} action
 */
export function linkLabels (
    state: State, action: types.LinkLabelsAction): State {
  // Add a new label to the state
  let item = state.task.items[action.itemIndex]
  if (action.labelIds.length === 0) {
    return state
  }
  const children = _.map(
    action.labelIds, (labelId) => getRootLabelId(item, labelId))
  let newLabel: LabelType = _.cloneDeep(item.labels[children[0]])
  newLabel.parent = -1
  newLabel.shapes = []
  newLabel.children = children
  state = addLabel(state, action.sessionId, action.itemIndex, newLabel)

  // assign the label properties
  item = state.task.items[action.itemIndex]
  const newLabelId = state.task.status.maxLabelId
  newLabel = item.labels[newLabelId]
  const labels: LabelType[] = _.map(children,
    (labelId) => _.cloneDeep(item.labels[labelId]))

  _.forEach(labels, (label) => {
    label.parent = newLabelId
    // sync the category and attributes of the labels
    label.category = _.cloneDeep(newLabel.category)
    label.attributes = _.cloneDeep(newLabel.attributes)
  })

  // update track information
  let tracks = state.task.tracks
  let trackId = -1
  for (const label of labels) {
    trackId = label.track
    if (trackId >= 0) break
  }
  if (trackId >= 0) {
    newLabel.track = trackId
    let track = tracks[trackId]
    track = updateObject(track, { labels: updateObject(
      track.labels, { [item.index]: newLabelId })})
    tracks = updateObject(tracks, { [trackId]: track })
  }

  // update the item
  item = updateObject(item, {
    labels: updateObject(item.labels, _.zipObject(children, labels))})
  const items = updateListItem(state.task.items, item.index, item)
  const task = updateObject(state.task, { items, tracks })
  return { ...state, task }
}

/**
 * Update the user selection
 * @param {State} state
 * @param {types.ChangeSelectAction} action
 */
export function changeSelect (
    state: State, action: types.ChangeSelectAction): State {
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

/**
 * Delete labels from one item
 * @param item
 * @param labelIds
 */
function deleteLabelsFromItem (
  item: ItemType, labelIds: number[]): [ItemType, LabelType[]] {
  let labels = item.labels
  const deletedLabels = pickObject(item.labels, labelIds)

  // find related labels and shapes
  const updatedLabels: {[key: number]: LabelType} = {}
  const updatedShapes: { [key: number]: IndexedShapeType } = {}
  const deletedShapes: { [key: number]: IndexedShapeType } = {}
  _.forEach(deletedLabels, (label) => {
    if (label.parent >= 0) {
      // TODO: consider multiple level parenting
      const parentLabel = _.cloneDeep(labels[label.parent])
      parentLabel.children = removeListItems(parentLabel.children, [label.id])
      updatedLabels[parentLabel.id] = parentLabel
    }
    label.shapes.forEach((shapeId) => {
      let shape = item.shapes[shapeId]
      shape = updateObject(
          shape, { label: removeListItems(shape.label, [label.id]) })
      updatedShapes[shape.id] = shape
    })
  })
  // remove widow labels
  _.forEach(updatedLabels, (label) => {
    if (label.children.length === 0) {
      deletedLabels[label.id] = label
    }
  })
  // remove orphan shapes
  _.forEach(updatedShapes, (shape) => {
    if (shape.label.length === 0) {
      deletedShapes[shape.id] = shape
    }
  })

  labels = removeObjectFields(updateObject(
    item.labels, updatedLabels), getObjectKeys(deletedLabels))
  const shapes = removeObjectFields(updateObject(
    item.shapes, updatedShapes), getObjectKeys(deletedShapes))
  return [{ ...item, labels, shapes }, _.values(deletedLabels)]
}

/**
 * Delete labels from one item
 * @param item
 * @param labelIds
 */
function deleteLabelsFromItems (
  items: ItemType[], labelIds: number[][]): [ItemType[], LabelType[]] {
  items = [...items]
  const deletedLabels: LabelType[] = []
  items.forEach((item, index) => {
    const [newItem, labels] = deleteLabelsFromItem(item, labelIds[index])
    items[index] = newItem
    deletedLabels.push(...labels)
  })
  return [items, deletedLabels]
}

/**
 * Delete labels from tracks
 * @param tracks
 * @param labels
 */
function deleteLabelsFromTracks (
  tracks: TrackMapType, labels: LabelType[]
): TrackMapType {
  tracks = { ...tracks }
  const deletedLabelsByTrack: TrackMapType = {}
  for (const l of labels) {
    if (!deletedLabelsByTrack.hasOwnProperty(l.track)) {
      deletedLabelsByTrack[l.track] = makeTrack(l.track)
    }
    deletedLabelsByTrack[l.track].labels[l.item] = l.id
  }
  _.forEach(deletedLabelsByTrack, (track, id) => {
    const trackId = Number(id)
    if (trackId >= 0 && trackId in tracks) {
      const oldTrack = tracks[trackId]
      const newTrack = updateObject(oldTrack,
        {
          labels: removeObjectFields(
            oldTrack.labels, getObjectKeys(track.labels))
        })
      if (_.size(newTrack.labels) > 0) {
        tracks[trackId] = newTrack
      } else {
        delete tracks[trackId]
      }
    }
  })
  return tracks
}

/**
 * Delete labels action
 * @param state
 * @param action
 */
export function deleteLabels (
  state: State, action: types.DeleteLabelsAction): State {
  const [newItems, deletedLabels] = deleteLabelsFromItems(
    pickArray(state.task.items, action.itemIndices), action.labelIds)
  const items = assignToArray(
    [...state.task.items], newItems, action.itemIndices)
  const tracks = deleteLabelsFromTracks(state.task.tracks, deletedLabels)
  const task = updateObject(state.task, { items, tracks })
  // Reset selected object
  let { user } = state
  const deletedIds = new Set()
  for (const ids of action.labelIds) {
    for (const labelId of ids) {
      deletedIds.add(labelId)
    }
  }
  const newIds = []
  for (const labelId of user.select.labels) {
    if (!deletedIds.has(labelId)) {
      newIds.push(labelId)
    }
  }
  user = updateUserSelect(user, { labels: newIds })
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
