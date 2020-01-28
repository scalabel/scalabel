/**
 * Main functions for transforming state
 * NOTE: All the functions should be pure
 * Pure Function: https://en.wikipedia.org/wiki/Pure_function
 */
import _ from 'lodash'
import * as types from '../action/types'
import { LabelTypeName, ViewerConfigTypeName } from '../common/types'
import { makeIndexedShape, makePane, makeTrack } from './states'
import {
  IndexedShapeType,
  ItemType,
  LabelType,
  LayoutType,
  PointCloudViewerConfigType,
  Select,
  ShapeType,
  State,
  TaskStatus,
  TaskType,
  TrackMapType,
  TrackType,
  UserType,
  ViewerConfigType
} from './types'
import {
  assignToArray,
  getObjectKeys,
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
  const items = state.task.items
  const itemStatuses = session.itemStatuses.slice()
  for (let i = 0; i < itemStatuses.length; i++) {
    const loadedMap: {[id: number]: boolean} = {}
    for (const key of Object.keys(items[i].urls)) {
      const sensorId = Number(key)
      loadedMap[sensorId] = false
    }
    itemStatuses[i] = updateObject(itemStatuses[i], loadedMap)
  }
  session = updateObject(session, { itemStatuses })
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
 * Delete parent label. The ids of label will be updated according to
 * the current state.
 * @param {State} state: current state
 * @param {types.DeleteLabelAction} action
 * @return {State}
 */
export function deleteLabelsById (
  state: State, sessionId: string, itemIndex: number, labelIds: number[])
  : State {
  const deleteLabelsAction: types.DeleteLabelsAction = {
    type: types.DELETE_LABELS,
    sessionId,
    itemIndices: [itemIndex],
    labelIds: [labelIds]
  }
  return deleteLabels(state, deleteLabelsAction)
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
        const selectedLabels: {[index: number]: number[]} = {}
        selectedLabels[user.select.item] = [label.id]
        user = updateUserSelect(user, {
          labels: selectedLabels,
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
        const selectedLabels: {[index: number]: number[]} = {}
        selectedLabels[user.select.item] = [l.id]
        user = updateUserSelect(user, { labels: selectedLabels })
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
  let task = state.task
  const user = state.user
  const shapeIds = action.shapeIds
  const newItems = changeShapesInItems(
    pickArray(task.items, action.itemIndices), shapeIds, action.shapes)
  const items = assignToArray(task.items, newItems, action.itemIndices)
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
 * get all linked label ids from one labelId
 * @param item
 * @param labelId
 */
export function getLinkedLabelIds (item: ItemType, labelId: number): number[] {
  return getChildLabelIds(item, getRootLabelId(item, labelId))
}

/**
 * get all linked label ids from the root
 * @param item
 * @param labelId
 */
function getChildLabelIds (item: ItemType, labelId: number): number[] {
  const labelIds: number[] = []
  const label = item.labels[labelId]
  if (label.children.length === 0) {
    labelIds.push(labelId)
  } else {
    for (const child of label.children) {
      const childLabelIds = getChildLabelIds(item, child)
      for (const childLabelId of childLabelIds) {
        labelIds.push(childLabelId)
      }
    }
  }
  return labelIds
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
 * Link labels on the same item
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
  newLabel.type = LabelTypeName.EMPTY
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
 * Unlink labels on the same item
 * @param {State} state
 * @param {types.UnlinkLabelsAction} action
 */
export function unlinkLabels (
  state: State, action: types.UnlinkLabelsAction): State {
  if (action.labelIds.length < 1) {
    return state
  }
  const deleteLabelList = []
  const labels = _.clone(state.task.items[action.itemIndex].labels)

  for (let labelId of action.labelIds) {

    let label = _.cloneDeep(labels[labelId])
    let parentId = label.parent
    let parentLabel
    label.parent = -1
    labels[labelId] = label

    while (parentId >= 0) {
      parentLabel = _.cloneDeep(labels[parentId])
      parentLabel.children = removeListItems(parentLabel.children, [labelId])
      labels[parentId] = parentLabel

      label = parentLabel
      labelId = parentId
      parentId = label.parent

      if (label.children.length !== 0) {
        break
      } else if (label.type === LabelTypeName.EMPTY) {
        deleteLabelList.push(labelId)
      } else {
        label.parent = -1
      }
    }
  }
  const items = updateListItem(state.task.items, action.itemIndex,
    updateObject(state.task.items[action.itemIndex], { labels }))
  const task = updateObject(state.task, { items })
  return deleteLabelsById(updateObject(state, { task }),
    action.sessionId, action.itemIndex, deleteLabelList)
}

/**
 * Update the user selection
 * @param {State} state
 * @param {types.ChangeSelectAction} action
 */
export function changeSelect (
    state: State, action: types.ChangeSelectAction): State {
  const newSelect = updateObject(state.user.select, action.select)
  for (const key of Object.keys(newSelect.labels)) {
    const index = Number(key)
    if (newSelect.labels[index].length === 0) {
      delete newSelect.labels[index]
    }
  }
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
    itemStatuses:
      updateListItem(session.itemStatuses, itemIndex,
        updateObject(session.itemStatuses[itemIndex], {
          sensorDataLoaded: updateObject(
            session.itemStatuses[itemIndex].sensorDataLoaded,
            { [action.sensorId]: true }
          )
        })
      )
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
  const newSelectedLabels: {[index: number]: number[]} = []
  for (const key of Object.keys(user.select.labels)) {
    const index = Number(key)
    const newSelectedIds = []
    for (const labelId of user.select.labels[index]) {
      if (!deletedIds.has(labelId)) {
        newSelectedIds.push(labelId)
      }
    }
    if (newSelectedIds.length > 0) {
      newSelectedLabels[index] = newSelectedIds
    }
  }
  user = updateUserSelect(user, { labels: newSelectedLabels })
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
 * Add new viewer config to state
 * @param state
 * @param action
 */
export function addViewerConfig (
  state: State, action: types.AddViewerConfigAction
) {
  const newViewerConfigs = {
    ...state.user.viewerConfigs,
    [action.id]: action.config
  }
  const newUser = updateObject(
    state.user,
    { viewerConfigs: newViewerConfigs }
  )
  return updateObject(state, { user: newUser })
}

/** Handle different synchronization modes for different viewer configs */
function handleViewerSynchronization (
  modifiedConfig: Readonly<ViewerConfigType>,
  config: ViewerConfigType
): ViewerConfigType {
  config = updateObject(config, { synchronized: modifiedConfig.synchronized })
  if (modifiedConfig.synchronized) {
    switch (config.type) {
      case ViewerConfigTypeName.POINT_CLOUD:
        const newTarget = (modifiedConfig as PointCloudViewerConfigType).target
        const oldTarget = (config as PointCloudViewerConfigType).target
        const position = (config as PointCloudViewerConfigType).position
        config = updateObject(
          config as PointCloudViewerConfigType,
          {
            position: {
              x: position.x - oldTarget.x + newTarget.x,
              y: position.y - oldTarget.y + newTarget.y,
              z: position.z - oldTarget.z + newTarget.z
            },
            target: { ...newTarget }
          }
        )
        break
    }
  }
  return config
}

/**
 * Update viewer configs in state w/ fields in action
 * @param state
 * @param action
 */
export function changeViewerConfig (
  state: State, action: types.ChangeViewerConfigAction
): State {
  if (action.viewerId in state.user.viewerConfigs) {
    const oldConfig = state.user.viewerConfigs[action.viewerId]
    const newViewerConfig = (action.config.type === oldConfig.type) ?
      updateObject(
        oldConfig,
        action.config
      ) : _.cloneDeep(action.config)
    const updatedViewerConfigs = { [action.viewerId]: newViewerConfig }
    for (const key of Object.keys(state.user.viewerConfigs)) {
      const id = Number(key)
      if (
        state.user.viewerConfigs[id].type === newViewerConfig.type &&
        id !== action.viewerId
      ) {
        const newConfig = handleViewerSynchronization(
          action.config, state.user.viewerConfigs[id]
        )
        updatedViewerConfigs[id] = newConfig
      }
    }
    const viewerConfigs =
      updateObject(state.user.viewerConfigs, updatedViewerConfigs)
    state = updateObject(
      state,
      { user: updateObject(state.user, { viewerConfigs }) }
    )
  }
  return state
}

/** Update existing pane */
export function updatePane (
  state: State, action: types.UpdatePaneAction
) {
  if (!(action.pane in state.user.layout.panes)) {
    return state
  }

  const newPane = updateObject(
    state.user.layout.panes[action.pane],
    action.props
  )

  const newLayout = updateObject(
    state.user.layout,
    {
      panes: updateObject(
        state.user.layout.panes,
        {
          [action.pane]: newPane
        }
      )
    }
  )

  return updateObject(
    state,
    {
      user: updateObject(
        state.user,
        { layout: newLayout }
      )
    }
  )
}

/**
 * Split existing pane into half
 * @param state
 * @param action
 */
export function splitPane (
  state: State, action: types.SplitPaneAction
): State {
  if (!(action.pane in state.user.layout.panes)) {
    return state
  }

  const child1Id = state.user.layout.maxPaneId + 1
  const child2Id = state.user.layout.maxPaneId + 2

  const oldViewerConfig = state.user.viewerConfigs[action.viewerId]
  const newViewerConfig = _.cloneDeep(oldViewerConfig)
  newViewerConfig.pane = child2Id
  const newViewerConfigId = state.user.layout.maxViewerConfigId + 1

  const oldPane = state.user.layout.panes[action.pane]
  const child1 = makePane(
    oldPane.viewerId,
    child1Id,
    oldPane.id
  )
  const child2 = makePane(
    newViewerConfigId,
    child2Id,
    oldPane.id
  )

  const newPane = updateObject(oldPane, {
    viewerId: -1,
    split: action.split,
    child1: child1Id,
    child2: child2Id
  })

  const newViewerConfigs = updateObject(
    state.user.viewerConfigs,
    {
      [action.viewerId]: updateObject(
        oldViewerConfig,
        { pane: child1Id }
      ),
      [newViewerConfigId]: newViewerConfig
    }
  )

  const newLayout = updateObject(
    state.user.layout,
    {
      maxViewerConfigId: newViewerConfigId,
      maxPaneId: child2Id,
      panes: updateObject(
        state.user.layout.panes,
        {
          [newPane.id]: newPane,
          [child1Id]: child1,
          [child2Id]: child2
        }
      )
    }
  )

  return updateObject(
    state,
    {
      user: updateObject(
        state.user,
        { viewerConfigs: newViewerConfigs, layout: newLayout }
      )
    }
  )
}

/** delete pane from state */
export function deletePane (
  state: State, action: types.DeletePaneAction
): State {
  const panes = state.user.layout.panes
  if (action.pane === state.user.layout.rootPane || !(action.pane in panes)) {
    return state
  }

  const parentId = panes[action.pane].parent

  if (!(parentId in panes)) {
    return state
  }

  const parent = panes[parentId]

  // Shallow copy of panes and modification of dictionary
  const newPanes = { ...panes }

  // Get id of the child that is not the pane that will be deleted
  let newLeafId: number = -1
  if (parent.child1 === action.pane && parent.child2) {
    newLeafId = parent.child2
  } else if (parent.child2 === action.pane && parent.child1) {
    newLeafId = parent.child1
  } else {
    return state
  }

  if (!(newLeafId in panes)) {
    return state
  }

  let newParentId = -1
  if (parentId !== state.user.layout.rootPane) {
    newParentId = parent.parent
    if (!(newParentId in panes)) {
      return state
    }

    let newParent = panes[newParentId]

    // Flatten tree by removing old parent from the parent of the old parent and
    // replacing with the new leaf
    if (parentId === newParent.child1) {
      newParent = updateObject(newParent, { child1: newLeafId })
    } else if (parentId === newParent.child2) {
      newParent = updateObject(newParent, { child2: newLeafId })
    } else {
      return state
    }

    newPanes[newParentId] = newParent
  }

  delete newPanes[parentId]
  delete newPanes[action.pane]
  newPanes[newLeafId] = updateObject(panes[newLeafId], { parent: newParentId })
  const updateParams: Partial<LayoutType> = { panes: newPanes }

  if (parentId === state.user.layout.rootPane) {
    updateParams.rootPane = newLeafId
  }

  return updateObject(
    state,
    {
      user: updateObject(
        state.user,
        { layout: updateObject(state.user.layout, updateParams) }
      )
    }
  )
}
