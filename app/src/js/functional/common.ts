/**
 * Main functions for transforming state
 * NOTE: All the functions should be pure
 * Pure Function: https://en.wikipedia.org/wiki/Pure_function
 */
import _ from 'lodash'
import * as types from '../action/types'
import { LabelTypeName, ViewerConfigTypeName } from '../common/types'
import { makeIndexedShape, makeLabel, makePane, makeTrack } from './states'
import {
  IndexedShapeType,
  ItemType,
  LabelType,
  LayoutType,
  PartialIndexedShapeType,
  PartialLabelType,
  PointCloudViewerConfigType,
  Select,
  ShapeType,
  State,
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
function addLabelsAndShapesToItem (
    item: ItemType,
    labels: LabelType[],
    indexedShapes: IndexedShapeType[]
): ItemType {
  // Check for invalid id's.
  // All id's should be assigned before calling this function
  if (labels.some((label) => label.id < 0)) {
    throw new Error('Trying to add labels with negative id')
  }
  if (indexedShapes.some((indexedShape) => indexedShape.id < 0)) {
    throw new Error('Trying to add shapes with negative id')
  }
  for (const label of labels) {
    if (label.shapes.some((shapeId) => shapeId < 0)) {
      throw new Error(
        `Trying to add label with id ${label.id} with negative shape id`
      )
    }
    if (label.children.some((childId) => childId < 0)) {
      throw new Error(
        `Trying to add label with id ${label.id} with negative child id`
      )
    }
  }

  const newLabels: {[id: number]: LabelType} = {}
  const newIndexedShapes: {[id: number]: IndexedShapeType} = {}
  for (const label of labels) {
    newLabels[label.id] = _.cloneDeep(label)
  }
  for (const indexedShape of indexedShapes) {
    newIndexedShapes[indexedShape.id] = _.cloneDeep(indexedShape)
  }

  item = updateObject(item,
    {
      labels: updateObject(item.labels, newLabels),
      indexedShapes: updateObject(item.indexedShapes, newIndexedShapes)
    }
  )
  return item
}

/**
 * Add labels to multiple items
 * @param item
 * @param taskStatus
 * @param newLabels
 * @param shapeTypes
 * @param shapes
 */
function addLabelsAndShapestoItems (
  items: ItemType[],
  labelsToAdd: LabelType[][],
  indexedShapes: IndexedShapeType[][]
): ItemType[] {
  items = [...items]
  items.forEach((item, index) => {
    const newItem = addLabelsAndShapesToItem(
      item,
      labelsToAdd[index],
      indexedShapes[index]
    )
    items[index] = newItem
  })
  return items
}

/**
 * Update state with new labels, shapes & tracks.
 * Negative id's represent newly created objects
 * which must be assigned valid id's before being added to state
 */
export function updateLabels (
  state: State,
  action: types.UpdateLabelsAction
): State {
  let { task, user } = state
  const session = state.session

  // Split inputs into new labels and updated labels
  const newLabels: LabelType[][] = []
  const newIndexedShapes: IndexedShapeType[][] = []

  const updatedLabels = []
  const updatedIndexedShapes = []

  // Map temporary ids to valid ids
  const labelIdMap: { [temporaryId: number]: number } = {}
  const shapeIdMap: { [temporaryId: number]: number } = {}
  let maxLabelId = task.status.maxLabelId
  let maxShapeId = task.status.maxShapeId

  for (const itemLabels of action.labels) {
    const [newItemLabels, updatedItemLabels] = _.partition(
      itemLabels, (label) => label.id < 0
    )
    newLabels.push(newItemLabels.map((label) => {
      const newLabel = makeLabel(label)
      const newId = maxLabelId + 1
      labelIdMap[label.id] = newId
      newLabel.id = newId
      maxLabelId++
      return newLabel
    }))
    updatedLabels.push(updatedItemLabels)
  }

  for (const itemIndexedShapes of action.indexedShapes) {
    const [newItemIndexedShapes, updatedItemIndexedShapes] = _.partition(
      itemIndexedShapes, (indexedShape) => indexedShape.id < 0
    )
    newIndexedShapes.push(newItemIndexedShapes.map((indexedShape) => {
      if (
        !indexedShape.id ||
        !indexedShape.labels ||
        !indexedShape.type ||
        !indexedShape.shape
      ) {
        throw new Error('New indexed shape with uninitialized fields')
      }
      const newIndexedShape = makeIndexedShape(
        indexedShape.id,
        indexedShape.item,
        indexedShape.labels,
        indexedShape.type,
        indexedShape.shape as ShapeType
      )
      const newId = maxShapeId + 1
      shapeIdMap[indexedShape.id] = newId
      newIndexedShape.id = newId
      maxShapeId++
      return newIndexedShape
    }))
    updatedIndexedShapes.push(updatedItemIndexedShapes)
  }

  const shapeIdReplacer = (label: PartialLabelType) =>
    label.shapes = ((label.shapes) ? label.shapes : []).map(
      (shapeId) => (shapeId < 0) ? shapeIdMap[shapeId] : shapeId
    )

  const labelIdReplacer = (indexedShape: PartialIndexedShapeType) =>
    indexedShape.labels =
      ((indexedShape.labels) ? indexedShape.labels : []).map((labelId) =>
        (labelId < 0) ? labelIdMap[labelId] : labelId
      )

  newLabels.forEach((itemLabels) => itemLabels.forEach(shapeIdReplacer))
  updatedLabels.forEach((itemLabels) => itemLabels.forEach(shapeIdReplacer))

  newIndexedShapes.forEach(
    (itemIndexedShapes) => itemIndexedShapes.forEach(labelIdReplacer)
  )
  updatedIndexedShapes.forEach(
    (itemIndexedShapes) => itemIndexedShapes.forEach(labelIdReplacer)
  )

  let maxTrackId = task.status.maxTrackId
  const newTracks = action.newTracks.map((track) => {
    if (track.id > 0) {
      throw new Error('Adding track with positive id')
    }
    const newLabelIds =
      Object.keys(track.labels).filter((key) => Number(key) < 0)
    track = makeTrack(maxTrackId, track.type, { ...track.labels })
    newLabelIds.forEach((key) => {
      const id = Number(key)
      track.labels[labelIdMap[id]] = track.labels[id]
      delete track.labels[id]
    })
    maxTrackId++
    return track
  })

  let items = [...task.items]
  const selectedItems = pickArray(items, action.itemIndices)
  let newItems = addLabelsAndShapestoItems(
    selectedItems, newLabels, newIndexedShapes
  )

  newItems = changeLabelsInItems(newItems, updatedLabels)
  newItems = changeShapesInItems(newItems, updatedIndexedShapes)
  items = assignToArray(items, newItems, action.itemIndices)
  // Find the first new label in the selected item if the labels are created
  // by this session.
  if (action.sessionId === session.id) {
    const selectedItemIndex = action.itemIndices.indexOf(user.select.item)
    if (selectedItemIndex >= 0) {
      const selectedLabels: {[index: number]: number[]} = {}
      selectedLabels[user.select.item] =
        newLabels[selectedItemIndex].map((newLabel) => newLabel.id)
      const itemNewLabels = newLabels[selectedItemIndex]
      if (itemNewLabels.length > 0) {
        user = updateUserSelect(user, {
          labels: selectedLabels,
          category: itemNewLabels[0].category[0],
          attributes: itemNewLabels[0].attributes
        })
      }
    }
  }
  const status = updateObject(task.status, { maxLabelId, maxShapeId })
  task = updateObject(task, { status, items })
  task = addTracksToTask(task, newTracks)
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
function addTracksToTask (
  task: TaskType,
  newTracks: TrackType[]
): TaskType {
  if (newTracks.some((track) => track.id < 0)) {
    throw new Error('Tried to add track with negative id')
  }
  const newTrackMap: { [id: number]: TrackType } = {}
  newTracks.forEach((track) => newTrackMap[track.id] = track)
  const tracks = updateObject(task.tracks, newTracks)
  task = { ...task, tracks }
  return task
}

/**
 * Merge tracks and items
 * @param tracks
 * @param items
 */
function mergeTracksInItems (
  tracks: TrackType[], items: ItemType[]): [TrackType, ItemType[]] {
  tracks = [...tracks]
  const props: PartialLabelType[][] = _.range(
    items.length).map(() => [])
  const track = _.cloneDeep(tracks[0])
  for (let i = 1; i < tracks.length; i += 1) {
    _.forEach(tracks[i].labels, (labelId, itemIndex) => {
      props[Number(itemIndex)].push(
        { id: labelId, item: Number(itemIndex), track: tracks[0].id }
      )
    })
    track.labels = { ...track.labels, ...tracks[i].labels }
  }
  items = changeLabelsInItems(items, props)
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
  item: ItemType,
  indexedShapes: PartialIndexedShapeType[]
): ItemType {
  const newIndexedShapes: {[key: number]: IndexedShapeType} = {}
  indexedShapes.forEach((indexedShape) => {
    const newIndexedShape = { ...item.indexedShapes[indexedShape.id] }
    if (indexedShape.type) {
      newIndexedShape.type = indexedShape.type
    }
    if (indexedShape.labels) {
      newIndexedShape.labels = indexedShape.labels
    }
    if (indexedShape.shape) {
      newIndexedShape.shape =
        updateObject(newIndexedShape.shape, indexedShape.shape)
    }
    newIndexedShapes[indexedShape.id] = newIndexedShape
  })
  return {
    ...item, indexedShapes: updateObject(item.indexedShapes, newIndexedShapes)
  }
}

/**
 * changes shapes in items
 * @param items
 * @param shapeIds
 * @param shapes
 */
function changeShapesInItems (
  items: ItemType[],
  indexedShapes: PartialIndexedShapeType[][]
): ItemType[] {
  items = [...items]
  items.forEach((item, index) => {
    items[index] = changeShapesInItem(item, indexedShapes[index])
  })
  return items
}

/**
 * Change properties of labels in one item
 * @param item
 * @param labels
 */
function changeLabelsInItem (
  item: ItemType,
  labels: PartialLabelType[]
  ): ItemType {
  const newLabels: {[key: number]: LabelType} = {}
  labels.forEach((label) => {
    if (label.children && label.children.some((childId) => childId < 0)) {
      throw new Error('Trying to update label with negative child id')
    }
    const oldLabel = item.labels[label.id]
    // avoid changing the shape field in the label
    newLabels[label.id] = updateObject(oldLabel, { ..._.cloneDeep(label) })
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
  items: ItemType[],
  labels: PartialLabelType[][]): ItemType[] {
  items = [...items]
  items.forEach((item, index) => {
    items[index] = changeLabelsInItem(item, labels[index])
  })
  return items
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
  item = addLabelsAndShapesToItem(
    state.task.items[action.itemIndex], [newLabel], []
  )

  // assign the label properties
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
      let shape = item.indexedShapes[shapeId]
      shape = updateObject(
          shape, { labels: removeListItems(shape.labels, [label.id]) })
      updatedShapes[shape.id] = shape
    })
  })
  // remove widow labels if label type is empty
  _.forEach(updatedLabels, (label) => {
    if (label.type === LabelTypeName.EMPTY && label.children.length === 0) {
      deletedLabels[label.id] = label
    }
  })
  // remove orphan shapes
  _.forEach(updatedShapes, (shape) => {
    if (shape.labels.length === 0) {
      deletedShapes[shape.id] = shape
    }
  })

  labels = removeObjectFields(updateObject(
    item.labels, updatedLabels), getObjectKeys(deletedLabels))
  const indexedShapes = removeObjectFields(updateObject(
    item.indexedShapes, updatedShapes), getObjectKeys(deletedShapes))
  return [{ ...item, labels, indexedShapes }, _.values(deletedLabels)]
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
      deletedLabelsByTrack[l.track] = makeTrack(l.track, l.type)
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
