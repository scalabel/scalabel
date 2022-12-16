/**
 * Main functions for transforming state
 * NOTE: All the functions should be pure
 * Pure Function: https://en.wikipedia.org/wiki/Pure_function
 */
import { IdType } from "aws-sdk/clients/workdocs"
import _ from "lodash"

import { uid } from "../common/uid"
import md5 from "blueimp-md5"
import * as actionConsts from "../const/action"
import { LabelTypeName, ViewerConfigTypeName } from "../const/common"
import * as actionTypes from "../types/action"
import {
  INVALID_ID,
  ItemType,
  LabelType,
  LayoutType,
  PaneType,
  PointCloudViewerConfigType,
  Select,
  ShapeType,
  SplitType,
  State,
  TaskStatus,
  TaskType,
  TrackIdMap,
  TrackType,
  UserType,
  ViewerConfigType
} from "../types/state"
import { isValidId, makeLabel, makePane, makeTrack } from "./states"
import {
  assignToArray,
  pickArray,
  pickObject,
  removeListItems,
  removeObjectFields,
  updateListItem,
  updateObject
} from "./util"

/**
 * Initialize session component of state
 *
 * @param {State} state
 * @returns {State}
 */
export function initSession(state: State): State {
  // Initialize state
  let session = state.session
  const items = state.task.items
  const itemStatuses = session.itemStatuses.slice()
  for (let i = 0; i < itemStatuses.length; i++) {
    const loadedMap: { [id: number]: boolean } = {}
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
 *
 * @param {UserType} user
 * @param {Partial<Select>} pselect partial selection
 */
function updateUserSelect(user: UserType, pselect: Partial<Select>): UserType {
  const select = updateObject(user.select, pselect)
  return updateObject(user, { select })
}

/**
 * Update the task in state
 *
 * @param {State} state: current state
 * @param state
 * @param {actionTypes.UpdateTaskAction} action
 */
export function updateTask(
  state: State,
  action: actionTypes.UpdateTaskAction
): State {
  return updateObject(state, { task: _.cloneDeep(action.newTask) })
}

/**
 * Override the original state with the new state
 *
 * @param {State} state: current state
 * @param state
 * @param {actionTypes.UpdateStateAction} action
 */
export function updateState(
  state: State,
  action: actionTypes.UpdateStateAction
): State {
  return _.merge(state, _.cloneDeep(action.newState))
}

/**
 * Add new label. The ids of label and shapes will be updated according to
 * the current state.
 *
 * @param {State} state: current state
 * @param {actionTypes.AddLabelAction} action
 * @param state
 * @param itemIndex
 * @param label
 * @param shapes
 * @returns {State}
 */
export function addLabel(
  state: State,
  itemIndex: number,
  label: LabelType,
  shapes: ShapeType[] = []
): State {
  const addLabelsAction: actionTypes.AddLabelsAction = {
    actionId: uid(),
    type: actionConsts.ADD_LABELS,
    sessionId: state.session.id,
    userId: state.user.id,
    timestamp: Date.now(),
    itemIndices: [itemIndex],
    labels: [[label]],
    shapes: [[shapes]]
  }
  return addLabels(state, addLabelsAction)
}

/**
 * Delete parent label. The ids of label will be updated according to
 * the current state.
 *
 * @param {State} state: current state
 * @param {actionTypes.DeleteLabelAction} action
 * @param state
 * @param itemIndex
 * @param labelIds
 * @returns {State}
 */
export function deleteLabelsById(
  state: State,
  itemIndex: number,
  labelIds: IdType[]
): State {
  const deleteLabelsAction: actionTypes.DeleteLabelsAction = {
    type: actionConsts.DELETE_LABELS,
    actionId: uid(),
    sessionId: state.session.id,
    userId: state.user.id,
    timestamp: Date.now(),
    itemIndices: [itemIndex],
    labelIds: [labelIds]
  }
  return deleteLabels(state, deleteLabelsAction)
}

/**
 * Add news labels to one item
 *
 * @param item
 * @param taskStatus
 * @param label
 * @param shapeTypes
 * @param newLabels
 * @param shapes
 */
function addLabelsToItem(
  item: ItemType,
  taskStatus: TaskStatus,
  newLabels: LabelType[],
  shapes: ShapeType[][]
): [ItemType, LabelType[], TaskStatus] {
  newLabels = [...newLabels]
  const newLabelIds: IdType[] = []
  const newShapeIds: IdType[] = []
  const newShapes: ShapeType[] = []
  newLabels.forEach((label, index) => {
    const shapeIds = shapes[index].map((shape) => shape.id)
    const newLabelShapes = _.cloneDeep(shapes[index])
    const order = taskStatus.maxOrder + 1 + index
    const validChildren = label.children.filter((id) => isValidId(id))
    label = updateObject(label, {
      item: item.index,
      order,
      // Shapes: label.shapes.concat(shapeIds),
      children: validChildren
    })
    newLabels[index] = label
    newLabelIds.push(label.id)
    newShapes.push(...newLabelShapes)
    newShapeIds.push(...shapeIds)
  })
  const labels = updateObject(item.labels, _.zipObject(newLabelIds, newLabels))
  const allShapes = updateObject(
    item.shapes,
    _.zipObject(newShapeIds, newShapes)
  )
  item = updateObject(item, { labels, shapes: allShapes })
  taskStatus = updateObject(taskStatus, {
    maxOrder: taskStatus.maxOrder + newLabels.length
  })
  return [item, newLabels, taskStatus]
}

/**
 * Add labels to multiple items
 *
 * @param item
 * @param items
 * @param taskStatus
 * @param newLabels
 * @param shapeTypes
 * @param labelsToAdd
 * @param shapes
 */
function addLabelstoItems(
  items: ItemType[],
  taskStatus: TaskStatus,
  labelsToAdd: LabelType[][],
  shapes: ShapeType[][][]
): [ItemType[], LabelType[], TaskStatus] {
  const allNewLabels: LabelType[] = []
  items = [...items]
  items.forEach((item, index) => {
    const [newItem, newLabels, newStatus] = addLabelsToItem(
      item,
      taskStatus,
      labelsToAdd[index],
      shapes[index]
    )
    items[index] = newItem
    taskStatus = newStatus
    allNewLabels.push(...newLabels)
  })
  return [items, allNewLabels, taskStatus]
}

/**
 * Add new label. The ids of label and shapes will be updated according to
 * the current state.
 *
 * @param {State} state: current state
 * @param state
 * @param {actionTypes.AddLabelsAction} action
 * @returns {State}
 */
export function addLabels(
  state: State,
  action: actionTypes.AddLabelsAction
): State {
  let { task, user } = state
  const session = state.session
  let items = [...task.items]
  const selectedItems = pickArray(items, action.itemIndices)
  const [newItems, newLabels, status] = addLabelstoItems(
    selectedItems,
    task.status,
    action.labels,
    action.shapes
  )
  items = assignToArray(items, newItems, action.itemIndices)
  // Find the first new label in the selected item if the labels are created
  // by this session.
  if (action.sessionId === session.id) {
    for (const label of newLabels) {
      if (label.item === user.select.item) {
        if (label.children.length === 0) {
          // Skip virtual parent label
          const selectedLabels: { [index: number]: IdType[] } = {}
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
  }
  task = updateObject(task, { status, items })
  return { task, user, session }
}

/**
 * Add one track to task
 *
 * @param task
 * @param type
 * @param itemIndices
 * @param labels
 * @param shapeTypes
 * @param shapes
 */
function addTrackToTask(
  task: TaskType,
  type: string,
  itemIndices: number[],
  labels: LabelType[],
  shapes: ShapeType[][]
): [TaskType, TrackType, LabelType[]] {
  const track = makeTrack({ type, id: labels[0].track }, false)
  const labelList = labels.map((l) => [l])
  const shapeList = shapes.map((s) => [s])
  const [newItems, newLabels, status] = addLabelstoItems(
    pickArray(task.items, itemIndices),
    task.status,
    labelList,
    shapeList
  )
  const items = assignToArray(task.items, newItems, itemIndices)
  newLabels.forEach((l) => {
    track.labels[l.item] = l.id
  })
  const tracks = updateObject(task.tracks, { [track.id]: track })
  task = { ...task, items, status, tracks }
  return [task, track, newLabels]
}

/**
 * Add track action
 *
 * @param {State} state
 * @param {actionTypes.AddTrackAction} action
 */
export function addTrack(
  state: State,
  action: actionTypes.AddTrackAction
): State {
  let { user } = state
  const [task, , newLabels] = addTrackToTask(
    state.task,
    action.trackType,
    action.itemIndices,
    action.labels,
    action.shapes
  )
  // Select the label on the current item
  if (action.sessionId === state.session.id) {
    for (const l of newLabels) {
      if (l.item === user.select.item) {
        const selectedLabels: { [index: number]: IdType[] } = {}
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
 *
 * @param tracks
 * @param items
 */
function mergeTracksInItems(
  tracks: TrackType[],
  items: ItemType[]
): [TrackType, ItemType[]] | [null, []] {
  if (tracks.length === 0) {
    return [null, []]
  }

  tracks = [...tracks]
  const labelIds: IdType[][] = _.range(items.length).map(() => [])
  const props: Array<Array<Partial<LabelType>>> = _.range(items.length).map(
    () => []
  )

  const firstItem = Number(Object.keys(tracks[0].labels)[0])
  const firstLabelId = tracks[0].labels[firstItem]
  const firstLabel = items[firstItem].labels[firstLabelId]

  const prop: Partial<LabelType> = {
    track: tracks[0].id,
    category: firstLabel.category
  }

  const track = _.cloneDeep(tracks[0])
  for (let i = 1; i < tracks.length; i += 1) {
    _.forEach(tracks[i].labels, (labelId, itemIndex) => {
      const idx = Number(itemIndex)
      const item = items[idx]

      // Change current label as well as all its desendants.
      const desendants = getChildLabelIds(item, labelId, true)
      desendants.forEach((lid) => {
        labelIds[idx].push(lid)
        props[idx].push(prop)
      })
    })
    track.labels = { ...track.labels, ...tracks[i].labels }
  }
  items = changeLabelsInItems(items, labelIds, props)
  return [track, items]
}

/**
 * Merge tracks action
 *
 * @param state
 * @param action
 */
export function mergeTracks(
  state: State,
  action: actionTypes.MergeTrackAction
): State {
  let { task, session } = state
  const mergedTracks = action.trackIds.map((trackId) => task.tracks[trackId])
  const tracks = removeObjectFields(task.tracks, action.trackIds)
  const [track, items] = mergeTracksInItems(mergedTracks, task.items)

  if (track !== null && items.length > 0) {
    tracks[track.id] = track
    task = updateObject(task, { items, tracks })
  }

  session = updateObject(session, { trackLinking: false })
  return { ...state, task, session }
}

/**
 * Split tracks and items
 *
 * @param track
 * @param splitIndex
 * @param newTrackId
 * @param items
 */
function splitTrackInItems(
  track: TrackType,
  splitIndex: number,
  newTrackId: IdType,
  items: ItemType[]
): [TrackType[], ItemType[]] {
  const splitedTrack0 = makeTrack({ type: track.type, id: track.id }, false)
  const splitedTrack1 = makeTrack({ type: track.type, id: newTrackId }, false)

  const labelIds: IdType[][] = _.range(items.length).map(() => [])
  const props: Array<Array<Partial<LabelType>>> = _.range(items.length).map(
    () => []
  )

  const prop: Partial<LabelType> = {
    track: splitedTrack1.id
  }

  _.forEach(track.labels, (labelId, itemIndex) => {
    const idx = Number(itemIndex)
    const item = items[idx]

    if (idx < splitIndex) {
      splitedTrack0.labels[idx] = labelId
    } else {
      splitedTrack1.labels[idx] = labelId

      const desendants = getChildLabelIds(item, labelId, true)
      desendants.forEach((lid) => {
        labelIds[idx].push(lid)
        if (idx === splitIndex) {
          props[idx].push({ ...prop, manual: true })
        } else {
          props[idx].push({ ...prop })
        }
      })
    }
  })

  items = changeLabelsInItems(items, labelIds, props)
  return [[splitedTrack0, splitedTrack1], items]
}

/**
 * Split track action
 *
 * @param state
 * @param action
 */
export function splitTrack(
  state: State,
  action: actionTypes.SplitTrackAction
): State {
  let task = state.task
  const trackToBeSplited = task.tracks[action.trackId]
  const tracks = removeObjectFields(task.tracks, [action.trackId])
  const [splitedTracks, items] = splitTrackInItems(
    trackToBeSplited,
    action.splitIndex,
    action.newTrackId,
    task.items
  )
  splitedTracks.forEach((track) => {
    if (Object.keys(track.labels).length > 0) {
      tracks[track.id] = track
    }
  })
  task = updateObject(task, { items, tracks })
  return { ...state, task }
}

/**
 * update shapes in an item
 *
 * @param item
 * @param shapeIds
 * @param shapes
 */
function changeShapesInItem(
  item: ItemType,
  shapeIds: IdType[],
  shapes: Array<Partial<ShapeType>>
): ItemType {
  const newShapes = { ...item.shapes }
  shapeIds.forEach((shapeId, index) => {
    newShapes[shapeId] = updateObject(newShapes[shapeId], shapes[index])
    newShapes[shapeId].id = shapeId
  })
  return { ...item, shapes: newShapes }
}

/**
 * changes shapes in items
 *
 * @param items
 * @param shapeIds
 * @param shapes
 */
function changeShapesInItems(
  items: ItemType[],
  shapeIds: IdType[][],
  shapes: Array<Array<Partial<ShapeType>>>
): ItemType[] {
  items = [...items]
  items.forEach((item, index) => {
    items[index] = changeShapesInItem(item, shapeIds[index], shapes[index])
  })
  return items
}

/**
 * Change shapes action
 *
 * @param state
 * @param action
 */
export function changeShapes(
  state: State,
  action: actionTypes.ChangeShapesAction
): State {
  let task = state.task
  const user = state.user
  const shapeIds = action.shapeIds
  const newItems = changeShapesInItems(
    pickArray(task.items, action.itemIndices),
    shapeIds,
    action.shapes
  )
  const items = assignToArray(task.items, newItems, action.itemIndices)
  task = updateObject(task, { items })
  return { ...state, task, user }
}

/**
 * Change properties of labels in one item
 *
 * @param item
 * @param labels
 * @param labelIds
 * @param props
 */
function changeLabelsInItem(
  item: ItemType,
  labelIds: IdType[],
  props: Array<Partial<LabelType>>
): ItemType {
  const newLabels: { [key: string]: LabelType } = {}
  const allShapes = item.shapes
  const allDeletedShapes: IdType[] = []
  const allChangedShapes: { [key: string]: ShapeType } = {}
  labelIds.forEach((labelId, index) => {
    const children = props[index].children
    if (children !== undefined) {
      props[index].children = children.filter((id) => isValidId(id))
    }
    const oldLabel = item.labels[labelId]
    const newLabel = updateObject(oldLabel, _.cloneDeep(props[index]))
    newLabels[labelId] = newLabel
    // Find the shapes to change and delete from the old label
    const newLabelShapeIds = new Set(newLabel.shapes)
    const changedShapeIds = _.filter(
      oldLabel.shapes,
      (s) => !newLabelShapeIds.has(s)
    )
    const changedShapes = changedShapeIds.map((s) => _.cloneDeep(allShapes[s]))
    _.forEach(changedShapes, (s) => {
      s.label = removeListItems(s.label, [oldLabel.id])
      if (s.label.length === 0) {
        allDeletedShapes.push(s.id)
      } else {
        allChangedShapes[s.id] = s
      }
    })
  })
  item = updateObject(item, {
    labels: updateObject(item.labels, newLabels),
    shapes: updateObject(
      removeObjectFields(allShapes, allDeletedShapes),
      allChangedShapes
    )
  })
  return item
}

/**
 * Change properties of labels in one item
 *
 * @param items
 * @param labels
 * @param labelIds
 * @param props
 */
function changeLabelsInItems(
  items: ItemType[],
  labelIds: IdType[][],
  props: Array<Array<Partial<LabelType>>>
): ItemType[] {
  items = [...items]
  items.forEach((item, index) => {
    items[index] = changeLabelsInItem(item, labelIds[index], props[index])
  })
  return items
}

/**
 * Change labels action
 *
 * @param state
 * @param action
 */
export function changeLabels(
  state: State,
  action: actionTypes.ChangeLabelsAction
): State {
  let items = pickArray(state.task.items, action.itemIndices)
  items = changeLabelsInItems(items, action.labelIds, action.props)
  items = assignToArray(state.task.items, items, action.itemIndices)
  const task = updateObject(state.task, { items })
  return { ...state, task }
}

/**
 * Get the label id of the root of a label by tracing its ancestors
 *
 * @param item
 * @param labelId
 */
export function getRootLabelId(item: ItemType, labelId: IdType): string {
  let parent = item.labels[labelId].parent

  while (isValidId(parent)) {
    if (item.labels[parent] !== undefined) {
      labelId = parent
      parent = item.labels[labelId].parent
    } else {
      break
    }
  }
  return labelId
}

/**
 * get all linked label ids from one labelId
 *
 * @param item
 * @param labelId
 */
export function getLinkedLabelIds(item: ItemType, labelId: IdType): string[] {
  return getChildLabelIds(item, getRootLabelId(item, labelId))
}

/**
 * get all linked label ids from the root
 *
 * @param item
 * @param labelId
 * @param includeRoot
 */
function getChildLabelIds(
  item: ItemType,
  labelId: IdType,
  includeRoot = false
): string[] {
  const labelIds: IdType[] = []
  const label = item.labels[labelId]
  if (label.children.length === 0) {
    labelIds.push(labelId)
  } else {
    for (const child of label.children) {
      const childLabelIds = getChildLabelIds(item, child, includeRoot)
      for (const childLabelId of childLabelIds) {
        labelIds.push(childLabelId)
      }
      if (includeRoot) {
        labelIds.push(labelId)
      }
    }
  }
  return labelIds
}

/**
 * Get the trackId of the root of a label by tracing its ancestors
 *
 * @param item
 * @param labelId
 */
export function getRootTrackId(item: ItemType, labelId: IdType): IdType {
  let parent = item.labels[labelId].parent
  while (isValidId(parent)) {
    if (item.labels[parent] !== undefined) {
      labelId = parent
      parent = item.labels[labelId].parent
    } else {
      break
    }
  }
  return item.labels[labelId].track
}

/**
 * Create a parent label for labels in idList at item[index]
 * with a given track id. And save track and labels to state
 *
 * @param {State} state Redux state
 * @param {number} index Index of the item
 * @param {string[]} idList ID list of labels to the parent
 * @param {LabelType} label Label template for the parent label
 * @param {string} [trackId] track id of the parent label if given
 */
function createParentLabel(
  state: State,
  index: number,
  idList: string[],
  label: LabelType,
  trackId?: string
): State {
  let item = state.task.items[index]
  let tracks = state.task.tracks
  const labelsToMerge = idList.map((id) => item.labels[id])

  // Randomly generating an id for the parent label will cause inconsistency
  // between the client and the server. To rescue, we determinstically assign
  // an id to this parent label based on its children.
  const seed = labelsToMerge
    .map((l) => l.id)
    .sort()
    .join("_")
  const pid = md5(seed)

  // Make parent label
  const parentLabel: LabelType = makeLabel({ ...label, id: pid }, false)
  parentLabel.parent = INVALID_ID
  parentLabel.shapes = []
  parentLabel.children = [...idList]
  if (trackId !== undefined) {
    parentLabel.track = trackId
  }
  parentLabel.type = LabelTypeName.EMPTY
  state = addLabel(state, index, parentLabel)
  item = state.task.items[index]

  if (trackId !== undefined) {
    // Update track information
    let track = state.task.tracks[trackId]
    track = updateObject(track, {
      labels: updateObject(track.labels, { [index]: parentLabel.id })
    })
    tracks = updateObject(tracks, { [trackId]: track })
  }

  // Assign the children label properties
  const newParentLabel = item.labels[parentLabel.id]
  const rootSet = new Set(idList)
  const childrenList = idList.map((lid) => getChildLabelIds(item, lid, true))
  const desendants = ([] as string[]).concat(...childrenList)
  const newDescents = desendants.map((lid) => {
    const lbl = item.labels[lid]
    const nLabel = _.cloneDeep(lbl)
    // Only the directly linking labels shall update their parents.
    if (rootSet.has(lbl.id)) {
      nLabel.parent = parentLabel.id
    }
    nLabel.category = _.cloneDeep(newParentLabel.category)
    nLabel.attributes = _.cloneDeep(newParentLabel.attributes)
    if (trackId !== undefined) {
      nLabel.track = trackId
    }
    return nLabel
  })

  // Update the item
  item = updateObject(item, {
    labels: updateObject(item.labels, _.zipObject(desendants, newDescents))
  })
  const items = updateListItem(state.task.items, index, item)
  const task = updateObject(state.task, { items, tracks })
  return { ...state, task }
}

/**
 * Link labels on the same item
 * The new label properties are the same as label1 in action
 *
 * @param {State} state
 * @param {actionTypes.LinkLabelsAction} action
 */
export function linkLabels(
  state: State,
  action: actionTypes.LinkLabelsAction
): State {
  // No selection or only 1 item selected will not get linked
  if (action.labelIds.length < 2) {
    return state
  }
  // Add a new label to the state
  const item = state.task.items[action.itemIndex]

  // Some of the labels may be already linked. Thus we deduplicate all roots.
  const roots = action.labelIds.map((labelId) => getRootLabelId(item, labelId))
  const uniqueRoots = [...new Set(roots)]
  if (uniqueRoots.length === 1) {
    // No need to link
    return state
  }

  const baseLabel = item.labels[uniqueRoots[0]]
  const toLinkTrackIds = [
    ...new Set(
      uniqueRoots
        .map((labelId) => item.labels[labelId].track)
        .filter((trackId) => trackId !== "")
    )
  ]
  if (toLinkTrackIds.length > 1) {
    // In track mode, will only be linked between multiple tracks
    // It is impossible to have only one track
    const baseTrackId = toLinkTrackIds[0]
    const toLinkTracks = toLinkTrackIds.map((tId) => state.task.tracks[tId])
    // If multiple tracks to be linked, all of the labels should be linked.
    state.task.items.forEach((taskItem, idx) => {
      const labelIdsToMerge = toLinkTracks
        .map((track) =>
          idx in track.labels
            ? getRootLabelId(taskItem, track.labels[idx])
            : null
        )
        .filter((lbl) => lbl !== null) as string[]

      if (labelIdsToMerge.length > 0) {
        state = createParentLabel(
          state,
          idx,
          labelIdsToMerge,
          baseLabel,
          baseTrackId
        )
      }
    })
    // Unused tracks should be deleted
    state.task.tracks = removeObjectFields(
      state.task.tracks,
      toLinkTrackIds.slice(1)
    )
  } else {
    // No track. Only link labels.
    state = createParentLabel(state, action.itemIndex, uniqueRoots, baseLabel)
  }
  return { ...state }
}

/**
 * Unlink labels on the same item
 *
 * @param {State} state
 * @param {actionTypes.UnlinkLabelsAction} action
 */
export function unlinkLabels(
  state: State,
  action: actionTypes.UnlinkLabelsAction
): State {
  if (action.labelIds.length < 1) {
    return state
  }
  const deleteLabelList = []
  const labels = _.clone(state.task.items[action.itemIndex].labels)

  for (let labelId of action.labelIds) {
    let label = _.cloneDeep(labels[labelId])
    let parentId = label.parent
    let parentLabel
    label.parent = INVALID_ID
    labels[labelId] = label

    while (isValidId(parentId)) {
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
        label.parent = INVALID_ID
      }
    }
  }
  const items = updateListItem(
    state.task.items,
    action.itemIndex,
    updateObject(state.task.items[action.itemIndex], { labels })
  )
  const task = updateObject(state.task, { items })
  return deleteLabelsById(
    updateObject(state, { task }),
    action.itemIndex,
    deleteLabelList
  )
}

/**
 * Update the user selection
 *
 * @param {State} state
 * @param {actionTypes.ChangeSelectAction} action
 */
export function changeSelect(
  state: State,
  action: actionTypes.ChangeSelectAction
): State {
  // Keep selected label selected in new items in tracking mode
  if (
    state.task.config.tracking &&
    state.user.select.item !== action.select.item
  ) {
    if (action.select.labels !== undefined) {
      for (const key of Object.keys(state.user.select.labels)) {
        const index = Number(key)
        const selectedLabelIds = state.user.select.labels[index]
        const newItem =
          action.select.item !== undefined ? action.select.item : 0
        const newLabelId = selectedLabelIds
          .map((labelId) => {
            if (labelId in state.task.items[index].labels) {
              const track = state.task.items[index].labels[labelId].track
              return state.task.tracks[track].labels[newItem]
            }
            return ""
          })
          .filter(Boolean)
        if (newLabelId.length > 0) {
          if (action.select.labels === undefined) {
            action.select.labels = {}
          }
          action.select.labels[newItem] = newLabelId
        }
      }
    }
  }
  if (state.session.trackLinking) {
    for (const key of Object.keys(state.user.select.labels)) {
      const index = Number(key)
      const selectedLabelIds = state.user.select.labels[index]
      const newItem = action.select.item !== undefined ? action.select.item : 0
      if (newItem === index && state.user.select.item === action.select.item) {
        continue
      }
      const newLabelId = selectedLabelIds
        .map((labelId) => {
          if (labelId in state.task.items[index].labels) {
            const track = state.task.items[index].labels[labelId].track
            if (newItem in state.task.tracks[track].labels) {
              return state.task.tracks[track].labels[newItem]
            }
          }
          return ""
        })
        .filter(Boolean)
      if (action.select.labels === undefined) {
        action.select.labels = {}
      }
      if (newLabelId.length > 0) {
        action.select.labels[newItem] = newLabelId
      } else {
        action.select.labels[index] = selectedLabelIds
      }
    }
  }
  const newSelect = updateObject(state.user.select, action.select)
  for (const key of Object.keys(newSelect.labels)) {
    const index = Number(key)
    if (newSelect.labels[index].length === 0) {
      // eslint-disable-next-line @typescript-eslint/no-dynamic-delete
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
 *
 * @param {State} state
 * @param {actionTypes.LoadItemAction} action
 * @returns {State}
 */
export function loadItem(
  state: State,
  action: actionTypes.LoadItemAction
): State {
  const itemIndex = action.itemIndex
  let session = state.session
  session = updateObject(session, {
    itemStatuses: updateListItem(
      session.itemStatuses,
      itemIndex,
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
 *
 * @param item
 * @param labelIds
 * @returns new item and the deleted labels
 */
function deleteLabelsFromItem(
  item: ItemType,
  labelIds: IdType[]
): [ItemType, LabelType[]] {
  let labels = item.labels

  // Collect all desendants
  const childrenList = labelIds.map((lid) => getChildLabelIds(item, lid, true))
  const desendants = ([] as string[]).concat(...childrenList)
  const deletedLabels = pickObject(item.labels, desendants)

  // Find related labels and shapes
  const updatedLabels: { [key: string]: LabelType } = {}
  const updatedShapes: { [key: string]: ShapeType } = {}
  const deletedShapes: { [key: string]: ShapeType } = {}
  _.forEach(deletedLabels, (label) => {
    if (isValidId(label.parent)) {
      // TODO: consider multiple level parenting
      const parentLabel =
        label.parent in updatedLabels
          ? updatedLabels[label.parent]
          : _.cloneDeep(labels[label.parent])
      parentLabel.children = removeListItems(parentLabel.children, [label.id])
      updatedLabels[parentLabel.id] = parentLabel
    }
    label.shapes.forEach((shapeId: IdType) => {
      if (!(shapeId in updatedShapes)) {
        updatedShapes[shapeId] = item.shapes[shapeId]
      }
      let shape = updatedShapes[shapeId]
      shape = updateObject(shape, {
        label: removeListItems(shape.label, [label.id])
      })
      updatedShapes[shapeId] = shape
    })
  })
  // Remove widow labels if label type is empty
  _.forEach(updatedLabels, (label) => {
    if (label.type === LabelTypeName.EMPTY && label.children.length === 0) {
      deletedLabels[label.id] = label
    }
  })
  // Remove orphan shapes
  _.forEach(updatedShapes, (shape) => {
    if (shape.label.length === 0) {
      deletedShapes[shape.id] = shape
    }
  })

  labels = removeObjectFields(
    updateObject(item.labels, updatedLabels),
    _.keys(deletedLabels)
  )
  const shapes = removeObjectFields(
    updateObject(item.shapes, updatedShapes),
    _.keys(deletedShapes)
  )
  return [{ ...item, labels, shapes }, _.values(deletedLabels)]
}

/**
 * Delete labels from one item
 *
 * @param item
 * @param items
 * @param labelIds
 */
function deleteLabelsFromItems(
  items: ItemType[],
  labelIds: IdType[][]
): [ItemType[], LabelType[]] {
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
 *
 * @param tracks
 * @param labels
 */
function deleteLabelsFromTracks(
  tracks: TrackIdMap,
  labels: LabelType[]
): TrackIdMap {
  tracks = { ...tracks }
  const deletedLabelsByTrack: TrackIdMap = {}
  for (const l of labels) {
    if (!(l.track in deletedLabelsByTrack)) {
      // Create a temporary track to contain the labels to delete
      deletedLabelsByTrack[l.track] = makeTrack({ id: l.track }, false)
    }
    deletedLabelsByTrack[l.track].labels[l.item] = l.id
  }
  _.forEach(deletedLabelsByTrack, (track, trackId) => {
    if (isValidId(trackId) && trackId in tracks) {
      const oldTrack = tracks[trackId]
      const newTrack = updateObject(oldTrack, {
        labels: removeObjectFields(
          oldTrack.labels,
          _.keys(track.labels).map((k) => Number(k))
        )
      })
      if (_.size(newTrack.labels) > 0) {
        tracks[trackId] = newTrack
      } else {
        // eslint-disable-next-line @typescript-eslint/no-dynamic-delete
        delete tracks[trackId]
      }
    }
  })
  return tracks
}

/**
 * Delete labels action
 *
 * @param state
 * @param action
 */
export function deleteLabels(
  state: State,
  action: actionTypes.DeleteLabelsAction
): State {
  const [newItems, deletedLabels] = deleteLabelsFromItems(
    pickArray(state.task.items, action.itemIndices),
    action.labelIds
  )
  const items = assignToArray(
    [...state.task.items],
    newItems,
    action.itemIndices
  )
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
  const newSelectedLabels: { [index: number]: string[] } = []
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
 *
 * @param {State} state
 * @param {number} _labelId
 * @param {object} _attributeOptions
 * @returns {State}
 */
export function changeAttribute(state: State): State {
  return state
}

/**
 * Notify all the subscribers to update. it is an no-op now.
 *
 * @param {State} state
 * @returns {State}
 */
export function updateAll(state: State): State {
  return state
}

/**
 * Add new viewer config to state
 *
 * @param state
 * @param action
 */
export function addViewerConfig(
  state: State,
  action: actionTypes.AddViewerConfigAction
): State {
  const newViewerConfigs = {
    ...state.user.viewerConfigs,
    [action.id]: action.config
  }
  const newUser = updateObject(state.user, { viewerConfigs: newViewerConfigs })
  return updateObject(state, { user: newUser })
}

/**
 * Handle different synchronization modes for different viewer configs
 *
 * @param modifiedConfig
 * @param config
 */
function handleViewerSynchronization(
  modifiedConfig: Readonly<ViewerConfigType>,
  config: ViewerConfigType
): ViewerConfigType {
  config = updateObject(config, { synchronized: modifiedConfig.synchronized })
  if (modifiedConfig.synchronized) {
    switch (config.type) {
      case ViewerConfigTypeName.POINT_CLOUD: {
        const newTarget = (modifiedConfig as PointCloudViewerConfigType).target
        const oldTarget = (config as PointCloudViewerConfigType).target
        const position = (config as PointCloudViewerConfigType).position
        config = updateObject(config as PointCloudViewerConfigType, {
          position: {
            x: position.x - oldTarget.x + newTarget.x,
            y: position.y - oldTarget.y + newTarget.y,
            z: position.z - oldTarget.z + newTarget.z
          },
          target: { ...newTarget }
        })
        break
      }
    }
  }
  return config
}

/**
 * Update viewer configs in state w/ fields in action
 *
 * @param state
 * @param action
 */
export function changeViewerConfig(
  state: State,
  action: actionTypes.ChangeViewerConfigAction
): State {
  if (action.viewerId in state.user.viewerConfigs) {
    const oldConfig = state.user.viewerConfigs[action.viewerId]
    const newViewerConfig =
      action.config.type === oldConfig.type
        ? updateObject(oldConfig, action.config)
        : _.cloneDeep(action.config)
    const updatedViewerConfigs = { [action.viewerId]: newViewerConfig }
    for (const key of Object.keys(state.user.viewerConfigs)) {
      const id = Number(key)
      if (
        state.user.viewerConfigs[id].type === newViewerConfig.type &&
        id !== action.viewerId
      ) {
        const newConfig = handleViewerSynchronization(
          action.config,
          state.user.viewerConfigs[id]
        )
        updatedViewerConfigs[id] = newConfig
      }
    }
    const viewerConfigs = updateObject(
      state.user.viewerConfigs,
      updatedViewerConfigs
    )
    state = updateObject(state, {
      user: updateObject(state.user, { viewerConfigs })
    })
  }
  return state
}

/**
 * Propagate hidden flag from starting pane upward through the tree
 * A non-leaf pane is hidden iff both of its children are hidden
 * TODO: this is not functional now
 *
 * @param paneId
 */
function propagateHiddenPane(
  paneId: number,
  panes: { [id: number]: PaneType }
): void {
  let pane = panes[paneId]
  while (pane.parent >= 0) {
    const parent = panes[pane.parent]
    if (parent.child1 !== undefined && parent.child2 !== undefined) {
      // Set pane to be hidden if both children are hidden
      const hide = panes[parent.child1].hide && panes[parent.child2].hide
      panes[pane.parent] = updateObject(parent, { hide })
      pane = panes[pane.parent]
    } else {
      break
    }
  }
}

/**
 * Update existing pane
 *
 * @param state
 * @param action
 */
export function updatePane(
  state: State,
  action: actionTypes.UpdatePaneAction
): State {
  const panes = state.user.layout.panes

  if (!(action.pane in panes)) {
    return state
  }

  const newPane = updateObject(panes[action.pane], action.props)

  const newPanes = updateObject(panes, {
    [action.pane]: newPane
  })

  propagateHiddenPane(newPane.id, newPanes)

  const newLayout = updateObject(state.user.layout, { panes: newPanes })

  return updateObject(state, {
    user: updateObject(state.user, { layout: newLayout })
  })
}

/**
 * Update children split counts upwards to root
 *
 * @param paneId
 */
function updateSplitCounts(
  paneId: number,
  panes: { [id: number]: PaneType }
): void {
  while (paneId >= 0) {
    let pane = panes[paneId]
    let parent = panes[pane.parent]
    if (pane.parent >= 0) {
      parent = updateObject(parent, {
        numHorizontalChildren: pane.numHorizontalChildren,
        numVerticalChildren: pane.numVerticalChildren
      })
      if (pane.split === SplitType.HORIZONTAL) {
        parent.numHorizontalChildren++
      } else if (pane.split === SplitType.VERTICAL) {
        parent.numVerticalChildren++
      }
      panes[parent.id] = parent
    } else {
      break
    }
    pane = parent
    paneId = pane.id
  }
}

/**
 * Split existing pane into half
 *
 * @param state
 * @param action
 */
export function splitPane(
  state: State,
  action: actionTypes.SplitPaneAction
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
  const child1 = makePane(oldPane.viewerId, child1Id, oldPane.id)
  const child2 = makePane(newViewerConfigId, child2Id, oldPane.id)

  const newPane = updateObject(oldPane, {
    viewerId: -1,
    split: action.split,
    child1: child1Id,
    child2: child2Id
  })

  const newViewerConfigs = updateObject(state.user.viewerConfigs, {
    [action.viewerId]: updateObject(oldViewerConfig, { pane: child1Id }),
    [newViewerConfigId]: newViewerConfig
  })

  const newPanes = updateObject(state.user.layout.panes, {
    [newPane.id]: newPane,
    [child1Id]: child1,
    [child2Id]: child2
  })

  updateSplitCounts(newPane.id, newPanes)

  const newLayout = updateObject(state.user.layout, {
    maxViewerConfigId: newViewerConfigId,
    maxPaneId: child2Id,
    panes: newPanes
  })

  return updateObject(state, {
    user: updateObject(state.user, {
      viewerConfigs: newViewerConfigs,
      layout: newLayout
    })
  })
}

/**
 * delete pane from state
 *
 * @param state
 * @param action
 */
export function deletePane(
  state: State,
  action: actionTypes.DeletePaneAction
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

  // Get id of the child that is not the pane to be deleted
  let newLeafId: number = -1
  if (parent.child1 === action.pane && parent.child2 !== undefined) {
    newLeafId = parent.child2
  } else if (parent.child2 === action.pane && parent.child1 !== undefined) {
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

  // eslint-disable-next-line @typescript-eslint/no-dynamic-delete
  delete newPanes[parentId]
  // eslint-disable-next-line @typescript-eslint/no-dynamic-delete
  delete newPanes[action.pane]

  newPanes[newLeafId] = updateObject(panes[newLeafId], { parent: newParentId })
  updateSplitCounts(newLeafId, newPanes)

  const updateParams: Partial<LayoutType> = { panes: newPanes }

  if (parentId === state.user.layout.rootPane) {
    updateParams.rootPane = newLeafId
  }

  return updateObject(state, {
    user: updateObject(state.user, {
      layout: updateObject(state.user.layout, updateParams)
    })
  })
}

/**
 * adds a new submission
 *
 * @param state
 * @param action
 */
export function submit(state: State, action: actionTypes.SubmitAction): State {
  const submissions = [
    ...state.task.progress.submissions,
    _.cloneDeep(action.submitData)
  ]
  const newProgress = updateObject(state.task.progress, {
    submissions
  })
  const newTask = updateObject(state.task, {
    progress: newProgress
  })
  return updateObject(state, {
    task: newTask
  })
}

/**
 * Start to link track.
 *
 * @param state Previous state
 */
export function startLinkTrack(state: State): State {
  const newSession = updateObject(state.session, {
    trackLinking: true
  })
  return updateObject(state, {
    session: newSession
  })
}

/**
 * Stop linking track.
 *
 * @param state Previous state
 */
export function stopLinkTrack(state: State): State {
  const newSession = updateObject(state.session, {
    trackLinking: false
  })
  return updateObject(state, {
    session: newSession
  })
}

/**
 * Update boundary clone.
 *
 * @param state
 * @param action
 */
export function updatePolygon2DBoundaryCloneStatus(
  state: State,
  action: actionTypes.UpdatePolygon2DBoundaryCloneStatusAction
): State {
  const { status: polygon2DBoundaryClone } = action
  const { session: oldSession } = state
  const session = updateObject(oldSession, { polygon2DBoundaryClone })
  return updateObject(state, { ...state, session })
}

/**
 * Update session status, if it should be updated
 *
 * @param state
 * @param action
 */
export function updateSessionStatus(
  state: State,
  action: actionTypes.UpdateSessionStatusAction
): State {
  const newStatus = action.newStatus

  const oldSession = state.session
  // Update mod 1000 since only nearby differences are important
  const numUpdates = (oldSession.numUpdates + 1) % 1000

  const newSession = updateObject(oldSession, {
    status: newStatus,
    numUpdates
  })
  return updateObject(state, {
    session: newSession
  })
}

/**
 * Change overlay visibility
 * 
 * @param state
 * @param action
 */
export function changeOverlays(
  state: State,
  action: actionTypes.ChangeOverlaysAction
): State {
  const newOverlayStatus = action.newOverlayStatus
  const oldSession = state.session
  const newSession = updateObject(oldSession, {
    overlayStatus: newOverlayStatus
  })
  return updateObject(state, {
    session: newSession
  })
}


/**
 * Change session mode, 'annotating' or 'selecting'
 * When in 'selecting' mode, user could drag annotation by clicking mask
 *
 * @param state
 * @param action
 */
export function changeSessionMode(
  state: State,
  action: actionTypes.ChangeSessionModeAction
): State {
  const newMode = action.newMode

  const oldSession = state.session

  const newSession = updateObject(oldSession, {
    mode: newMode
  })
  return updateObject(state, {
    session: newSession
  })
}

/**
 * Add alert to state
 *
 * @param state
 * @param action
 */
export function addAlert(
  state: State,
  action: actionTypes.AddAlertAction
): State {
  const oldSession = state.session
  const newAlerts = oldSession.alerts
  newAlerts.push(action.alert)
  const newSession = updateObject(oldSession, {
    ...state.session,
    alerts: newAlerts
  })
  return updateObject(state, {
    session: newSession
  })
}

/**
 * Remove alert from state
 *
 * @param state
 * @param action
 */
export function removeAlert(
  state: State,
  action: actionTypes.RemoveAlertAction
): State {
  const oldSession = state.session
  const newAlerts = oldSession.alerts.filter(
    (alert) => alert.id !== action.alertId
  )
  const newSession = updateObject(oldSession, {
    ...state.session,
    alerts: newAlerts
  })
  return updateObject(state, {
    session: newSession
  })
}

/**
 * Toggle ground plane
 *
 * @param state
 * @param action
 */
export function toggleGroundPlane(state: State): State {
  const oldInfo3D = state.session.info3D
  const newInfo3D = updateObject(oldInfo3D, {
    ...oldInfo3D,
    showGroundPlane: !oldInfo3D.showGroundPlane
  })
  const oldSession = state.session
  const newSession = updateObject(oldSession, {
    ...oldSession,
    info3D: newInfo3D
  })
  return updateObject(state, {
    session: newSession
  })
}
