import _ from "lodash"

import {
  ItemType,
  LabelIdMap,
  LabelType,
  ShapeIdMap,
  ShapeType,
  TaskType,
  TrackIdMap,
  TrackType
} from "../types/state"

/**
 * Convert the Ids in the task to string
 *
 * @param task
 */
export function taskIdToString(task: TaskType): TaskType {
  const newTask = _.clone(task)
  newTask.items = itemsIdToString(task.items)
  newTask.tracks = tracksIdToString(task.tracks)
  return newTask
}

/**
 * Convert the Ids in the tracks to string
 *
 * @param tracks
 */
function tracksIdToString(tracks: TrackIdMap): TrackIdMap {
  const newTracks: TrackIdMap = {}
  _.forEach(tracks, (track, trackId) => {
    newTracks[trackId.toString()] = trackIdToString(track)
  })
  return newTracks
}

/**
 * Convert the Ids in the track to string
 *
 * @param track
 */
function trackIdToString(track: TrackType): TrackType {
  const newTrack = { ...track }
  newTrack.id = newTrack.id.toString()
  const labels = _.clone(newTrack.labels)
  for (const i of Object.keys(labels)) {
    const index = Number(i)
    labels[index] = labels[index].toString()
  }
  newTrack.labels = labels
  return newTrack
}

/**
 * Convert the Ids in the items to string
 *
 * @param items
 */
function itemsIdToString(items: ItemType[]): ItemType[] {
  return items.map((item) => itemIdToString(item))
}

/**
 * Convert the Ids in the item to string
 *
 * @param item
 */
function itemIdToString(item: ItemType): ItemType {
  const newItem = _.clone(item)
  newItem.labels = labelsIdToString(item.labels)
  newItem.shapes = shapesIdToString(item.shapes)
  return newItem
}

/**
 * Convert the Ids in the labels to string
 *
 * @param labels
 */
function labelsIdToString(labels: LabelIdMap): LabelIdMap {
  const newLabels: LabelIdMap = {}
  _.forEach(labels, (label, labelId) => {
    newLabels[labelId.toString()] = labelIdToString(label)
  })
  return newLabels
}

/**
 * Convert the Ids in the label to string
 *
 * @param label
 */
function labelIdToString(label: LabelType): LabelType {
  const newLabel: LabelType = { ...label }
  newLabel.id = newLabel.id.toString()
  newLabel.parent = newLabel.parent.toString()
  newLabel.children = newLabel.children.map((c) => c.toString())
  newLabel.shapes = newLabel.shapes.map((c) => c.toString())
  newLabel.track = newLabel.track.toString()
  return newLabel
}

/**
 * Convert the Ids in the shapes to string
 *
 * @param shapes
 */
function shapesIdToString(shapes: ShapeIdMap): ShapeIdMap {
  const newShapes: ShapeIdMap = {}
  _.forEach(shapes, (shape, shapeId) => {
    newShapes[shapeId.toString()] = shapeIdToString(shape)
  })
  return newShapes
}

/**
 * Convert the Ids in the shape to string
 *
 * @param shape
 */
function shapeIdToString(shape: ShapeType): ShapeType {
  let newShape: ShapeType = { ...shape }
  newShape.id = shape.id.toString()
  newShape.label = shape.label.map((l) => l.toString())
  /** to be compatible with the old indexed shape type */
  if ("shape" in shape) {
    // eslint-disable-next-line @typescript-eslint/dot-notation
    const shapeInfo = shape["shape"] as ShapeType
    newShape = { ...newShape, ...shapeInfo }
  }
  if ("shape" in newShape) {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    delete (newShape as any).shape
  }
  return newShape
}
